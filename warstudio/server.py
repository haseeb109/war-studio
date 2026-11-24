"""Local GUI server: http://127.0.0.1:8765

One worker thread runs jobs one at a time (the Mac's GPU/encoder is the bottleneck anyway).
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import ROOT, VERSION, ai33, edits
from .config import (DATA_DIR, FAVORITES_FILE, PROJECTS_DIR, KEY_NAMES, list_styles, load_keys, load_settings,
                     masked_keys, save_keys, save_settings)
from .pipeline import create_project, run_project, STEPS
from .util import read_json, write_json, download

app = FastAPI(title="WarStudio")
JOBS = queue.Queue()
PREVIEW_TEXT = "At seventeen fourteen local time, a pair of F-15E Strike Eagles holds at twenty-eight thousand feet."


def _worker():
    while True:
        proj, from_step = JOBS.get()
        try:
            if from_step == "__apply__":
                _run_apply(proj)
            else:
                run_project(proj, from_step)
        except Exception:
            pass
        JOBS.task_done()


def _run_apply(proj):
    import traceback
    from .pipeline import Logger
    log = Logger(proj)
    t0 = time.time()
    log.set(state="running", step="render", progress=0, error=None, apply_started=t0)
    try:
        edits.apply(proj, log)
        log.set(state="done", step=None, progress=1, last_apply_seconds=round(time.time() - t0, 1))
    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc()[-1500:])
        log.set(state="error", error=str(e)[:500])


threading.Thread(target=_worker, daemon=True).start()


# ---------- models ----------

class Keys(BaseModel):
    keys: dict


class Settings(BaseModel):
    settings: dict


class Fav(BaseModel):
    voice_id: str
    name: str = ""
    desc: str = ""
    favorite: bool = True


class NewJob(BaseModel):
    title: str
    script: str
    voice_id: str
    style: str = "beyond_military"
    dummy: bool = False
    sources: list[str] = []          # footage sources for this video only; empty = all available


class Resume(BaseModel):
    from_step: str | None = None


# ---------- helpers ----------

def _git(*args, timeout=60):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def _favorites():
    return read_json(FAVORITES_FILE, {"favorites": [], "custom": []})


def _tools():
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout if shutil.which("ffmpeg") else ""
    return {"ffmpeg": bool(shutil.which("ffmpeg")), "videotoolbox": "h264_videotoolbox" in enc,
            "hyperframes": bool(shutil.which("hyperframes") or shutil.which("npx")), "git": bool(shutil.which("git")),
            "uv": bool(shutil.which("uv"))}


# ---------- routes ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "warstudio" / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    from .config import BRAIN_PRESETS
    return {"version": VERSION, "keys": masked_keys(), "key_names": KEY_NAMES, "styles": list_styles(),
            "brain_presets": {k: v["label"] for k, v in BRAIN_PRESETS.items()},
            "settings": load_settings(), "tools": _tools(), "steps": STEPS, "queue": JOBS.qsize()}


@app.post("/api/keys")
def set_keys(k: Keys):
    save_keys(k.keys)
    return {"ok": True, "keys": masked_keys()}


class TestReq(BaseModel):
    key_name: str
    key: str = ""


@app.post("/api/keys/test")
def test_key(t: TestReq):
    from . import connect
    ok, msg = connect.test(t.key_name, t.key.strip() or None)
    return {"ok": ok, "message": msg}


@app.post("/api/settings")
def set_settings(s: Settings):
    return {"ok": True, "settings": save_settings(s.settings)}


@app.get("/api/voices")
def voices(refresh: bool = False):
    cache = DATA_DIR / "voices_cache.json"
    data = read_json(cache)
    if refresh or not data:
        vs, source = ai33.list_voices(load_keys().get("AI33_API_KEY"))
        data = {"voices": vs, "source": source, "time": time.time()}
        write_json(cache, data)
    fav = _favorites()
    favs = {f["voice_id"] for f in fav["favorites"]}
    allv = fav["custom"] + [v for v in data["voices"] if v["voice_id"] not in {c["voice_id"] for c in fav["custom"]}]
    for v in allv:
        v["favorite"] = v["voice_id"] in favs
    allv.sort(key=lambda v: (not v["favorite"], v["name"].lower()))
    return {"voices": allv, "source": data["source"]}


@app.post("/api/voices/favorite")
def favorite(f: Fav):
    fav = _favorites()
    vid = ai33.normalize_voice_id(f.voice_id)
    fav["favorites"] = [x for x in fav["favorites"] if x["voice_id"] != vid]
    if f.favorite:
        fav["favorites"].append({"voice_id": vid, "name": f.name})
    write_json(FAVORITES_FILE, fav)
    return {"ok": True}


@app.post("/api/voices/custom")
def custom_voice(f: Fav):
    fav = _favorites()
    vid = ai33.normalize_voice_id(f.voice_id)
    fav["custom"] = [x for x in fav["custom"] if x["voice_id"] != vid] + [{"voice_id": vid, "name": f.name or vid, "desc": f.desc or "custom", "preview_url": ""}]
    write_json(FAVORITES_FILE, fav)
    return {"ok": True}


@app.post("/api/voices/preview")
def voice_preview(f: Fav):
    """Generate (once, then cached) a short sample of the voice through ai33."""
    vid = ai33.normalize_voice_id(f.voice_id)
    out = DATA_DIR / "voice_previews" / f"{vid}.mp3"
    if not out.exists():
        key = load_keys().get("AI33_API_KEY")
        if not key:
            raise HTTPException(400, "AI33_API_KEY missing (Settings tab)")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            ai33.tts_many(key, [(out, PREVIEW_TEXT)], vid, load_settings().get("tts_model"), 1, timeout=120)
        except Exception as e:
            raise HTTPException(500, str(e)[:300])
    return {"url": f"/api/voices/preview/{vid}.mp3"}


@app.get("/api/voices/preview/{name}")
def voice_preview_file(name: str):
    p = DATA_DIR / "voice_previews" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/mpeg")


@app.post("/api/jobs")
def new_job(j: NewJob):
    if not j.script.strip():
        raise HTTPException(400, "script is empty")
    from .steps.search import SOURCES, source_ready
    from .config import load_keys, load_settings
    bad = [s for s in j.sources if s not in SOURCES]
    if bad:
        raise HTTPException(400, f"unknown footage source: {', '.join(bad)}")
    if j.sources:
        keys, settings = load_keys(), dict(load_settings(), **{f"use_{s}": True for s in j.sources})
        if not any(source_ready(s, keys, settings) for s in j.sources):
            raise HTTPException(400, "none of the selected footage sources has an API key — add it in Settings")
    proj = create_project(j.title or "Untitled", j.script, j.voice_id, j.style, j.dummy, j.sources)
    JOBS.put((proj, None))
    return {"id": proj.name}


@app.get("/api/jobs")
def jobs():
    out = []
    for p in sorted(PROJECTS_DIR.glob("*/project.json"), reverse=True)[:200]:
        meta = read_json(p) or {}
        st = read_json(p.parent / "status.json", {}) or {}
        qc = read_json(p.parent / "qc.json", {}) or {}
        rv = read_json(p.parent / "review.json", {}) or {}
        out.append({"id": p.parent.name, "title": meta.get("title"), "state": st.get("state"), "step": st.get("step"),
                    "progress": st.get("progress", 0), "has_final": (p.parent / "final.mp4").exists(),
                    "created": meta.get("created"), "seconds": qc.get("video_seconds"), "style": meta.get("style"),
                    "no_footage": len(qc.get("beats_without_footage", [])), "review": rv.get("state", "unreviewed"),
                    "total_seconds": st.get("total_seconds"), "run_started": st.get("run_started"),
                    "dummy": meta.get("dummy", False)})
    return out


@app.get("/api/jobs/{jid}/poster")
def poster(jid: str):
    proj = PROJECTS_DIR / Path(jid).name
    final = proj / "final.mp4"
    if not final.exists():
        raise HTTPException(404)
    out = proj / "poster.jpg"
    if not out.exists() or out.stat().st_mtime < final.stat().st_mtime:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "3", "-i", str(final), "-frames:v", "1",
                        "-vf", "scale=480:-1", str(out)], capture_output=True)
    if not out.exists():
        raise HTTPException(404)
    return FileResponse(out, media_type="image/jpeg")


# ---------- review editor ----------

class EditReq(BaseModel):
    action: str
    bid: int | None = None
    sid: int | None = None
    data: dict = {}


class SearchReq(BaseModel):
    query: str


class ReviewedReq(BaseModel):
    reviewed: bool = True


def _proj(jid):
    p = PROJECTS_DIR / Path(jid).name
    if not (p / "project.json").exists():
        raise HTTPException(404, "video not found")
    return p


@app.get("/api/review/{jid}")
def review(jid: str):
    proj = _proj(jid)
    if not (proj / "beats.json").exists():
        raise HTTPException(400, "this video has no timeline yet")
    d = edits.review_data(proj)
    d["status"] = (read_json(proj / "status.json", {}) or {}).get("state")
    d["final_mtime"] = int((proj / "final.mp4").stat().st_mtime) if (proj / "final.mp4").exists() else 0
    return d


@app.post("/api/review/{jid}/edit")
def review_edit(jid: str, r: EditReq):
    proj = _proj(jid)
    x = r.data or {}
    try:
        if r.action == "clip":
            edits.swap_clip(proj, r.bid, x.get("candidate") or {})
        elif r.action == "offset":
            edits.set_offset(proj, r.bid, float(x.get("start", 0)))
        elif r.action == "grade":
            edits.set_grade(proj, r.bid, x.get("footage_class", ""))
        elif r.action == "overlay":
            edits.set_overlay(proj, r.bid, x.get("kind", "measure"), x.get("line1", ""), x.get("line2", ""))
        elif r.action == "overlay_remove":
            edits.remove_overlay(proj, r.bid)
        elif r.action == "sfx_add":
            edits.sfx_add(proj, r.bid, x.get("cat", ""), float(x.get("offset", 0.12)))
        elif r.action == "sfx_remove":
            edits.sfx_remove(proj, int(x.get("index", -1)))
        elif r.action == "narration":
            return {"ok": True, "result": edits.edit_narration(proj, r.sid, x.get("text", ""))}
        else:
            raise HTTPException(400, f"unknown action {r.action}")
    except edits.EditError as e:
        raise HTTPException(400, str(e))
    except ai33.Ai33Error as e:
        raise HTTPException(502, str(e))
    return {"ok": True}


@app.post("/api/review/{jid}/search")
def review_search(jid: str, r: SearchReq):
    _proj(jid)
    if not r.query.strip():
        raise HTTPException(400, "type something to search")
    try:
        return {"results": edits.search(_proj(jid), r.query.strip())}
    except edits.EditError as e:
        raise HTTPException(502, str(e))


@app.post("/api/review/{jid}/upload")
async def review_upload(jid: str, bid: int, name: str, request: Request):
    proj = _proj(jid)
    tmp = proj / "review" / f"upload_{int(time.time())}_{Path(name).name}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    try:
        from .config import load_style
        style = load_style(read_json(proj / "project.json")["style"])
        await __import__("anyio").to_thread.run_sync(lambda: edits.use_upload(proj, bid, tmp, Path(name).name, style))
    except edits.EditError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(400, "could not read this video file: " + str(e)[-200:])
    finally:
        tmp.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/review/{jid}/undo")
def review_undo(jid: str):
    try:
        return {"ok": True, "undone": edits.undo(_proj(jid))}
    except edits.EditError as e:
        raise HTTPException(400, str(e))


@app.post("/api/review/{jid}/apply")
def review_apply(jid: str):
    proj = _proj(jid)
    rv = read_json(proj / "review.json", {}) or {}
    if not rv.get("dirty") and not rv.get("audio_dirty"):
        raise HTTPException(400, "no changes to apply")
    st = read_json(proj / "status.json", {}) or {}
    if st.get("state") in ("running", "queued"):
        raise HTTPException(409, "already processing")
    st["state"] = "queued"
    write_json(proj / "status.json", st)
    JOBS.put((proj, "__apply__"))
    return {"ok": True}


@app.post("/api/review/{jid}/reviewed")
def review_mark(jid: str, r: ReviewedReq):
    return {"ok": True, "state": edits.set_reviewed(_proj(jid), r.reviewed)}


@app.get("/api/review/{jid}/thumb/{bid}")
def review_thumb(jid: str, bid: int):
    proj = _proj(jid)
    seg = proj / "segments" / f"seg_{bid:05d}.mp4"
    if not seg.exists():
        raise HTTPException(404)
    tdir = proj / "thumbs"
    tdir.mkdir(exist_ok=True)
    out = tdir / f"{bid}_{int(seg.stat().st_mtime)}.jpg"
    if not out.exists():
        for old in tdir.glob(f"{bid}_*.jpg"):
            old.unlink()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.4", "-i", str(seg), "-frames:v", "1",
                        "-vf", "scale=320:-1", "-q:v", "5", str(out)], capture_output=True)
    if not out.exists():
        raise HTTPException(404)
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/files/{jid}/previews/{name}")
def preview_file(jid: str, name: str):
    p = _proj(jid) / "previews" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/jobs/{jid}")
def job(jid: str):
    proj = PROJECTS_DIR / Path(jid).name
    if not proj.exists():
        raise HTTPException(404)
    st = read_json(proj / "status.json", {}) or {}
    qc = read_json(proj / "qc.json")
    files = [f for f in ("final.mp4", "contact_sheet.jpg", "footage_log.csv") if (proj / f).exists()]
    return {"id": jid, "meta": read_json(proj / "project.json"), "status": st, "qc": qc, "files": files}


@app.post("/api/jobs/{jid}/resume")
def resume(jid: str, r: Resume):
    proj = PROJECTS_DIR / Path(jid).name
    if not proj.exists():
        raise HTTPException(404)
    if r.from_step and r.from_step not in STEPS:
        raise HTTPException(400, "unknown step")
    JOBS.put((proj, r.from_step))
    return {"ok": True}


@app.get("/files/{jid}/{name}")
def files(jid: str, name: str):
    p = PROJECTS_DIR / Path(jid).name / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/open/{jid}")
def open_folder(jid: str):
    p = PROJECTS_DIR / Path(jid).name
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    return {"ok": True}


DIAG = {"running": False, "results": [], "report": "", "full": False, "started": 0}


class DiagReq(BaseModel):
    full: bool = False


@app.post("/api/diagnostics")
def diag_start(d: DiagReq):
    if DIAG["running"]:
        raise HTTPException(409, "a check is already running")
    from . import diagnostics

    def work():
        DIAG.update(running=True, results=[], report="", full=d.full, started=time.time())
        try:
            _, text = diagnostics.run_all(d.full, on_update=lambda r: DIAG.update(results=[dict(x) for x in r]))
            DIAG["report"] = text
        except Exception as e:
            DIAG["report"] = f"System check crashed: {e}"
        finally:
            DIAG["running"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


@app.get("/api/diagnostics")
def diag_status():
    if not DIAG["report"] and not DIAG["running"]:
        p = DATA_DIR / "diagnostics" / "latest.txt"
        if p.exists():
            DIAG["report"] = p.read_text(encoding="utf-8")
    return {k: DIAG[k] for k in ("running", "results", "report", "full")}


@app.get("/api/diagnostics/report.txt")
def diag_report():
    p = DATA_DIR / "diagnostics" / "latest.txt"
    if not p.exists():
        raise HTTPException(404, "no report yet")
    return FileResponse(p, media_type="text/plain", filename="warstudio-report.txt")


@app.get("/api/update/check")
def update_check():
    code, out = _git("fetch", "--quiet", "origin")
    if code != 0:
        return {"ok": False, "message": out[-300:]}
    c1, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    c2, behind = _git("rev-list", "--count", f"HEAD..origin/{branch}")
    if c1 or c2 or not behind.strip().isdigit():
        return {"ok": False, "message": "cannot compare with GitHub: " + (behind or branch)[-200:]}
    _, log = _git("log", "--oneline", f"HEAD..origin/{branch}", "-n", "15")
    return {"ok": True, "behind": int(behind), "changes": log.splitlines(), "version": VERSION}


@app.post("/api/update/apply")
def update_apply():
    if JOBS.unfinished_tasks:
        raise HTTPException(409, "a video is rendering; update after it finishes")
    # uv rewrites uv.lock on the Mac (different uv version / platform); that local copy is never wanted,
    # and it blocked `git pull`. Throw it away, and park any other local edits in a stash instead of failing.
    _git("checkout", "--", "uv.lock")
    stashed = ""
    _, dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty.strip():
        c, o = _git("stash", "push", "-m", f"warstudio auto-update {time.strftime('%Y-%m-%d %H:%M')}")
        stashed = "local edits saved with git stash: " + dirty.strip().replace("\n", ", ")[:200] if c == 0 else ""
    code, out = _git("pull", "--ff-only", timeout=300)
    if code != 0:
        raise HTTPException(500, "git pull failed: " + out[-400:])
    sync = ""
    if shutil.which("uv"):
        r = subprocess.run(["uv", "sync", "--frozen"], cwd=ROOT, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0:
            r = subprocess.run(["uv", "sync"], cwd=ROOT, capture_output=True, text=True, timeout=1200)
        sync = (r.stdout + r.stderr)[-400:]
        _git("checkout", "--", "uv.lock")
    threading.Timer(1.0, _restart).start()
    return {"ok": True, "pull": out[-400:], "sync": sync, "stashed": stashed}


def _restart():
    os.environ["WARSTUDIO_NO_BROWSER"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def recover_interrupted():
    """Jobs live in memory; if the app was closed mid-run, mark those videos as interrupted."""
    n = 0
    for st_path in PROJECTS_DIR.glob("*/status.json"):
        st = read_json(st_path, {}) or {}
        if st.get("state") in ("running", "queued"):
            st["state"] = "error"
            st["error"] = "interrupted (WarStudio was closed while this was processing). Use 'Re-run from step' to continue."
            st.setdefault("log", []).append("interrupted: app restarted")
            write_json(st_path, st)
            n += 1
    return n


def serve(port=8765, open_browser=True):
    import uvicorn
    from .sfx import ensure_defaults
    ensure_defaults()
    recover_interrupted()
    if open_browser and not os.environ.get("WARSTUDIO_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
