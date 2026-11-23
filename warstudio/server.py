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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import ROOT, VERSION, ai33
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
            run_project(proj, from_step)
        except Exception:
            pass
        JOBS.task_done()


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
    return {"version": VERSION, "keys": masked_keys(), "key_names": KEY_NAMES, "styles": list_styles(),
            "settings": load_settings(), "tools": _tools(), "steps": STEPS, "queue": JOBS.qsize()}


@app.post("/api/keys")
def set_keys(k: Keys):
    save_keys(k.keys)
    return {"ok": True, "keys": masked_keys()}


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
    proj = create_project(j.title or "Untitled", j.script, j.voice_id, j.style, j.dummy)
    JOBS.put((proj, None))
    return {"id": proj.name}


@app.get("/api/jobs")
def jobs():
    out = []
    for p in sorted(PROJECTS_DIR.glob("*/project.json"), reverse=True)[:50]:
        meta = read_json(p)
        st = read_json(p.parent / "status.json", {}) or {}
        out.append({"id": p.parent.name, "title": meta.get("title"), "state": st.get("state"), "step": st.get("step"),
                    "progress": st.get("progress", 0), "has_final": (p.parent / "final.mp4").exists()})
    return out


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
    code, out = _git("pull", "--ff-only", timeout=300)
    if code != 0:
        raise HTTPException(500, "git pull failed: " + out[-400:])
    sync = ""
    if shutil.which("uv"):
        r = subprocess.run(["uv", "sync"], cwd=ROOT, capture_output=True, text=True, timeout=1200)
        sync = (r.stdout + r.stderr)[-400:]
    threading.Timer(1.0, _restart).start()
    return {"ok": True, "pull": out[-400:], "sync": sync}


def _restart():
    os.execv(sys.executable, [sys.executable] + sys.argv)


def serve(port=8765, open_browser=True):
    import uvicorn
    from .sfx import ensure_defaults
    ensure_defaults()
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
