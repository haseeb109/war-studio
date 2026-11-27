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
from .config import (MUSIC_DIR, DATA_DIR, FAVORITES_FILE, PROJECTS_DIR, KEY_NAMES, list_styles, load_keys, load_settings,
                     masked_keys, save_keys, save_settings)
from .pipeline import create_project, run_project, STEPS
from .util import read_json, write_json, download

app = FastAPI(title="WarStudio")
JOBS = queue.Queue()
PREVIEW_TEXT = "At seventeen fourteen local time, a pair of F-15E Strike Eagles holds at twenty-eight thousand feet."


def _worker():
    while True:
        proj, from_step = JOBS.get()
        from .pipeline import STOP_FILE
        if (Path(proj) / STOP_FILE).exists():            # stopped while it was waiting in the queue
            (Path(proj) / STOP_FILE).unlink(missing_ok=True)
            st = read_json(Path(proj) / "status.json", {}) or {}
            st.update(state="stopped", step=None)
            st.setdefault("log", []).append(time.strftime("%H:%M:%S") + "  STOPPED by user before it started")
            write_json(Path(proj) / "status.json", st)
            JOBS.task_done()
            continue
        try:
            if from_step == "__apply__":
                _run_apply(proj)
            else:
                run_project(proj, from_step)
        except Exception:
            pass
        try:
            _maybe_retry(Path(proj))
        except Exception:
            pass
        JOBS.task_done()


RETRY_WAIT = [300, 1200, 3600]


def _maybe_retry(proj):
    """A failed video runs again by itself (5 min, then 20 min later), from where it stopped."""
    st = read_json(proj / "status.json", {}) or {}
    if st.get("state") != "error":
        return
    n, limit = int(st.get("retries") or 0), int(load_settings().get("auto_retry", 2) or 0)
    if n >= limit:
        return
    wait = RETRY_WAIT[min(n, len(RETRY_WAIT) - 1)]
    st.update(state="scheduled", scheduled_for=time.time() + wait, retries=n + 1)
    st.setdefault("log", []).append(time.strftime("%H:%M:%S") + f"  auto-retry {n + 1}/{limit} in {wait // 60} min "
                                    f"(after: {str(st.get('error') or '')[:120]})")
    write_json(proj / "status.json", st)


WORK_FILES = ["segments", "previews", "clips", "maps", "overlays", "vo", "video_only.mp4", "mix.wav", "segments.txt"]


def cleanup_project(proj):
    """Remove the work files of a finished video; final.mp4, logs, script, voice and plans stay."""
    import shutil
    freed = 0
    for name in WORK_FILES:
        p = proj / name
        if p.is_dir():
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            freed += p.stat().st_size
            p.unlink()
    st = read_json(proj / "status.json", {}) or {}
    st["cleaned"] = time.time()
    write_json(proj / "status.json", st)
    return freed


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


# several videos run at once (footage, maps, render in parallel); the voice step itself is one video at a time
for _ in range(max(1, int(load_settings().get("parallel_videos", 2) or 2))):
    threading.Thread(target=_worker, daemon=True).start()


def _scheduler():
    """Videos scheduled for later: queue each one when its time comes (survives restarts: it lives in status.json).
    Also removes the work files of videos finished more than auto_cleanup_days ago."""
    last_clean = 0
    while True:
        now = time.time()
        if now - last_clean > 3600:
            last_clean = now
            days = float(load_settings().get("auto_cleanup_days", 3) or 0)
            for st_path in PROJECTS_DIR.glob("*/status.json") if days > 0 else []:
                try:
                    st = read_json(st_path, {}) or {}
                    fin = float(st.get("run_finished") or 0)
                    if st.get("state") == "done" and not st.get("cleaned") and fin and now - fin > days * 86400:
                        cleanup_project(st_path.parent)
                except Exception:
                    pass
        for st_path in PROJECTS_DIR.glob("*/status.json"):
            try:
                st = read_json(st_path, {}) or {}
                if st.get("state") == "scheduled" and float(st.get("scheduled_for") or 0) <= now:
                    st["state"] = "queued"
                    st.setdefault("log", []).append(time.strftime("%H:%M:%S") + "  scheduled time reached: starting")
                    write_json(st_path, st)
                    JOBS.put((st_path.parent, None))
            except Exception:
                pass
        time.sleep(15)


threading.Thread(target=_scheduler, daemon=True).start()


# ---------------------------------------------------------------- shot library
class LibBuild(BaseModel):
    target: int = 1000
    per_video: int = 4
    categories: list[str] = []
    max_gb: float = 100
    auto_continue: bool = True


class LibAuto(BaseModel):
    on: bool = True


@app.get("/api/library")
def library_info():
    from . import library
    return {"stats": library.stats(load_settings()), "job": library.job_status(),
            "categories": [c[0] for c in library.DEFAULT_CATEGORIES]}


@app.post("/api/library/build")
def library_build(b: LibBuild):
    from . import library
    if not load_keys().get("DVIDS_API_KEY"):
        raise HTTPException(400, "DVIDS_API_KEY is needed to build the library - add it in Settings")
    if b.target < 10:
        raise HTTPException(400, "target must be at least 10 shots")
    try:
        return library.start_build(b.target, b.categories or None, b.per_video, b.max_gb, b.auto_continue)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/library/stop")
def library_stop():
    from . import library
    return library.stop_build()


@app.post("/api/library/retry")
def library_retry():
    """Paused build -> continue from the same file; ended build -> start again where it stopped."""
    from . import library
    if not load_keys().get("DVIDS_API_KEY"):
        raise HTTPException(400, "DVIDS_API_KEY is needed to build the library - add it in Settings")
    try:
        return library.retry_build()
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/library/sound")
def library_sound(a: LibAuto):
    """Give the shots cut before v0.9.0 their own sound (audio only, from DVIDS)."""
    from . import library
    if not load_keys().get("DVIDS_API_KEY"):
        raise HTTPException(400, "DVIDS_API_KEY is needed - add it in Settings")
    try:
        return library.start_sound(a.on)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/library/auto")
def library_auto(a: LibAuto):
    from . import library
    return library.set_auto_continue(a.on)


@app.get("/api/library/shots")
def library_shots(q: str = "", category: str = "", favorites: bool = False, sort: str = "new", offset: int = 0,
                  limit: int = 60):
    from . import library
    return library.list_shots(load_settings(), q, category, favorites, sort, offset, min(limit, 200))


def _lib_file(shot_id, field):
    from . import library
    r = library.shot_row(load_settings(), shot_id)
    if not r or not r.get(field) or not Path(r[field]).exists():
        raise HTTPException(404)
    return FileResponse(r[field])


@app.get("/api/library/thumb/{shot_id}")
def library_thumb(shot_id: int):
    return _lib_file(shot_id, "keyframe")


@app.get("/api/library/video/{shot_id}")
def library_video(shot_id: int):
    return _lib_file(shot_id, "file")


class Fav(BaseModel):
    on: bool = True


@app.post("/api/library/shots/{shot_id}/favorite")
def library_favorite(shot_id: int, f: Fav):
    from . import library
    library.set_favorite(load_settings(), shot_id, f.on)
    return {"ok": True}


@app.post("/api/library/shots/{shot_id}/delete")
def library_delete(shot_id: int):
    from . import library
    if not library.delete_shot(load_settings(), shot_id):
        raise HTTPException(404, "shot not found")
    return {"ok": True}


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
    bgm: str = ""                    # file name in assets/music; "" = no music
    start_in_minutes: float = 0      # schedule: start this many minutes from now (0 = now)
    start_at: float = 0              # schedule: start at this unix time (overrides start_in_minutes)


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


@app.post("/api/maps/sample")
def map_sample():
    """Settings > Render test map: 6 s clip in the locked map style."""
    from . import maps
    from .config import load_style
    out = DATA_DIR / "map_sample.mp4"
    out.unlink(missing_ok=True)
    try:
        look = maps.render_sample(load_style("beyond_military"), load_settings(), load_keys(), out)
    except Exception as e:
        raise HTTPException(500, f"map test failed: {str(e)[:300]}")
    return {"ok": True, "look": look, "url": f"/api/maps/sample.mp4?t={int(time.time())}"}


@app.get("/api/maps/sample.mp4")
def map_sample_file():
    p = DATA_DIR / "map_sample.mp4"
    if not p.exists():
        raise HTTPException(404, "no test map yet")
    return FileResponse(p, media_type="video/mp4")


MAPTEST = {"queue": [], "running": [], "error": "", "batches": {}}
MAPTEST_DIR = DATA_DIR / "map_tests"
MAPTEST_LOCK = threading.Lock()


class MapTest(BaseModel):
    scene: str = "red_sea"
    camera: str = "all"               # a move name, "random", or "all" (every move at once + one combined grid video)
    label_fx: str = ""
    seconds: float = 5.0
    text: str = ""                    # optional: a script sentence, planned by the brain


def _maptest_grid(batch):
    """All clips of one 'all moves' batch side by side in one video (5 x 2 grid), each labelled with its move."""
    import subprocess
    b = MAPTEST["batches"].get(batch) or {}
    clips = [c for c in b.get("done", []) if (MAPTEST_DIR / c["file"]).exists()]
    if len(clips) < 2:
        return
    clips.sort(key=lambda c: c["order"])
    cols = 5 if len(clips) > 4 else len(clips)
    rows = -(-len(clips) // cols)
    w, h = 1920 // cols, (1920 // cols) * 9 // 16
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for c in clips:
        cmd += ["-i", str(MAPTEST_DIR / c["file"])]
    parts, lay = [], []
    for i in range(len(clips)):
        parts.append(f"[{i}:v]scale={w}:{h},setsar=1[v{i}]")
        lay.append(f"{(i % cols) * w}_{(i // cols) * h}")
    fill = cols * rows - len(clips)
    for j in range(fill):
        parts.append(f"color=c=black:s={w}x{h}:d=6[f{j}]")
    ins = "".join(f"[v{i}]" for i in range(len(clips))) + "".join(f"[f{j}]" for j in range(fill))
    lay += [f"{((len(clips) + j) % cols) * w}_{((len(clips) + j) // cols) * h}" for j in range(fill)]
    fc = ";".join(parts) + f";{ins}xstack=inputs={cols * rows}:layout={'|'.join(lay)}:shortest=1[out]"
    name = f"{batch}-ALL.mp4"
    try:
        subprocess.run(cmd + ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                              "-pix_fmt", "yuv420p", str(MAPTEST_DIR / name)], check=True, capture_output=True, timeout=600)
        write_json(MAPTEST_DIR / f"{batch}-ALL.json", {
            "camera": "ALL MOVES", "grid": [c["camera"] for c in clips], "label_fx": "mixed", "title": b.get("title", ""),
            "scene": b.get("scene", ""), "seconds": clips[0].get("seconds"), "render_seconds": round(time.time() - b["t0"]),
            "created": time.time(), "file": name, "text": ""})
    except Exception as e:
        MAPTEST["error"] = f"grid video: {str(e)[:200]}"


def _maptest_worker():
    from . import maps
    from .config import load_style
    while True:
        with MAPTEST_LOCK:
            job = MAPTEST["queue"].pop(0) if MAPTEST["queue"] else None
            if job:
                MAPTEST["running"].append(job)
        if not job:
            time.sleep(1)
            continue
        MAPTEST_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{job['batch']}-{job['order']:02d}-{job['camera']}"
        out = MAPTEST_DIR / f"{name}.mp4"
        t0 = time.time()
        try:
            info = maps.render_test(load_style("beyond_military"), load_settings(), load_keys(), out, job["scene"],
                                    job["camera"], job["label_fx"], job["seconds"], job["text"], caption=job.get("caption", False))
            info.update(scene=job["scene"], text=job["text"][:120], render_seconds=round(time.time() - t0),
                        created=time.time(), file=out.name, order=job["order"])
            write_json(MAPTEST_DIR / f"{name}.json", info)
        except Exception as e:
            info = None
            MAPTEST["error"] = f"{job['camera']}: {str(e)[:300]}"
        with MAPTEST_LOCK:
            MAPTEST["running"].remove(job)
            b = MAPTEST["batches"].get(job["batch"])
            if b is not None:
                b["left"] -= 1
                if info:
                    b["done"].append(info)
                    b.setdefault("title", info.get("title", ""))
                last = b["left"] == 0 and b["grid"]
        if b is not None and last:
            (_maptest_reel if b.get("reel") else _maptest_grid)(job["batch"])


for _ in range(max(1, int(load_settings().get("maptest_parallel", 3) or 3))):   # several clips render at the same time
    threading.Thread(target=_maptest_worker, daemon=True).start()


@app.post("/api/maptest")
def maptest_add(m: MapTest):
    from . import maps
    if m.scene not in maps.SCENES and not m.text.strip():
        raise HTTPException(400, "unknown place")
    moves = maps.CAMERAS if m.camera == "all" else [m.camera]
    for mv in moves:
        if mv != "random" and mv not in maps.CAMERAS:
            raise HTTPException(400, f"unknown camera move {mv}")
    batch = time.strftime("%Y%m%d-%H%M%S")
    with MAPTEST_LOCK:
        MAPTEST["batches"][batch] = {"left": len(moves), "done": [], "grid": len(moves) > 1, "t0": time.time(),
                                     "scene": m.scene}
        for i, mv in enumerate(moves):
            MAPTEST["queue"].append({"batch": batch, "order": i, "scene": m.scene, "camera": mv, "label_fx": m.label_fx,
                                     "seconds": max(4.0, min(5.0, float(m.seconds))), "text": m.text})
    return {"ok": True, "queued": len(moves), "batch": batch}


def _maptest_reel(batch):
    """Showreel: every clip of the batch one after another in one long video."""
    import subprocess
    b = MAPTEST["batches"].get(batch) or {}
    clips = sorted([c for c in b.get("done", []) if (MAPTEST_DIR / c["file"]).exists()], key=lambda c: c["order"])
    if not clips:
        return
    lst = MAPTEST_DIR / f"{batch}-list.txt"
    lst.write_text("".join(f"file '{(MAPTEST_DIR / c['file']).as_posix()}'\n" for c in clips))
    name = f"{batch}-SHOWREEL.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", str(MAPTEST_DIR / name)],
                       check=True, capture_output=True, timeout=1800)
        write_json(MAPTEST_DIR / f"{batch}-SHOWREEL.json", {
            "camera": "SHOWREEL", "grid": [f"{c['camera']}+{c['label_fx']}" for c in clips], "label_fx": "all",
            "title": f"{len(clips)} combos", "scene": "all places", "seconds": round(sum(c["seconds"] for c in clips)),
            "render_seconds": round(time.time() - b["t0"]), "created": time.time(), "file": name, "text": ""})
        for c in clips:                                    # the single combo clips are inside the reel now
            (MAPTEST_DIR / c["file"]).unlink(missing_ok=True)
            (MAPTEST_DIR / c["file"].replace(".mp4", ".json")).unlink(missing_ok=True)
    except Exception as e:
        MAPTEST["error"] = f"showreel: {str(e)[:200]}"
    lst.unlink(missing_ok=True)


@app.post("/api/maptest/showreel")
def maptest_showreel():
    """4-5 min video with every camera move x every label effect (30 combos x 9 s), places taking turns, each with
    its combo written on top."""
    from . import maps
    batch = time.strftime("%Y%m%d-%H%M%S") + "-reel"
    scenes = list(maps.SCENES)
    jobs = []
    for i, mv in enumerate(maps.CAMERAS):
        for j, fx in enumerate(maps.LABEL_FX):
            n = len(jobs)
            jobs.append({"batch": batch, "order": n, "scene": scenes[n % len(scenes)], "camera": mv, "label_fx": fx,
                         "seconds": 9.0, "text": "", "caption": True})
    with MAPTEST_LOCK:
        MAPTEST["batches"][batch] = {"left": len(jobs), "done": [], "grid": False, "reel": True, "t0": time.time()}
        MAPTEST["queue"].extend(jobs)
    return {"ok": True, "queued": len(jobs), "seconds": 9 * len(jobs)}


@app.get("/api/maptest")
def maptest_state():
    from . import maps
    MAPTEST_DIR.mkdir(parents=True, exist_ok=True)
    clips = []
    for j in sorted(MAPTEST_DIR.glob("*.json"), key=lambda p: (p.stat().st_mtime, "-ALL" in p.name), reverse=True)[:60]:
        d = read_json(j, {}) or {}
        if (MAPTEST_DIR / d.get("file", "")).exists():
            clips.append(d)
    return {"queue": len(MAPTEST["queue"]), "running": [r["camera"] for r in MAPTEST["running"]], "error": MAPTEST["error"],
            "clips": clips, "cameras": maps.CAMERAS, "label_fx": maps.LABEL_FX, "scenes": list(maps.SCENES)}


@app.get("/api/maptest/file/{name}")
def maptest_file(name: str):
    p = MAPTEST_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4")


@app.post("/api/maptest/clear")
def maptest_clear():
    with MAPTEST_LOCK:
        MAPTEST["queue"].clear()
    MAPTEST["error"] = ""
    for p in MAPTEST_DIR.glob("*"):
        if p.is_file():
            p.unlink()
    return {"ok": True}


@app.get("/api/music")
def music_list():
    """Background music files the user put into assets/music (one is chosen per video)."""
    from .sfx import AUDIO_EXT, MOODS, mood_tracks
    for m in MOODS:                                   # the mood folders for "Auto: by script"
        (MUSIC_DIR / m).mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in MUSIC_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT)
    moods = {m: len(v) for m, v in mood_tracks().items() if m != "_any"}
    return {"folder": str(MUSIC_DIR), "files": files, "moods": moods}


@app.post("/api/music/open")
def music_open():
    """Open the music folder in Finder."""
    import subprocess
    import sys
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(MUSIC_DIR)])
    return {"ok": True, "folder": str(MUSIC_DIR)}


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
    bad = [s for s in j.sources if s not in SOURCES and s != "library"]
    if bad:
        raise HTTPException(400, f"unknown footage source: {', '.join(bad)}")
    online = [s for s in j.sources if s != "library"]
    if online and "library" not in j.sources:
        keys, settings = load_keys(), dict(load_settings(), **{f"use_{s}": True for s in online})
        if not any(source_ready(s, keys, settings) for s in online):
            raise HTTPException(400, "none of the selected footage sources has an API key — add it in Settings")
    if j.bgm and j.bgm != "auto" and not (MUSIC_DIR / Path(j.bgm).name).is_file():
        raise HTTPException(400, f"music file not found in assets/music: {j.bgm}")
    proj = create_project(j.title or "Untitled", j.script, j.voice_id, j.style, j.dummy, j.sources,
                          "auto" if j.bgm == "auto" else (Path(j.bgm).name if j.bgm else ""))
    when = j.start_at or (time.time() + j.start_in_minutes * 60 if j.start_in_minutes > 0 else 0)
    if when and when > time.time() + 30:
        st = read_json(proj / "status.json", {}) or {}
        st.update(state="scheduled", scheduled_for=when, step=None)
        st.setdefault("log", []).append("scheduled to start at " + time.strftime("%Y-%m-%d %H:%M", time.localtime(when)))
        write_json(proj / "status.json", st)
        return {"id": proj.name, "scheduled_for": when}
    JOBS.put((proj, None))
    return {"id": proj.name}


@app.get("/api/jobs")
def jobs():
    from . import costs
    settings = load_settings()
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
                    "scheduled_for": st.get("scheduled_for") if st.get("state") == "scheduled" else None,
                    "dummy": meta.get("dummy", False), "cost": costs.summary(p.parent, settings),
                    "mix": qc.get("source_mix")})
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
    from . import costs
    return {"id": jid, "meta": read_json(proj / "project.json"), "status": st, "qc": qc, "files": files,
            "cost": costs.summary(proj, load_settings())}


@app.post("/api/jobs/{jid}/resume")
def resume(jid: str, r: Resume):
    proj = PROJECTS_DIR / Path(jid).name
    if not proj.exists():
        raise HTTPException(404)
    if r.from_step and r.from_step not in STEPS:
        raise HTTPException(400, "unknown step")
    (proj / "STOP").unlink(missing_ok=True)
    st = read_json(proj / "status.json", {}) or {}
    if st.get("retries"):
        st["retries"] = 0
        write_json(proj / "status.json", st)
    JOBS.put((proj, r.from_step))
    return {"ok": True}


def _stop(proj):
    st = read_json(proj / "status.json", {}) or {}
    state = st.get("state")
    if state == "scheduled":                         # not started yet: just cancel
        st.update(state="stopped", scheduled_for=None)
        st.setdefault("log", []).append(time.strftime("%H:%M:%S") + "  schedule cancelled by user")
        write_json(proj / "status.json", st)
        return True
    if state in ("running", "queued"):
        (proj / "STOP").write_text(str(time.time()))     # the worker stops at its next log line / step
        st["stop_requested"] = time.time()
        write_json(proj / "status.json", st)
        return True
    return False


@app.post("/api/jobs/{jid}/cleanup")
def cleanup_job(jid: str):
    proj = _proj(jid)
    st = read_json(proj / "status.json", {}) or {}
    if st.get("state") in ("running", "queued"):
        raise HTTPException(400, "stop the video first")
    return {"ok": True, "freed_gb": round(cleanup_project(proj) / 1e9, 2)}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    """Delete a video and all its files."""
    import shutil
    proj = _proj(jid)
    st = read_json(proj / "status.json", {}) or {}
    if st.get("state") in ("running", "queued"):
        raise HTTPException(400, "stop the video first")
    size = sum(f.stat().st_size for f in proj.rglob("*") if f.is_file())
    shutil.rmtree(proj)
    return {"ok": True, "freed_gb": round(size / 1e9, 2)}


@app.post("/api/jobs/stop_all")
def stop_all():
    """Stop every running, queued and scheduled video."""
    n = sum(_stop(st.parent) for st in PROJECTS_DIR.glob("*/status.json"))
    return {"ok": True, "stopped": n}


@app.post("/api/jobs/{jid}/stop")
def stop_job(jid: str):
    proj = _proj(jid)
    if not _stop(proj):
        raise HTTPException(400, "this video is not running, queued or scheduled")
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
    try:                                  # insurance: a copy of the library index before any git change
        from . import library
        library.backup_index(load_settings(), name=f"index-preupdate-{time.strftime('%Y%m%d-%H%M%S')}.db")
    except Exception:
        pass
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

    def _resume_library():
        try:
            from . import library
            library.recover_index(load_settings(), library.status_log)   # index lost but shot files on disk
        except Exception as e:
            print(f"library index check failed: {e}", flush=True)
        try:
            from . import library
            library.resume_if_interrupted(load_settings())
        except Exception:
            pass
    threading.Timer(3.0, _resume_library).start()      # a build cut off by a crash / restart continues by itself
    if open_browser and not os.environ.get("WARSTUDIO_NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
