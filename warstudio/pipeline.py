"""Runs the 8 steps in order. Every step writes its own JSON, so a failed run resumes where it stopped."""
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from .config import PROJECTS_DIR, load_keys, load_settings, load_style
from .steps import vo, beats, specs, search, rank, fetch, render, qc
from .util import write_json, read_json

STEPS = ["voice", "beats", "specs", "search", "rank", "fetch", "render", "qc"]


def slugify(title):
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "video"
    return f"{datetime.now():%Y%m%d-%H%M}-{s}"


def create_project(title, script, voice_id, style_id, dummy=False):
    proj = PROJECTS_DIR / slugify(title)
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "title.txt").write_text(title.strip() + "\n", encoding="utf-8")
    (proj / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    write_json(proj / "project.json", {"title": title, "voice_id": voice_id, "style": style_id, "dummy": dummy,
                                       "created": time.time()})
    write_json(proj / "status.json", {"state": "queued", "step": None, "progress": 0, "log": []})
    return proj


class Logger:
    def __init__(self, proj):
        self.proj = Path(proj)
        self.status = read_json(self.proj / "status.json", {}) or {}
        self.status.setdefault("log", [])

    def set(self, **kw):
        self.status.update(kw)
        write_json(self.proj / "status.json", self.status)

    def __call__(self, msg, progress=None):
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        self.status["log"] = (self.status["log"] + [line])[-400:]
        if progress is not None:
            self.status["progress"] = round(float(progress), 3)
        write_json(self.proj / "status.json", self.status)


def run_project(proj, from_step=None):
    proj = Path(proj)
    meta = read_json(proj / "project.json")
    keys, settings = load_keys(), load_settings()
    style = load_style(meta["style"])
    dummy = bool(meta.get("dummy"))
    log = Logger(proj)
    if from_step:
        idx = STEPS.index(from_step)
        cleanup = {"voice": ["vo.json", "vo.wav"], "beats": ["beats.json"], "specs": ["specs.json"],
                   "search": ["candidates.json"], "rank": ["picks.json"], "fetch": ["fetched.json"],
                   "render": ["overlays.json", "final.mp4"], "qc": []}
        for s in STEPS[idx:]:
            for f in cleanup[s]:
                (proj / f).unlink(missing_ok=True)
        if idx <= STEPS.index("render"):
            for p in (proj / "segments").glob("*.mp4") if (proj / "segments").exists() else []:
                p.unlink()
    t0 = time.time()
    log.set(state="running", error=None)
    try:
        for name in STEPS:
            log.set(step=name, progress=0)
            log(f"--- step: {name}")
            if name == "voice":
                vo.run_step(proj, keys, settings, meta["voice_id"], log, dummy=dummy)
            elif name == "beats":
                beats.run_step(proj, style, log)
            elif name == "specs":
                specs.run_step(proj, keys, settings, log, dummy=dummy)
            elif name == "search":
                search.run_step(proj, keys, settings, style, log)
            elif name == "rank":
                rank.run_step(proj, keys, settings, style, log)
            elif name == "fetch":
                fetch.run_step(proj, keys, style, log)
            elif name == "render":
                render.run_step(proj, settings, style, log)
            elif name == "qc":
                qc.run_step(proj, log)
        log.set(state="done", step=None, progress=1)
        log(f"DONE in {(time.time() - t0) / 60:.1f} min -> final.mp4")
    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc()[-1500:])
        log.set(state="error", error=str(e)[:500])
        raise
