"""Runs the 8 steps in order. Every step writes its own JSON, so a failed run resumes where it stopped."""
import json
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from .config import PROJECTS_DIR, load_keys, load_settings, load_style
from .steps import vo, beats, specs, search, rank, fetch, render, qc
from .util import write_json, read_json

STEPS = ["voice", "beats", "specs", "search", "rank", "fetch", "render", "qc"]


def fmt_duration(sec):
    sec = int(round(sec or 0))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def slugify(title):
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "video"
    base = f"{datetime.now():%Y%m%d-%H%M%S}-{s}"
    name, n = base, 2
    while (PROJECTS_DIR / name).exists():
        name, n = f"{base}-{n}", n + 1
    return name


def create_project(title, script, voice_id, style_id, dummy=False, sources=None, bgm=""):
    proj = PROJECTS_DIR / slugify(title)
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "title.txt").write_text(title.strip() + "\n", encoding="utf-8")
    (proj / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    write_json(proj / "project.json", {"title": title, "voice_id": voice_id, "style": style_id, "dummy": dummy,
                                       "sources": list(sources or []), "bgm": bgm or "", "created": time.time()})
    write_json(proj / "status.json", {"state": "queued", "step": None, "progress": 0, "log": []})
    return proj


class Logger:
    """Thread-safe: steps log from worker threads (fetch, specs, search)."""

    def __init__(self, proj):
        import threading
        self.proj = Path(proj)
        self.status = read_json(self.proj / "status.json", {}) or {}
        self.status.setdefault("log", [])
        self._lock = threading.Lock()

    def set(self, **kw):
        with self._lock:
            self.status.update(kw)
            write_json(self.proj / "status.json", self.status)

    def __call__(self, msg, progress=None):
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        with self._lock:
            self.status["log"] = (self.status["log"] + [line])[-400:]
            if progress is not None:
                self.status["progress"] = round(float(progress), 3)
            write_json(self.proj / "status.json", self.status)


VOICE_LOCK = threading.Lock()      # ai33 cannot take several videos at once: voices are made one video at a time


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
                   "render": ["overlays.json", "sfx_events.json", "final.mp4"], "qc": []}
        for s in STEPS[idx:]:
            for f in cleanup[s]:
                (proj / f).unlink(missing_ok=True)
        if idx <= STEPS.index("render"):
            for p in (proj / "segments").glob("*.mp4") if (proj / "segments").exists() else []:
                p.unlink()
    t0 = time.time()
    timings = dict(log.status.get("timings") or {})
    if from_step:                                   # steps that will run again lose their old time
        for s in STEPS[STEPS.index(from_step):]:
            timings.pop(s, None)
    log.set(state="running", error=None, run_started=t0, run_finished=None, timings=timings)
    try:
        for name in STEPS:
            ts = time.time()
            log.set(step=name, progress=0, step_started=ts)
            log(f"--- step: {name}")
            if name == "voice":
                if (proj / "vo.json").exists() or dummy:
                    vo.run_step(proj, keys, settings, meta["voice_id"], log, dummy=dummy)
                else:
                    if not VOICE_LOCK.acquire(blocking=False):
                        log("voice: waiting - another video is generating its voice (one at a time for ai33)")
                        log.set(step="voice (waiting)")
                        VOICE_LOCK.acquire()
                    try:
                        vo.run_step(proj, keys, settings, meta["voice_id"], log, dummy=dummy)
                    finally:
                        VOICE_LOCK.release()
            elif name == "beats":
                beats.run_step(proj, style, log)
            elif name == "specs":
                specs.run_step(proj, keys, settings, log, dummy=dummy)
            elif name == "search":
                search.run_step(proj, keys, dict(settings, job_sources=meta.get("sources") or []), style, log)
            elif name == "rank":
                rank.run_step(proj, keys, settings, style, log)
            elif name == "fetch":
                fetch.run_step(proj, keys, style, log, settings)
            elif name == "render":
                render.run_step(proj, settings, style, log)
            elif name == "qc":
                qc.run_step(proj, log)
            took = round(time.time() - ts, 1)
            if took >= 1 or name not in timings:      # a skipped (already done) step keeps its earlier time
                timings[name] = took
            log.set(timings=timings)
        total = round(sum(timings.values()), 1)
        log.set(state="done", step=None, progress=1, run_finished=time.time(), total_seconds=total,
                last_run_seconds=round(time.time() - t0, 1))
        if not dummy and str(settings.get("library_learn", True)).lower() not in ("false", "0", "no"):
            try:
                from .library import Builder
                lb = Builder(keys, settings, log)
                n_new = lb.learn_from_project(proj)
                log(f"library: +{n_new} shots learned from this video's online clips"
                    + (f", {lb.dupes} already in the library (duplicates skipped)" if lb.dupes else ""))
            except Exception as e:
                log(f"library: learning skipped ({str(e)[:120]})")
        try:                                   # what this video cost
            from . import costs
            cs = costs.summary(proj, settings)
            log.set(cost=cs)
            log(costs.line(cs))
        except Exception as e:
            log(f"cost: could not add up ({str(e)[:100]})")
        try:                                   # the footage mix is the last thing in the log
            q = read_json(proj / "qc.json", {}) or {}
            if q.get("source_mix"):
                log(qc.mix_line(q["source_mix"]))
        except Exception:
            pass
        log(f"DONE in {fmt_duration(time.time() - t0)}" + (f" (all steps together: {fmt_duration(total)})"
                                                           if abs(total - (time.time() - t0)) > 5 else "") + " -> final.mp4")
    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc()[-1500:])
        log.set(state="error", error=str(e)[:500], run_finished=time.time())
        raise
