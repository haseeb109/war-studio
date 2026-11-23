"""System check (smoke test): tests every installed piece one by one and writes a plain-text report.

Quick check : system, repo, tools, Python packages, encoders, overlays, SFX, audio mix, API keys valid.
Full check  : + CLIP model on MPS, HyperFrames render, every API end to end (search -> resolve -> 2 s download),
              ai33 TTS sample, a mini production run with real footage. Costs a few cents of credits.

The report never contains API keys. Send report.txt to Claude when something breaks.
"""
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import ROOT, VERSION
from .config import DATA_DIR, PROJECTS_DIR, load_keys, load_settings, load_style, masked_keys

REPORT_DIR = DATA_DIR / "diagnostics"


class Check:
    def __init__(self, on_update=None):
        self.results = []
        self.on_update = on_update
        self.tmp = Path(tempfile.mkdtemp(prefix="warstudio_check_"))

    def run(self, group, name, fn):
        t0 = time.time()
        entry = {"group": group, "name": name, "status": "running", "detail": "", "seconds": 0}
        self.results.append(entry)
        self._emit()
        try:
            res = fn()
            if isinstance(res, tuple):
                entry["status"], entry["detail"] = res
            else:
                entry["status"], entry["detail"] = "ok", str(res or "")
        except Exception as e:
            entry["status"] = "fail"
            entry["detail"] = f"{type(e).__name__}: {str(e)[:600]}"
            entry["trace"] = traceback.format_exc()[-1200:]
        entry["seconds"] = round(time.time() - t0, 2)
        self._emit()
        return entry

    def _emit(self):
        if self.on_update:
            self.on_update(self.results)


def _cmd(args, timeout=60):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def _first_line(s):
    return (s.splitlines() or [""])[0][:160]


# ---------------- individual checks ----------------

def chk_system():
    mac = platform.mac_ver()[0]
    ram = ""
    if sys.platform == "darwin":
        _, out = _cmd(["sysctl", "-n", "hw.memsize"])
        try:
            ram = f", RAM {int(out) / 1e9:.0f} GB"
        except ValueError:
            pass
        _, chip = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        ram += f", {chip}"
    free = shutil.disk_usage(ROOT).free / 1e9
    detail = f"{platform.system()} {mac or platform.release()} {platform.machine()}{ram}, Python {platform.python_version()}, free disk {free:.0f} GB"
    if free < 30:
        return "warn", detail + " (less than 30 GB free: long videos may fail)"
    return "ok", detail


def chk_repo():
    code, commit = _cmd(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"])
    _, branch = _cmd(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    _, dirty = _cmd(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    d = f"v{VERSION}, {branch} @ {commit}" if code == 0 else f"v{VERSION} (not a git checkout: update button will not work)"
    if dirty:
        return "warn", d + f"; locally modified files: {len(dirty.splitlines())} (update may refuse to pull)"
    return ("ok" if code == 0 else "warn"), d


def chk_update():
    code, out = _cmd(["git", "-C", str(ROOT), "fetch", "--quiet", "origin"], timeout=60)
    if code:
        return "warn", "cannot reach GitHub: " + out[-200:]
    _, branch = _cmd(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    _, behind = _cmd(["git", "-C", str(ROOT), "rev-list", "--count", f"HEAD..origin/{branch}"])
    if behind.strip().isdigit() and int(behind) > 0:
        return "warn", f"{behind} update(s) available: press 'Check for update'"
    return "ok", "up to date with GitHub"


def chk_tool(name, args, required=True):
    def f():
        if not shutil.which(args[0]):
            return ("fail" if required else "warn"), f"{args[0]} not found (run setup_mac.sh)"
        code, out = _cmd(args)
        return ("ok" if code == 0 else "fail"), _first_line(out)
    return f


def chk_ffmpeg_features():
    _, enc = _cmd(["ffmpeg", "-hide_banner", "-encoders"])
    _, dec = _cmd(["ffmpeg", "-hide_banner", "-decoders"])
    _, flt = _cmd(["ffmpeg", "-hide_banner", "-filters"])
    need = {"libx264 encoder": "libx264" in enc, "libvpx-vp9 decoder (HyperFrames overlays)": "libvpx-vp9" in dec,
            "libmp3lame": "libmp3lame" in enc, "loudnorm filter": "loudnorm" in flt, "aac": " aac " in enc}
    missing = [k for k, v in need.items() if not v]
    extra = "h264_videotoolbox available" if "h264_videotoolbox" in enc else "no h264_videotoolbox (slower libx264 renders)"
    if missing:
        return "fail", "missing: " + ", ".join(missing)
    return "ok", "all required codecs/filters present; " + extra


def chk_package(mod):
    def f():
        m = importlib.import_module(mod)
        return "ok", getattr(m, "__version__", "installed")
    return f


def chk_torch():
    import torch
    mps = torch.backends.mps.is_available()
    d = f"torch {torch.__version__}, MPS (Apple GPU) {'available' if mps else 'NOT available'}"
    return ("ok" if mps or sys.platform != "darwin" else "warn"), d


def chk_encoder(enc):
    def f():
        from .util import encoder_works
        ok = encoder_works(enc)
        return ("ok" if ok else ("warn" if enc == "h264_videotoolbox" else "fail")), ("encodes fine" if ok else "encode test failed")
    return f


def chk_selected_encoder():
    from .util import video_encoder
    s = load_settings().get("encoder", "auto")
    return "ok", f"setting '{s}' -> renders will use {video_encoder(s)}"


def chk_segment(c):
    def f():
        from .overlays import render_png, plan
        from .steps.render import segment_cmd
        from .util import run, duration, video_encoder
        st = load_style("beyond_military")
        src = c.tmp / "src.mp4"
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=3",
             "-c:v", "libx264", "-preset", "ultrafast", str(src)])
        png = render_png({"kind": "measure", "line1": "28,000 ft", "line2": "8,534 m", "duration": 2.0}, st, c.tmp / "ov.png")
        out = c.tmp / "seg.mp4"
        enc = video_encoder(load_settings().get("encoder", "auto"))
        run(segment_cmd(str(src), 75, 30, st, "flir", {"type": "png", "path": str(png), "duration": 2.0}, out, enc))
        _, frames = _cmd(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)])
        if frames.strip() != "75":
            return "fail", f"expected 75 frames, got {frames.strip()}"
        return "ok", f"drift + FLIR grade + PNG overlay, exact 75 frames ({enc})"
    return f


def chk_hyperframes(c):
    def f():
        from .overlays import hyperframes_available, render_webm
        if not hyperframes_available():
            return "warn", "HyperFrames/npx or libvpx-vp9 missing: PNG overlays will be used"
        st = load_style("beyond_military")
        out = render_webm({"kind": "time", "line1": "17:14", "line2": "", "duration": 1.5}, st, c.tmp / "hf.webm", c.tmp)
        if not out:
            return "warn", "HyperFrames render failed (first run downloads Chrome; try again). PNG overlays will be used"
        _, info = _cmd(["ffprobe", "-v", "error", "-show_entries", "stream_tags=alpha_mode", "-of", "csv=p=0", str(out)])
        return ("ok" if "1" in info else "warn"), "animated transparent WebM rendered" + ("" if "1" in info else " (no alpha flag!)")
    return f


def chk_sfx():
    from .sfx import CATS, ensure_defaults, sfx_files, SFX_DIR
    ensure_defaults()
    user = {c: len([p for p in (SFX_DIR / c).glob("*") if p.suffix.lower() in (".wav", ".mp3", ".aif", ".aiff", ".flac", ".m4a")])
            for c in CATS if (SFX_DIR / c).exists()}
    missing = [c for c in CATS if not sfx_files(c)]
    own = ", ".join(f"{c}:{n}" for c, n in user.items() if n) or "none (built-in synthesized SFX in use)"
    return ("fail" if missing else "ok"), (f"missing: {missing}" if missing else f"all {len(CATS)} categories ready; your files: {own}")


def chk_mix(c):
    def f():
        from .sfx import mix, write_wav
        import numpy as np
        st = load_style("beyond_military")
        vo = c.tmp / "vo.wav"
        t = np.arange(48000 * 3) / 48000
        write_wav(vo, (np.sin(2 * np.pi * 220 * t) * 0.3).astype("float32"))
        out = c.tmp / "mix.wav"
        mix(vo, [{"t": 0.5, "cat": "ui_click", "gain_db": -16}, {"t": 1.0, "cat": "explosion", "gain_db": -4}], 3.0, st, out)
        from .steps.render import loudnorm_filter
        af = loudnorm_filter(out, -14)
        return "ok", "voice + SFX mixed, loudnorm two-pass " + ("measured" if "measured_I" in af else "single-pass fallback")
    return f


def chk_music():
    from .config import MUSIC_DIR
    s = load_settings().get("music_file") or ""
    files = [p.name for p in MUSIC_DIR.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac", ".aiff")]
    if s and not Path(s).exists():
        return "warn", f"music_file setting points to a missing file: {s}"
    if s or files:
        return "ok", f"music: {s or files[0]}"
    return "warn", "no background music (put a track in assets/music/)"


def chk_keys():
    k = masked_keys()
    missing = [n for n, v in k.items() if not v]
    have = [n for n, v in k.items() if v]
    if not any(k[x] for x in ("DVIDS_API_KEY", "PEXELS_API_KEY", "PIXABAY_API_KEY")):
        return "fail", "no footage key (DVIDS / Pexels / Pixabay) — Settings tab"
    return ("warn" if missing else "ok"), f"set: {', '.join(have) or 'none'}" + (f"; missing: {', '.join(missing)}" if missing else "")


def _http_get(url, **kw):
    import httpx
    from .util import UA
    h = {"User-Agent": UA, **kw.pop("headers", {})}
    return httpx.get(url, headers=h, timeout=30, **kw)


def chk_ai33_key():
    key = load_keys().get("AI33_API_KEY")
    if not key:
        return "warn", "AI33_API_KEY not set (only dry runs possible)"
    from . import ai33
    vs, source = ai33.list_voices(key)
    import httpx
    r = httpx.get("https://api.ai33.pro/v1/task/00000000-0000-0000-0000-000000000000",
                  headers={"xi-api-key": key}, timeout=30)
    if r.status_code == 401:
        return "fail", "ai33 rejected the key (401): " + r.text[:120]
    return "ok", f"key accepted; voice library: {len(vs)} voices from {source}"


def chk_ai33_tts(c):
    def f():
        key = load_keys().get("AI33_API_KEY")
        if not key:
            return "warn", "skipped (no key)"
        from . import ai33
        from .util import duration
        out = c.tmp / "tts.mp3"
        credits = ai33.tts_many(key, [(out, "System check. One, two, three.")], "elevenlabs_nPczCjzI2devNBz1zQrb",
                                load_settings().get("tts_model"), 1, timeout=180)
        return "ok", f"TTS end to end OK: {duration(out):.1f}s audio, {credits} credits"
    return f


def chk_brain():
    from .config import brain_config
    from .connect import test_brain
    b = brain_config()
    if not b["key"]:
        return "warn", f"{b['key_name']} not set: shot specs will use simple keyword fallback (worse matches)"
    ok, msg = test_brain(b["provider"])
    return ("ok" if ok else "fail"), msg


def chk_gemini():
    keys, s = load_keys(), load_settings()
    key = keys.get("GEMINI_API_KEY")
    if not key:
        return "warn", "GEMINI_API_KEY not set: CLIP top-1 picks clips (no rerank)"
    r = _http_get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key, "pageSize": 200})
    if r.status_code >= 400:
        return "fail", f"Gemini key error HTTP {r.status_code}: {r.text[:200]}"
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    flash = [n for n in names if "flash" in n and "image" not in n and "tts" not in n]
    if s["gemini_model"] not in names:
        return "fail", f"model '{s['gemini_model']}' not available for this key. Set gemini_model in Settings to one of: {', '.join(flash[:8])}"
    return "ok", f"key OK, model {s['gemini_model']} available"


def chk_gemini_vision(c):
    def f():
        keys, s = load_keys(), load_settings()
        if not keys.get("GEMINI_API_KEY"):
            return "warn", "skipped (no key)"
        from PIL import Image
        from .llm import gemini_pick
        img = c.tmp / "g.jpg"
        Image.new("RGB", (512, 288), (255, 196, 0)).save(img)
        res, usage = gemini_pick(keys["GEMINI_API_KEY"], s["gemini_model"], img,
                                 'What is the main color of this image? Return JSON {"color": "<word>"}')
        return "ok", f"vision call OK: {res}, tokens {usage.get('totalTokenCount', '?')}"
    return f


def _source_check(c, src, deep):
    def f():
        from .steps import search, fetch
        keys = load_keys()
        kname, fn = search.SOURCES[src]
        if not keys.get(kname):
            return "warn", f"{kname} not set"
        res = fn("fighter jet", keys[kname], 5)
        if not res:
            return "warn", "key works but 0 results for 'fighter jet'"
        p = res[0]
        detail = f"search OK ({len(res)} results); preview {'yes' if p['preview_url'] else 'MISSING'}"
        if not p["preview_url"]:
            return "warn", detail + f"; raw first result: {json.dumps(p)[:300]}"
        if not deep:
            return "ok", detail
        from .util import download
        download(p["preview_url"], c.tmp / f"{src}_prev.jpg", timeout=30)
        if src == "dvids":
            fetch.resolve_dvids(p, keys[kname])
        if not p.get("video_url"):
            return "fail", detail + "; no video URL resolved"
        out = c.tmp / f"{src}_clip.mp4"
        fetch.fetch_one(p, 1.0, 2.0, out, 1920, 1080, 30)
        from .util import duration
        return "ok", detail + f"; preview downloaded; 2 s segment fetched + normalized ({duration(out):.2f}s, 1080p30)"
    return f


def chk_clip(c):
    def f():
        from PIL import Image, ImageDraw
        from .steps.rank import clip_scores, _clip
        s = load_settings()
        t0 = time.time()
        dev = _clip(s)["dev"]
        a, b = c.tmp / "sky.jpg", c.tmp / "grass.jpg"
        Image.new("RGB", (224, 224), (90, 150, 230)).save(a)
        Image.new("RGB", (224, 224), (40, 160, 40)).save(b)
        sc = clip_scores(s, "a clear blue sky", [str(a), str(b)])
        ok = sc[0] > sc[1]
        return ("ok" if ok else "warn"), f"{s['clip_model']} on {dev}, loaded+scored in {time.time() - t0:.1f}s; sanity {'passed' if ok else 'FAILED'}"
    return f


def chk_mini_run(c):
    def f():
        from .pipeline import create_project, run_project
        from .util import read_json, duration
        keys = load_keys()
        if not any(keys.get(k) for k in ("DVIDS_API_KEY", "PEXELS_API_KEY", "PIXABAY_API_KEY")):
            return "warn", "skipped (no footage key)"
        script = ("At 17:14 local time, an F-15E Strike Eagle holds at 28,000 feet over the coast. "
                  "But unknown to the crew, a missile launches from the shore. The jet breaks hard and drops flares.")
        proj = create_project("System check mini run", script, "elevenlabs_nPczCjzI2devNBz1zQrb", "beyond_military", dummy=True)
        run_project(proj)
        qc = read_json(proj / "qc.json", {})
        drift = qc.get("av_drift_seconds", 99)
        missing = len(qc.get("beats_without_footage", []))
        from . import edits
        t0 = time.time()
        edits.set_overlay(proj, 0, "label", "SYSTEM CHECK")
        edits.apply(proj, lambda *a, **k: None)
        qc2 = read_json(proj / "qc.json", {})
        edit_ok = abs(qc2.get("av_drift_seconds", 99)) < 0.1
        st = "ok" if drift < 0.1 and missing == 0 and edit_ok else "warn"
        return st, f"review edit + partial re-render {'OK' if edit_ok else 'FAILED'} in {time.time() - t0:.0f}s; dry-run video OK: {qc.get('video_seconds')}s, {qc.get('beats')} beats, {missing} without footage, A/V drift {drift}s, {qc.get('integrated_lufs')} LUFS -> projects/{proj.name}"
    return f


def recent_errors(n=3):
    out = []
    for p in sorted(PROJECTS_DIR.glob("*/status.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            st = json.loads(p.read_text())
        except Exception:
            continue
        if st.get("state") == "error":
            out.append({"project": p.parent.name, "step": st.get("step"), "error": st.get("error"), "log": st.get("log", [])[-15:]})
        if len(out) >= n:
            break
    return out


# ---------------- runner + report ----------------

def run_all(full=False, on_update=None):
    c = Check(on_update)
    c.run("System", "Machine", chk_system)
    c.run("System", "WarStudio version", chk_repo)
    c.run("System", "GitHub update", chk_update)
    c.run("Tools", "ffmpeg", chk_tool("ffmpeg", ["ffmpeg", "-version"]))
    c.run("Tools", "ffprobe", chk_tool("ffprobe", ["ffprobe", "-version"]))
    c.run("Tools", "ffmpeg codecs/filters", chk_ffmpeg_features)
    c.run("Tools", "git", chk_tool("git", ["git", "--version"]))
    c.run("Tools", "uv", chk_tool("uv", ["uv", "--version"], required=False))
    c.run("Tools", "node / npx (HyperFrames)", chk_tool("npx", ["npx", "--version"], required=False))
    for m in ("fastapi", "uvicorn", "httpx", "numpy", "PIL", "open_clip"):
        c.run("Python", m, chk_package(m))
    c.run("Python", "torch + Apple GPU", chk_torch)
    c.run("Render", "h264_videotoolbox encode", chk_encoder("h264_videotoolbox"))
    c.run("Render", "libx264 encode", chk_encoder("libx264"))
    c.run("Render", "encoder in use", chk_selected_encoder)
    c.run("Render", "segment render (grade + overlay + frame count)", chk_segment(c))
    c.run("Audio", "sound effects", chk_sfx)
    c.run("Audio", "mix + loudness", chk_mix(c))
    c.run("Audio", "background music", chk_music)
    c.run("APIs", "keys", chk_keys)
    c.run("APIs", "ai33 key + voice library", chk_ai33_key)
    c.run("APIs", "brain (script planner)", chk_brain)
    c.run("APIs", "Gemini key + model", chk_gemini)
    for src in ("dvids", "pexels", "pixabay"):
        c.run("Footage", f"{src}", _source_check(c, src, full))
    if full:
        c.run("Full", "HyperFrames overlay render", chk_hyperframes(c))
        c.run("Full", "CLIP model (Apple GPU)", chk_clip(c))
        c.run("Full", "Gemini vision call", chk_gemini_vision(c))
        c.run("Full", "ai33 voice generation", chk_ai33_tts(c))
        c.run("Full", "mini production (dry run, real footage)", chk_mini_run(c))
    shutil.rmtree(c.tmp, ignore_errors=True)
    return c.results, write_report(c.results, full)


def write_report(results, full):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    icon = {"ok": "PASS", "warn": "WARN", "fail": "FAIL", "running": "...."}
    counts = {k: sum(1 for r in results if r["status"] == k) for k in ("ok", "warn", "fail")}
    lines = [f"WarStudio system check ({'FULL' if full else 'QUICK'}) — {datetime.now():%Y-%m-%d %H:%M}",
             f"v{VERSION} | {counts['ok']} pass, {counts['warn']} warn, {counts['fail']} fail", ""]
    group = None
    for r in results:
        if r["group"] != group:
            group = r["group"]
            lines.append(f"[{group}]")
        lines.append(f"  {icon[r['status']]}  {r['name']}: {r['detail']}  ({r['seconds']}s)")
        if r.get("trace") and r["status"] == "fail":
            lines += ["      " + x for x in r["trace"].strip().splitlines()[-6:]]
    s = load_settings()
    lines += ["", "[Settings]", "  " + ", ".join(f"{k}={v}" for k, v in s.items())]
    errs = recent_errors()
    if errs:
        lines += ["", "[Recent failed videos]"]
        for e in errs:
            lines.append(f"  {e['project']} — step {e['step']}: {e['error']}")
            lines += ["      " + x for x in e["log"]]
    text = "\n".join(lines) + "\n"
    path = REPORT_DIR / f"report-{datetime.now():%Y%m%d-%H%M%S}.txt"
    path.write_text(text, encoding="utf-8")
    (REPORT_DIR / "latest.txt").write_text(text, encoding="utf-8")
    return text
