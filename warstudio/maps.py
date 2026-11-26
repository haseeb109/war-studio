"""Map beats -> animated map clips (MapLibre GL JS + HyperFrames, LOCKED Tactical dark style).

1. runs:   consecutive map beats of the same scene become one run (one clip, cut into the beats).
2. plan:   the brain writes one map plan per run (camera from/to, enemy/ally countries, places, arrows, rings).
           Every number is checked; a run with no valid plan falls back to stock footage as before.
3. render: the runs are rendered at the same time (maps_parallel, default 3), one HyperFrames worker each.
4. cut:    each beat gets its own piece of its run's clip -> fetched.json entry with source "map".
Nothing here may break a video: any failure logs a line and those beats use stock footage instead.
"""
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT
from .util import read_json, write_json, duration

MAPS_DIR = ROOT / "styles" / "beyond_military" / "maps"
HF_DIR = ROOT / "styles" / "beyond_military" / "hf"
CHUNK = 8                   # runs per brain call (short replies come back complete)
MAX_RUN = 15.0              # seconds: a longer run of map beats is split (one camera move stays readable)
PAD = 1.0
SIDES = ("enemy", "ally", "neutral")


def map_look(style, settings=None, keys=None):
    """-> (theme, extra plan fields): the locked map theme."""
    return style.get("maps", {}).get("theme") or {}, {}


def tilt(plan, theme):
    """3D fly-in: the camera tilts down while it dives in (only for themes that ask for it)."""
    if theme.get("pitch"):
        plan = dict(plan, to=dict(plan["to"], pitch=float(theme.get("pitch_deg", 55)), bearing=-15))
    return plan


SAMPLE = {"title": "RED SEA", "from": {"center": [44.0, 17.0], "zoom": 3.2}, "to": {"center": [42.96, 14.80], "zoom": 9.2},
          "highlight": [{"iso": "YEM", "color": "#b3261e"}, {"iso": "ERI", "color": "#3a5a80"}],
          "countries": [{"name": "YEMEN", "at": [44.2, 15.3]}, {"name": "ERITREA", "at": [39.0, 15.2]}],
          "points": [{"name": "Hodeidah", "lon": 42.95, "lat": 14.80}],
          "arrows": [{"from": [42.95, 14.80], "to": [42.55, 14.55]}], "rings": [{"lon": 42.95, "lat": 14.80, "km": 25}]}


def render_sample(style, settings, keys, out):
    """Settings > Render test map: a 6 s Red Sea clip with the current map look."""
    if not hf_cmd():
        raise RuntimeError("HyperFrames/npx not found")
    theme, extra = map_look(style, settings, keys)
    plan = tilt(dict(SAMPLE, **extra), theme)
    work = build_page(plan, 6.0, Path(out).parent / "map_sample_work", theme)
    render_one(work, out, style["canvas"]["fps"], timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    return "tactical dark (locked)"


PROMPT = """You plan animated military news maps (dark tactical style). For every RUN below, read the narration and
return where the camera looks and what is drawn. Use real geography. Coordinates are [longitude, latitude].
Reply with one JSON object only, exactly in this shape:
{"runs": [{"run": 0, "title": "RED SEA",
  "from": {"center": [42.0, 18.0], "zoom": 4.2}, "to": {"center": [43.3, 13.2], "zoom": 5.6},
  "highlight": [{"iso": "YEM", "side": "enemy"}],
  "countries": [{"name": "YEMEN", "at": [47.5, 15.6]}],
  "points": [{"name": "Hodeidah", "lon": 42.95, "lat": 14.8}],
  "arrows": [{"from": [42.95, 14.8], "to": [43.3, 12.8]}],
  "rings": [{"lon": 42.95, "lat": 14.8, "km": 180}]}]}
Fields:
- title: 1-3 words, the region, CAPITALS.
- from: wide start view. to: ends closer on the place the narration is about, and shows every point, arrow and ring.
  zoom 2 = continent, 8 = city.
- highlight: max 4 countries, ISO 3166 alpha-3 code, side enemy / ally / neutral ("enemy" = the side the narration
  treats as the threat).
- countries: max 5 country names to print, "at" inside the country.
- points: max 4 cities, bases, islands or straits that are named.
- arrows: max 3, movement / attack / route. rings: max 2, a range or distance that is named.
Only draw what the narration says; use an empty list when nothing fits. Every run must be in the reply."""


def _iso_set():
    try:
        geo = json.loads((MAPS_DIR / "countries.geojson").read_text())
        return {f["properties"].get("iso") for f in geo["features"]}
    except Exception:
        return set()


def find_runs(beats, specs, vo_total):
    """Consecutive map beats in the same scene -> [{"beats": [ids], "start", "end", "text"}]."""
    sp = {str(s["id"]): s for s in specs}
    runs, cur = [], None
    for i, b in enumerate(beats):
        bid = str(b["id"])
        s = sp.get(bid) or {}
        end = vo_total if i == len(beats) - 1 else beats[i + 1]["start"]
        if s.get("footage_class") != "map":
            cur = None
            continue
        if cur and cur["scene"] == (s.get("scene") or "") and cur["last"] == i - 1 and end - cur["start"] <= MAX_RUN:
            cur["beats"].append(bid)
            cur["end"], cur["last"] = end, i
            cur["text"] += " " + b["text"]
        else:
            cur = {"beats": [bid], "start": b["start"], "end": end, "last": i, "scene": s.get("scene") or "",
                   "text": b["text"]}
            runs.append(cur)
    return [{k: r[k] for k in ("beats", "start", "end", "text")} for r in runs]


def _num(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None


def _ll(v):
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    lon, lat = _num(v[0], -180, 180), _num(v[1], -85, 85)
    return None if lon is None or lat is None else [round(lon, 3), round(lat, 3)]


def _view(v):
    if not isinstance(v, dict):
        return None
    c, z = _ll(v.get("center")), _num(v.get("zoom"), 1, 9)
    return None if c is None or z is None else {"center": c, "zoom": round(z, 2)}


def validate_plan(p, isos, colors):
    """One run's plan from the brain -> a clean plan for the template, or None if the camera is unusable."""
    if not isinstance(p, dict):
        return None
    a, b = _view(p.get("from")), _view(p.get("to"))
    if not b:
        return None
    a = a or {"center": b["center"], "zoom": max(1.0, b["zoom"] - 1.2)}
    clean = lambda s, n: re.sub(r"[<>&\"`$\\]", "", str(s or "")).strip()[:n]
    out = {"title": clean(p.get("title"), 24).upper(), "from": a, "to": b,
           "highlight": [], "countries": [], "points": [], "arrows": [], "rings": []}
    for h in (p.get("highlight") or [])[:4]:
        iso = str((h or {}).get("iso", "")).upper() if isinstance(h, dict) else ""
        side = str(h.get("side", "neutral")).lower() if isinstance(h, dict) else ""
        if iso in isos and iso not in [x["iso"] for x in out["highlight"]]:
            out["highlight"].append({"iso": iso, "color": colors.get(side if side in SIDES else "neutral")})
    for c in (p.get("countries") or [])[:5]:
        at = _ll((c or {}).get("at")) if isinstance(c, dict) else None
        if at and clean(c.get("name"), 22):
            out["countries"].append({"name": clean(c["name"], 22).upper(), "at": at})
    for q in (p.get("points") or [])[:4]:
        if isinstance(q, dict):
            ll = _ll([q.get("lon"), q.get("lat")])
            if ll and clean(q.get("name"), 24):
                out["points"].append({"name": clean(q["name"], 24), "lon": ll[0], "lat": ll[1]})
    for r in (p.get("arrows") or [])[:3]:
        if isinstance(r, dict):
            f, t = _ll(r.get("from")), _ll(r.get("to"))
            if f and t and f != t:
                out["arrows"].append({"from": f, "to": t})
    for r in (p.get("rings") or [])[:2]:
        if isinstance(r, dict):
            ll, km = _ll([r.get("lon"), r.get("lat")]), _num(r.get("km"), 5, 3000)
            if ll and km:
                out["rings"].append({"lon": ll[0], "lat": ll[1], "km": round(km)})
    return out


def plan_runs(runs, title, settings, keys, log):
    """Ask the brain for every run's plan in one call. Returns ({run index: plan}, usd)."""
    from .config import brain_config
    from .llm import chat, extract_json
    brain = brain_config(settings, keys)
    if not brain["key"]:
        log("maps: no brain key - map beats use stock footage")
        return {}, 0.0
    style = json.loads((ROOT / "styles" / "beyond_military" / "style.json").read_text())
    colors = style["maps"].get("highlight_colors") or {"enemy": "#b3261e", "ally": "#3a5a80", "neutral": "#4b5a6b"}
    isos = _iso_set()
    plans, usd, lock = {}, 0.0, threading.Lock()
    price_in = float(settings.get("brain_price_in_per_m", 0.3) or 0.3)
    price_out = float(settings.get("brain_price_out_per_m", 1.2) or 1.2)

    def one(c0):
        nonlocal usd
        idx = list(range(c0, min(c0 + CHUNK, len(runs))))
        user = f"VIDEO TITLE: {title}\n\nRUNS:\n" + "\n".join(
            json.dumps({"run": i, "narration": runs[i]["text"][:900]}, ensure_ascii=False) for i in idx)
        for attempt in range(3):
            reply = ""
            try:
                # DeepSeek's JSON mode sometimes answers with an empty message: the later tries go without it
                reply, usage = chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": user}], brain,
                                    max_tokens=16000, json_mode=(attempt == 0))
                with lock:
                    usd += ((usage.get("prompt_tokens") or 0) / 1e6 * price_in +
                            (usage.get("completion_tokens") or 0) / 1e6 * price_out)
                data = extract_json(reply)
                got = 0
                for p in (data.get("runs") if isinstance(data, dict) else data) or []:
                    try:
                        i = int(p.get("run"))
                    except (TypeError, ValueError, AttributeError):
                        continue
                    v = validate_plan(p, isos, colors)
                    if v and i in idx:
                        with lock:
                            plans[i] = v
                        got += 1
                if got:
                    return
                raise ValueError("no usable run in the reply")
            except Exception as e:
                log(f"maps: plan for runs {idx[0]}-{idx[-1]}, try {attempt + 1} failed ({str(e)[:100]}; "
                    f"reply: {(reply or 'empty').strip()[:80]!r})")

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, range(0, len(runs), CHUNK)))
    return plans, usd


def hf_cmd():
    if shutil.which("hyperframes"):
        return ["hyperframes"]
    if shutil.which("npx"):
        return ["npx", "-y", "hyperframes"]
    return None


def build_page(plan, dur, work, theme):
    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in ("maplibre-gl.js", "maplibre-gl.css", "countries.geojson"):
        shutil.copy(MAPS_DIR / f, work / f)
    shutil.copy(HF_DIR / "gsap.min.js", work / "gsap.min.js")
    P = dict(plan, duration=round(dur, 3), theme=theme)
    page = (MAPS_DIR / "map_template.html").read_text().replace("__DUR__", f"{dur:.3f}").replace("__PLAN__", json.dumps(P))
    (work / "index.html").write_text(page)
    return work


def render_one(work, out, fps, timeout=1200):
    cmd = hf_cmd() + ["render", str(work), "--format", "mp4", "-o", str(out), "--fps", str(fps), "--workers", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not Path(out).exists():
        Path(out).unlink(missing_ok=True)
        raise RuntimeError((r.stderr or r.stdout or "hyperframes failed").strip()[-300:])


def cut(src, start, need, out, fps):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(src),
                    "-t", f"{need:.3f}", "-an", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                    "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, timeout=300)


def run_step(proj, beats, specs, vo_total, settings, keys, style, log, fetched):
    """Fill `fetched` for map beats with rendered map clips. Returns how many beats got a map."""
    if str(settings.get("maps_enabled", True)).lower() in ("false", "0", "no", "off"):
        return 0
    proj = Path(proj)
    runs = [r for r in find_runs(beats, specs, vo_total) if not all(fetched.get(b) for b in r["beats"])]
    if not runs:
        return 0
    if not hf_cmd():
        log("maps: HyperFrames/npx not found - map beats use stock footage")
        return 0
    mdir = proj / "maps"
    mdir.mkdir(exist_ok=True)
    plan_path = mdir / "plans.json"
    saved = read_json(plan_path, {}) or {}
    key = lambda r: ",".join(r["beats"])
    plans = {i: saved[key(r)] for i, r in enumerate(runs) if key(r) in saved}
    todo = [i for i in range(len(runs)) if i not in plans]
    if todo:
        title = (proj / "title.txt").read_text(encoding="utf-8").strip() if (proj / "title.txt").exists() else ""
        sub = [runs[i] for i in todo]
        got, usd = plan_runs(sub, title, settings, keys, log)
        for j, p in got.items():
            plans[todo[j]] = p
            saved[key(runs[todo[j]])] = p
        write_json(plan_path, saved)
        if usd:
            try:
                from . import costs
                costs.add(proj, "brain", usd=usd, maps=len(got))
            except Exception:
                pass
    theme, extra = map_look(style, settings, keys)
    fps = style["canvas"]["fps"]
    jobs = [(i, runs[i], plans[i]) for i in sorted(plans)]
    if not jobs:
        log("maps: no usable map plan - map beats use stock footage")
        return 0
    conc = max(1, int(settings.get("maps_parallel", 3) or 3))
    log(f"maps: rendering {len(jobs)} map clip(s) for {sum(len(r['beats']) for _, r, _ in jobs)} beats, {conc} at a time")
    lock, done, n_beats = threading.Lock(), [0], [0]
    t0 = time.time()

    def one(job):
        i, r, plan = job
        dur = (r["end"] - r["start"]) + PAD
        clip = mdir / f"run_{r['beats'][0]}_{len(r['beats'])}.mp4"
        try:
            if not clip.exists():
                render_one(build_page(tilt(dict(plan, **extra), theme), dur, mdir / f"work_{i}", theme), clip, fps)
                shutil.rmtree(mdir / f"work_{i}", ignore_errors=True)
            got = {}
            bstart = {str(b["id"]): b for b in beats}
            order = [str(b["id"]) for b in beats]
            for bid in r["beats"]:
                b = bstart[bid]
                k = order.index(bid)
                end = vo_total if k == len(order) - 1 else beats[k + 1]["start"]
                off, need = b["start"] - r["start"], (end - b["start"]) + PAD
                out = mdir / f"beat_{bid}.mp4"
                if not out.exists():
                    cut(clip, off, need, out, fps)
                got[bid] = {"file": str(out), "start": 0.0, "length": duration(out), "source": "map",
                            "id": f"map_{r['beats'][0]}", "page_url": "", "credit": "Natural Earth / MapLibre",
                            "title": plan.get("title") or "map", "license": "public domain", "src_height": 1080}
        except Exception as e:
            log(f"maps: run at beat {r['beats'][0]} failed, stock footage instead ({str(e)[:160]})")
            return
        with lock:
            fetched.update(got)
            done[0] += 1
            n_beats[0] += len(got)
            log(f"maps {done[0]}/{len(jobs)}", progress=done[0] / len(jobs))

    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, jobs))
    log(f"maps: {n_beats[0]} beats got a map ({done[0]}/{len(jobs)} clips, {time.time() - t0:.0f} s)")
    return n_beats[0]
