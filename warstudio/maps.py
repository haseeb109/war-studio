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

from .config import ROOT, DATA_DIR

CACHE_DIR = DATA_DIR / "map_cache"      # same map (plan + style + length) in another video -> no new render
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


CAMERAS = ["follow_arrow", "point_hop", "orbit", "push_in", "pull_back", "pan_in", "side_slide", "low_rise",
           "crane", "drift"]
LABEL_FX = ["drop", "type", "slide"]


def camera(plan, k, seed="", move=None):
    """Every map gets its own camera move and label effect (k = its place in the video). Nothing starts from the
    whole map any more: every move begins near the place. Moves that need an arrow or two places are used only when
    the plan has them; the order is shuffled per video and the same move never comes twice in a row."""
    import hashlib
    import random
    rnd = random.Random(hashlib.md5(str(seed).encode()).hexdigest())
    moves = CAMERAS[:]
    rnd.shuffle(moves)
    ok = [m for m in moves if not (m == "follow_arrow" and not plan.get("arrows"))
          and not (m == "point_hop" and len(plan.get("points") or []) < 2)]
    move = move if move in ok else ok[k % len(ok)]
    b = dict(plan["to"])
    z, (lon, lat) = b["zoom"], b["center"]
    span = 360.0 / (2 ** z) * 0.9                      # about one screen width at the end zoom, in degrees
    near = lambda dz=0, dx=0.0, dy=0.0, **kw: dict({"center": [lon + span * dx, lat + span * dy], "zoom": round(z + dz, 2),
                                                    "pitch": 0, "bearing": 0}, **kw)
    ease = "inout"
    if move == "push_in":                              # short push in and a slight tip forward (not from the whole map)
        f, t = near(-1.2), near(0, pitch=22)
    elif move == "orbit":                              # circles around the place
        f, t = near(-0.4, pitch=50, bearing=-55), near(0, pitch=50, bearing=40)
    elif move == "pull_back":                          # starts on the place, pulls out to show the region around it
        f, t = near(0.6, pitch=45, bearing=20), near(-1.3, pitch=8)
    elif move == "pan_in":                             # slides in from the side while closing in
        f, t = near(-0.6, 1.3, 0.3), near()
    elif move == "side_slide":                         # tilted, sliding across
        f, t = near(-0.2, -1.0, -0.4, pitch=40, bearing=-14), near(0, pitch=40, bearing=12)
    elif move == "low_rise":                           # low angle, rises to look straight down
        f, t = near(-0.1, pitch=62, bearing=35), near(0.2)
    elif move == "crane":                              # from straight down, tilts up toward the horizon
        f, t = near(0.3), near(-0.4, pitch=58, bearing=-25)
    elif move == "drift":                              # slow steady drift across the area
        f, t = near(0, 0.6, 0.35, pitch=30, bearing=12), near(0.2, -0.6, -0.35, pitch=30, bearing=-12)
        ease = "linear"
    else:                                              # follow_arrow / point_hop are driven by the page; keep a sane frame
        f, t = near(-0.3, pitch=40), near(0, pitch=45)
    for v in (f, t):
        v["zoom"] = max(1.0, min(9.5, v["zoom"]))
    return dict(plan, **{"from": f, "to": t, "camera": move, "cam": move, "ease": ease,
                         "label_fx": LABEL_FX[(k + rnd.randrange(3)) % 3]})


def move_fit(plan):
    """How well each camera move suits this map (higher = better; 0 = cannot be used)."""
    pts, arr, rings = len(plan.get("points") or []), len(plan.get("arrows") or []), len(plan.get("rings") or [])
    hl, z = len(plan.get("highlight") or []), float((plan.get("to") or {}).get("zoom", 6))
    wide = z <= 5.0 or hl >= 2
    one = pts <= 1 and not arr
    return {
        "follow_arrow": 3.0 if arr else 0,                          # an attack / route: ride along the arrow
        "point_hop": (2.6 + (0.4 if not arr else 0)) if pts >= 2 else 0,   # several named places: hop between them
        "orbit": 2.4 if rings else (1.6 if one else 1.0),           # a range / strike zone: circle around it
        "push_in": 2.2 if one else (1.6 if rings else 1.2),         # one target: close in on it
        "low_rise": 1.8 if one else 1.0,
        "pull_back": 2.2 if wide else 1.0,                          # a region / several countries: reveal the context
        "crane": 1.8 if wide else 0.9,
        "drift": 2.0 if z <= 4.5 else 1.0,                          # big sea / ocean areas: slow drift across
        "pan_in": 1.6 if arr else 1.3,
        "side_slide": 1.6 if (arr or pts >= 2) else 1.1,
    }


def choose_moves(plans, seed=""):
    """One move per map, in video order: the best-suited move wins, a move is used at most twice per video and
    never twice in a row, so every map looks different but each one gets a move that fits what it shows."""
    import hashlib
    import random
    rnd = random.Random(hashlib.md5(("moves" + str(seed)).encode()).hexdigest())
    uses, out, last = {}, [], None
    for plan in plans:
        fit = move_fit(plan)
        best = max((m for m in CAMERAS if fit[m] > 0 and uses.get(m, 0) < 2 and m != last),
                   key=lambda m: fit[m] - 1.2 * uses.get(m, 0) + rnd.random() * 0.3, default=None)
        best = best or max(CAMERAS, key=lambda m: fit[m])
        uses[best] = uses.get(best, 0) + 1
        out.append(best)
        last = best
    return out


_GEO = None


def _country_at(lon, lat):
    """(iso, name, label point) of the country under a point, from the bundled countries.geojson."""
    global _GEO
    if _GEO is None:
        try:
            _GEO = json.loads((MAPS_DIR / "countries.geojson").read_text())["features"]
        except Exception:
            _GEO = []

    def inside(ring):
        c, n = False, len(ring)
        for i in range(n):
            (x1, y1), (x2, y2) = ring[i][:2], ring[i - 1][:2]
            if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1:
                c = not c
        return c
    for f in _GEO:
        g = f.get("geometry") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
        for poly in polys:
            if poly and inside(poly[0]):
                xs, ys = [p[0] for p in poly[0]], [p[1] for p in poly[0]]
                return f["properties"].get("iso"), f["properties"].get("name", ""), [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2]
    best, bd = None, 9.0                                     # at sea (a strait, a port's water): the nearest coast
    for f in _GEO:
        g = f.get("geometry") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
        for poly in polys:
            for x, y in (q[:2] for q in poly[0]):
                d = (x - lon) ** 2 + (y - lat) ** 2
                if d < bd:
                    bd, best = d, (f, poly[0])
    if best:
        f, ring = best
        xs, ys = [q[0] for q in ring], [q[1] for q in ring]
        return f["properties"].get("iso"), f["properties"].get("name", ""), [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2]
    return None, None, None


def ensure_detail(plan, colors=None):
    """Every map always carries the full detailing: a pin with its label, a highlighted country with its name,
    a range ring on the main place, and an arrow when there are two places. Things the brain left out are added."""
    colors = colors or {"enemy": "#b3261e", "ally": "#3a5a80", "neutral": "#4b5a6b"}
    p = json.loads(json.dumps(plan))
    for k in ("highlight", "countries", "points", "arrows", "rings"):
        p[k] = list(p.get(k) or [])
    lon, lat = p["to"]["center"]
    if not p["points"]:
        nm = (p.get("title") or "TARGET").title()
        if p["arrows"]:
            lon, lat = p["arrows"][0]["to"]
        p["points"].append({"name": nm[:24], "lon": round(lon, 3), "lat": round(lat, 3)})
    main = p["points"][0]
    if not p["highlight"] or not p["countries"]:
        iso, name, at = _country_at(main["lon"], main["lat"])
        if not iso:
            iso, name, at = _country_at(lon, lat)
        if iso and not p["highlight"]:
            p["highlight"].append({"iso": iso, "color": colors.get("enemy")})
        if name and not p["countries"]:
            p["countries"].append({"name": name.upper()[:22], "at": at})
    if not p["arrows"] and len(p["points"]) >= 2:
        a, b = p["points"][0], p["points"][1]
        p["arrows"].append({"from": [a["lon"], a["lat"]], "to": [b["lon"], b["lat"]]})
    if not p["rings"]:
        z = float(p["to"].get("zoom", 6))
        km = max(5, min(1500, round(360.0 / (2 ** z) * 111 * 0.12)))   # ~1/8 of the screen width
        tgt = p["arrows"][0]["to"] if p["arrows"] else [main["lon"], main["lat"]]
        p["rings"].append({"lon": tgt[0], "lat": tgt[1], "km": km})
    return p


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


SCENES = {
    "red_sea": dict(SAMPLE, title="RED SEA", to={"center": [43.2, 13.7], "zoom": 6.6},
                    arrows=[{"from": [42.95, 14.8], "to": [43.4, 12.65]}],
                    points=[{"name": "Hodeidah", "lon": 42.95, "lat": 14.8}, {"name": "Bab-el-Mandeb", "lon": 43.4, "lat": 12.6}],
                    rings=[{"lon": 43.4, "lat": 12.6, "km": 60}]),
    "hormuz": {"title": "STRAIT OF HORMUZ", "from": {"center": [55.5, 26.5], "zoom": 5}, "to": {"center": [56.3, 26.4], "zoom": 7.4},
               "highlight": [{"iso": "IRN", "color": "#b3261e"}, {"iso": "OMN", "color": "#3a5a80"}, {"iso": "ARE", "color": "#3a5a80"}],
               "countries": [{"name": "IRAN", "at": [57.5, 27.6]}, {"name": "OMAN", "at": [56.9, 24.6]}],
               "points": [{"name": "Bandar Abbas", "lon": 56.27, "lat": 27.18}, {"name": "Musandam", "lon": 56.25, "lat": 26.2}],
               "arrows": [{"from": [56.27, 27.18], "to": [56.5, 26.45]}], "rings": [{"lon": 56.27, "lat": 27.18, "km": 70}]},
    "falklands": {"title": "FALKLAND ISLANDS", "from": {"center": [-62, -51.5], "zoom": 4}, "to": {"center": [-62.5, -51.7], "zoom": 6.2},
                  "highlight": [{"iso": "ARG", "color": "#b3261e"}, {"iso": "FLK", "color": "#3a5a80"}],
                  "countries": [{"name": "ARGENTINA", "at": [-68.5, -49.5]}, {"name": "FALKLANDS", "at": [-59.5, -51.3]}],
                  "points": [{"name": "Rio Grande", "lon": -67.7, "lat": -53.8}, {"name": "Stanley", "lon": -57.85, "lat": -51.69}],
                  "arrows": [{"from": [-67.7, -53.8], "to": [-58.2, -51.8]}], "rings": [{"lon": -57.85, "lat": -51.69, "km": 120}]},
    "midway": {"title": "PACIFIC OCEAN", "from": {"center": [-170, 25], "zoom": 3.5}, "to": {"center": [-172, 26], "zoom": 4.6},
               "highlight": [], "countries": [],
               "points": [{"name": "Midway Atoll", "lon": -177.37, "lat": 28.21}, {"name": "Pearl Harbor", "lon": -157.95, "lat": 21.35}],
               "arrows": [{"from": [-157.95, 21.35], "to": [-176.8, 28.0]}], "rings": [{"lon": -177.37, "lat": 28.21, "km": 300}]},
}


def render_test(style, settings, keys, out, scene="red_sea", move="random", label_fx="", dur=5.0, text="", log=print,
                caption=False):
    """Maps Test playground: one clip with a chosen place, camera move and label effect. Returns what was used."""
    import random as _r
    if not hf_cmd():
        raise RuntimeError("HyperFrames/npx not found")
    theme, _ = map_look(style, settings, keys)
    if text.strip():                                   # the real path: the brain plans the map from a sentence
        plans, _usd = plan_runs([{"text": text.strip()}], "", settings, keys, log)
        if not plans:
            raise RuntimeError("the brain gave no usable map plan for this text")
        base = plans[0]
    else:
        base = json.loads(json.dumps(SCENES.get(scene) or SCENES["red_sea"]))
    seed = str(_r.random())
    if move in ("", "random"):
        plan = camera(base, _r.randrange(20), seed)
    else:
        plan = next((camera(base, k, seed) for k in range(40) if camera(base, k, seed)["cam"] == move), None)
        if plan is None:
            raise RuntimeError(f"camera move '{move}' needs an arrow / two places that this map does not have")
    if label_fx:
        plan["label_fx"] = label_fx
    if caption:                                        # showreel: say on screen which combo this is
        nice = lambda x: x.replace("_", " ").upper()
        plan["caption"] = [f"CAMERA: {nice(plan['cam'])}", f"LABELS: {nice(plan['label_fx'])}",
                           "EFFECTS: GLOW · PING · FLOW · RADAR"]
    work = build_page(plan, float(dur), Path(out).parent / ("work_" + Path(out).stem), theme)
    render_one(work, out, style["canvas"]["fps"], timeout=600)
    shutil.rmtree(work, ignore_errors=True)
    return {"camera": plan["cam"], "label_fx": plan["label_fx"], "title": plan.get("title", ""), "seconds": float(dur)}


def render_sample(style, settings, keys, out):
    """Settings > Render test map: a 6 s Red Sea clip with the current map look."""
    if not hf_cmd():
        raise RuntimeError("HyperFrames/npx not found")
    theme, extra = map_look(style, settings, keys)
    import time as _t
    plan = camera(dict(SAMPLE, **extra), int(_t.time()) % len(CAMERAS), seed=_t.time())
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
Every map MUST have: at least one point (the place the narration is about), at least one highlighted country,
its name in countries, and a ring around the main place or the strike / threat zone; add arrows for any attack,
movement or route. Use real places from the narration. Every run must be in the reply."""


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


def select_runs(runs, beats, specs, vo_total, settings, log=None):
    """User rule (v1.7): a 20-30 min video gets 5-6 maps, each 4-5 s, and one inside the first minute.
    Picks map runs spread over the video; each run is stretched/trimmed to 4-5 s of beats."""
    mins = vo_total / 60.0
    n = int(settings.get("maps_per_video", 0) or 0) or max(5 if mins >= 18 else 1, min(6, round(mins / 3.8)))
    lo, hi = float(settings.get("map_min_seconds", 4.0)), float(settings.get("map_max_seconds", 5.0))
    order = [str(b["id"]) for b in beats]
    starts = {str(b["id"]): b["start"] for b in beats}
    ends = {str(b["id"]): (vo_total if i == len(beats) - 1 else beats[i + 1]["start"]) for i, b in enumerate(beats)}
    texts = {str(b["id"]): b["text"] for b in beats}

    def shape(first):
        """consecutive beats starting at `first` (or up to 2 beats earlier) whose total is lo..hi seconds, nearest to
        the middle; None when no such set exists (LOCKED rule: a map is never shorter than lo or longer than hi)"""
        i0 = order.index(first)
        best = None
        for st in range(i0, max(-1, i0 - 3), -1):
            for en in range(st, min(len(order), st + 6)):
                d = ends[order[en]] - starts[order[st]]
                if d > hi + 1e-6:
                    break
                if d >= lo - 1e-6 and en >= i0:
                    score = abs(d - (lo + hi) / 2) + 0.2 * (i0 - st)
                    if best is None or score < best[0]:
                        best = (score, st, en)
        if not best:
            return None
        ids = order[best[1]:best[2] + 1]
        return {"beats": ids, "start": starts[ids[0]], "end": ends[ids[-1]], "text": " ".join(texts[x] for x in ids)}

    cands = [c for c in (shape(r["beats"][0]) for r in runs) if c]
    chosen = []
    hook = 30.0 if vo_total >= 120 else 0.0                            # first 30 s = clean hook (real footage only)
    early = [c for c in cands if hook <= c["start"] < 60]
    if early:
        chosen.append(early[0])
    elif beats:
        # no geography in the first minute: the map goes on the second sentence of the hook, about the title's region
        for b in beats:
            c = shape(str(b["id"])) if max(4.0, hook) <= b["start"] < 58 else None
            if c:
                chosen.append(dict(c, forced=True))
                break
    # the rest: as evenly spread as possible over the video
    for k in range(1, n):
        target = vo_total * k / n + (vo_total / n) / 2
        space = min(45.0, vo_total / (2 * n))
        free = [c for c in cands if all(abs(c["start"] - x["start"]) > space and not (set(c["beats"]) & set(x["beats"]))
                                        for x in chosen)]
        if not free:
            break
        chosen.append(min(free, key=lambda c: abs(c["start"] - target)))
    chosen.sort(key=lambda c: c["start"])
    if log:
        log(f"maps: {len(chosen)} map(s) for a {mins:.0f} min video (rule: {n}, {lo:.0f}-{hi:.0f} s each, one in the first "
            f"minute) out of {len(runs)} map moments")
    return chosen


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
    return ensure_detail(out, colors)


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


def cache_key(plan, dur, theme):
    import hashlib
    tpl = (MAPS_DIR / "map_template.html").read_bytes()
    raw = json.dumps([plan, round(dur, 2), theme], sort_keys=True).encode() + tpl
    return hashlib.sha1(raw).hexdigest()[:16]


def render_cached(plan, dur, theme, clip, work, fps):
    """Render one map clip, or copy it from the cache if the exact same map was made before."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hit = CACHE_DIR / f"{cache_key(plan, dur, theme)}.mp4"
    if hit.exists():
        shutil.copy(hit, clip)
        return True
    render_one(build_page(plan, dur, work, theme), clip, fps)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copy(clip, hit)
    return False


def build_page(plan, dur, work, theme):
    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in ("maplibre-gl.js", "maplibre-gl.css", "countries.geojson"):
        shutil.copy(MAPS_DIR / f, work / f)
    shutil.copy(HF_DIR / "gsap.min.js", work / "gsap.min.js")
    P = dict(plan, duration=round(dur, 3), theme=theme)
    P.setdefault("pixel_ratio", 0.75)       # the map is drawn at 3/4 size and scaled up: faster, a little softer
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
    runs = select_runs(find_runs(beats, specs, vo_total), beats, specs, vo_total, settings, log)
    runs = [r for r in runs if not all((fetched.get(b) or {}).get("source") == "map" for b in r["beats"])]
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
    moves = choose_moves([plans[i] for i, _, _ in jobs], seed=proj.name)
    log("maps: camera per map (by what it shows): " + ", ".join(moves))
    conc = max(1, int(settings.get("maps_parallel", 3) or 3))
    log(f"maps: rendering {len(jobs)} map clip(s) for {sum(len(r['beats']) for _, r, _ in jobs)} beats, {conc} at a time")
    lock, done, n_beats, cached = threading.Lock(), [0], [0], [0]
    t0 = time.time()

    def one(job):
        i, r, plan = job
        dur = (r["end"] - r["start"]) + PAD
        clip = mdir / f"run_{r['beats'][0]}_{len(r['beats'])}.mp4"
        try:
            pos = [j for j, _, _ in jobs].index(i)
            if not clip.exists() and render_cached(camera(dict(plan, **extra), pos, seed=proj.name, move=moves[pos]), dur, theme, clip,
                                                    mdir / f"work_{i}", fps):
                with lock:
                    cached[0] += 1
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
    log(f"maps: {n_beats[0]} beats got a map ({done[0]}/{len(jobs)} clips, {cached[0]} reused from earlier videos, "
        f"{time.time() - t0:.0f} s)")
    return n_beats[0]
