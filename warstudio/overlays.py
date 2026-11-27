"""On-screen callouts: time stamps, measures (imperial + metric), system ID labels.

Numbers are detected with regex (deterministic, never invented by the LLM). Labels come from the
shot spec (first mention of a named aircraft/ship/weapon). Rendering: HyperFrames transparent WebM
when available, Pillow PNG fallback otherwise.
"""
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import ROOT
from .config import DATA_DIR

OV_CACHE = DATA_DIR / "overlay_cache"     # rendered callouts shared between videos

UNITS = [
    (r"nautical miles?", "mi", lambda v: (v * 1.852, "km")),
    (r"miles an hour|miles per hour|mph", "mph", lambda v: (v * 1.609, "km/h")),
    (r"knots?", "knots", lambda v: (v * 1.852, "km/h")),
    (r"miles?|mi\b", "mi", lambda v: (v * 1.609, "km")),
    (r"feet|foot|ft\b", "ft", lambda v: (v * 0.3048, "m")),
    (r"yards?|yds?\b", "yd", lambda v: (v * 0.9144, "m")),
    (r"pounds|lbs?\b", "lbs", lambda v: (v * 0.4536, "kg")),
    (r"kilometers?|km\b", "km", None),
    (r"meters?|metres?", "m", None),
    (r"seconds", "sec", None),
]
NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
MEASURE_RE = re.compile(NUM + r"[\s-]*(" + "|".join(u[0] for u in UNITS) + r")", re.I)
_HHMM = r"(?:([01]?\d|2[0-3]):([0-5]\d)|([01]\d|2[0-3])([0-5]\d))(?!\d|[,.]\d|\s*(?:%s))" % "|".join(u[0] for u in UNITS)
TIME_RE = re.compile(r"\b(?:at|by|until|around)\s+" + _HHMM + r"|\b" + _HHMM + r"\s*(?:local time|hours|zulu)", re.I)


WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
SCALE = r"(million|billion|thousand|hundred)"
# "20 million barrels", "hundreds of fast boats", "3,000 troops", "twenty-one miles" handled by MEASURE first
STAT_RE = re.compile(r"\b(" + NUM + r"|" + "|".join(WORDNUM) + r")(?:[\s-]+" + SCALE + r")?\s+((?:[a-z-]+\s){0,2}?(?:"
                     r"barrels|troops|soldiers|sailors|marines|ships|warships|tankers|vessels|boats|aircraft|jets|"
                     r"fighters|bombers|drones|missiles|rockets|mines|torpedoes|submarines|tanks|launchers|"
                     r"people|casualties|bases|carriers|destroyers|frigates|helicopters|rounds|shells|warheads))\b", re.I)
PCT_RE = re.compile(NUM + r"\s*(%|percent)", re.I)
FRAC_RE = re.compile(r"\b(one|a) (fifth|quarter|third|half)\b(?= of)", re.I)
MONEY_RE = re.compile(r"\$\s?" + NUM + r"(?:\s*(million|billion|trillion))?", re.I)
YEAR_RE = re.compile(r"\b(?:in|since|by|of|back in)\s+((?:January|February|March|April|May|June|July|August|September|"
                     r"October|November|December)\s+)?((?:19[4-9]|20[0-3])\d)\b", re.I)


def _fmt(v):
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:,.1f}".rstrip("0").rstrip(".")
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def detect(text):
    """Return the first overlay found in text: {'kind': 'time'|'measure', 'line1', 'line2'} or None."""
    m = TIME_RE.search(text)
    if m:
        g = [x for x in m.groups() if x is not None]
        h, mi = g[0], g[1]
        return {"kind": "time", "line1": f"{int(h):02d}:{mi}", "line2": ""}
    m = MEASURE_RE.search(text)
    if not m:
        return detect_more(text)
    if m:
        raw, unit_txt = m.group(1), m.group(2).lower()
        val = float(raw.replace(",", ""))
        for pat, short, conv in UNITS:
            if re.fullmatch(pat, unit_txt, re.I) or re.match(pat, unit_txt, re.I):
                line1 = f"{raw} {short}"
                line2 = ""
                if conv:
                    mv, mu = conv(val)
                    line2 = f"{_fmt(mv)} {mu}"
                return {"kind": "measure", "line1": line1, "line2": line2}
    return None


def detect_more(text):
    """Extra callouts (v1.7): money, percentages / fractions, counts of military things, years."""
    m = MONEY_RE.search(text)
    if m:
        return {"kind": "measure", "line1": "$" + m.group(1) + ((" " + m.group(2).upper()) if m.group(2) else ""), "line2": "",
                "type": "money"}
    m = PCT_RE.search(text)
    if m:
        return {"kind": "measure", "line1": f"{m.group(1)}%", "line2": "", "type": "percent"}
    m = FRAC_RE.search(text)
    if m:
        return {"kind": "measure", "line1": {"fifth": "20%", "quarter": "25%", "third": "33%", "half": "50%"}[m.group(2).lower()],
                "line2": f"one {m.group(2).lower()}".upper(), "type": "percent"}
    m = YEAR_RE.search(text)
    if m:
        return {"kind": "time", "line1": ((m.group(1) or "").strip()[:3].upper() + " " if m.group(1) else "") + m.group(2),
                "line2": "", "type": "year"}
    m = STAT_RE.search(text)
    if m:
        n = m.group(1)
        if re.fullmatch(r"(19|20)\d\d", n) and not m.group(3):
            return None
        n = str(WORDNUM.get(n.lower(), n)) if not n[0].isdigit() else n
        scale = (" " + m.group(3).upper()) if m.group(3) else ""
        if n not in ("1",) or scale:
            return {"kind": "measure", "line1": f"{n}{scale}", "line2": m.group(4).strip().upper()[:26], "type": "count"}
    return None


SEAS = ["Strait of Hormuz", "Persian Gulf", "Gulf of Oman", "Red Sea", "Black Sea", "South China Sea", "Taiwan Strait",
        "Bab-el-Mandeb", "Suez Canal", "Arabian Sea", "Mediterranean", "Baltic Sea", "Pacific", "Atlantic", "Indian Ocean",
        "Gulf of Aden", "Caspian Sea", "Sea of Japan", "North Sea", "Arctic", "English Channel", "Gaza", "West Bank",
        "Crimea", "Donbas", "Kyiv", "Moscow", "Tehran", "Washington", "Beijing", "Pentagon", "Kremlin", "White House"]
_PLACES = None


def places():
    """Country names (from the bundled map) + seas, straits, capitals - longest first so 'Saudi Arabia' beats 'Arabia'."""
    global _PLACES
    if _PLACES is None:
        names = list(SEAS)
        try:
            from .config import ROOT
            geo = json.loads((ROOT / "styles" / "beyond_military" / "maps" / "countries.geojson").read_text())
            names += [f["properties"]["name"] for f in geo["features"] if len(f["properties"].get("name", "")) > 3]
        except Exception:
            pass
        names = sorted(set(names), key=len, reverse=True)
        _PLACES = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    return _PLACES


DESIGNATOR = re.compile(r"\b([A-Z]{1,4}-\d{1,3}[A-Z]?(?:\s+[A-Z][a-z]+)?)\b")      # MQ-9 Reaper, F-35, S-400
GROUPS = re.compile(r"\b(Revolutionary Guards?|Hezbollah|Houthis?|Hamas|NATO|CENTCOM|Pentagon|IRGC|Navy SEALs?|Marines)\b")


def place_label(text):
    m = DESIGNATOR.search(text) or GROUPS.search(text)
    if m:
        return {"kind": "label", "line1": m.group(1).upper()[:32], "line2": "", "type": "system"}
    m = places().search(text)
    return {"kind": "label", "line1": m.group(1).upper()[:32], "line2": "", "type": "place"} if m else None


def plan(beats, specs, style):
    """Choose at most one overlay per beat, respecting the minimum gap. Returns {beat_id: overlay}."""
    oc = style["overlays"]
    spec_by_id = {s["id"]: s for s in specs}
    out, last_end, seen_labels, place_count = {}, -99.0, {}, {}
    per_min = float(oc.get("max_per_minute", 99))
    times = []
    total = max((b.get("end") or b["start"]) for b in beats) if beats else 0
    hook = float(oc.get("hook_free_seconds", 30) or 0) if total >= 120 else 0.0   # short tests: no hook rule
    for b in beats:
        if b["start"] < hook:                        # the hook is clean: real footage + music only, no text
            continue
        if b["start"] - last_end < oc["min_gap_seconds"]:
            continue
        if sum(1 for t in times if b["start"] - t < 60) >= per_min:
            continue
        ov = detect(b["text"])
        if not ov:
            lab = (spec_by_id.get(b["id"], {}).get("label") or "").strip()
            again = float(oc.get("label_repeat_seconds", 0) or 0)       # a system label may come back much later
            last = seen_labels.get(lab.lower())
            if lab and len(lab) <= 32 and (last is None or (again and b["start"] - last >= again)):
                seen_labels[lab.lower()] = b["start"]
                ov = {"kind": "label", "line1": lab, "line2": ""}
        if not ov:                                   # a named place: country, sea, strait, capital
            pl = place_label(b["text"])
            again = float(oc.get("place_repeat_seconds", 150) or 0)
            if pl:
                last = seen_labels.get(pl["line1"].lower())
                cnt = place_count.get(pl["line1"], 0)
                if cnt < 3 and (last is None or (again and b["start"] - last >= again)):
                    place_count[pl["line1"]] = cnt + 1
                    seen_labels[pl["line1"].lower()] = b["start"]
                    ov = pl
        if ov:
            dur = min(b["end"] - b["start"], oc["max_hold_seconds"])
            ov["duration"] = round(max(0.8, dur), 3)
            ov["line1"] = (ov.get("line1") or "").upper()   # all overlay text in capitals
            ov["line2"] = (ov.get("line2") or "").upper()
            out[b["id"]] = ov
            last_end = b["start"] + ov["duration"]
            times.append(b["start"])
    return out


TYPE_START, TYPE_MAX = 0.12, 0.9


def type_times(ov):
    """Seconds (from the overlay's start) at which each letter appears - the template types at exactly these
    times and the typewriter sound clicks on them."""
    text = (ov.get("line1") or "") + (ov.get("line2") or "")
    step = min(0.055, TYPE_MAX / max(1, len(text)))
    return [round(TYPE_START + i * step, 3) for i, ch in enumerate(text) if not ch.isspace()]


# ---------- rendering ----------

MAC_FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
             "/Library/Fonts/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "C:/Windows/Fonts/arialbd.ttf"]


def _font(size):
    for f in MAC_FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render_png(ov, style, out_png):
    ov = dict(ov, line1=(ov.get("line1") or "").upper(), line2=(ov.get("line2") or "").upper())
    oc = style["overlays"]
    W, H = style["canvas"]["width"], style["canvas"]["height"]
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if ov["kind"] == "label":
        f = _font(54)
        tw = d.textlength(ov["line1"], font=f)
        x, y = 70, 70
        d.rectangle([x, y, x + 16, y + 92], fill=oc["box_color"])
        d.rectangle([x + 16, y, x + 16 + tw + 60, y + 92], fill=oc["label_bg"])
        d.text((x + 46, y + 16), ov["line1"], font=f, fill=oc["label_text"])
    else:
        f1, f2 = _font(64), _font(40)
        w = max(d.textlength(ov["line1"], font=f1), d.textlength(ov["line2"], font=f2) if ov["line2"] else 0) + 70
        h = 100 if not ov["line2"] else 160
        pos = oc["positions"][ov["kind"]]
        x = {"top_right": W - w - 70, "top_center": (W - w) / 2, "top_left": 70}[pos]
        y = 60
        d.rounded_rectangle([x, y, x + w, y + h], 12, fill=oc["box_color"])
        d.text((x + (w - d.textlength(ov["line1"], font=f1)) / 2, y + 14), ov["line1"], font=f1, fill=oc["text_color"])
        if ov["line2"]:
            d.text((x + (w - d.textlength(ov["line2"], font=f2)) / 2, y + 96), ov["line2"], font=f2, fill=oc["text_color"])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return out_png


def hyperframes_available():
    if not (shutil.which("hyperframes") or shutil.which("npx")):
        return False
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libvpx-vp9" in enc


def render_webm(ov, style, out_webm, work):
    """HyperFrames: HTML/GSAP composition -> transparent WebM. Returns path or None."""
    hf_dir = Path(style["_dir"]) / "hf"
    tpl_path = hf_dir / "overlay_template.html"
    if not tpl_path.exists():
        return None
    key = hashlib.md5(json.dumps(ov, sort_keys=True).encode()).hexdigest()[:10]
    proj = Path(work) / f"hf_{key}"
    proj.mkdir(parents=True, exist_ok=True)
    for name in ("fonts", "gsap.min.js"):
        dst = proj / name
        if not dst.exists():
            src = hf_dir / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)
    oc = style["overlays"]
    page = (tpl_path.read_text(encoding="utf-8")
            .replace("__DUR__", f"{ov['duration']:.3f}")
            .replace("__KIND__", ov["kind"])
            .replace("__POS__", oc["positions"].get(ov["kind"], "top_left"))
            .replace("__L1__", html.escape(ov["line1"]))
            .replace("__L2__", html.escape(ov.get("line2", ""))))
    (proj / "index.html").write_text(page, encoding="utf-8")
    cmd = ["hyperframes"] if shutil.which("hyperframes") else ["npx", "-y", "hyperframes"]
    cmd += ["render", str(proj), "--format", "webm", "-o", str(out_webm), "--fps", "30", "--quiet", "--workers", "2"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception:
        return None
    return out_webm if r.returncode == 0 and Path(out_webm).exists() else None


def _tpl_hash():
    try:
        return hashlib.md5((ROOT / "styles" / "beyond_military" / "hf" / "overlay_template.html").read_bytes()).hexdigest()[:8]
    except Exception:
        return "x"


def render_all(overlay_plan, style, work, log, use_hf=True):
    """Render every planned overlay. Returns {beat_id: {'path', 'type': 'webm'|'png', **ov}}."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    hf = use_hf and hyperframes_available()
    log(f"overlays: {len(overlay_plan)} ({'HyperFrames' if hf else 'PNG fallback'})")
    out = {}
    todo = []
    for bid, ov in overlay_plan.items():
        key = hashlib.md5(json.dumps({k: ov[k] for k in ("kind", "line1", "line2", "duration")}, sort_keys=True).encode()).hexdigest()[:12]
        webm, png = work / f"ov_{key}.webm", work / f"ov_{key}.png"
        shared = OV_CACHE / f"ov_{key}_{_tpl_hash()}.webm"          # same callout in an earlier video: copy it
        if hf and not webm.exists() and shared.exists():
            shutil.copy(shared, webm)
        if hf and webm.exists():
            out[bid] = dict(ov, path=str(webm), type="webm")
        elif not hf and png.exists():
            out[bid] = dict(ov, path=str(png), type="png")
        else:
            todo.append((bid, ov, webm, png))

    def do_hf(item):
        bid, ov, webm, png = item
        return bid, render_webm(ov, style, webm, work)

    if hf and todo:
        uniq = list({str(t[2]): t for t in todo}.values())     # identical callouts render once
        first = do_hf(uniq[0])                       # one alone: proves HyperFrames works before going parallel
        made = {str(uniq[0][2]): first[1]}
        if first[1] is None:
            log("HyperFrames render failed; using PNG overlays")
            hf = False
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(5) as ex:
                for item, (bid, res) in zip(uniq[1:], ex.map(do_hf, uniq[1:])):
                    made[str(item[2])] = res
        for bid, ov, webm, png in todo:
            if made.get(str(webm)):
                out[bid] = dict(ov, path=str(webm), type="webm")
                try:
                    OV_CACHE.mkdir(parents=True, exist_ok=True)
                    dst = OV_CACHE / f"{webm.stem}_{_tpl_hash()}.webm"
                    if not dst.exists():
                        shutil.copy(webm, dst)
                except Exception:
                    pass
    for bid, ov, webm, png in todo:
        if bid not in out:
            out[bid] = dict(ov, path=str(png if png.exists() else render_png(ov, style, png)), type="png")
    return out
