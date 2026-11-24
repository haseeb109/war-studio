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


def plan(beats, specs, style):
    """Choose at most one overlay per beat, respecting the minimum gap. Returns {beat_id: overlay}."""
    oc = style["overlays"]
    spec_by_id = {s["id"]: s for s in specs}
    out, last_end, seen_labels = {}, -99.0, set()
    for b in beats:
        if b["start"] - last_end < oc["min_gap_seconds"]:
            continue
        ov = detect(b["text"])
        if not ov:
            lab = (spec_by_id.get(b["id"], {}).get("label") or "").strip()
            if lab and lab.lower() not in seen_labels and len(lab) <= 32:
                seen_labels.add(lab.lower())
                ov = {"kind": "label", "line1": lab, "line2": ""}
        if ov:
            dur = min(b["end"] - b["start"], oc["max_hold_seconds"])
            ov["duration"] = round(max(0.8, dur), 3)
            out[b["id"]] = ov
            last_end = b["start"] + ov["duration"]
    return out


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
            with ThreadPoolExecutor(3) as ex:
                for item, (bid, res) in zip(uniq[1:], ex.map(do_hf, uniq[1:])):
                    made[str(item[2])] = res
        for bid, ov, webm, png in todo:
            if made.get(str(webm)):
                out[bid] = dict(ov, path=str(webm), type="webm")
    for bid, ov, webm, png in todo:
        if bid not in out:
            out[bid] = dict(ov, path=str(png if png.exists() else render_png(ov, style, png)), type="png")
    return out
