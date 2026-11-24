"""Checks the segment that was actually cut (not the search thumbnail) for things that don't belong in a
clean B-roll shot: burned-in text, lower-thirds, subtitles, title cards, talking-head interviews,
black frames, slates/blank cards and colour bars.

Layers (all free):
- always: ffmpeg blackdetect + pixel statistics (blank card, colour bars)
- macOS: Apple Vision text recognition + face detection (built into the OS, no download, no API)
- optional: the vision reranker (Groq/Gemini) looking at the 3 frames, when segment_check = "vlm"

Result: {"problems": [...], "crop_bottom": <fraction or 0>}. A lower-third that sits low enough is not a
problem: render zooms in from the top so the strip is pushed out of frame.
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PROBLEMS = ("burned_text", "lower_third", "subtitles", "title_card", "logo_banner", "talking_head",
            "blur", "black", "color_bars", "blank_card")
MAX_CROP = 0.18             # at most 18 % of the frame height may be zoomed away to hide a lower-third
SAMPLE_AT = (0.15, 0.5, 0.85)


# ---------------------------------------------------------------- frames + pixel checks
def extract_frames(path, length, outdir):
    outs = []
    for i, f in enumerate(SAMPLE_AT):
        o = Path(outdir) / f"f{i}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, length * f):.2f}", "-i", str(path),
                        "-frames:v", "1", "-vf", "scale=640:-2", str(o)], capture_output=True, timeout=60)
        if o.exists():
            outs.append(str(o))
    return outs


def black_fraction(path, length):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vf", "blackdetect=d=0.2:pix_th=0.10",
                        "-an", "-f", "null", "-"], capture_output=True, text=True, timeout=120)
    tot = sum(float(x) for x in re.findall(r"black_duration:([\d.]+)", p.stderr))
    return tot / max(length, 0.01)


def pixel_problems(img_path):
    a = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32)
    g = a.mean(axis=2)
    if g.std() < 10:
        return ["blank_card"]
    top = a[: int(a.shape[0] * 0.6)]
    cols = np.array_split(top, 7, axis=1)
    means = np.array([c.reshape(-1, 3).mean(0) for c in cols])
    stds = np.array([c.reshape(-1, 3).std(0).mean() for c in cols])
    sat = (means.max(1) - means.min(1))
    if (stds < 18).all() and (sat > 60).sum() >= 4 and np.abs(np.diff(means, axis=0)).sum(1).min() > 40:
        return ["color_bars"]
    return []


# ---------------------------------------------------------------- Apple Vision (macOS only)
def vision_available():
    try:
        import Vision  # noqa: F401
        return True
    except Exception:
        return False


def _vision_handler(path):
    import Vision
    from Foundation import NSURL
    return Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(path)), None)


def vision_text(path):
    """[(text, confidence, (x, y, w, h))], boxes normalised, origin bottom-left (Vision convention)."""
    import Vision
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(1)          # fast
    req.setUsesLanguageCorrection_(False)
    _vision_handler(path).performRequests_error_([req], None)
    out = []
    for r in req.results() or []:
        c = r.topCandidates_(1)
        if not c:
            continue
        bb = r.boundingBox()
        out.append((str(c[0].string()), float(c[0].confidence()),
                    (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)))
    return out


def vision_faces(path):
    """[area fraction of each face]"""
    import Vision
    req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    _vision_handler(path).performRequests_error_([req], None)
    return [float(r.boundingBox().size.width * r.boundingBox().size.height) for r in (req.results() or [])]


def classify_text(boxes):
    """Turn OCR boxes into problems. Small markings (hull numbers, 'U.S. AIR FORCE' on a fuselage, a corner
    bug) are allowed; wide strips at the bottom/top and big text are not."""
    probs, crop = set(), 0.0
    boxes = [b for b in boxes if b[1] >= 0.5 and len(b[0].strip()) >= 2]
    if not boxes:
        return [], 0.0
    chars = sum(len(t) for t, _, _ in boxes)
    area = sum(w * h for _, _, (x, y, w, h) in boxes)
    if area > 0.04 or chars >= 45:
        probs.add("title_card")
    for t, _, (x, y, w, h) in boxes:
        top_edge = y + h
        if top_edge <= 0.35 and (w >= 0.2 or len(t) >= 14):
            if 0.3 <= x + w / 2 <= 0.7 and w >= 0.35:
                probs.add("subtitles")
            else:
                probs.add("lower_third")
            crop = max(crop, top_edge)
        elif y >= 0.82 and w >= 0.3:
            probs.add("logo_banner")
        elif w >= 0.3 and h >= 0.05:
            probs.add("burned_text")
    return sorted(probs), round(crop, 3)


# ---------------------------------------------------------------- VLM layer
VLM_PROMPT = ("These are 3 frames from one B-roll shot for a military documentary. List what makes it NOT a clean "
              "B-roll shot. Allowed problems: burned_text (captions or text added in editing), lower_third (name/title "
              "strip), subtitles, title_card (text slate or card), logo_banner, talking_head (a person speaking to camera "
              "or interview), blur (blurred/pixelated area), black, color_bars. Markings painted on vehicles are fine. "
              'Return JSON {"problems": [..]} (empty list if clean).')


def vlm_problems(reranker, frames, workdir):
    from .steps.rank import make_collage
    res, _ = reranker.pick(lambda tile: make_collage(frames, Path(workdir) / "check.jpg", tile), VLM_PROMPT,
                           tag="segment check: ")
    if not res:
        return None
    return [p for p in (res.get("problems") or []) if p in PROBLEMS]


# ---------------------------------------------------------------- main entry
def check_segment(path, length, mode="auto", reranker=None):
    """mode: auto (pixels + Apple Vision when present) | vlm (also ask the vision model) | local | off"""
    if mode == "off":
        return {"problems": [], "crop_bottom": 0.0, "by": "off"}
    side = Path(str(path) + ".check.json")
    if side.exists():
        try:
            return json.loads(side.read_text())
        except Exception:
            pass
    probs, crop, by = set(), 0.0, ["pixels"]
    if black_fraction(path, length) > 0.25:
        probs.add("black")
    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames(path, length, td)
        for f in frames:
            probs.update(pixel_problems(f))
        if frames and vision_available():
            by.append("apple-vision")
            faces_hit = 0
            for f in frames:
                try:
                    p, c = classify_text(vision_text(f))
                    probs.update(p)
                    crop = max(crop, c)
                    if max(vision_faces(f) or [0]) >= 0.03:
                        faces_hit += 1
                except Exception:
                    pass
            if faces_hit >= 2:
                probs.add("talking_head")
        if frames and mode == "vlm" and reranker is not None and reranker.active:
            v = vlm_problems(reranker, frames, td)
            if v is not None:
                by.append(reranker.active)
                probs.update(v)
    result = {"problems": sorted(probs), "crop_bottom": 0.0, "by": "+".join(by)}
    # fix 4: a low lower-third alone is hidden by zooming in from the top
    if set(result["problems"]) <= {"lower_third"} and result["problems"] and 0 < crop <= MAX_CROP - 0.02:
        result = {"problems": [], "crop_bottom": round(crop + 0.02, 3), "by": result["by"]}
    try:
        side.write_text(json.dumps(result))
    except Exception:
        pass
    return result
