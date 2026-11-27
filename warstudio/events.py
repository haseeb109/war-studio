"""Event packs: breaking-news footage (1-2 viral clips) + news screenshots for a current attack.

assets/events/<slug>/videos/*  assets/events/<slug>/images/*  assets/events/<slug>/event.json
A video with an event pack shows the hero footage 4 times (first inside the first 20 s, then spread out, each time
with another edit: original / slow-motion replay / framed / montage) and the screenshots as article cards with the
matching lines highlighted, alternating with the footage: V . I . V . I . V . I . V. Everything else stays DVIDS
library footage. Any failure here leaves those beats with library footage (never breaks a video).
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .config import ROOT
from .util import duration, read_json, write_json

EVENTS_DIR = ROOT / "assets" / "events"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".heic"}
COUNTRIES = {"USA": "United States", "RUS": "Russia", "UKR": "Ukraine", "IRN": "Iran", "SAU": "Saudi Arabia",
             "ARE": "United Arab Emirates"}
HERO_USES = 4
STYLES = ["original", "replay", "framed", "montage"]      # the 4 uses of the hero footage, in this order
W, H = 1920, 1080


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:60] or "event"


def pack_dir(slug):
    return EVENTS_DIR / Path(slug).name


def files(slug):
    d = pack_dir(slug)
    vids = sorted(p for p in (d / "videos").glob("*") if p.suffix.lower() in VIDEO_EXT) if (d / "videos").is_dir() else []
    imgs = sorted(p for p in (d / "images").glob("*") if p.suffix.lower() in IMAGE_EXT) if (d / "images").is_dir() else []
    return vids, imgs


def list_packs():
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(EVENTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir():
            meta = read_json(d / "event.json", {}) or {}
            v, i = files(d.name)
            out.append(dict(meta, slug=d.name, videos=len(v), images=len(i)))
    return out


def create_pack(title, countries=None, place="", date=""):
    slug = time.strftime("%Y%m%d") + "-" + slugify(title)
    d = pack_dir(slug)
    (d / "videos").mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(parents=True, exist_ok=True)
    meta = {"title": title, "countries": [c for c in (countries or []) if c in COUNTRIES], "place": place,
            "date": date or time.strftime("%Y-%m-%d"), "credits": {},
            "_note": "credits: {\"file name\": \"Source / owner\"} - shown on screen when that file is used"}
    write_json(d / "event.json", meta)
    return dict(meta, slug=slug)


# ---------------------------------------------------------------- where the footage and the cards go
def plan_slots(vo_total, n_videos, n_images, hero_uses=HERO_USES):
    """[(seconds, kind, file index, style)]: V1 inside the first 20 s, then I . V . I . V . I . V spread evenly
    from ~1 min to ~85% of the video (the user's pattern). Images are each used once."""
    if not n_videos and not n_images:
        return []
    T = float(vo_total)
    seq = []
    if n_videos:
        seq.append((min(4.0, T * 0.1), "video", 0, STYLES[0]))   # 0-4 s: voice over the cold open
    rest, vi, ii = [], 1, 0
    while (n_videos and vi < hero_uses) or ii < n_images:
        if ii < n_images:
            rest.append(("image", ii, "card"))
            ii += 1
        if n_videos and vi < hero_uses:
            rest.append(("video", vi % n_videos, STYLES[vi % len(STYLES)]))
            vi += 1
        if len(rest) > 12:
            break
    if not rest:
        return seq
    a, b = (min(60.0, T * 0.15), T * 0.85) if T > 120 else (T * 0.3, T * 0.9)
    step = (b - a) / max(1, len(rest) - 1) if len(rest) > 1 else 0
    for k, (kind, idx, style) in enumerate(rest):
        seq.append((a + k * step, kind, idx, style))
    return seq


def assign_beats(slots, beats, vo_total, taken, lens):
    """Each slot -> whole beats from the first beat at/after its time, covering lens[kind] seconds (4-8 s)."""
    order = [str(b["id"]) for b in beats]
    starts = {str(b["id"]): b["start"] for b in beats}
    ends = {str(b["id"]): (vo_total if i == len(beats) - 1 else beats[i + 1]["start"]) for i, b in enumerate(beats)}
    used, out = set(taken), []
    for t, kind, idx, style in slots:
        i = next((k for k, bid in enumerate(order) if starts[bid] >= t - 0.01 and bid not in used), None)
        if i is None:
            continue
        ids = []
        while i < len(order) and order[i] not in used:
            ids.append(order[i])
            if ends[order[i]] - starts[ids[0]] >= lens[kind]:
                break
            i += 1
        if not ids:
            continue
        used |= set(ids)
        out.append({"kind": kind, "idx": idx, "style": style, "beats": ids, "start": starts[ids[0]],
                    "end": ends[ids[-1]]})
    return out


# ---------------------------------------------------------------- drawing helpers
def _font(size):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label_png(path, tag, text="", accent=(255, 196, 0)):
    """Transparent 1920x1080 PNG with a tag box (e.g. FOOTAGE) and a credit line, bottom left."""
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f1, f2 = _font(34), _font(28)
    x, y = 60, H - 130
    tw = d.textlength(tag, font=f1)
    d.rounded_rectangle([x, y, x + tw + 36, y + 56], 8, fill=accent + (255,))
    d.text((x + 18, y + 9), tag, font=f1, fill=(0, 0, 0, 255))
    if text:
        cw = d.textlength(text, font=f2)
        d.rounded_rectangle([x + tw + 48, y + 6, x + tw + 48 + cw + 30, y + 50], 8, fill=(0, 0, 0, 190))
        d.text((x + tw + 63, y + 13), text, font=f2, fill=(255, 255, 255, 255))
    im.save(path)
    return path


def _ff(args, timeout=600):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True, capture_output=True, timeout=timeout)


FILL = (f"split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=28:2,"
        f"eq=brightness=-0.12[bg];[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2")       # vertical phone video -> blurred copy fills the sides


def render_video(src, style, dur, out, fps, label, place=""):
    """One use of the hero footage, `dur` seconds, 1920x1080, with sound where it makes sense."""
    L = max(0.5, duration(src))
    tag = {"original": "FOOTAGE", "replay": "REPLAY", "framed": "FOOTAGE", "montage": "FOOTAGE"}[style]
    lab = label_png(Path(out).with_suffix(".png"), tag, (place + " · " if place and style == "framed" else "") + label)
    enc = ["-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", f"{dur:.3f}"]
    silent = ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    if style == "original":                      # as it is, from the start
        fc = (f"[0:v]{FILL},tpad=stop_mode=clone:stop_duration=30,"          # hard cut in on a white flash
              f"geq=lum='min(255,lum(X,Y)+255*max(0,1-T/0.25))':cb='cb(X,Y)':cr='cr(X,Y)'[v0];[v0][1:v]overlay[v]")
        _ff(["-i", str(src), "-i", str(lab), "-filter_complex", fc, "-map", "[v]", "-map", "0:a:0?"] + enc + [str(out)])
    elif style == "replay":                      # the middle moment at half speed, the last second frozen, red frame
        take = max(0.5, (dur - 1.0) / 2)
        ss = max(0.0, min(L - take, L / 2 - take / 2))
        fc = (f"[0:v]trim={ss:.3f}:{ss + take:.3f},setpts=2*(PTS-STARTPTS),{FILL},"
              f"tpad=stop_mode=clone:stop_duration=30,drawbox=x=0:y=0:w=iw:h=ih:color=red@0.85:t=14[v0];[v0][1:v]overlay[v]")
        _ff(["-i", str(src), "-i", str(lab)] + silent + ["-filter_complex", fc, "-map", "[v]", "-map", "2:a"] + enc + [str(out)])
    elif style == "framed":                      # smaller, white frame, over its own blurred copy
        ss = max(0.0, min(L - 1.0, L * 0.35)) if L > dur + 1 else 0.0
        fc = (f"[0:v]trim=start={ss:.3f},setpts=PTS-STARTPTS,split[a][b];"
              f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:2,eq=brightness=-0.2[bg];"
              f"[b]scale=1400:788:force_original_aspect_ratio=decrease,pad=iw+16:ih+16:8:8:white[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2-30,tpad=stop_mode=clone:stop_duration=30[v0];[v0][1:v]overlay[v]")
        _ff(["-ss", "0", "-i", str(src), "-i", str(lab), "-filter_complex", fc, "-map", "[v]", "-map", "0:a:0?",
             "-af", f"atrim=start={ss:.3f},asetpts=PTS-STARTPTS"] + enc + [str(out)])
    else:                                        # montage: three quick cuts from different moments
        part = dur / 3
        offs = [max(0.0, min(L - part, L * f)) for f in (0.1, 0.45, 0.75)]
        v = "".join(f"[0:v]trim={o:.3f}:{o + part:.3f},setpts=PTS-STARTPTS,{FILL},"
                    f"tpad=stop_mode=clone:stop_duration={part:.3f},trim=0:{part:.3f},setpts=PTS-STARTPTS[v{i}];"
                    for i, o in enumerate(offs))
        fc = v + "[v0][v1][v2]concat=n=3:v=1:a=0[vc];[vc][1:v]overlay[v]"
        _ff(["-i", str(src), "-i", str(lab)] + silent + ["-filter_complex", fc, "-map", "[v]", "-map", "2:a"] + enc + [str(out)])
    Path(lab).unlink(missing_ok=True)
    return out


def headline_boxes(img_path, words):
    """Up to 2 lines to highlight: the biggest text (the headline) and the line that best matches the script.
    Boxes in pixels of the image, top-left origin. [] when Apple Vision is not available."""
    try:
        from .cleancheck import vision_available, vision_text
        if not vision_available():
            return []
        lines = vision_text(img_path)
    except Exception:
        return []
    from PIL import Image
    iw, ih = Image.open(img_path).size
    pix = [(t, (x * iw, (1 - y - h) * ih, w * iw, h * ih)) for t, _c, (x, y, w, h) in lines if t.strip()]
    if not pix:
        return []
    head = max(pix, key=lambda p: p[1][3])
    wset = {w for w in re.findall(r"[a-z]{4,}", " ".join(words).lower())}
    best = max(pix, key=lambda p: len(wset & set(re.findall(r"[a-z]{4,}", p[0].lower()))))
    return [head[1]] + ([best[1]] if best is not head and wset & set(re.findall(r"[a-z]{4,}", best[0].lower())) else [])


def render_card(img_path, dur, out, fps, label, words):
    """News screenshot as an article card: dark background, the shot slowly pushing in, a yellow highlighter
    wiping across the headline and the line the narration talks about."""
    from PIL import Image, ImageDraw, ImageFilter
    src = Image.open(img_path).convert("RGB")
    boxes = headline_boxes(img_path, words)
    maxw, maxh = 1500, 830
    s = min(maxw / src.width, maxh / src.height)
    card = src.resize((max(1, int(src.width * s)), max(1, int(src.height * s))), Image.LANCZOS)
    boxes = [(x * s, y * s, w * s, h * s) for x, y, w, h in boxes]
    tag = label_png(Path(out).with_suffix(".png"), "REPORT", label)
    lab = Image.open(tag).convert("RGBA")
    n = max(1, int(dur * fps))
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
                          "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], stdin=subprocess.PIPE)
    bg = Image.new("RGB", (W, H), (11, 15, 20))
    shadow = Image.new("RGBA", (card.width + 60, card.height + 60), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle([30, 30, card.width + 30, card.height + 30], fill=(0, 0, 0, 160))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    for i in range(n):
        t = i / fps
        k = 1.0 + 0.05 * (t / max(dur, 0.1))                          # slow push in
        c = card.resize((int(card.width * k), int(card.height * k)), Image.BILINEAR).convert("RGBA")
        hl = Image.new("RGBA", c.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(hl)
        for j, (x, y, w, h) in enumerate(boxes):                       # highlighter wipes in, one line after the other
            u = min(1.0, max(0.0, (t - 0.6 - j * 0.7) / 0.6))
            if u > 0:
                pad = h * 0.18
                dr.rectangle([x * k - pad, y * k - pad, (x + w * u) * k + pad, (y + h) * k + pad], fill=(255, 214, 0, 105))
        c = Image.alpha_composite(c, hl)
        f = bg.copy()
        ox, oy = (W - c.width) // 2, (H - c.height) // 2 - 20
        f.paste(shadow.resize((c.width + 60, c.height + 60)), (ox - 30, oy - 30), shadow.resize((c.width + 60, c.height + 60)))
        f.paste(c, (ox, oy), c)
        f.paste(lab, (0, 0), lab)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait(timeout=600)
    Path(tag).unlink(missing_ok=True)
    if p.returncode != 0 or not Path(out).exists():
        raise RuntimeError("card render failed")
    return out


def render_intro(dur, out, fps, meta, words):
    """Cold open, voice only: black screen, red BREAKING tag blinking, the event title typing in letter by letter,
    place + date below, a thin red line growing, the narration's key words fading under it. Ends on black so the
    hero footage hits hard on its white flash."""
    from PIL import Image, ImageDraw
    title = (meta.get("title") or "").upper()[:60]
    sub = " · ".join(x for x in [meta.get("place", ""), meta.get("date", "")] if x).upper()
    kw = " ".join(w for w in re.findall(r"[A-Za-z']+", " ".join(words)) if len(w) > 3)[:70].upper()
    fT, fB, fS, fK = _font(92), _font(40), _font(34), _font(30)
    from PIL import ImageDraw as _D
    for size in (92, 80, 68, 58):                                      # long titles: smaller, never off screen
        fT = _font(size)
        if _D.Draw(Image.new("RGB", (8, 8))).textlength(title, font=fT) <= W - 160:
            break
    n = max(1, int(dur * fps))
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
                          "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / fps
        fade = max(0.0, min(1.0, (dur - 0.9 - t) / 0.4))                  # last ~0.5 s goes to black
        im = Image.new("RGB", (W, H), (6, 8, 11))
        d = ImageDraw.Draw(im)
        a = lambda v: int(v * fade)
        if int(t * 2.5) % 2 == 0 or t > 1.6:                               # BREAKING blinks, then stays
            bw = d.textlength("BREAKING", font=fB)
            d.rectangle([W / 2 - bw / 2 - 24, 330, W / 2 + bw / 2 + 24, 390], fill=(a(210), a(20), a(30)))
            d.text((W / 2 - bw / 2, 336), "BREAKING", font=fB, fill=(a(255),) * 3)
        k = int(len(title) * min(1.0, max(0.0, (t - 0.4) / 1.6)))            # title types in
        shown = title[:k] + ("_" if k < len(title) and int(t * 6) % 2 else "")
        tw = d.textlength(title, font=fT)
        d.text((W / 2 - tw / 2, 440), shown, font=fT, fill=(a(255),) * 3)
        lw = (tw + 80) * min(1.0, max(0.0, (t - 1.2) / 1.0))                 # red line grows from the middle
        d.rectangle([W / 2 - lw / 2, 565, W / 2 + lw / 2, 571], fill=(a(210), a(20), a(30)))
        if sub and t > 2.0:
            u = min(1.0, (t - 2.0) / 0.5)
            sw = d.textlength(sub, font=fS)
            d.text((W / 2 - sw / 2, 600), sub, font=fS, fill=(a(200 * u), a(200 * u), a(200 * u)))
        if kw and t > 2.6:
            u = min(1.0, (t - 2.6) / 0.6)
            kwid = d.textlength(kw, font=fK)
            d.text((W / 2 - kwid / 2, 680), kw, font=fK, fill=(a(255 * u), a(196 * u), 0))
        p.stdin.write(im.tobytes())
    p.stdin.close()
    p.wait(timeout=600)
    if p.returncode != 0 or not Path(out).exists():
        raise RuntimeError("intro render failed")
    return out


def cut(src, start, need, out, fps):
    _ff(["-ss", f"{max(0.0, start):.3f}", "-i", str(src), "-t", f"{need:.3f}", "-r", str(fps), "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(out)])


# ---------------------------------------------------------------- the step (runs inside fetch, after maps)
def run_step(proj, beats, vo_total, settings, style, log, fetched):
    proj = Path(proj)
    slug = (read_json(proj / "project.json", {}) or {}).get("event") or ""
    if not slug:
        return 0
    if not pack_dir(slug).is_dir():
        log(f"event pack '{slug}' not found - library footage only")
        return 0
    meta = read_json(pack_dir(slug) / "event.json", {}) or {}
    credits = meta.get("credits") or {}
    vids, imgs = files(slug)
    if not vids and not imgs:
        log(f"event pack '{slug}' is empty - library footage only")
        return 0
    fps = style["canvas"]["fps"]
    edir = proj / "event"
    edir.mkdir(exist_ok=True)
    plan_path = edir / "plan.json"
    plan = read_json(plan_path, None)
    if not plan:
        taken = {b for b, v in fetched.items() if (v or {}).get("source") == "map"}
        lens = {"video": min(8.0, max(4.0, duration(vids[0]) if vids else 6.0)), "image": 5.0}
        plan = assign_beats(plan_slots(vo_total, len(vids), len(imgs)), beats, vo_total, taken, lens)
        if plan and plan[0]["kind"] == "video" and plan[0]["start"] > 1.0:   # cold open: voice only before the hit
            pre = [str(b["id"]) for b in beats if b["start"] < plan[0]["start"] - 0.01 and str(b["id"]) not in taken]
            if pre and pre[0] == str(beats[0]["id"]):
                plan.insert(0, {"kind": "intro", "idx": 0, "style": "cold_open", "beats": pre,
                                "start": beats[0]["start"], "end": plan[0]["start"]})
        write_json(plan_path, plan)
    log(f"event pack: {meta.get('title') or slug} - {sum(1 for s in plan if s['kind'] == 'video')} footage uses, "
        f"{sum(1 for s in plan if s['kind'] == 'image')} report cards"
        + (", cold open (voice only) until the first footage" if plan and plan[0]["kind"] == "intro" else ""))
    texts = {str(b["id"]): b["text"] for b in beats}
    order = [str(b["id"]) for b in beats]
    got = 0
    for k, s in enumerate(plan):
        if all((fetched.get(b) or {}).get("source") == "event" for b in s["beats"]):
            continue
        dur = (s["end"] - s["start"]) + 1.0
        clip = edir / f"slot_{k:02d}_{s['kind']}_{s['style']}.mp4"
        try:
            if not clip.exists():
                if s["kind"] == "intro":
                    render_intro(dur, clip, fps, meta, [texts[b] for b in s["beats"]])
                elif s["kind"] == "video":
                    f = vids[s["idx"] % len(vids)]
                    render_video(f, s["style"], dur, clip, fps, credits.get(f.name, ""), meta.get("place", ""))
                else:
                    f = imgs[s["idx"] % len(imgs)]
                    words = [texts[b] for b in s["beats"]]
                    render_card(f, dur, clip, fps, credits.get(f.name, ""), words)
            src_name = ("intro" if s["kind"] == "intro" else
                        (vids[s["idx"] % len(vids)] if s["kind"] == "video" else imgs[s["idx"] % len(imgs)]).name)
            for bid in s["beats"]:
                i = order.index(bid)
                b = beats[i]
                end = vo_total if i == len(order) - 1 else beats[i + 1]["start"]
                out = edir / f"beat_{bid}.mp4"
                if not out.exists():
                    cut(clip, b["start"] - s["start"], (end - b["start"]) + 1.0, out, fps)
                fetched[bid] = {"file": str(out), "start": 0.0, "length": duration(out), "source": "event",
                                "id": f"event:{slug}:{src_name}:{k}", "page_url": "", "title": meta.get("title", ""),
                                "credit": credits.get(src_name, ""), "license": "event pack (check the source)",
                                "src_height": 1080}
                got += 1
        except Exception as e:
            log(f"event pack: slot {k + 1} ({s['kind']} {s['style']}) failed, library footage instead ({str(e)[:140]})")
    log(f"event pack: {got} beats use the event footage / cards")
    return got
