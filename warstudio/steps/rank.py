"""Step 5: pick one clip per beat.

1. Download each candidate's preview image (small JPEG).
2. CLIP (local, free) scores preview vs beat text -> top K.
3. The K previews become ONE numbered collage -> a vision model (Groq, else Gemini) scores them 0-10 -> winner.
   (One image per call instead of K keeps rerank cost ~10x lower.)
Without a Groq/Gemini key the CLIP top-1 wins. A clip from the same source video is not reused within
`same_source_cooldown_seconds` of the timeline.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..rerank import Reranker
from ..util import download, num, read_json, write_json

_CLIP = {}


def _clip(settings):
    if "model" not in _CLIP:
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        import warnings
        warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
        import open_clip
        import torch
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(settings["clip_model"], pretrained=settings["clip_pretrained"])
        model.eval().to(dev)
        _CLIP.update(model=model, preprocess=preprocess, tok=open_clip.get_tokenizer(settings["clip_model"]), dev=dev, torch=torch)
    return _CLIP


def clip_scores(settings, text, image_paths):
    c = _clip(settings)
    torch = c["torch"]
    imgs = []
    for p in image_paths:
        try:
            imgs.append(c["preprocess"](Image.open(p).convert("RGB")))
        except Exception:
            imgs.append(None)
    ok = [i for i, x in enumerate(imgs) if x is not None]
    scores = [-1.0] * len(image_paths)
    if not ok:
        return scores
    with torch.no_grad():
        ie = c["model"].encode_image(torch.stack([imgs[i] for i in ok]).to(c["dev"]))
        te = c["model"].encode_text(c["tok"]([text[:300]]).to(c["dev"]))
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        sims = (ie @ te.T).squeeze(-1).float().cpu().tolist()
    for i, s in zip(ok, sims if isinstance(sims, list) else [sims]):
        scores[i] = s
    return scores


def diverse_top(cands, k, per_source_min=2):
    """CLIP top-k, but every source that has candidates keeps up to `per_source_min` seats, so one
    source with sharper previews can't crowd the others out before Gemini sees them."""
    ranked = sorted(cands, key=lambda c: -c.get("clip_score", -1))
    top = ranked[:k]
    for src in {c.get("source") for c in ranked}:
        have = sum(1 for c in top if c.get("source") == src)
        extra = [c for c in ranked if c.get("source") == src and c not in top][:max(0, per_source_min - have)]
        for c in extra:
            counts = {}
            for t in top:
                counts[t.get("source")] = counts.get(t.get("source"), 0) + 1
            victims = [t for t in reversed(top) if counts[t.get("source")] > per_source_min]
            if not victims:
                break
            top.remove(victims[0])
            top.append(c)
    return sorted(top, key=lambda c: -c.get("clip_score", -1))


def make_collage(paths, out, tile=(512, 288)):
    tw, th = tile
    cols = 3 if len(paths) > 4 else 2
    rows = (len(paths) + cols - 1) // cols
    im = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", int(th * 0.15))
    except Exception:
        f = ImageFont.load_default()
    for i, p in enumerate(paths):
        try:
            t = Image.open(p).convert("RGB").resize((tw, th))
        except Exception:
            t = Image.new("RGB", (tw, th), (40, 40, 40))
        x, y = (i % cols) * tw, (i // cols) * th
        im.paste(t, (x, y))
        bw = int(th * 0.21)
        d.rectangle([x, y, x + bw + 4, y + bw], fill=(255, 196, 0))
        d.text((x + bw * 0.28, y + bw * 0.1), str(i + 1), font=f, fill=(0, 0, 0))
    im.save(out, quality=85)
    return out


# Master-prompt ranking rules (see prompts/master_prompt.md)
TIER_BONUS = {"exact": 0.02, "substitute": 0.01, "category": 0.0}
REEL_SECONDS = 90          # longer source videos are multi-shot reels (interviews, slates): prefer single shots
REEL_PENALTY = 0.02
SCENE_REUSE_BONUS = 0.015  # same scene: reusing a strong clip from another moment beats a weaker new clip
SCENE_REUSE_MAX = 3        # uses of one clip inside one scene


BROLL_WORDS = ("b-roll", "broll", "b roll")
NOT_BROLL_WORDS = ("package", "interview", "briefing", "press conference", "podcast", "town hall", "news update",
                   "update", "speech", "remarks", "testimony", "q&a")
BROLL_BONUS = 0.015
NOT_BROLL_PENALTY = 0.025


def title_adjust(title):
    """DVIDS (and others) label raw footage 'B-Roll'; 'Package'/'Interview' videos carry narration, lower-thirds
    and talking heads."""
    t = f" {(title or '').lower()} "
    if any(w in t for w in BROLL_WORDS):
        return BROLL_BONUS
    if any(f" {w}" in t or f"{w} " in t for w in NOT_BROLL_WORDS):
        return -NOT_BROLL_PENALTY
    return 0.0


def adjusted_score(c):
    s = c.get("clip_score_raw", c.get("clip_score", 0)) + TIER_BONUS.get(c.get("tier"), 0.0)
    s += title_adjust(c.get("title"))
    if num(c.get("duration")) > REEL_SECONDS and c.get("tier") != "exact":
        s -= REEL_PENALTY
    if c.get("scene_reuse"):
        s += SCENE_REUSE_BONUS
    return s


def _prompt(beat, spec, n):
    avoid = ", ".join(spec.get("avoid") or []) or "none"
    ent = spec.get("entity") or "not specified"
    ladder = (f"1) the exact subject ({ent}) = best; 2) the look-alike '{spec.get('substitute') or '-'}' = good; "
              f"3) the broad class '{spec.get('category') or '-'}' = acceptable; 4) anything else = max 2.")
    return (f"You are picking B-roll for a military documentary.\nNarration: \"{beat['text']}\"\n"
            f"What the frame should show: {spec.get('visual') or beat['text']}\n"
            f"Footage type wanted: {spec['footage_class']}. Wrong matches: {avoid}.\n"
            f"Preference ladder: {ladder}\n"
            f"The image has {n} numbered tiles. Score each tile 0-10 by that ladder "
            f"(wrong aircraft/ship type = max 3; airshow, toy, game or 3D render for a real event = max 3; "
            f"text/watermark/logos = minus 3; a single clean cinematic shot beats a busy one).\n"
            f'Return JSON: {{"scores": [<{n} integers in tile order>], "best": <tile number>}}')


def run_step(proj, keys, settings, style, log):
    proj = Path(proj)
    out_path = proj / "picks.json"
    picks = read_json(out_path, {}) or {}
    beats = read_json(proj / "beats.json")
    specs = {s["id"]: s for s in read_json(proj / "specs.json")}
    cands = read_json(proj / "candidates.json")
    pdir = proj / "previews"
    pdir.mkdir(exist_ok=True)
    k = int(settings.get("rerank_top_k", 5))
    cool = style["footage"]["same_source_cooldown_seconds"]
    from ..config import save_settings
    rr = Reranker(keys, settings, log, save_settings)
    log(f"ranking with {rr.label}")
    recent = []  # (time, source_key, scene)
    scene_picks = {}  # scene -> list of winners in that scene (for reuse)
    for b in beats:
        bid = str(b["id"])
        sp = specs[b["id"]]
        scene = sp.get("scene") or ""
        if bid in picks:
            p = picks[bid]
            if p:
                recent.append((b["start"], f"{p['source']}:{p['id']}", scene))
                if scene:
                    scene_picks.setdefault(scene, []).append(p)
            continue
        # cooldown only blocks repeats across DIFFERENT scenes; inside one scene a clip may come back
        blocked = {sk for t, sk, scn in recent if b["start"] - t < cool and not (scene and scn == scene)}
        cl = [dict(c) for c in cands.get(bid, []) if f"{c['source']}:{c['id']}" not in blocked]
        if scene and scene_picks.get(scene):
            have = {f"{c['source']}:{c['id']}" for c in cl}
            uses = {}
            for p in scene_picks[scene]:
                k_ = f"{p['source']}:{p['id']}"
                uses[k_] = uses.get(k_, 0) + 1
            for p in reversed(scene_picks[scene]):
                k_ = f"{p['source']}:{p['id']}"
                if k_ in have or uses[k_] >= SCENE_REUSE_MAX:
                    continue
                have.add(k_)
                cl.append({x: v for x, v in p.items() if x not in ("alternates", "gemini_score", "rerank_by")}
                          | {"scene_reuse": True})
        if not cl:
            picks[bid] = None
            continue
        jobs = []
        for c in cl:
            c["preview_file"] = str(pdir / f"{c['source']}_{c['id'].replace(':', '_')}.jpg")
            if not Path(c["preview_file"]).exists():
                jobs.append(c)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda c: _safe_dl(c["preview_url"], c["preview_file"]), jobs))
        cl = [c for c in cl if Path(c["preview_file"]).exists()]
        if not cl:
            picks[bid] = None
            continue
        # CLIP compares against the concrete visual line from the master prompt, not the narration
        text = (sp.get("visual") or f"{sp.get('entity', '')} {b['text']}").strip()
        sc = clip_scores(settings, text, [c["preview_file"] for c in cl])
        for c, s in zip(cl, sc):
            title = (c.get("title") or "").lower()
            if any(a.lower() in title for a in sp.get("avoid") or [] if a):
                s -= 0.05
            c["clip_score_raw"] = round(s, 4)
            c["clip_score"] = round(adjusted_score(c), 4)
        top = diverse_top(cl, k)
        winner, gscore, scores, by = top[0], None, None, None
        if rr.active and len(top) > 1:
            res, by = rr.pick(lambda tile: make_collage([c["preview_file"] for c in top], pdir / f"collage_{bid}.jpg", tile),
                              _prompt(b, sp, len(top)), tag=f"beat {bid}: ")
            if res:
                scores = res.get("scores") or []
                try:
                    best = int(res.get("best") or (scores.index(max(scores)) + 1 if scores else 1))
                except (TypeError, ValueError):
                    best = 1
                if 1 <= best <= len(top):
                    winner = top[best - 1]
                    gscore = scores[best - 1] if len(scores) >= best else None
        keep = ("source", "id", "title", "preview_url", "preview_file", "video_url", "duration", "page_url", "credit",
                "clip_score", "query", "tier", "scene_reuse", "license", "min_start")
        alts = [{x: c.get(x) for x in keep} for c in top]
        if isinstance(scores, list):
            for a, s_ in zip(alts, scores):
                a["gemini_score"] = s_
        winner = dict(winner, gemini_score=gscore, rerank_by=by or "clip", alternates=alts, scene=scene)
        picks[bid] = winner
        recent.append((b["start"], f"{winner['source']}:{winner['id']}", scene))
        if scene:
            scene_picks.setdefault(scene, []).append(winner)
        if int(bid) % 10 == 0:
            write_json(out_path, picks)
            log(f"rank {int(bid) + 1}/{len(beats)}", progress=(int(bid) + 1) / len(beats))
    write_json(out_path, picks)
    from collections import Counter
    mix = Counter((v or {}).get("source", "none") for v in picks.values())
    by = Counter((v or {}).get("rerank_by", "clip") for v in picks.values() if v)
    log("reranked by: " + ", ".join(f"{a} {b}" for a, b in by.most_common()))
    tiers = Counter(("reuse" if (v or {}).get("scene_reuse") else (v or {}).get("tier", "?")) for v in picks.values() if v)
    log("match level: " + ", ".join(f"{a} {b}" for a, b in tiers.most_common()))
    log(f"rank done; {sum(1 for v in picks.values() if not v)} beats without a pick; picks per source: " +
        ", ".join(f"{a} {b}" for a, b in mix.most_common()))
    return picks


def _safe_dl(url, dest):
    try:
        download(url, dest, timeout=30)
    except Exception:
        pass
