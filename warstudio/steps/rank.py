"""Step 5: pick one clip per beat.

1. Download each candidate's preview image (small JPEG).
2. CLIP (local, free) scores preview vs beat text -> top K.
3. The K previews become ONE numbered collage -> Gemini Flash scores them 0-10 -> winner.
   (One image per call instead of K keeps rerank cost ~10x lower.)
Without a Gemini key the CLIP top-1 wins. A clip from the same source video is not reused within
`same_source_cooldown_seconds` of the timeline.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..llm import gemini_pick, resolve_gemini_model, ModelGone
from ..util import download, read_json, write_json

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


def make_collage(paths, out):
    tw, th = 512, 288
    cols = 3 if len(paths) > 4 else 2
    rows = (len(paths) + cols - 1) // cols
    im = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 44)
    except Exception:
        f = ImageFont.load_default()
    for i, p in enumerate(paths):
        try:
            t = Image.open(p).convert("RGB").resize((tw, th))
        except Exception:
            t = Image.new("RGB", (tw, th), (40, 40, 40))
        x, y = (i % cols) * tw, (i // cols) * th
        im.paste(t, (x, y))
        d.rectangle([x, y, x + 64, y + 60], fill=(255, 196, 0))
        d.text((x + 18, y + 6), str(i + 1), font=f, fill=(0, 0, 0))
    im.save(out, quality=85)
    return out


def _prompt(beat, spec, n):
    avoid = ", ".join(spec.get("avoid") or []) or "none"
    ent = spec.get("entity") or "not specified"
    return (f"You are picking B-roll for a military documentary.\nNarration: \"{beat['text']}\"\n"
            f"Required subject: {ent}. Footage type wanted: {spec['footage_class']}. Wrong matches: {avoid}.\n"
            f"The image has {n} numbered tiles. Score each tile 0-10 for how well it shows what the narration says "
            f"(exact system named = most important; wrong aircraft/ship type = max 3; text/watermark/logos = minus 3).\n"
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
    gem = keys.get("GEMINI_API_KEY")
    gmodel = None
    if gem:
        try:
            gmodel = resolve_gemini_model(gem, settings.get("gemini_model", "auto"))
            if settings.get("gemini_model", "auto") not in ("auto", gmodel):
                log(f"Gemini model '{settings.get('gemini_model')}' is not available for this key; using {gmodel}")
        except Exception as e:
            log(f"Gemini unavailable ({str(e)[:160]}); ranking with CLIP only")
            gem = None
    log(f"ranking with CLIP{f' + Gemini rerank ({gmodel})' if gem else ' only (no Gemini key)'}")
    recent = []  # (time, source_key)
    gem_fail = 0
    for b in beats:
        bid = str(b["id"])
        if bid in picks:
            p = picks[bid]
            if p:
                recent.append((b["start"], f"{p['source']}:{p['id']}"))
            continue
        cl = [c for c in cands.get(bid, []) if f"{c['source']}:{c['id']}" not in
              {sk for t, sk in recent if b["start"] - t < cool}]
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
        sp = specs[b["id"]]
        text = f"{sp.get('entity', '')} {b['text']}".strip()
        sc = clip_scores(settings, text, [c["preview_file"] for c in cl])
        for c, s in zip(cl, sc):
            title = (c.get("title") or "").lower()
            if any(a.lower() in title for a in sp.get("avoid") or [] if a):
                s -= 0.05
            c["clip_score"] = round(s, 4)
        top = sorted(cl, key=lambda c: -c["clip_score"])[:k]
        winner, gscore, scores = top[0], None, None
        if gem and gem_fail >= 3:
            log("Gemini failed 3 times in a row; using CLIP only for the rest of this run")
            gem = None
        if gem and len(top) > 1:
            col = make_collage([c["preview_file"] for c in top], pdir / f"collage_{bid}.jpg")
            try:
                try:
                    res, _ = gemini_pick(gem, gmodel, col, _prompt(b, sp, len(top)))
                except ModelGone as gone:
                    new = resolve_gemini_model(gem, "auto")
                    log(f"Gemini model {gmodel} is not available ({str(gone)[-120:]}); switching to {new}")
                    gmodel = new
                    if settings.get("gemini_model", "auto") != "auto":
                        from ..config import save_settings
                        save_settings({"gemini_model": "auto"})
                    res, _ = gemini_pick(gem, gmodel, col, _prompt(b, sp, len(top)))
                scores = res.get("scores") or []
                best = int(res.get("best") or (scores.index(max(scores)) + 1 if scores else 1))
                if 1 <= best <= len(top):
                    winner = top[best - 1]
                    gscore = scores[best - 1] if len(scores) >= best else None
                gem_fail = 0
            except Exception as e:
                gem_fail += 1
                log(f"beat {bid}: Gemini rerank failed ({str(e)[:80]}), using CLIP top-1")
        keep = ("source", "id", "title", "preview_url", "preview_file", "video_url", "duration", "page_url", "credit", "clip_score", "query")
        alts = [{x: c.get(x) for x in keep} for c in top]
        if isinstance(scores, list):
            for a, s_ in zip(alts, scores):
                a["gemini_score"] = s_
        winner = dict(winner, gemini_score=gscore, alternates=alts)
        picks[bid] = winner
        recent.append((b["start"], f"{winner['source']}:{winner['id']}"))
        if int(bid) % 10 == 0:
            write_json(out_path, picks)
            log(f"rank {int(bid) + 1}/{len(beats)}", progress=(int(bid) + 1) / len(beats))
    write_json(out_path, picks)
    log(f"rank done; {sum(1 for v in picks.values() if not v)} beats without a pick")
    return picks


def _safe_dl(url, dest):
    try:
        download(url, dest, timeout=30)
    except Exception:
        pass
