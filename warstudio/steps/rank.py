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
    s -= float(c.get("lib_penalty") or 0)          # library shot used in other videos recently
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


MAX_PREVIEWS = 24          # previews per beat that get downloaded + CLIP-scored (search returns ~35-40)
MIN_PER_TIER = 4           # ...but every search tier (exact / substitute / category) keeps a few seats
REUSE_IF_BELOW = 6         # in-scene reuse beats the reranked clips only when their best score is under this


def _cap_candidates(cl, n=MAX_PREVIEWS, per_tier=MIN_PER_TIER):
    """Keep the first n candidates in search order (sources return them by relevance), but make sure each tier
    keeps up to `per_tier`, so the broad-category fallback is never cut away entirely."""
    if len(cl) <= n:
        return cl
    keep = cl[:n]
    for tier in {c.get("tier") for c in cl}:
        have = sum(1 for c in keep if c.get("tier") == tier)
        extra = [c for c in cl[n:] if c.get("tier") == tier][:max(0, per_tier - have)]
        for c in extra:
            counts = {}
            for x in keep:
                counts[x.get("tier")] = counts.get(x.get("tier"), 0) + 1
            victim = next((x for x in reversed(keep) if counts[x.get("tier")] > per_tier), None)
            if victim is None:
                break
            keep.remove(victim)
            keep.append(c)
    return keep


def _key(c):
    return f"{c['source']}:{c['id']}"


def run_step(proj, keys, settings, style, log):
    """Two phases so the slow network parts run in parallel:
    A) for every beat at once: download previews (24 at a time), CLIP-score them, and send the top-K collage to the
       reranker (8 beats at a time).
    B) walk the timeline in order and choose, using the scores from A: same-clip cooldown across scenes, in-scene
       reuse, fall back to the next reranked clip when the best one is blocked. No network calls in B."""
    import threading
    from collections import Counter
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
    todo = [b for b in beats if str(b["id"]) not in picks]

    # ---------------- phase A1: previews for every beat, all at once
    pool = {}
    for b in todo:
        bid = str(b["id"])
        cl = [dict(c) for c in _cap_candidates(cands.get(bid, []) or [])]
        for c in cl:
            if c.get("source") == "library":       # library shots carry their own keyframe on disk
                continue
            c["preview_file"] = str(pdir / f"{c['source']}_{c['id'].replace(':', '_')}.jpg")
        pool[bid] = cl
    need = {c["preview_file"]: c["preview_url"] for cl in pool.values() for c in cl
            if c.get("preview_url") and not Path(c["preview_file"]).exists()}
    if need:
        from ..util import wait_for_internet
        wait_for_internet(log)
        log(f"downloading {len(need)} previews (24 at a time)")
        with ThreadPoolExecutor(24) as ex:
            list(ex.map(lambda kv: _safe_dl(kv[1], kv[0]), need.items()))

    # ---------------- phase A2: CLIP scores (local GPU, fast)
    tops = {}
    for b in todo:
        bid = str(b["id"])
        sp = specs[b["id"]]
        cl = [c for c in pool[bid] if Path(c["preview_file"]).exists()]
        pool[bid] = cl
        if not cl:
            continue
        text = (sp.get("visual") or f"{sp.get('entity', '')} {b['text']}").strip()
        for c, s_ in zip(cl, clip_scores(settings, text, [c["preview_file"] for c in cl])):
            title = (c.get("title") or "").lower()
            if any(a.lower() in title for a in sp.get("avoid") or [] if a):
                s_ -= 0.05
            c["clip_score_raw"] = round(s_, 4)
            c["clip_score"] = round(adjusted_score(c), 4)
        cl.sort(key=lambda c: -c["clip_score"])
        tops[bid] = diverse_top(cl, k)

    # ---------------- phase A3: vision rerank, several beats at once
    reranked = {}           # bid -> (scores list aligned with tops[bid], provider)
    conc = max(1, int(settings.get("rerank_concurrency", 8)))
    lock = threading.Lock()
    done_n = [0]
    # library beats are matched already (CLIP + named-system rule + captions): no paid rerank for them
    lib_beats = {str(b["id"]) for b in todo if tops.get(str(b["id"])) and
                 all(c.get("source") == "library" for c in tops[str(b["id"])])}
    if lib_beats:
        log(f"{len(lib_beats)} beats come from the library: no rerank call for them")
    jobs = [b for b in todo if len(tops.get(str(b["id"]), [])) > 1 and str(b["id"]) not in lib_beats]

    def rerank_one(b):
        bid = str(b["id"])
        top = tops[bid]
        res, by = (None, None)
        if rr.active:
            res, by = rr.pick(lambda tile: make_collage([c["preview_file"] for c in top], pdir / f"collage_{bid}.jpg", tile),
                              _prompt(b, specs[b["id"]], len(top)), tag=f"beat {bid}: ")
        if res:
            scores = res.get("scores") or []
            try:
                best = int(res.get("best") or (scores.index(max(scores)) + 1 if scores else 1))
            except (TypeError, ValueError):
                best = 1
            sc = []
            for x in scores[:len(top)]:
                try:
                    sc.append(float(x))
                except (TypeError, ValueError):
                    sc.append(None)
            sc += [None] * (len(top) - len(sc))
            valid = [x for x in sc if x is not None]
            if 1 <= best <= len(top):          # the model's own "best" wins over its scores if they disagree
                if not valid:
                    sc[best - 1] = 10.0
                elif sc[best - 1] is None or sc[best - 1] < max(valid):
                    sc[best - 1] = max(valid) + 0.5
            with lock:
                reranked[bid] = (sc, by)
        with lock:
            done_n[0] += 1
            if done_n[0] % 10 == 0 or done_n[0] == len(jobs):
                log(f"rerank {done_n[0]}/{len(jobs)}", progress=done_n[0] / max(len(jobs), 1))

    if jobs and rr.active:
        log(f"reranking {len(jobs)} beats, {conc} at a time")
        with ThreadPoolExecutor(conc) as ex:
            list(ex.map(rerank_one, jobs))

    # ---------------- phase B: walk the timeline, choose with cooldown + scene reuse (no network)
    recent = []       # (time, source_key, scene)
    scene_picks = {}  # scene -> winners in that scene
    keep = ("source", "id", "title", "preview_url", "preview_file", "video_url", "duration", "page_url", "credit",
            "clip_score", "query", "tier", "scene_reuse", "license", "min_start", "lib_penalty")
    for b in beats:
        bid = str(b["id"])
        sp = specs[b["id"]]
        scene = sp.get("scene") or ""
        if bid in picks and b not in todo:
            p = picks[bid]
            if p:
                recent.append((b["start"], _key(p), scene))
                if scene:
                    scene_picks.setdefault(scene, []).append(p)
            continue
        blocked = {sk for t, sk, scn in recent if b["start"] - t < cool and not (scene and scn == scene)}
        scene_uses = Counter(_key(p) for p in scene_picks.get(scene, [])) if scene else Counter()
        blocked |= {sk for sk, n in scene_uses.items() if n >= SCENE_REUSE_MAX}     # max 3 uses inside one scene
        top = tops.get(bid) or []
        sc, by = reranked.get(bid, (None, None))
        scored = [(c, (sc[i] if sc and i < len(sc) else None)) for i, c in enumerate(top)]
        options = [(c, s_) for c, s_ in scored if _key(c) not in blocked]
        # in-scene reuse candidates (strong clips already used in this scene, from another moment)
        reuse = []
        if scene and scene_picks.get(scene):
            uses = scene_uses
            seen = {_key(c) for c, _ in options}
            for p in reversed(scene_picks[scene]):
                if _key(p) in seen or uses[_key(p)] >= SCENE_REUSE_MAX or not Path(p.get("preview_file") or "").exists():
                    continue
                seen.add(_key(p))
                reuse.append({x: v for x, v in p.items() if x not in ("alternates", "gemini_score", "rerank_by")}
                             | {"scene_reuse": True})
        if reuse:
            text = (sp.get("visual") or f"{sp.get('entity', '')} {b['text']}").strip()
            for c, s_ in zip(reuse, clip_scores(settings, text, [c["preview_file"] for c in reuse])):
                c["clip_score_raw"] = round(s_, 4)
                c["clip_score"] = round(adjusted_score(c), 4)
        winner, gscore = None, None
        rated = [(c, s_) for c, s_ in options if s_ is not None]
        if rated:
            c, s_ = max(rated, key=lambda x: x[1])
            if reuse and s_ < REUSE_IF_BELOW:
                winner = max(reuse, key=lambda c: c["clip_score"])
            else:
                winner, gscore = c, s_
        else:
            pool_b = [c for c, _ in options] + reuse
            if not pool_b:     # every reranked clip is in cooldown: best CLIP clip that is not
                pool_b = [c for c in (pool.get(bid) or []) if _key(c) not in blocked]
            if pool_b:
                winner = max(pool_b, key=lambda c: c["clip_score"])
        if winner is None:
            picks[bid] = None
            continue
        alts = [{x: c.get(x) for x in keep} | ({"gemini_score": s_} if s_ is not None else {})
                for c, s_ in sorted(options, key=lambda x: -(x[1] if x[1] is not None else -1))]
        alts += [{x: c.get(x) for x in keep} for c, _ in scored if _key(c) in blocked]
        winner = dict(winner, gemini_score=gscore, rerank_by=(by if gscore is not None else "clip"),
                      alternates=alts, scene=scene)
        if scene_uses.get(_key(winner)):
            winner["scene_reuse"] = True        # same clip again in this scene (fetch cuts another moment)
        picks[bid] = winner
        recent.append((b["start"], _key(winner), scene))
        if scene:
            scene_picks.setdefault(scene, []).append(winner)
    write_json(out_path, picks)

    mix = Counter((v or {}).get("source", "none") for v in picks.values())
    by_ = Counter(("library" if (v or {}).get("source") == "library" else (v or {}).get("rerank_by", "clip"))
                  for v in picks.values() if v)
    log("reranked by: " + ", ".join(f"{a} {b}" for a, b in by_.most_common()))
    if rr.chain or rr.stats["ok"]:
        log(rr.summary())
        try:                                   # the cost record never breaks the step
            from .. import costs
            costs.add(proj, "rerank", usd=getattr(rr, "cost", lambda: 0.0)(), calls=rr.stats.get("ok", 0))
        except Exception as e:
            log(f"cost: rerank not recorded ({str(e)[:80]})")
    tiers = Counter(("reuse" if (v or {}).get("scene_reuse") else (v or {}).get("tier", "?")) for v in picks.values() if v)
    log("match level: " + ", ".join(f"{a} {b}" for a, b in tiers.most_common()))
    log(f"rank done; {sum(1 for v in picks.values() if not v)} beats without a pick; picks per source: " +
        ", ".join(f"{a} {b}" for a, b in mix.most_common()))
    return picks


def _safe_dl(url, dest):
    from ..util import with_net_retry
    try:
        with_net_retry(lambda: download(url, dest, timeout=30), what="preview", online_tries=2)
    except Exception:
        pass
