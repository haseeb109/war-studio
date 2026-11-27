"""Step 4: find candidate clips for every beat (metadata + preview image only, nothing big downloaded).

Sources in style order (DVIDS first for real military footage). Every API reply is cached in
data/search_cache so a repeated query costs no request. A source that returns 429 is skipped for the
rest of the run (the other sources carry on).
"""
import hashlib
import json
import time
from pathlib import Path

import httpx

from ..config import DATA_DIR
from ..util import read_json, write_json, num, UA

CACHE = DATA_DIR / "search_cache"


class SourceDown(Exception):
    pass


def _cached_get(source, url, params, headers=None, ttl=7 * 86400):
    key = hashlib.md5((source + url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    p = CACHE / f"{source}_{key}.json"
    if p.exists() and time.time() - p.stat().st_mtime < ttl:
        d = json.loads(p.read_text())
        if not _empty(d):
            return d
        p.unlink(missing_ok=True)              # an empty answer cached by an older version: ask again
    from ..util import with_net_retry

    def get():
        resp = httpx.get(url, params=params, headers={"User-Agent": UA, **(headers or {})}, timeout=30)
        if resp.status_code in (500, 502, 503, 504):
            resp.raise_for_status()          # server hiccup: retried below
        return resp
    r = with_net_retry(get, what=f"{source} search", online_tries=3)
    if r.status_code == 429:
        raise SourceDown(f"{source} rate limit (429)")
    if r.status_code in (401, 403):
        raise SourceDown(f"{source} key rejected ({r.status_code})")
    r.raise_for_status()
    d = r.json()
    if isinstance(d, dict) and (d.get("errors") or d.get("error")):
        raise RuntimeError(f"{source} API error: {str(d.get('errors') or d.get('error'))[:200]}")
    if not _empty(d):                           # never cache "no results": a bad request would stick for 7 days
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    return d


def _empty(d):
    if not isinstance(d, dict):
        return False
    for k in ("results", "hits", "videos"):
        if k in d and not d[k]:
            return True
    q = d.get("query")
    return isinstance(q, dict) and "pages" in q and not q["pages"]


def _thumb(v):
    if isinstance(v, dict):
        return v.get("url") or v.get("src") or ""
    return v or ""


def search_dvids(q, key, n, page=1):
    # thumb_width/height: DVIDS defaults to 122x92 thumbnails, far too small for CLIP/Gemini to judge,
    # which made DVIDS clips lose every ranking against Pixabay's 640px previews.
    # (v0.5 also sent aspect_ratio=landscape + thumb_height; on real runs that returned nothing for videos.)
    # Ask with big thumbnails first; if that gives nothing, the plain query that has always worked;
    # then the first words only (DVIDS full-text search needs every word to match).
    base = {"api_key": key, "type": "video", "max_results": min(50, n)}
    if page > 1:
        base["page"] = int(page)
    words = q.split()
    attempts = [dict(base, q=q, thumb_width=640, sort="score", sortdir="desc"), dict(base, q=q)]
    if len(words) > 2:
        attempts.append(dict(base, q=" ".join(words[:2]), thumb_width=640))
    results = []
    for params in attempts:
        results = _cached_get("dvids", "https://api.dvidshub.net/search", params).get("results") or []
        if results:
            break
    out = []
    for r in results:
        rid = str(r.get("id") or "")
        if not rid:
            continue
        out.append({"source": "dvids", "id": rid, "title": r.get("title") or "",
                    "preview_url": _thumb(r.get("thumbnail")) or _thumb(r.get("image")),
                    "video_url": "", "duration": num(r.get("duration")),
                    "page_url": r.get("url") or f"https://www.dvidshub.net/video/{rid.split(':')[-1]}",
                    "credit": r.get("credit") or r.get("unit_name") or "DVIDS (public domain)",
                    "license": "Public domain (US DoD / DVIDS)"})
    return out


def search_pexels(q, key, n):
    d = _cached_get("pexels", "https://api.pexels.com/videos/search",
                    {"query": q, "per_page": n, "orientation": "landscape"}, headers={"Authorization": key})
    out = []
    for v in d.get("videos") or []:
        files = [f for f in v.get("video_files") or [] if f.get("file_type") == "video/mp4" and f.get("width")]
        files.sort(key=lambda f: (abs((f.get("height") or 0) - 1080), -(f.get("width") or 0)))
        if not files:
            continue
        out.append({"source": "pexels", "id": str(v["id"]), "title": v.get("url", "").rstrip("/").split("/")[-1],
                    "preview_url": v.get("image") or "", "video_url": files[0]["link"],
                    "duration": num(v.get("duration")), "page_url": v.get("url") or "",
                    "credit": f"{(v.get('user') or {}).get('name', '')} / Pexels"})
    return out


def search_pixabay(q, key, n):
    # film only (no 3D/animation renders) and full-HD sources, closer to the cinematic Pexels look
    d = _cached_get("pixabay", "https://pixabay.com/api/videos/", {"key": key, "q": q, "per_page": max(3, n),
                                                                  "video_type": "film", "min_width": 1920})
    out = []
    for h in d.get("hits") or []:
        vids = h.get("videos") or {}
        best = vids.get("large") if (vids.get("large") or {}).get("url") else vids.get("medium") or {}
        if not best.get("url"):
            continue
        thumb = (vids.get("medium") or {}).get("thumbnail") or best.get("thumbnail") or ""
        out.append({"source": "pixabay", "id": str(h["id"]), "title": h.get("tags") or "",
                    "preview_url": thumb, "video_url": best["url"], "duration": num(h.get("duration")),
                    "page_url": h.get("pageURL") or "", "credit": f"{h.get('user', '')} / Pixabay"})
    return out


FREE_LICENSES = ("public domain", "pd-", "cc0", "cc-zero", "cc by", "cc-by")


def commons_license_ok(lic, allow_sa=False):
    l = (lic or "").lower().replace("-", " ")
    if any(x in l for x in ("public domain", "pd ", "cc0", "cc zero")) or l.startswith("pd"):
        return True
    if "cc by" in l and "nc" not in l.split() and "nd" not in l.split():
        return allow_sa or "sa" not in l.split()
    return False


def _strip_html(s):
    import re
    return re.sub(r"<[^>]+>", "", s or "").strip()


def search_commons(q, key, n, allow_sa=False):
    """Wikimedia Commons videos (no key). Only public domain, CC0 and CC BY (CC BY-SA if allowed in Settings).
    Has Russian MoD (mil.ru, CC BY 4.0), Tasnim (Iran, CC BY 4.0), many DoD and national-forces clips."""
    d = _cached_get("commons", "https://commons.wikimedia.org/w/api.php",
                    {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                     "gsrsearch": f"filetype:video {q}", "gsrlimit": min(50, n * 2), "prop": "imageinfo",
                     "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 640},
                    headers={"User-Agent": "WarStudio/0.5 (https://github.com/haseeb109/war-studio)"})
    out = []
    pages = sorted(((d.get("query") or {}).get("pages") or {}).values(), key=lambda p: p.get("index", 0))
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        mime = ii.get("mime") or ""
        if not mime.startswith("video/") or not ii.get("url") or not ii.get("thumburl"):
            continue
        meta = ii.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not commons_license_ok(lic, allow_sa):
            continue
        artist = _strip_html((meta.get("Artist") or {}).get("value", "")) or "Wikimedia Commons"
        title = (p.get("title") or "").replace("File:", "").rsplit(".", 1)[0]
        out.append({"source": "commons", "id": str(p.get("pageid")), "title": title,
                    "preview_url": ii["thumburl"], "video_url": ii["url"], "duration": num(ii.get("duration")),
                    "page_url": ii.get("descriptionurl") or "", "credit": f"{artist} / Wikimedia Commons / {lic}",
                    "license": lic})
        if len(out) >= n:
            break
    return out


SOURCES = {"dvids": ("DVIDS_API_KEY", search_dvids), "commons": (None, search_commons),
           "pexels": ("PEXELS_API_KEY", search_pexels), "pixabay": ("PIXABAY_API_KEY", search_pixabay)}


def source_ready(src, keys, settings=None):
    k = SOURCES[src][0]
    if k is None:
        return str((settings or {}).get(f"use_{src}", True)).lower() not in ("false", "0", "no")
    return bool(keys.get(k))


def call_source(src, q, keys, n, settings=None):
    kname, fn = SOURCES[src]
    if src == "commons":
        allow = str((settings or {}).get("commons_allow_sharealike", False)).lower() in ("true", "1", "yes")
        return fn(q, None, n, allow_sa=allow)
    return fn(q, keys[kname], n)


def order_for(fc, style):
    order = list(style["footage"]["source_order"])
    for s in SOURCES:
        if s not in order:
            order.append(s)
    if fc == "map":
        order = [s for s in order if s != "dvids"] + ["dvids"]
    return order


def library_first(specs, done, settings, log, project=None, only=False):
    """Match every beat against the local shot library. A good match fills the beat's candidates from the
    library only (no online search, no Gemini rerank for that beat). Returns how many beats matched.
    only=True (the video uses nothing but the library): there is no online fallback, so every beat takes the best
    library shot above a much lower bar - a weak but fresh shot beats holding the previous clip past 3 s."""
    try:
        from ..library import LibraryIndex, clip_text_vecs
        idx = LibraryIndex(settings, exclude_project=project)
    except Exception as e:
        log(f"library: not used ({str(e)[:120]})")
        return 0
    if not len(idx):
        log("library: empty - every beat goes online (build it on the Library page)")
        return 0
    thr = float(settings.get("library_min_clip", 0.24))
    floor = float(settings.get("library_only_min_clip", 0.15)) if only else thr
    todo = [sp for sp in specs if not done.get(str(sp["id"]))]
    if not todo:
        return 0
    if not only:
        todo = [sp for sp in todo if sp.get("footage_class") != "map"]      # maps are searched online
    texts = [("aerial view from high altitude over coastline and sea" if sp.get("footage_class") == "map" else
              (sp.get("visual") or " ".join(sp.get("queries") or [])[:200] or "military")).strip() for sp in todo]
    vecs = clip_text_vecs(settings, texts)
    hits = weak = subs = 0
    for sp, v, t in zip(todo, vecs, texts):
        res = idx.match(v, sp.get("entity") or "", t, k=12)
        if only and sp.get("entity") and (not res or res[0]["clip_sim"] < floor):
            # the named system is not in the library: the closest shot of the same kind stands in
            res = idx.match(v, "", t, k=12)
            subs += bool(res and res[0]["clip_sim"] >= floor)
        elif only and sp.get("entity") and res:
            # reserve: same-kind shots of other systems, used only when the named system's few shots are used up
            # (one clip max 3 times per video); without them a long video about one aircraft ran out of footage
            seen_ids = {r["id"] for r in res}
            res = res + [dict(r, reserve=True) for r in idx.match(v, "", t, k=10) if r["id"] not in seen_ids]
        if res and res[0]["clip_sim"] >= floor:
            done[str(sp["id"])] = res
            hits += 1
            weak += res[0]["clip_sim"] < thr
    if idx.rotated:
        log(f"library: {len(idx.rotated)} shots rest this time (used in the last "
            f"{int(settings.get('library_rotation_videos', 5))} videos)")
    log(f"library: {hits}/{len(todo)} beats matched from {len(idx)} shots (no online search, no Gemini for these)"
        + (f"; {weak} of them are weaker matches (library only: best shot available instead of holding the "
           f"previous clip)" if weak else "")
        + (f"; {subs} use a stand-in because the named system is not in the library" if subs else "")
        + (f"; {len(todo) - hits} go online" if hits < len(todo) and not only else "")
        + (f"; {len(todo) - hits} without any library shot" if hits < len(todo) and only else ""))
    return hits


def run_step(proj, keys, settings, style, log):
    proj = Path(proj)
    out_path = proj / "candidates.json"
    done = read_json(out_path, {}) or {}
    specs = read_json(proj / "specs.json")
    n = int(settings.get("candidates_per_query", 12))
    down = set()
    picked = list(settings.get("job_sources") or [])
    use_library = (not picked or "library" in picked) and str(settings.get("use_library", True)).lower() not in ("false", "0", "no")
    lib_hits = library_first(specs, done, settings, log, proj.name, only=picked == ["library"]) if use_library else 0
    if picked == ["library"]:
        # this video uses only the library: beats without a library match keep the previous clip
        write_json(out_path, done)
        empty = sum(1 for sp in specs if not done.get(str(sp["id"])))
        log(f"search done (library only); {empty} beats without a library match")
        if specs and empty == len(specs):
            raise RuntimeError("the library has no match for any beat - build the library first or pick online sources")
        return done
    chosen = [s for s in picked if s in SOURCES]
    if chosen:
        # sources picked for this video on the Create page; a picked no-key source counts as switched on
        settings = dict(settings, **{f"use_{s}": True for s in chosen})
        active = [s for s in chosen if source_ready(s, keys, settings)]
        skipped = [s for s in chosen if s not in active]
        if skipped:
            log(f"selected source(s) without an API key, skipped: {', '.join(skipped)}")
    else:
        active = [s for s in SOURCES if source_ready(s, keys, settings)]
    if not active and not lib_hits:
        raise RuntimeError("No footage source available (DVIDS / Pexels / Pixabay keys, or Wikimedia Commons) — check Settings")
    log(f"searching footage on: {', '.join(active)}")
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()

    def one(sp):
        bid = str(sp["id"])
        cands, seen = [], set()
        queries = list(sp["queries"])
        tiers = list(sp.get("tiers") or (["exact"] + ["category"] * (len(queries) - 1)))
        if sp["footage_class"] == "map":
            queries = [q + " aerial" for q in queries[:2]] + ["satellite earth view", "aerial coastline"]
            tiers = tiers[:2] + ["category", "category"]
        # Master prompt: every tier is searched (exact -> substitute -> category) so rank can fall back to a
        # look-alike or the broad class instead of a random clip.
        for q, tier in zip(queries, tiers):
            for src in order_for(sp["footage_class"], style):
                if src in down or src not in active:
                    continue
                try:
                    res = call_source(src, q, keys, n, settings)
                    with lock:
                        z = zero.setdefault(src, [0, 0])
                        z[0] += 1
                        z[1] += 0 if res else 1
                except SourceDown as e:
                    with lock:
                        if src not in down:
                            down.add(src)
                            log(f"{e}; continuing without {src}")
                    continue
                except Exception as e:
                    log(f"{src} search error for '{q}': {str(e)[:120]}")
                    continue
                for c in res:
                    k = (c["source"], c["id"])
                    if k not in seen and c["preview_url"]:
                        seen.add(k)
                        c["query"] = q
                        c["tier"] = tier
                        cands.append(c)
        return bid, cands

    zero = {}  # source -> [queries asked, queries with no result]

    todo = [sp for sp in specs if not done.get(str(sp["id"]))]
    with ThreadPoolExecutor(6) as ex:
        for i, (bid, cands) in enumerate(ex.map(one, todo), 1):
            done[bid] = cands
            if i % 10 == 0 or i == len(todo):
                write_json(out_path, done)
                log(f"search {len(specs) - len(todo) + i}/{len(specs)}", progress=(len(specs) - len(todo) + i) / len(specs))
    write_json(out_path, done)
    empty = sum(1 for v in done.values() if not v)
    from collections import Counter
    mix = Counter(c["source"] for v in done.values() for c in (v or []))
    log(f"search done; {empty} beats without candidates; candidates per source: " +
        (", ".join(f"{s} {n}" for s, n in mix.most_common()) or "none"))
    for s, (asked, none) in zero.items():
        if asked and none == asked:
            log(f"WARNING: {s} returned no results for any of its {asked} searches — run System Check / Test the {s} key")
    if specs and empty == len(specs):
        # stop here instead of rendering a video that is black from start to end
        raise RuntimeError("footage search found nothing for any beat (" + ", ".join(active) +
                           "). Check the keys in Settings, run System Check, or pick more sources on the Create page.")
    if specs and empty > len(specs) * 0.5:
        log(f"WARNING: {empty}/{len(specs)} beats have no footage candidates; they will repeat the previous clip")
    return done
