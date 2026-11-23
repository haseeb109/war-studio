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
from ..util import read_json, write_json, UA

CACHE = DATA_DIR / "search_cache"


class SourceDown(Exception):
    pass


def _cached_get(source, url, params, headers=None, ttl=7 * 86400):
    key = hashlib.md5((source + url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    p = CACHE / f"{source}_{key}.json"
    if p.exists() and time.time() - p.stat().st_mtime < ttl:
        return json.loads(p.read_text())
    r = httpx.get(url, params=params, headers={"User-Agent": UA, **(headers or {})}, timeout=30)
    if r.status_code == 429:
        raise SourceDown(f"{source} rate limit (429)")
    if r.status_code in (401, 403):
        raise SourceDown(f"{source} key rejected ({r.status_code})")
    r.raise_for_status()
    d = r.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return d


def _thumb(v):
    if isinstance(v, dict):
        return v.get("url") or v.get("src") or ""
    return v or ""


def search_dvids(q, key, n):
    d = _cached_get("dvids", "https://api.dvidshub.net/search",
                    {"api_key": key, "type": "video", "q": q, "max_results": n, "sort": "score"})
    out = []
    for r in d.get("results") or []:
        rid = str(r.get("id") or "")
        if not rid:
            continue
        out.append({"source": "dvids", "id": rid, "title": r.get("title") or "",
                    "preview_url": _thumb(r.get("thumbnail")) or _thumb(r.get("image")),
                    "video_url": "", "duration": float(r.get("duration") or 0),
                    "page_url": r.get("url") or f"https://www.dvidshub.net/video/{rid.split(':')[-1]}",
                    "credit": r.get("credit") or r.get("unit_name") or "DVIDS (public domain)"})
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
                    "duration": float(v.get("duration") or 0), "page_url": v.get("url") or "",
                    "credit": f"{(v.get('user') or {}).get('name', '')} / Pexels"})
    return out


def search_pixabay(q, key, n):
    d = _cached_get("pixabay", "https://pixabay.com/api/videos/", {"key": key, "q": q, "per_page": max(3, n)})
    out = []
    for h in d.get("hits") or []:
        vids = h.get("videos") or {}
        best = vids.get("large") if (vids.get("large") or {}).get("url") else vids.get("medium") or {}
        if not best.get("url"):
            continue
        thumb = (vids.get("medium") or {}).get("thumbnail") or best.get("thumbnail") or ""
        out.append({"source": "pixabay", "id": str(h["id"]), "title": h.get("tags") or "",
                    "preview_url": thumb, "video_url": best["url"], "duration": float(h.get("duration") or 0),
                    "page_url": h.get("pageURL") or "", "credit": f"{h.get('user', '')} / Pixabay"})
    return out


SOURCES = {"dvids": ("DVIDS_API_KEY", search_dvids), "pexels": ("PEXELS_API_KEY", search_pexels),
           "pixabay": ("PIXABAY_API_KEY", search_pixabay)}


def order_for(fc, style):
    order = list(style["footage"]["source_order"])
    if fc == "map":
        order = [s for s in order if s != "dvids"] + ["dvids"]
    return order


def run_step(proj, keys, settings, style, log):
    proj = Path(proj)
    out_path = proj / "candidates.json"
    done = read_json(out_path, {}) or {}
    specs = read_json(proj / "specs.json")
    n = int(settings.get("candidates_per_query", 12))
    down = set()
    active = [s for s, (k, _) in SOURCES.items() if keys.get(k)]
    if not active:
        raise RuntimeError("No footage API key set (DVIDS / Pexels / Pixabay) — add one in Settings")
    log(f"searching footage on: {', '.join(active)}")
    for i, sp in enumerate(specs):
        bid = str(sp["id"])
        if bid in done and done[bid]:
            continue
        cands, seen = [], set()
        queries = list(sp["queries"])
        if sp["footage_class"] == "map":
            queries = [q + " aerial" for q in queries[:2]] + ["satellite earth view", "aerial coastline"]
        for q in queries:
            for src in order_for(sp["footage_class"], style):
                if src in down or src not in active:
                    continue
                try:
                    res = SOURCES[src][1](q, keys[SOURCES[src][0]], n)
                except SourceDown as e:
                    log(f"{e}; continuing without {src}")
                    down.add(src)
                    continue
                except Exception as e:
                    log(f"{src} search error for '{q}': {str(e)[:120]}")
                    continue
                for c in res:
                    k = (c["source"], c["id"])
                    if k not in seen and c["preview_url"]:
                        seen.add(k)
                        c["query"] = q
                        cands.append(c)
            if len(cands) >= n:
                break
        done[bid] = cands
        if i % 10 == 0 or i == len(specs) - 1:
            write_json(out_path, done)
            log(f"search {i + 1}/{len(specs)}", progress=(i + 1) / len(specs))
    write_json(out_path, done)
    empty = sum(1 for v in done.values() if not v)
    log(f"search done; {empty} beats without candidates")
    return done
