"""Shot library: clean, captioned 1080p shots kept on this Mac, matched to beats before any online search.

Build (one time / weekly top-up, runs overnight):
  for each category (F-15E, MQ-9 Reaper, carriers, ...): DVIDS search (B-Roll first) ->
  for each new video: small preview file (360-480p) -> scene cuts (ffmpeg) -> candidate shots 3-8 s ->
  clean check on the preview (black, blank, colour bars, Apple Vision text/face) -> the clean ones are cut ONCE
  in 1080p from the full-quality file -> keyframe -> CLIP vector -> one caption from the vision model.
Use (every video, per beat):
  CLIP(visual line) vs every shot's vector + named-system rule + caption words - recent-use penalty.
  A good enough match = library clip, no Gemini call. Otherwise the online search + rerank runs as before.
Learning: after a video, its online clips that Gemini scored >= 7 and passed the clean check join the library.
"""
import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from .config import DATA_DIR, ROOT
from .util import UA, download, duration, encoder_args, num, read_json, video_encoder, write_json

STATUS_FILE = DATA_DIR / "library_status.json"
_DB_LOCK = threading.RLock()
_CLIP_LOCK = threading.Lock()

# name, DVIDS search, entity written on shots of this category when the caption can't name one
DEFAULT_CATEGORIES = [
    ("F-15E Strike Eagle", "F-15E Strike Eagle", "F-15E Strike Eagle"),
    ("F-16 Fighting Falcon", "F-16 Fighting Falcon", "F-16 Fighting Falcon"),
    ("F-35 Lightning II", "F-35 Lightning II", "F-35 Lightning II"),
    ("F-22 Raptor", "F-22 Raptor", "F-22 Raptor"),
    ("F/A-18 Super Hornet", "F/A-18 Super Hornet", "F/A-18 Super Hornet"),
    ("A-10 Thunderbolt II", "A-10 Thunderbolt", "A-10 Thunderbolt II"),
    ("B-52 Stratofortress", "B-52 Stratofortress", "B-52 Stratofortress"),
    ("B-1B Lancer", "B-1B Lancer", "B-1B Lancer"),
    ("B-2 Spirit", "B-2 Spirit", "B-2 Spirit"),
    ("MQ-9 Reaper", "MQ-9 Reaper", "MQ-9 Reaper"),
    ("KC-135 tanker / aerial refueling", "KC-135 aerial refueling", "KC-135 Stratotanker"),
    ("KC-46 Pegasus", "KC-46 Pegasus", "KC-46 Pegasus"),
    ("C-17 Globemaster III", "C-17 Globemaster", "C-17 Globemaster III"),
    ("E-3 Sentry AWACS", "E-3 Sentry AWACS", "E-3 Sentry"),
    ("Aircraft carrier flight deck", "aircraft carrier flight deck", ""),
    ("Destroyer at sea", "guided missile destroyer underway", ""),
    ("Submarine", "submarine surfaced", ""),
    ("Amphibious assault ship", "amphibious assault ship", ""),
    ("Patriot missile battery", "Patriot missile", "Patriot"),
    ("HIMARS launch", "HIMARS live fire", "HIMARS"),
    ("M1 Abrams tank", "M1 Abrams", "M1 Abrams"),
    ("Bradley fighting vehicle", "Bradley fighting vehicle", "M2 Bradley"),
    ("AH-64 Apache", "AH-64 Apache", "AH-64 Apache"),
    ("UH-60 Black Hawk", "UH-60 Black Hawk", "UH-60 Black Hawk"),
    ("CH-47 Chinook", "CH-47 Chinook", "CH-47 Chinook"),
    ("V-22 Osprey", "V-22 Osprey", "V-22 Osprey"),
    ("M777 artillery firing", "M777 howitzer live fire", "M777 howitzer"),
    ("Ship missile launch", "missile launch ship", ""),
    ("Soldiers on patrol", "soldiers patrol", ""),
    ("Special forces training", "special forces training", ""),
    ("Targeting pod / FLIR feed", "targeting pod video", ""),
    ("Operations center / radar screens", "operations center", ""),
    ("Jets takeoff and landing", "takeoff afterburner", ""),
]

SKIP_TITLE = ("interview", "briefing", "press conference", "podcast", "town hall", "speech", "remarks",
              "testimony", "ceremony", "promotion", "retirement", "graduation", "change of command", "q&a", "pao")
STOP = set("the a an of and or on in at to for with by from into over under near its their this that is are was "
           "were be as it his her they them he she we you night day during while".split())


# ---------------------------------------------------------------- paths / settings
def lib_dir(settings=None):
    d = (settings or {}).get("library_dir") or ""
    p = Path(d).expanduser() if d else ROOT / "library" / "shots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path(settings=None):
    return lib_dir(settings) / "index.db"


def _db(settings=None):
    con = sqlite3.connect(db_path(settings), timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript("""
    CREATE TABLE IF NOT EXISTS shots(
      id INTEGER PRIMARY KEY, source TEXT, src_id TEXT, title TEXT, category TEXT, src_start REAL, length REAL,
      file TEXT UNIQUE, keyframe TEXT, caption TEXT, entity TEXT, tags TEXT, shot_type TEXT, quality REAL,
      credit TEXT, license TEXT, page_url TEXT, emb BLOB, added REAL, uses INTEGER DEFAULT 0,
      last_used REAL DEFAULT 0, origin TEXT);
    CREATE UNIQUE INDEX IF NOT EXISTS shot_src ON shots(source, src_id, src_start);
    CREATE TABLE IF NOT EXISTS videos(source TEXT, src_id TEXT, category TEXT, status TEXT, shots INTEGER,
      note TEXT, done REAL, PRIMARY KEY(source, src_id));
    CREATE TABLE IF NOT EXISTS uses(shot_id INTEGER, project TEXT, at REAL);
    """)
    return con


def stats(settings=None):
    with _DB_LOCK:
        con = _db(settings)
        n = con.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
        cats = {r[0] or "?": r[1] for r in con.execute("SELECT category, COUNT(*) FROM shots GROUP BY category")}
        vids = {r[0]: r[1] for r in con.execute("SELECT status, COUNT(*) FROM videos GROUP BY status")}
        learned = con.execute("SELECT COUNT(*) FROM shots WHERE origin LIKE 'video:%'").fetchone()[0]
        con.close()
    size = sum(f.stat().st_size for f in lib_dir(settings).glob("*.mp4"))
    free = shutil.disk_usage(lib_dir(settings)).free
    return {"shots": n, "by_category": cats, "videos": vids, "learned_from_videos": learned,
            "gb": round(size / 1e9, 2), "free_gb": round(free / 1e9, 1), "dir": str(lib_dir(settings))}


# ---------------------------------------------------------------- CLIP helpers (one model, used from threads)
def clip_image_vecs(settings, paths):
    from .steps.rank import _clip
    from PIL import Image
    with _CLIP_LOCK:
        c = _clip(settings)
        torch = c["torch"]
        ims = [c["preprocess"](Image.open(p).convert("RGB")) for p in paths]
        with torch.no_grad():
            e = c["model"].encode_image(torch.stack(ims).to(c["dev"]))
            e = e / e.norm(dim=-1, keepdim=True)
        return e.float().cpu().numpy().astype(np.float32)


def clip_text_vecs(settings, texts):
    from .steps.rank import _clip
    with _CLIP_LOCK:
        c = _clip(settings)
        torch = c["torch"]
        with torch.no_grad():
            e = c["model"].encode_text(c["tok"]([t[:300] for t in texts]).to(c["dev"]))
            e = e / e.norm(dim=-1, keepdim=True)
        return e.float().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- matching
def key_tokens(entity):
    """Tokens that name a system: 'F-15E Strike Eagle' -> {'f-15e', 'f15e', 'strike', 'eagle'};
    'USS Carney' -> {'carney'}. Matching any one of the designator tokens is required."""
    toks = re.findall(r"[a-z0-9][a-z0-9\-/]*", (entity or "").lower())
    des = [t for t in toks if any(ch.isdigit() for ch in t)]
    if des:
        out = set()
        for t in des:
            out.add(t)
            out.add(t.replace("-", "").replace("/", ""))
        return out
    words = [t for t in toks if t not in STOP and t not in ("uss", "usns", "hms", "the") and len(t) > 2]
    return set(words[-1:]) if words else set()


def _hay(row):
    return " ".join(str(row.get(k) or "") for k in ("entity", "caption", "tags", "title", "category")).lower()


class LibraryIndex:
    def __init__(self, settings=None):
        self.settings = settings or {}
        with _DB_LOCK:
            con = _db(settings)
            rows = [dict(r) for r in con.execute("SELECT * FROM shots WHERE emb IS NOT NULL")]
            week = time.time() - 7 * 86400
            recent = {r[0]: r[1] for r in con.execute(
                "SELECT shot_id, COUNT(DISTINCT project) FROM uses WHERE at > ? GROUP BY shot_id", (week,))}
            con.close()
        self.rows = [r for r in rows if Path(r["file"]).exists() and Path(r["keyframe"] or "").exists()]
        self.recent = recent
        self.E = (np.stack([np.frombuffer(r["emb"], dtype=np.float32) for r in self.rows])
                  if self.rows else np.zeros((0, 512), np.float32))
        self.hay = [_hay(r) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def match(self, text_vec, entity="", visual="", k=8):
        """Top-k library shots for one beat. A beat that names a system only gets shots of that system."""
        if not self.rows:
            return []
        sims = self.E @ text_vec.astype(np.float32)
        keys = key_tokens(entity)
        words = {w for w in re.findall(r"[a-z0-9\-]+", (visual or "").lower()) if w not in STOP and len(w) > 2}
        out = []
        for i, r in enumerate(self.rows):
            hay = self.hay[i]
            ent_ok = bool(keys) and any(t in hay or t in hay.replace("-", "") for t in keys)
            if keys and not ent_ok:
                continue
            bonus = 0.04 if ent_ok else 0.0
            bonus += min(0.03, 0.01 * sum(1 for w in words if w in hay))
            pen = min(0.04, 0.01 * self.recent.get(r["id"], 0))          # used in other videos this week
            if time.time() - (r.get("last_used") or 0) < 86400:
                pen += 0.01
            out.append((float(sims[i]) + bonus - pen, float(sims[i]), ent_ok, pen, r))
        out.sort(key=lambda x: -x[0])
        res = []
        for score, sim, ent_ok, pen, r in out[:k]:
            res.append({"source": "library", "id": str(r["id"]), "title": r["caption"] or r["title"] or "",
                        "preview_url": "", "preview_file": r["keyframe"], "video_url": r["file"],
                        "duration": r["length"], "tier": "exact" if ent_ok else "category", "query": "library",
                        "credit": r["credit"] or "", "license": r["license"] or "", "page_url": r["page_url"] or "",
                        "lib_score": round(score, 4), "clip_sim": round(sim, 4), "lib_penalty": round(pen, 4)})
        return res


def record_uses(proj, fetched, settings=None):
    ids = [int(v["id"]) for v in (fetched or {}).values() if v and v.get("source") == "library"]
    if not ids:
        return 0
    now = time.time()
    with _DB_LOCK:
        con = _db(settings)
        con.executemany("INSERT INTO uses(shot_id, project, at) VALUES(?,?,?)", [(i, Path(proj).name, now) for i in ids])
        con.executemany("UPDATE shots SET uses = uses + 1, last_used = ? WHERE id = ?", [(now, i) for i in ids])
        con.commit()
        con.close()
    return len(ids)


# ---------------------------------------------------------------- shot finding (pure functions, tested)
def detect_cuts(path, thresh=0.3):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vf",
                        f"select='gt(scene,{thresh})',showinfo", "-an", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=900)
    return [float(x) for x in re.findall(r"pts_time:([\d.]+)", p.stderr)]


def plan_shots(cuts, dur, min_start=0.0, min_len=3.0, max_len=8.0, pad=0.3):
    """Shots between scene cuts, trimmed, 3-8 s, out of the opening slate and the closing credits.
    One long continuous take gives up to 2 windows. Longest first."""
    end = dur - min(8.0, dur * 0.05) if dur > 60 else dur
    b = [max(0.0, min_start)] + [c for c in sorted(cuts) if min_start + 0.5 < c < end - 0.5] + [end]
    out = []
    for a, z in zip(b, b[1:]):
        a, z = a + pad, z - pad
        L = z - a
        if L < min_len:
            continue
        if L <= max_len:
            out.append((round(a, 2), round(L, 2)))
        elif L < 2.5 * max_len:
            mid = a + L / 2
            out.append((round(mid - max_len / 2, 2), max_len))
        else:
            out.append((round(a + L * 0.15, 2), max_len))
            out.append((round(a + L * 0.6, 2), max_len))
    return sorted(out, key=lambda s: -s[1])


def pick_files(files):
    """(proxy, full): a small file to analyse (<= 480p) and the file closest to 1080p to cut from."""
    mp4 = [f for f in files if isinstance(f, dict) and str(f.get("src", "")).split("?")[0].endswith(".mp4")]
    if not mp4:
        return None, None
    h = lambda f: int(num(f.get("height")))
    full = sorted(mp4, key=lambda f: (abs(h(f) - 1080), -int(num(f.get("width")))))[0]
    small = [f for f in mp4 if 240 <= h(f) <= 480] or sorted(mp4, key=h)[:1]
    proxy = sorted(small, key=h)[-1] if small and h(small[0]) <= 480 else sorted(mp4, key=h)[0]
    return proxy["src"], full["src"]


# ---------------------------------------------------------------- captions
CAPTION_PROMPT = (
    "This is a frame from a military B-roll shot. Category hint: {cat}. Source title: \"{title}\".\n"
    "Describe it for a footage search index. Name an aircraft/ship/vehicle/weapon type only if you can really "
    "identify it in the image (the hint can be wrong).\n"
    'Return JSON {{"caption": "<one line: subject, action, angle, setting, light>", "entity": "<exact type or empty>", '
    '"tags": ["5-10 lowercase search words"], "shot": "<wide|medium|close|aerial|cockpit|flir|interior>", '
    '"quality": <1-10 how clean and cinematic>}}')


class Captioner:
    def __init__(self, keys, settings, log):
        self.rr = None
        mode = str(settings.get("library_captions", "auto")).lower()
        if mode != "off":
            from .rerank import Reranker
            rr = Reranker(keys, settings, log)
            self.rr = rr if rr.active else None

    def __call__(self, keyframe, cat, title):
        if not self.rr:
            return None
        res, _ = self.rr.pick(lambda tile: keyframe, CAPTION_PROMPT.format(cat=cat, title=(title or "")[:120]))
        return res if isinstance(res, dict) else None


# ---------------------------------------------------------------- building
class Builder:
    def __init__(self, keys, settings, log, stop_event=None, status=None):
        self.keys, self.settings, self.log = keys, settings, log
        self.stop = stop_event or threading.Event()
        self.status = status or (lambda **kw: None)
        self.dir = lib_dir(settings)
        self.enc = video_encoder(settings.get("encoder", "auto"))
        self.per_video = int(settings.get("library_shots_per_video", 4))
        self.max_gb = float(settings.get("library_max_gb", 200))
        self.captioner = None

    # -- DVIDS
    def _dvids_search(self, query, n=50):
        from .steps.search import search_dvids
        return search_dvids(query + " B-Roll", self.keys["DVIDS_API_KEY"], n) + search_dvids(query, self.keys["DVIDS_API_KEY"], n)

    def _dvids_asset(self, src_id):
        r = httpx.get("https://api.dvidshub.net/asset", params={"api_key": self.keys["DVIDS_API_KEY"], "id": src_id},
                      headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        d = r.json().get("results") or r.json()
        files = d.get("files") or []
        if isinstance(files, dict):
            files = list(files.values())
        return d, files

    def _seen(self, src_id):
        with _DB_LOCK:
            con = _db(self.settings)
            r = con.execute("SELECT 1 FROM videos WHERE source='dvids' AND src_id=?", (src_id,)).fetchone()
            con.close()
        return bool(r)

    def _mark(self, src_id, cat, status, shots=0, note=""):
        with _DB_LOCK:
            con = _db(self.settings)
            con.execute("INSERT OR REPLACE INTO videos VALUES(?,?,?,?,?,?,?)",
                        ("dvids", src_id, cat, status, shots, note[:300], time.time()))
            con.commit()
            con.close()

    def _disk_ok(self):
        free = shutil.disk_usage(self.dir).free / 1e9
        used = sum(f.stat().st_size for f in self.dir.glob("*.mp4")) / 1e9
        if free < 15:
            self.log(f"STOP: only {free:.0f} GB free on this disk (keeps 15 GB free)")
            return False
        if used > self.max_gb:
            self.log(f"STOP: library is {used:.0f} GB, the limit is {self.max_gb:.0f} GB (Settings: library_max_gb)")
            return False
        return True

    def candidates(self, cat_query):
        """DVIDS results for a category: B-Roll first, no interviews/briefings, 20 s - 15 min, not done before."""
        from .steps.rank import title_adjust
        seen_ids, out = set(), []
        for r in self._dvids_search(cat_query):
            t = (r.get("title") or "").lower()
            d = num(r.get("duration"))
            if r["id"] in seen_ids or any(w in t for w in SKIP_TITLE) or (d and not 20 <= d <= 900):
                continue
            seen_ids.add(r["id"])
            if not self._seen(r["id"]):
                out.append(r)
        return sorted(out, key=lambda r: -title_adjust(r.get("title")))

    def process_video(self, cat, r):
        """One DVIDS video -> up to per_video clean 1080p shots in the library. Returns shots added."""
        from .cleancheck import check_segment
        name, _, cat_entity = cat
        src_id = r["id"]
        asset, files = self._dvids_asset(src_id)
        proxy_url, full_url = pick_files(files)
        if not proxy_url:
            self._mark(src_id, name, "failed", note="no mp4 file")
            return 0
        min_start = num(asset.get("time_start"))
        with tempfile.TemporaryDirectory() as td:
            proxy = Path(td) / "proxy.mp4"
            download(proxy_url, proxy, timeout=600)
            dur = duration(proxy)
            shots = plan_shots(detect_cuts(proxy), dur, min_start)
            clean = []
            for i, (st, L) in enumerate(shots[: self.per_video * 3]):
                if self.stop.is_set() or len(clean) >= self.per_video:
                    break
                part = Path(td) / f"p{i}.mp4"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{st}", "-i", str(proxy), "-t", f"{L}",
                                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", str(part)],
                               capture_output=True, timeout=300)
                if not part.exists():
                    continue
                chk = check_segment(part, L, "auto")
                if chk.get("problems"):
                    continue
                clean.append((st, L))
            added = 0
            for st, L in sorted(clean):
                if self.stop.is_set():
                    break
                if self.add_shot_from_url(full_url, st, L, name, cat_entity, asset, r):
                    added += 1
        self._mark(src_id, name, "done", added)
        return added

    def add_shot_from_url(self, url, st, L, cat_name, cat_entity, asset, r):
        base = f"dvids_{re.sub(r'[^0-9A-Za-z]+', '_', r['id'])}_{int(st * 100)}"
        out, kf = self.dir / f"{base}.mp4", self.dir / f"{base}.jpg"
        tmp = self.dir / f"{base}.part.mp4"
        vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,format=yuv420p"
        try:
            ua = ["-user_agent", UA] if str(url).startswith("http") else []
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{st}", *ua, "-i", url,
                            "-t", f"{L}", "-an", "-vf", vf, *encoder_args(self.enc), str(tmp)],
                           capture_output=True, timeout=900, check=True)
            if duration(tmp) < min(2.0, L * 0.5):
                raise RuntimeError("cut too short")
            tmp.replace(out)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            self.log(f"  shot {base}: 1080p cut failed ({str(e)[-120:]})")
            return False
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{L / 2:.2f}", "-i", str(out), "-frames:v", "1",
                        "-vf", "scale=640:-2", str(kf)], capture_output=True, timeout=120)
        return self._insert(out, kf, source="dvids", src_id=r["id"], src_start=st, length=duration(out),
                            title=r.get("title") or "", cat_name=cat_name, cat_entity=cat_entity,
                            credit=r.get("credit") or "DVIDS", license="Public domain (US DoD / DVIDS)",
                            page_url=r.get("page_url") or asset.get("url") or "", origin="build")

    def _insert(self, out, kf, source, src_id, src_start, length, title, cat_name, cat_entity, credit, license,
                page_url, origin, caption=None):
        if not Path(kf).exists():
            Path(out).unlink(missing_ok=True)
            return False
        vec = clip_image_vecs(self.settings, [kf])[0]
        cap = caption
        if cap is None and self.captioner:
            try:
                cap = self.captioner(str(kf), cat_name, title)
            except Exception:
                cap = None
        cap = cap or {}
        tags = cap.get("tags") if isinstance(cap.get("tags"), list) else []
        row = (source, src_id, title[:200], cat_name, src_start, length, str(out), str(kf),
               str(cap.get("caption") or title)[:300], str(cap.get("entity") or cat_entity or "")[:80],
               " ".join(str(t) for t in tags)[:300], str(cap.get("shot") or "")[:20], num(cap.get("quality"), 0),
               credit[:200], license[:120], page_url[:300], vec.astype(np.float32).tobytes(), time.time(), origin)
        with _DB_LOCK:
            con = _db(self.settings)
            try:
                con.execute("""INSERT INTO shots(source, src_id, title, category, src_start, length, file, keyframe,
                    caption, entity, tags, shot_type, quality, credit, license, page_url, emb, added, origin)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
                con.commit()
                ok = True
            except sqlite3.IntegrityError:
                ok = False
            con.close()
        return ok

    def build(self, target, categories=None):
        """Grow the library to `target` shots, round-robin over categories, 4 videos at a time. Resumable:
        videos already done are skipped, so a stopped/crashed build continues where it was."""
        from concurrent.futures import ThreadPoolExecutor
        if not self.keys.get("DVIDS_API_KEY"):
            raise RuntimeError("DVIDS_API_KEY is needed to build the library")
        cats = [c for c in DEFAULT_CATEGORIES if not categories or c[0] in categories]
        self.captioner = Captioner(self.keys, self.settings, self.log)
        self.log(f"library build: target {target} shots, {len(cats)} categories, {self.per_video} shots per video, "
                 f"captions: {'on (' + self.captioner.rr.label + ')' if self.captioner.rr else 'off'}; folder {self.dir}")
        have = stats(self.settings)["shots"]
        if have >= target:
            self.log(f"library already has {have} shots (target {target}) - raise the target to add more")
            return have
        queues = {}
        for c in cats:
            try:
                queues[c[0]] = self.candidates(c[1])
            except Exception as e:
                self.log(f"  {c[0]}: DVIDS search failed ({str(e)[-120:]})")
                queues[c[0]] = []
            self.log(f"  {c[0]}: {len(queues[c[0]])} new videos found")
        by_name = {c[0]: c for c in cats}
        lock = threading.Lock()
        count = [have]
        t0 = time.time()

        def job(item):
            name, r = item
            if self.stop.is_set() or count[0] >= target:
                return
            try:
                n = self.process_video(by_name[name], r)
            except Exception as e:
                self._mark(r["id"], name, "failed", note=str(e)[-200:])
                self.log(f"  {name}: video {r['id']} failed ({str(e)[-100:]})")
                return
            with lock:
                count[0] += n
                c_ = count[0]
            rate = (c_ - have) / max(time.time() - t0, 1) * 3600
            self.log(f"  {name}: +{n} shots ({c_}/{target}, ~{rate:.0f} shots/hour)", )
            self.status(shots=c_, target=target, progress=min(1.0, c_ / target))

        while count[0] < target and not self.stop.is_set() and any(queues.values()):
            batch = []
            for name in list(queues):
                if queues[name]:
                    batch.append((name, queues[name].pop(0)))
            for i in range(0, len(batch), 4):
                if not self._disk_ok():
                    self.stop.set()
                    break
                with ThreadPoolExecutor(4) as ex:
                    list(ex.map(job, batch[i:i + 4]))
                if self.stop.is_set() or count[0] >= target:
                    break
        if not any(queues.values()) and count[0] < target:
            self.log("no more new DVIDS videos in these categories - add categories or lower the target")
        self.log(f"library build {'stopped' if self.stop.is_set() and count[0] < target else 'done'}: "
                 f"{count[0]} shots in {(time.time() - t0) / 60:.0f} min")
        return count[0]

    # -- learning from a finished video
    def learn_from_project(self, proj):
        """Online clips of a finished video that Gemini scored >= 7 and that passed the clean check join the
        library (caption = the beat's visual line). No extra API cost."""
        proj = Path(proj)
        picks = read_json(proj / "picks.json", {}) or {}
        fetched = read_json(proj / "fetched.json", {}) or {}
        specs = {str(s["id"]): s for s in (read_json(proj / "specs.json", []) or [])}
        added = 0
        for bid, f in fetched.items():
            p = picks.get(bid) or {}
            if not f or f.get("source") in ("library", "user") or f.get("problems"):
                continue
            if num(p.get("gemini_score"), 0) < 7 or not Path(f["file"]).exists() or num(f.get("length")) < 2.0:
                continue
            sp = specs.get(bid, {})
            base = f"{f['source']}_{re.sub(r'[^0-9A-Za-z]+', '_', str(f['id']))}_{int(num(f.get('start')) * 100)}"
            out, kf = self.dir / f"{base}.mp4", self.dir / f"{base}.jpg"
            if out.exists():
                continue
            shutil.copy2(f["file"], out)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{num(f.get('length')) / 2:.2f}", "-i", str(out),
                            "-frames:v", "1", "-vf", "scale=640:-2", str(kf)], capture_output=True, timeout=120)
            cap = {"caption": sp.get("visual") or f.get("title") or "", "entity": sp.get("entity") or "",
                   "tags": " ".join(sp.get("queries") or []).lower().split()[:10], "quality": p.get("gemini_score")}
            if self._insert(out, kf, source=f["source"], src_id=str(f["id"]), src_start=num(f.get("start")),
                            length=num(f.get("length")), title=f.get("title") or "", cat_name="learned",
                            cat_entity=sp.get("entity") or "", credit=f.get("credit") or "", license=f.get("license") or "",
                            page_url=f.get("page_url") or "", origin=f"video:{proj.name}", caption=cap):
                added += 1
        return added


# ---------------------------------------------------------------- background job (one at a time)
_JOB = {"thread": None, "stop": threading.Event()}


def job_status():
    st = read_json(STATUS_FILE, {}) or {}
    st["running"] = bool(_JOB["thread"] and _JOB["thread"].is_alive())
    return st


def start_build(target, categories=None, per_video=None):
    from .config import load_keys, load_settings
    if _JOB["thread"] and _JOB["thread"].is_alive():
        raise RuntimeError("a library build is already running")
    settings = load_settings()
    if per_video:
        settings = dict(settings, library_shots_per_video=int(per_video))
    st = {"state": "running", "started": time.time(), "target": int(target), "log": [], "progress": 0}
    write_json(STATUS_FILE, st)
    lock = threading.Lock()

    def log(msg, **_):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with lock:
            st["log"] = (st["log"] + [line])[-300:]
            write_json(STATUS_FILE, st)

    def status(**kw):
        with lock:
            st.update(kw)
            write_json(STATUS_FILE, st)

    _JOB["stop"] = threading.Event()

    def run():
        try:
            n = Builder(load_keys(), settings, log, _JOB["stop"], status).build(int(target), categories)
            status(state="stopped" if _JOB["stop"].is_set() and n < int(target) else "done", shots=n,
                   finished=time.time())
        except Exception as e:
            import traceback
            log(f"ERROR: {e}")
            log(traceback.format_exc()[-1200:])
            status(state="error", error=str(e)[:400], finished=time.time())

    _JOB["thread"] = threading.Thread(target=run, daemon=True)
    _JOB["thread"].start()
    return st


def stop_build():
    _JOB["stop"].set()
    return job_status()
