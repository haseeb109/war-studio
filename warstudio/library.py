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
from .util import (UA, NetGate, download, download_resume, duration, encoder_args, ffmpeg_net_args, is_network_error,
                   num, read_json, run, video_encoder, with_net_retry, write_json)

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
    cols = [r[1] for r in con.execute("PRAGMA table_info(shots)")]
    for col, typ in (("phash", "TEXT"), ("favorite", "INTEGER DEFAULT 0"), ("action", "TEXT"),
                     ("frames", "INTEGER DEFAULT 1"), ("src_height", "INTEGER"), ("audio", "INTEGER")):
        if col not in cols:
            con.execute(f"ALTER TABLE shots ADD COLUMN {col} {typ}")
    con.execute("""CREATE TABLE IF NOT EXISTS deleted(phash TEXT, source TEXT, src_id TEXT, src_start REAL,
                   length REAL, emb BLOB, at REAL)""")
    con.commit()
    return con


# ---------------------------------------------------------------- actions (what happens in the shot)
ACTIONS = [
    ("refueling", ["refuel", "tanker", "boom operator", "drogue", "take on fuel", "takes on fuel", "fuel up"]),
    ("takeoff", ["takeoff", "take off", "takes off", "taking off", "lift off", "lifts off", "liftoff", "catapult"]),
    ("landing", ["landing", " lands", "touchdown", "touch down", "trap on", "arrested"]),
    ("launch", ["launch", "leaves the rail", "missile leaves", "fires a missile", "fired a missile", "missile away"]),
    ("explosion", ["explosion", "explode", "blast", "detonat", "fireball", "turns white", "impact", "bombs fall"]),
    ("firing", ["firing", "fires", "fired", "gunfire", "howitzer", "salvo", "rounds", "live fire", "shoots"]),
    ("console", ["screen", "display", "operator", "control station", "operations center", "radar", "console",
                 "monitor", "feed", "targeting pod", "pod"]),
    ("hover", ["hover", "rappel", "fast rope", "sling load"]),
    ("taxi", ["taxi", "flight line", "flightline", "parked", "on the ramp"]),
    ("sailing", ["underway", "sailing", "at sea", "steaming", "wake", "turns into the wind", "off the coast"]),
    ("patrol", ["patrol", "convoy", "soldiers move", "advance", "dismount", "trucks", "driving", "road"]),
    ("loading", ["loading", "munitions", "ordnance", "maintenance", "crew chief", "weapons load"]),
    ("formation", ["formation", "wingman", "pair of", "two jets", "flight of"]),
    ("flying", ["flying", "flies", "in flight", "airborne", "banking", "banks", "circling", "orbit", "cruise",
                "overhead", "climb", "low through", "push their throttles"]),
]
ACTION_NAMES = [a for a, _ in ACTIONS]


def detect_action(text):
    t = f" {(text or '').lower()} "
    for name, words in ACTIONS:
        if any(w in t for w in words):
            return name
    return ""


def dhash(path):
    """64-bit difference hash of an image: near-identical frames differ in only a few bits."""
    from PIL import Image
    px = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(int(a) ^ int(b)).count("1")


DUP_HASH_BITS = 6        # keyframes this close are the same picture
DUP_CLIP = 0.965         # ...or this similar for CLIP with a loosely similar hash
DUP_CLIP_HASH_BITS = 14


def stats(settings=None):
    with _DB_LOCK:
        con = _db(settings)
        n = con.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
        cats = {r[0] or "?": r[1] for r in con.execute("SELECT category, COUNT(*) FROM shots GROUP BY category")}
        vids = {r[0]: r[1] for r in con.execute("SELECT status, COUNT(*) FROM videos GROUP BY status")}
        learned = con.execute("SELECT COUNT(*) FROM shots WHERE origin LIKE 'video:%'").fetchone()[0]
        favs = con.execute("SELECT COUNT(*) FROM shots WHERE favorite = 1").fetchone()[0]
        deleted = con.execute("SELECT COUNT(*) FROM deleted").fetchone()[0]
        sound = {"with": con.execute("SELECT COUNT(*) FROM shots WHERE audio = 1").fetchone()[0],
                 "none": con.execute("SELECT COUNT(*) FROM shots WHERE audio = 0").fetchone()[0],
                 "unverified": con.execute("SELECT COUNT(*) FROM shots WHERE audio = -1").fetchone()[0],
                 "unknown": con.execute("SELECT COUNT(*) FROM shots WHERE audio IS NULL").fetchone()[0]}
        con.close()
    size = sum(f.stat().st_size for f in lib_dir(settings).glob("*.mp4"))
    free = shutil.disk_usage(lib_dir(settings)).free
    return {"shots": n, "by_category": cats, "videos": vids, "learned_from_videos": learned,
            "avg_mb_per_shot": round(size / 1e6 / n, 1) if n else 9.0, "favorites": favs, "deleted": deleted,
            "sound": sound,
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


def shot_vec(settings, video, length, keyframe=None):
    """CLIP vector of a shot = average of 3 frames (25/50/75 %), so moving shots (takeoff, launch) are described
    by more than one moment. Falls back to the keyframe alone. Returns (vector, frames_used)."""
    with tempfile.TemporaryDirectory() as td:
        frames = []
        for i, f in enumerate((0.25, 0.5, 0.75)):
            o = Path(td) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, num(length) * f):.2f}", "-i",
                            str(video), "-frames:v", "1", "-vf", "scale=448:-2", str(o)], capture_output=True, timeout=120)
            if o.exists() and o.stat().st_size > 0:
                frames.append(str(o))
        if len(frames) >= 2:
            v = clip_image_vecs(settings, frames).mean(axis=0)
            return (v / (np.linalg.norm(v) or 1)).astype(np.float32), len(frames)
    return clip_image_vecs(settings, [keyframe])[0], 1


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
    def __init__(self, settings=None, exclude_project=None):
        self.settings = settings or {}
        n_rot = int(self.settings.get("library_rotation_videos", 5))
        with _DB_LOCK:
            con = _db(settings)
            rows = [dict(r) for r in con.execute("SELECT * FROM shots WHERE emb IS NOT NULL")]
            week = time.time() - 7 * 86400
            recent = {r[0]: r[1] for r in con.execute(
                "SELECT shot_id, COUNT(DISTINCT project) FROM uses WHERE at > ? GROUP BY shot_id", (week,))}
            last = [r[0] for r in con.execute(
                "SELECT project FROM uses WHERE project != ? GROUP BY project ORDER BY MAX(at) DESC LIMIT ?",
                (exclude_project or "", n_rot))] if n_rot > 0 else []
            self.rotated = {r[0] for r in con.execute(
                f"SELECT DISTINCT shot_id FROM uses WHERE project IN ({','.join('?' * len(last))})", last)} if last else set()
            con.close()
        self.rows = [r for r in rows if Path(r["file"]).exists() and Path(r["keyframe"] or "").exists()]
        self.recent = recent
        self.E = (np.stack([np.frombuffer(r["emb"], dtype=np.float32) for r in self.rows])
                  if self.rows else np.zeros((0, 512), np.float32))
        self.hay = [_hay(r) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def match(self, text_vec, entity="", visual="", k=8, rotation=True):
        """Top-k library shots for one beat. A beat that names a system only gets shots of that system.
        Shots used in the last `library_rotation_videos` videos are left out (used only if nothing else fits)."""
        if not self.rows:
            return []
        sims = self.E @ text_vec.astype(np.float32)
        keys = key_tokens(entity)
        words = {w for w in re.findall(r"[a-z0-9\-]+", (visual or "").lower()) if w not in STOP and len(w) > 2}
        want_action = detect_action(visual)
        out, held = [], []
        for i, r in enumerate(self.rows):
            hay = self.hay[i]
            ent_ok = bool(keys) and any(t in hay or t in hay.replace("-", "") for t in keys)
            if keys and not ent_ok:
                continue
            bonus = 0.04 if ent_ok else 0.0
            act = r.get("action") or detect_action(hay)
            if want_action and act:
                bonus += 0.03 if act == want_action else -0.015      # takeoff beat -> takeoff shots first
            if r.get("favorite"):
                bonus += 0.01
            bonus += min(0.03, 0.01 * sum(1 for w in words if w in hay))
            q = num(r.get("quality"), 0)
            if q:
                bonus += max(-0.01, min(0.01, 0.004 * (q - 6)))           # cleaner / more cinematic shots first
            pen = min(0.04, 0.01 * self.recent.get(r["id"], 0))          # used in other videos this week
            if time.time() - (r.get("last_used") or 0) < 86400:
                pen += 0.01
            item = (float(sims[i]) + bonus - pen, float(sims[i]), ent_ok, pen, r)
            if rotation and r["id"] in self.rotated:
                held.append(item)
            else:
                out.append(item)
        if not out and held:            # nothing fresh fits this beat: a recent shot, with a penalty
            out = [(s_ - 0.05, sim, e, pen + 0.05, r) for s_, sim, e, pen, r in held]
        out.sort(key=lambda x: -x[0])
        res = []
        for score, sim, ent_ok, pen, r in out[:k]:
            res.append({"source": "library", "id": str(r["id"]), "title": r["caption"] or r["title"] or "",
                        "preview_url": f"/api/library/thumb/{r['id']}", "preview_file": r["keyframe"],
                        "video_url": r["file"], "action": r.get("action") or "",
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


MIN_SOURCE_HEIGHT = 720


def has_audio(path):
    """True when the media file (local path or URL) has an audio stream."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", *ffmpeg_net_args(path), "-select_streams", "a",
                            "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=90)
    except Exception:
        return False
    if r.returncode != 0 and is_network_error(RuntimeError(r.stderr)):
        raise RuntimeError(f"ffprobe network error: {r.stderr[-200:]}")
    return bool(r.stdout.strip())


ALIGN_W, ALIGN_H, ALIGN_FPS = 96, 54, 30       # tiny grey frames used to prove the sound belongs to the picture
ALIGN_MIN = 0.85                                 # mean frame correlation needed (same video ~0.95+, another <0.6)


def _grey_frames(path):
    """The shot's own frames, tiny and grey (it is already 1920x1080 @ 30 fps)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
                          f"scale={ALIGN_W}:{ALIGN_H}:force_original_aspect_ratio=increase,crop={ALIGN_W}:{ALIGN_H},"
                          f"fps={ALIGN_FPS},format=gray", "-f", "rawvideo", "-"], capture_output=True, timeout=120).stdout
    a = np.frombuffer(raw, dtype=np.uint8)
    n = len(a) // (ALIGN_W * ALIGN_H)
    return a[:n * ALIGN_W * ALIGN_H].reshape(n, ALIGN_W * ALIGN_H)


def _norm_frames(f):
    x = f.astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > 1e-3, x / np.maximum(n, 1e-3), 0.0)


MOTION_MIN = 1.5       # mean grey-level change between frames that counts as movement


def _shift_scores(A, B, pre, lo, hi, max_shift):
    out = {}
    for d in range(-max_shift, max_shift + 1):
        k = np.arange(lo, hi)
        j = k + pre + d
        m = (j >= 0) & (j < len(B))
        if m.sum() >= 0.8 * len(k) and len(k):
            out[d] = float((A[k[m]] * B[j[m]]).sum(axis=1).mean())
    return out


def verify_alignment(shot, src, pre_frames, max_shift=15, exact_file=False):
    """Prove that `src` (frames of a source file starting `pre_frames` before the recorded shot start) shows this
    shot at these seconds. Two proofs:
      picture - the frames look the same (correlation >= ALIGN_MIN): the same video;
      timing  - the frame-to-frame MOVEMENT lines up at one shift, clearly better than every other shift, and both
                halves of the shot agree on it: the same seconds, to one frame.
    A moving shot needs both. A still shot cannot prove its timing: it is accepted only from the exact file the shot
    was cut from (`exact_file`, same timeline by construction) at the recorded start. Returns
    {ok, offset (frames), score, motion, still}."""
    no = {"ok": False, "offset": 0, "score": 0.0, "motion": 0.0, "still": True}
    if len(shot) < 4 or len(src) < 4:
        return no
    A, B = _norm_frames(shot), _norm_frames(src)
    dS = np.diff(shot.astype(np.int16), axis=0)
    dR = np.diff(src.astype(np.int16), axis=0)
    moving = float((np.abs(dS).mean(axis=1) > MOTION_MIN).mean())
    still = moving < 0.3
    look = _shift_scores(A, B, pre_frames, 0, len(A), max_shift)
    if still:
        sc = look.get(0, 0.0)
        return {"ok": bool(exact_file and sc >= ALIGN_MIN), "offset": 0, "score": round(sc, 4), "motion": 0.0,
                "still": True}
    mA, mB = _norm_frames(dS), _norm_frames(dR)
    mot = _shift_scores(mA, mB, pre_frames, 0, len(mA), max_shift)
    if not mot:
        return no
    best = max(mot, key=mot.get)
    others = [v for d, v in mot.items() if abs(d - best) > 2]
    margin = mot[best] - (max(others) if others else 0.0)
    ok = mot[best] >= 0.5 and margin >= 0.15 and look.get(best, 0.0) >= ALIGN_MIN
    if ok and len(mA) >= 20:
        h1 = _shift_scores(mA, mB, pre_frames, 0, len(mA) // 2, max_shift)
        h2 = _shift_scores(mA, mB, pre_frames, len(mA) // 2, len(mA), max_shift)
        if h1 and h2 and abs(max(h1, key=h1.get) - max(h2, key=h2.get)) > 1:
            ok = False                                       # the halves disagree: the timelines drift
    return {"ok": bool(ok), "offset": best, "score": round(look.get(best, 0.0), 4), "motion": round(mot[best], 4),
            "still": False}


def pick_files(files):
    """(proxy, full, full_height): a small file to analyse (<= 480p) and the file closest to 1080p to cut from."""
    mp4 = [f for f in files if isinstance(f, dict) and str(f.get("src", "")).split("?")[0].endswith(".mp4")]
    if not mp4:
        return None, None, 0
    h = lambda f: int(num(f.get("height")))
    full = sorted(mp4, key=lambda f: (abs(h(f) - 1080), -int(num(f.get("width")))))[0]
    small = [f for f in mp4 if 240 <= h(f) <= 480] or sorted(mp4, key=h)[:1]
    proxy = sorted(small, key=h)[-1] if small and h(small[0]) <= 480 else sorted(mp4, key=h)[0]
    return proxy["src"], full["src"], h(full)


# ---------------------------------------------------------------- captions
CAPTION_PROMPT = (
    "This is a frame from a military B-roll shot. Category hint: {cat}. Source title: \"{title}\".\n"
    "Describe it for a footage search index. Name an aircraft/ship/vehicle/weapon type only if you can really "
    "identify it in the image (the hint can be wrong).\n"
    'Return JSON {{"caption": "<one line: subject, action, angle, setting, light>", "entity": "<exact type or empty>", '
    '"action": "<one of: ' + "|".join(ACTION_NAMES) + '|other>", '
    '"tags": ["5-10 lowercase search words"], "shot": "<wide|medium|close|aerial|cockpit|flir|interior>", '
    '"quality": <1-10 how clean and cinematic>}}')


def best_lite_model(key):
    """Newest stable Gemini Flash-Lite model this key can use (about half the price of Flash), or None."""
    from .llm import _GONE, gemini_models

    def ver(m):
        x = re.search(r"gemini-(\d+(?:\.\d+)?)", m)
        return float(x.group(1)) if x else 0.0
    lite = [m for m in gemini_models(key) if "flash-lite" in m and m not in _GONE
            and not any(b in m for b in ("preview", "exp", "tts", "image", "audio", "live"))]
    return sorted(lite, key=ver, reverse=True)[0] if lite else None


class Captioner:
    def __init__(self, keys, settings, log):
        self.rr = None
        mode = str(settings.get("library_captions", "auto")).lower()
        if mode != "off":
            from .rerank import Reranker
            s = dict(settings)
            if str(settings.get("library_caption_model", "lite")).lower() == "lite" and keys.get("GEMINI_API_KEY"):
                try:
                    lite = best_lite_model(keys["GEMINI_API_KEY"])
                except Exception:
                    lite = None
                if lite:
                    s.update(reranker="gemini", gemini_model=lite)
            rr = Reranker(keys, s, log)
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
        self._dup_lock = threading.Lock()
        self._dup = None            # [(id, phash, vec, source, src_id, start, length)]
        self.dupes = 0
        self.pages = {}             # category -> next DVIDS results page
        # 2 network / DNS failures in a row pause the whole build (no next file is tried) until Retry is pressed
        # or - auto-continue on - the internet really answers again
        self.gate = NetGate(self.log, self.status, self.stop,
                            max_fails=int(settings.get("library_pause_after", 2)),
                            auto=str(settings.get("library_auto_continue", True)).lower() not in ("false", "0", "no", "off"))

    # -- DVIDS
    def _dvids_search(self, query, n=50, page=1):
        from .steps.search import search_dvids
        k = self.keys["DVIDS_API_KEY"]
        return with_net_retry(lambda: search_dvids(query + " B-Roll", k, n, page) + search_dvids(query, k, n, page),
                              self.log, self.stop, what=f"DVIDS search '{query}'", status=self.status, gate=self.gate)

    def _dvids_asset(self, src_id):
        def get():
            r = httpx.get("https://api.dvidshub.net/asset", params={"api_key": self.keys["DVIDS_API_KEY"], "id": src_id},
                          headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            return r.json()
        j = with_net_retry(get, self.log, self.stop, what=f"DVIDS asset {src_id}", status=self.status, gate=self.gate)
        d = j.get("results") or j
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

    def candidates(self, cat_query, want=12, max_page=40):
        """New DVIDS videos for a category: B-Roll first, no interviews/briefings, 20 s - 15 min, not done before.
        Walks the result pages (50 per page) until `want` new videos are found, remembering the page for the
        next call - so the library can keep growing far past the first page of every category."""
        from .steps.rank import title_adjust
        seen_ids, out = set(), []
        page = self.pages.get(cat_query, 1)
        raw = []
        while len(out) < want and page <= max_page:
            got = self._dvids_search(cat_query, 50, page)
            page += 1
            if not got:
                page = max_page + 1
                break
            raw = got
            out += self._filter(raw, seen_ids)
        self.pages[cat_query] = page
        return sorted(out, key=lambda r: -title_adjust(r.get("title")))

    def _filter(self, results, seen_ids):
        out = []
        for r in results:
            t = (r.get("title") or "").lower()
            d = num(r.get("duration"))
            if r["id"] in seen_ids or any(w in t for w in SKIP_TITLE) or (d and not 20 <= d <= 900):
                continue
            seen_ids.add(r["id"])
            if not self._seen(r["id"]):
                out.append(r)
        return out

    def process_video(self, cat, r):
        """One DVIDS video -> up to per_video clean 1080p shots in the library. Returns shots added."""
        from .cleancheck import check_segment
        name, _, cat_entity = cat
        src_id = r["id"]
        asset, files = self._dvids_asset(src_id)
        proxy_url, full_url, full_h = pick_files(files)
        if not proxy_url:
            self._mark(src_id, name, "failed", note="no mp4 file")
            return 0
        if full_h and full_h < MIN_SOURCE_HEIGHT:
            self._mark(src_id, name, "skipped", note=f"only {full_h}p (needs {MIN_SOURCE_HEIGHT}p+)")
            return 0
        min_start = num(asset.get("time_start"))
        with tempfile.TemporaryDirectory() as td:
            proxy = Path(td) / "proxy.mp4"
            # resumable: a dropped connection continues from the bytes already saved
            download_resume(proxy_url, proxy, self.log, self.stop, status=self.status, gate=self.gate)
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
                if self.add_shot_from_url(full_url, st, L, name, cat_entity, asset, r, full_h):
                    added += 1
        self._mark(src_id, name, "done", added)
        return added

    def add_shot_from_url(self, url, st, L, cat_name, cat_entity, asset, r, src_height=0):
        base = f"dvids_{re.sub(r'[^0-9A-Za-z]+', '_', r['id'])}_{int(st * 100)}"
        out, kf = self.dir / f"{base}.mp4", self.dir / f"{base}.jpg"
        tmp = self.dir / f"{base}.part.mp4"
        vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,format=yuv420p"
        def cut():
            # ffmpeg reconnects by itself after a drop; a stalled read gives up after 30 s and is retried here
            run(["ffmpeg", "-y", "-loglevel", "error", *ffmpeg_net_args(url), "-ss", f"{st}", "-i", url,
                 "-t", f"{L}", "-vf", vf, *encoder_args(self.enc), "-map", "0:v:0", "-map", "0:a:0?",
                 "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(tmp)], timeout=900)
            if duration(tmp) < min(2.0, L * 0.5):
                raise RuntimeError("cut too short")
        try:
            with_net_retry(cut, self.log, self.stop, what=f"1080p cut {base}", status=self.status, gate=self.gate)
            tmp.replace(out)
        except Exception as e:
            if is_network_error(e):
                tmp.unlink(missing_ok=True)
                raise                     # the whole video is retried later, not marked failed
            tmp.unlink(missing_ok=True)
            self.log(f"  shot {base}: 1080p cut failed ({str(e)[-120:]})")
            return False
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{L / 2:.2f}", "-i", str(out), "-frames:v", "1",
                        "-vf", "scale=640:-2", str(kf)], capture_output=True, timeout=120)
        return self._insert(out, kf, source="dvids", src_id=r["id"], src_start=st, length=duration(out),
                            title=r.get("title") or "", cat_name=cat_name, cat_entity=cat_entity,
                            credit=r.get("credit") or "DVIDS", license="Public domain (US DoD / DVIDS)",
                            page_url=r.get("page_url") or asset.get("url") or "", origin="build",
                            src_height=src_height)

    def _load_dups(self):
        with _DB_LOCK:
            con = _db(self.settings)
            rows = [dict(r) for r in con.execute(
                "SELECT id, phash, emb, keyframe, source, src_id, src_start, length FROM shots")]
            # shots you deleted in the browser stay out: they count as duplicates forever
            rows += [dict(r, id=-2, keyframe=None) for r in con.execute(
                "SELECT phash, emb, source, src_id, src_start, length FROM deleted")]
            fix = []
            for r in rows:
                if not r["phash"] and r["keyframe"] and Path(r["keyframe"]).exists():
                    r["phash"] = str(dhash(r["keyframe"]))
                    fix.append((r["phash"], r["id"]))
            if fix:
                con.executemany("UPDATE shots SET phash = ? WHERE id = ?", fix)
                con.commit()
            con.close()
        self._dup = [(r["id"], int(r["phash"]) if r["phash"] else None,
                      np.frombuffer(r["emb"], dtype=np.float32) if r["emb"] else None,
                      r["source"], r["src_id"], num(r["src_start"]), num(r["length"])) for r in rows]

    def duplicate_of(self, vec, ph, source, src_id, st, L):
        """Id of an existing shot this one duplicates, else None.
        Same source video with the time ranges overlapping by half or more, or the same picture:
        DVIDS often posts the same B-roll inside several packages, so the picture check matters."""
        if self._dup is None:
            self._load_dups()
        for sid, h, v, src, sidv, s0, l0 in self._dup:
            if src == source and sidv == src_id and L and l0:
                ov = min(st + L, s0 + l0) - max(st, s0)
                if ov >= 0.5 * min(L, l0):
                    return sid
            if h is None:
                continue
            d = hamming(h, ph)
            if d <= DUP_HASH_BITS:
                return sid
            if v is not None and d <= DUP_CLIP_HASH_BITS and float(np.dot(v, vec)) >= DUP_CLIP:
                return sid
        return None

    def _insert(self, out, kf, source, src_id, src_start, length, title, cat_name, cat_entity, credit, license,
                page_url, origin, caption=None, src_height=0):
        if not Path(kf).exists():
            Path(out).unlink(missing_ok=True)
            return False
        vec, n_frames = shot_vec(self.settings, out, length, kf)
        ph = dhash(kf)
        with self._dup_lock:
            dup = self.duplicate_of(vec, ph, source, src_id, num(src_start), num(length))
            if dup is not None:
                self.dupes += 1
                Path(out).unlink(missing_ok=True)
                Path(kf).unlink(missing_ok=True)
                return False
            # reserve the slot now so a parallel worker can't add the same picture meanwhile
            self._dup.append((-1, ph, vec, source, src_id, num(src_start), num(length)))
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
               credit[:200], license[:120], page_url[:300], vec.astype(np.float32).tobytes(), time.time(), origin,
               str(ph))
        action = str(cap.get("action") or "").lower()
        if action not in ACTION_NAMES:
            action = detect_action(" ".join([str(cap.get("caption") or ""), " ".join(map(str, tags)), title]))
        row = row + (action, n_frames, int(num(src_height)), 1 if has_audio(out) else None)
        with _DB_LOCK:
            con = _db(self.settings)
            try:
                con.execute("""INSERT INTO shots(source, src_id, title, category, src_start, length, file, keyframe,
                    caption, entity, tags, shot_type, quality, credit, license, page_url, emb, added, origin, phash,
                    action, frames, src_height, audio)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
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
        backup_index(self.settings)
        self.upgrade_vectors()
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
        retry_later = []
        interrupted = {}

        def job(item):
            name, r = item
            if self.stop.is_set() or count[0] >= target:
                return
            if not self.gate.wait():            # paused: no next file starts until Retry / internet back
                return
            try:
                n = self.process_video(by_name[name], r)
            except Exception as e:
                if self.stop.is_set() or (is_network_error(e) and interrupted.get(r["id"], 0) < 3):
                    # a connection problem is not the video's fault: it goes back in the queue (3 times at most,
                    # so one broken video can never loop forever)
                    with lock:
                        retry_later.append(item)
                        interrupted[r["id"]] = interrupted.get(r["id"], 0) + 1
                    if not self.stop.is_set():
                        self.log(f"  {name}: video {r['id']} interrupted by the network - will be retried "
                                 f"({interrupted[r['id']]}/3)")
                    return
                self._mark(r["id"], name, "failed", note=str(e)[-200:])
                self.log(f"  {name}: video {r['id']} failed ({str(e)[-100:]})")
                return
            with lock:
                count[0] += n
                c_ = count[0]
            rate = (c_ - have) / max(time.time() - t0, 1) * 3600
            self.log(f"  {name}: +{n} shots ({c_}/{target}, ~{rate:.0f} shots/hour, {self.dupes} duplicates skipped)")
            self.status(shots=c_, target=target, progress=min(1.0, c_ / target))

        while count[0] < target and not self.stop.is_set() and any(queues.values()):
            batch = []
            for name in list(queues):
                if not queues[name] and self.pages.get(by_name[name][1], 1) <= 40:
                    try:
                        queues[name] = self.candidates(by_name[name][1])      # next result pages
                    except Exception:
                        queues[name] = []
                if queues[name]:
                    batch.append((name, queues[name].pop(0)))
            for i in range(0, len(batch), 4):
                if not self._disk_ok():
                    self.stop.set()
                    break
                if not self.gate.wait():
                    break
                with ThreadPoolExecutor(4) as ex:
                    list(ex.map(job, batch[i:i + 4]))
                with lock:
                    for name, r in retry_later:          # interrupted videos go first next round
                        queues.setdefault(name, []).insert(0, r)
                    retry_later.clear()
                if self.stop.is_set() or count[0] >= target:
                    break
        if not any(queues.values()) and count[0] < target:
            self.log("no more new DVIDS videos in these categories - add categories or lower the target")
        self.log(f"library build {'stopped' if self.stop.is_set() and count[0] < target else 'done'}: "
                 f"{count[0]} shots in {(time.time() - t0) / 60:.0f} min; {self.dupes} duplicate shots skipped")
        return count[0]

    # -- sound for shots cut before v0.9.0 (they were cut without their audio)
    def add_sound(self, workers=4):
        """Give every DVIDS shot its own sound back: the same seconds are cut, audio only, from the smallest
        rendition of its source video and muxed into the shot (the picture is not touched). Sources without sound
        are remembered (audio = 0). Resumable: only shots not done yet are visited."""
        from concurrent.futures import ThreadPoolExecutor
        with _DB_LOCK:
            con = _db(self.settings)
            rows = [dict(r) for r in con.execute(
                "SELECT id, source, src_id, src_start, length, file FROM shots WHERE audio IS NULL")]
            con.close()
        rows = [r for r in rows if Path(r["file"]).exists()]
        if not rows:
            self.log("sound: every shot already has its sound (or its source has none)")
            return 0
        local = 0                                  # shots cut by v0.9.0+ already carry their sound
        for r in rows:
            if has_audio(r["file"]):
                self._set_audio([r["id"]], 1)
                local += 1
        todo = [r for r in rows if r["source"] == "dvids" and not self._audio_known(r["id"])]
        other = [r["id"] for r in rows if r["source"] != "dvids" and not self._audio_known(r["id"])]
        if other:
            self._set_audio(other, 0)
        groups = {}
        for r in todo:
            groups.setdefault(r["src_id"], []).append(r)
        total = len(todo)
        self.log(f"sound: {local} shots already had sound; {total} shots from {len(groups)} DVIDS videos get theirs "
                 f"now (audio only, from the smallest file of each video)")
        done, silent, unver, t0 = [0], [0], [0], time.time()
        lock = threading.Lock()

        def one(item):
            src_id, shots = item
            if self.stop.is_set() or not self.gate.wait():
                return
            try:
                _, files = self._dvids_asset(src_id)
                proxy_url, full_url, _ = pick_files(files)
                urls = []
                for u in (proxy_url, full_url):          # small file first; the exact file the shot came from second
                    if u and u not in urls and with_net_retry(lambda u=u: has_audio(u), self.log, self.stop,
                                                              what=f"sound check {src_id}", gate=self.gate):
                        urls.append(u)
                if not urls:
                    self._set_audio([r["id"] for r in shots], 0)
                    with lock:
                        silent[0] += len(shots)
                        done[0] += len(shots)
                    return
                for r in shots:
                    if self.stop.is_set():
                        return
                    ok = self._mux_sound(r, urls, full_url)
                    with lock:
                        done[0] += 1
                        if not ok:
                            unver[0] += 1
            except Exception as e:
                if self.stop.is_set():
                    return
                self.log(f"  sound {src_id}: skipped this time ({str(e)[-120:]})")
                return
            with lock:
                d, sl = done[0], silent[0]
            rate = d / max(time.time() - t0, 1) * 3600
            self.log(f"  sound: {d}/{total} shots done (~{rate:.0f}/hour), {sl} from videos without sound, "
                     f"{unver[0]} left silent (picture did not match the source exactly)")
            self.status(shots=d, target=total, progress=min(1.0, d / max(total, 1)))

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, list(groups.items())))
        with _DB_LOCK:
            con = _db(self.settings)
            have = con.execute("SELECT COUNT(*) FROM shots WHERE audio = 1").fetchone()[0]
            none = con.execute("SELECT COUNT(*) FROM shots WHERE audio = 0").fetchone()[0]
            bad = con.execute("SELECT COUNT(*) FROM shots WHERE audio = -1").fetchone()[0]
            left = con.execute("SELECT COUNT(*) FROM shots WHERE audio IS NULL").fetchone()[0]
            con.close()
        self.log(f"sound {'stopped' if self.stop.is_set() else 'done'}: {have} shots have their own sound "
                 f"(every one checked frame by frame against its source), {none} come from videos without sound, "
                 f"{bad} left silent because the picture could not be matched"
                 + (f", {left} still to do (press Retry)" if left else ""))
        return done[0]

    def _audio_known(self, shot_id):
        with _DB_LOCK:
            con = _db(self.settings)
            v = con.execute("SELECT audio FROM shots WHERE id = ?", (shot_id,)).fetchone()
            con.close()
        return v is not None and v[0] is not None

    def _set_audio(self, ids, val):
        with _DB_LOCK:
            con = _db(self.settings)
            con.executemany("UPDATE shots SET audio = ? WHERE id = ?", [(val, i) for i in ids])
            con.commit()
            con.close()

    def _mux_sound(self, r, urls, exact=None):
        """Add the shot's own sound - only when it is PROVEN to belong to this picture (verify_alignment: same
        video AND same seconds to one frame; still shots only from the exact file they were cut from). One read
        of the source per file tried (audio + tiny grey frames of the same seconds +-0.5 s). No proof in any file
        -> the shot stays silent (audio = -1). Returns True when sound was added."""
        f = Path(r["file"])
        tmp = f.with_name(f.stem + ".sound.part.mp4")
        aud = f.with_name(f.stem + ".sound.part.m4a")
        try:
            mine = _grey_frames(f)
            for url in urls:
                try:
                    got = self._source_window(r, url)
                except Exception as e:
                    if is_network_error(e) or self.stop.is_set():
                        raise                              # not this shot's fault: tried again after Retry
                    got = None                             # this file cannot give these seconds: next file
                if got is None:
                    continue
                wav, frames, pre = got
                try:
                    v = verify_alignment(mine, frames, int(round(pre * ALIGN_FPS)), exact_file=url == exact)
                    if not v["ok"]:
                        continue
                    start = pre + v["offset"] / ALIGN_FPS
                    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.4f}", "-i", str(wav),
                         "-t", f"{num(r['length'])}", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                         str(aud)], timeout=120)
                finally:
                    wav.unlink(missing_ok=True)
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(f), "-i", str(aud), "-map", "0:v:0",
                     "-map", "1:a:0", "-c", "copy", "-shortest", "-movflags", "+faststart", str(tmp)], timeout=120)
                if abs(duration(tmp) - duration(f)) > 0.25 or not has_audio(tmp):
                    raise RuntimeError("muxed shot does not match the original length")
                tmp.replace(f)
                self._set_audio([r["id"]], 1)
                return True
            self._set_audio([r["id"]], -1)
            return False
        finally:
            aud.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)

    def _source_window(self, r, url):
        """The shot's seconds +-0.5 s from one source file, in ONE read: (wav, grey frames, seconds before)."""
        st, L = num(r["src_start"]), num(r["length"])
        pre = min(0.5, st)
        f = Path(r["file"])
        wav = f.with_name(f.stem + ".sound.part.wav")
        raw = f.with_name(f.stem + ".sound.part.raw")
        try:
            def read():
                run(["ffmpeg", "-y", "-loglevel", "error", *ffmpeg_net_args(url), "-ss", f"{st - pre:.4f}",
                     "-t", f"{L + pre + 0.5:.4f}", "-i", url,
                     "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(wav),
                     "-map", "0:v:0", "-an", "-vf", f"scale={ALIGN_W}:{ALIGN_H}:force_original_aspect_ratio=increase,"
                     f"crop={ALIGN_W}:{ALIGN_H},fps={ALIGN_FPS},format=gray", "-f", "rawvideo", str(raw)], timeout=300)
            with_net_retry(read, self.log, self.stop, what=f"sound {f.name}", gate=self.gate)
            frames = np.fromfile(raw, dtype=np.uint8)
            n = len(frames) // (ALIGN_W * ALIGN_H)
            if n < 3 or not wav.exists():
                wav.unlink(missing_ok=True)
                return None
            return wav, frames[:n * ALIGN_W * ALIGN_H].reshape(n, ALIGN_W * ALIGN_H), pre
        finally:
            raw.unlink(missing_ok=True)

    # -- learning from a finished video
    def learn_from_project(self, proj):
        """Every online clip of a finished video that passed the clean check joins the library (caption = the
        beat's visual line, quality = its Gemini score). Only from `library_learn_sources` (DVIDS by default) and
        at least `library_learn_min_score`. Duplicates are skipped. No extra API cost."""
        proj = Path(proj)
        picks = read_json(proj / "picks.json", {}) or {}
        fetched = read_json(proj / "fetched.json", {}) or {}
        specs = {str(s["id"]): s for s in (read_json(proj / "specs.json", []) or [])}
        allowed = set(self.settings.get("library_learn_sources") or ["dvids"])
        min_score = num(self.settings.get("library_learn_min_score"), 0)
        added = 0
        for bid, f in fetched.items():
            p = picks.get(bid) or {}
            if not f or f.get("source") not in allowed or f.get("problems"):
                continue
            if num(p.get("gemini_score"), 0) < min_score or not Path(f["file"]).exists() or num(f.get("length")) < 1.5:
                continue
            if num(f.get("src_height")) and num(f.get("src_height")) < MIN_SOURCE_HEIGHT:
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
                            page_url=f.get("page_url") or "", origin=f"video:{proj.name}", caption=cap,
                            src_height=num(f.get("src_height"))):
                added += 1
        backup_index(self.settings)
        return added

    def upgrade_vectors(self):
        """Shots added before 3-frame vectors existed get them now (once)."""
        with _DB_LOCK:
            con = _db(self.settings)
            rows = [dict(r) for r in con.execute("SELECT id, file, keyframe, length FROM shots WHERE COALESCE(frames,1) < 2")]
            con.close()
        done = 0
        for r in rows:
            if self.stop.is_set():
                break
            if not Path(r["file"]).exists():
                continue
            v, n = shot_vec(self.settings, r["file"], r["length"], r["keyframe"])
            if n < 2:
                continue
            with _DB_LOCK:
                con = _db(self.settings)
                con.execute("UPDATE shots SET emb = ?, frames = ? WHERE id = ?", (v.tobytes(), n, r["id"]))
                con.commit()
                con.close()
            done += 1
        if done:
            self.log(f"library: {done} older shots upgraded to 3-frame vectors")
        return done


# ---------------------------------------------------------------- backup
def backup_index(settings=None, keep=7, name=None):
    """One copy of index.db per day (sqlite backup API: consistent even while in use); the last `keep` days kept.
    `name` makes an extra copy (e.g. before an update) that the daily rotation leaves alone."""
    src = db_path(settings)
    if not src.exists():
        return None
    bdir = lib_dir(settings) / "backups"
    bdir.mkdir(exist_ok=True)
    dst = bdir / (name or f"index-{time.strftime('%Y%m%d')}.db")
    if dst.exists():
        return dst
    with _DB_LOCK:
        a, b = sqlite3.connect(src), sqlite3.connect(dst)
        a.backup(b)
        a.close()
        b.close()
    daily = [f for f in sorted(bdir.glob("index-2*.db"))]
    for old in daily[:-keep]:
        old.unlink()
    for old in sorted(bdir.glob("index-preupdate-*.db"))[:-keep]:
        old.unlink()
    return dst


def _shot_files(settings=None):
    return [f for f in lib_dir(settings).glob("*.mp4") if not f.name.endswith(".part.mp4")]


def _saved_index_copies(settings=None, root=None):
    """Every older copy of index.db we can find: git stashes (v0.8.0-v0.9.3 had index.db in git, so the Update
    button stashed the real one away) and the daily backups. Returns temp / backup file paths."""
    root = Path(root or ROOT)
    dbp = db_path(settings)
    out = []
    try:
        rel = dbp.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = None
    if rel and shutil.which("git") and (root / ".git").exists():
        r = subprocess.run(["git", "-C", str(root), "stash", "list", "--format=%gd"], capture_output=True, text=True)
        tmpd = Path(tempfile.mkdtemp(prefix="ws-index-"))
        for i, ref in enumerate(x for x in r.stdout.split() if x.strip()):
            b = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{rel}"], capture_output=True)
            if b.returncode == 0 and b.stdout[:16] == b"SQLite format 3\x00":
                f = tmpd / f"stash{i}.db"
                f.write_bytes(b.stdout)
                out.append((f"git {ref}", f))
    bdir = lib_dir(settings) / "backups"
    for f in sorted(bdir.glob("index-*.db")) if bdir.exists() else []:
        out.append((f"backup {f.name}", f))
    return out


def index_health(settings=None):
    """(rows whose shot file is on disk, shot files on disk)."""
    files = _shot_files(settings)
    with _DB_LOCK:
        con = _db(settings)
        rows = [r[0] for r in con.execute("SELECT file FROM shots")]
        con.close()
    return sum(1 for f in rows if Path(f).exists()), len(files)


def recover_index(settings=None, log=None, root=None, force=False):
    """The index lost its rows while the shot files are still on disk: merge every saved copy (git stashes, daily
    backups) back in, keeping only rows whose shot file still exists. Safe to run any time; returns shots added."""
    log = log or (lambda m: None)
    have, files = index_health(settings)
    if not files or (not force and have >= 0.9 * files):
        return 0
    copies = _saved_index_copies(settings, root)
    if not copies:
        log(f"library index: {have} of {files} shot files are indexed and no saved copy was found")
        return 0
    dbp = db_path(settings)
    keep = lib_dir(settings) / "backups"
    keep.mkdir(exist_ok=True)
    shutil.copy2(dbp, keep / f"index-before-recover-{time.strftime('%Y%m%d-%H%M%S')}.db")
    added, used = 0, []
    with _DB_LOCK:
        con = _db(settings)
        cols = {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})")] for t in ("shots", "videos", "deleted")}
        for name, f in copies:
            try:
                con.execute("ATTACH DATABASE ? AS c", (str(f),))
            except sqlite3.Error:
                continue
            try:
                ccols = {t: [r[1] for r in con.execute(f"PRAGMA c.table_info({t})")] for t in cols}
                common = [c for c in cols["shots"] if c in ccols["shots"] and c != "id"]
                n0 = con.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
                if "file" in common:
                    q = ",".join(common)
                    for row in con.execute(f"SELECT {q} FROM c.shots").fetchall():
                        r = dict(zip(common, row))
                        if Path(str(r.get("file") or "")).exists() and Path(str(r.get("keyframe") or "")).exists():
                            con.execute(f"INSERT OR IGNORE INTO shots({q}) VALUES({','.join('?' * len(common))})", row)
                vc = [c for c in cols["videos"] if c in ccols["videos"]]
                if vc:
                    con.execute(f"INSERT OR IGNORE INTO videos({','.join(vc)}) SELECT {','.join(vc)} FROM c.videos")
                dc = [c for c in cols["deleted"] if c in ccols["deleted"]]
                if dc and {"source", "src_id", "src_start"} <= set(dc):
                    con.execute(f"""INSERT INTO deleted({','.join(dc)}) SELECT {','.join('d.' + c for c in dc)}
                        FROM c.deleted d WHERE NOT EXISTS (SELECT 1 FROM deleted x WHERE x.source = d.source
                        AND x.src_id = d.src_id AND x.src_start = d.src_start)""")
                con.commit()
                got = con.execute("SELECT COUNT(*) FROM shots").fetchone()[0] - n0
                if got:
                    used.append(f"{name} (+{got})")
                    added += got
            finally:
                con.execute("DETACH DATABASE c")
        # rows of shots that are not on this disk (e.g. a copy made on another machine) are useless
        gone = [r[0] for r in con.execute("SELECT id, file FROM shots").fetchall() if not Path(r[1]).exists()]
        if gone and added:
            con.executemany("DELETE FROM shots WHERE id = ?", [(i,) for i in gone])
            con.commit()
        con.close()
    now, files = index_health(settings)
    log(f"library index recovered: {now} of {files} shot files indexed (was {have})"
        + (f" - from {', '.join(used)}" if used else ""))
    return added


# ---------------------------------------------------------------- browser (Library tab)
def list_shots(settings=None, q="", category="", favorites=False, sort="new", offset=0, limit=60):
    where, args = [], []
    for w in (q or "").lower().split():
        where.append("LOWER(COALESCE(caption,'')||' '||COALESCE(entity,'')||' '||COALESCE(tags,'')||' '||"
                     "COALESCE(title,'')||' '||COALESCE(category,'')||' '||COALESCE(action,'')) LIKE ?")
        args.append(f"%{w}%")
    if category:
        where.append("category = ?")
        args.append(category)
    if favorites:
        where.append("favorite = 1")
    order = {"new": "added DESC", "uses": "uses DESC, added DESC", "unused": "uses ASC, added DESC",
             "quality": "quality DESC, added DESC"}.get(sort, "added DESC")
    sql = ("SELECT id, caption, entity, action, category, length, uses, last_used, favorite, quality, origin, title, "
           "source, src_id FROM shots" + (" WHERE " + " AND ".join(where) if where else "") +
           f" ORDER BY {order} LIMIT ? OFFSET ?")
    with _DB_LOCK:
        con = _db(settings)
        rows = [dict(r) for r in con.execute(sql, args + [int(limit), int(offset)])]
        total = con.execute("SELECT COUNT(*) FROM shots" + (" WHERE " + " AND ".join(where) if where else ""),
                            args).fetchone()[0]
        con.close()
    return {"total": total, "shots": rows}


def shot_row(settings, shot_id):
    with _DB_LOCK:
        con = _db(settings)
        r = con.execute("SELECT * FROM shots WHERE id = ?", (int(shot_id),)).fetchone()
        con.close()
    return dict(r) if r else None


def set_favorite(settings, shot_id, on=True):
    with _DB_LOCK:
        con = _db(settings)
        con.execute("UPDATE shots SET favorite = ? WHERE id = ?", (1 if on else 0, int(shot_id)))
        con.commit()
        con.close()


def delete_shot(settings, shot_id):
    """Removes the files and the entry, and remembers the picture so the builder never adds it again."""
    r = shot_row(settings, shot_id)
    if not r:
        return False
    with _DB_LOCK:
        con = _db(settings)
        con.execute("INSERT INTO deleted VALUES(?,?,?,?,?,?,?)", (r.get("phash"), r["source"], r["src_id"],
                    r["src_start"], r["length"], r["emb"], time.time()))
        con.execute("DELETE FROM shots WHERE id = ?", (int(shot_id),))
        con.commit()
        con.close()
    for f in (r.get("file"), r.get("keyframe")):
        if f:
            Path(f).unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------- background job (one at a time)
_JOB = {"thread": None, "stop": threading.Event(), "gate": None}


def status_log(msg):
    """Write one line into the Library tab's log (used outside a build, e.g. at app start)."""
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    st = read_json(STATUS_FILE, {}) or {}
    st["log"] = (st.get("log") or [])[-299:] + [line]
    write_json(STATUS_FILE, st)


def job_status():
    st = read_json(STATUS_FILE, {}) or {}
    st["running"] = bool(_JOB["thread"] and _JOB["thread"].is_alive())
    return st


def _start_job(params, work, settings_extra=None):
    """Run one library job in the background (build or add-sound): status file, log, keep-awake, pause gate."""
    from .config import load_keys, load_settings
    if _JOB["thread"] and _JOB["thread"].is_alive():
        raise RuntimeError("a library job is already running")
    settings = dict(load_settings(), **(settings_extra or {}))
    settings["library_auto_continue"] = bool(params.get("auto_continue", True))
    st = {"state": "running", "started": time.time(), "target": int(params.get("target") or 0), "log": [],
          "progress": 0, "net": "", "kind": params.get("kind", "build"), "params": params}
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
        awake = keep_awake(log)
        try:
            b = Builder(load_keys(), settings, log, _JOB["stop"], status)
            _JOB["gate"] = b.gate
            finished = work(b)
            status(state="stopped" if _JOB["stop"].is_set() and not finished else "done", finished=time.time(), net="")
        except Exception as e:
            import traceback
            log(f"ERROR: {e}")
            log(traceback.format_exc()[-1200:])
            status(state="error", error=str(e)[:400], finished=time.time(), net="")
        finally:
            _JOB["gate"] = None
            if awake:
                awake.terminate()

    _JOB["thread"] = threading.Thread(target=run, daemon=True)
    _JOB["thread"].start()
    return st


def start_build(target, categories=None, per_video=None, max_gb=None, auto_continue=True):
    extra = {}
    if per_video:
        extra["library_shots_per_video"] = int(per_video)
    if max_gb:
        extra["library_max_gb"] = float(max_gb)
    params = {"kind": "build", "target": int(target), "categories": categories, "per_video": per_video,
              "max_gb": max_gb, "auto_continue": bool(auto_continue)}

    def work(b):
        n = b.build(int(target), categories)
        b.status(shots=n)
        return n >= int(target)
    return _start_job(params, work, extra)


def start_sound(auto_continue=True):
    """Give the shots cut before v0.9.0 their own sound (audio only, from DVIDS)."""
    def work(b):
        b.add_sound()
        with _DB_LOCK:
            con = _db(b.settings)
            left = con.execute("SELECT COUNT(*) FROM shots WHERE audio IS NULL").fetchone()[0]
            con.close()
        return left == 0
    return _start_job({"kind": "sound", "auto_continue": bool(auto_continue)}, work)


def _restart_from(p, st):
    if p.get("kind") == "sound":
        return start_sound(p.get("auto_continue", True))
    return start_build(p.get("target") or st.get("target") or 1000, p.get("categories"), p.get("per_video"),
                       p.get("max_gb"), p.get("auto_continue", True))


def stop_build():
    _JOB["stop"].set()
    return job_status()


def retry_build():
    """Retry button. A paused build (internet / DNS problem) continues from the same file. A build that ended
    early (stopped, error, app closed) starts again with the same choices: shots already in the library stay,
    videos already processed are skipped, so it carries on where it stopped."""
    if _JOB["thread"] and _JOB["thread"].is_alive():
        g = _JOB.get("gate")
        if g is not None and g.retry():
            return {"action": "resumed", "job": job_status()}
        return {"action": "running", "job": job_status()}
    st = read_json(STATUS_FILE, {}) or {}
    p = st.get("params") or {}
    if st.get("state") in ("stopped", "error", "running") and (p.get("kind") == "sound" or p.get("target")
                                                               or st.get("target")):
        _restart_from(p, st)
        return {"action": "restarted", "job": job_status()}
    return {"action": "nothing", "job": job_status()}


def set_auto_continue(on):
    """Flip auto-continue on a running build (the checkbox in the Library tab)."""
    g = _JOB.get("gate")
    if g is not None:
        g.auto = bool(on)
    st = read_json(STATUS_FILE, {}) or {}
    if st.get("params") is not None:
        st["params"]["auto_continue"] = bool(on)
        st["net_auto"] = bool(on)
        write_json(STATUS_FILE, st)
    return job_status()


def keep_awake(log=None):
    """macOS: stop the Mac from going to sleep while the library builds (caffeinate, ended with the build).
    The lid must stay open on battery-only Macs - a closed lid still sleeps."""
    import sys
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    try:
        p = subprocess.Popen(["caffeinate", "-ims"])
        if log:
            log("Mac sleep is blocked while the library builds (keep the lid open and the charger in)")
        return p
    except Exception:
        return None


def resume_if_interrupted(settings=None):
    """The app (or the Mac) stopped in the middle of a build: start it again with the same choices."""
    st = read_json(STATUS_FILE, {}) or {}
    if st.get("state") != "running" or (_JOB["thread"] and _JOB["thread"].is_alive()):
        return False
    if not (settings or {}).get("library_auto_resume", True):
        return False
    p = st.get("params") or {}
    try:
        _restart_from(p, st)
        return True
    except Exception:
        return False
