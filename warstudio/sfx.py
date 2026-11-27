"""Sound design v2 (v0.9.0), modelled on how Beyond Military uses sound. Everything is local: no API cost.

WHAT PLAYS WHERE (plan_events)
  * SFX follow what is ON SCREEN: the chosen shot's action (library action tag, else the beat's visual line)
      takeoff / landing -> jet_roar      flying / formation -> flyby        hover / helicopters -> rotor
      launch / rocket artillery -> launch                  explosion -> explosion + bass_hit
      small arms -> gunfire   big guns -> impact + bass_hit   console / radar screen -> radar beep
      flares or chaff in the shot -> flares                drones (MQ-9 ...) -> nothing (they are quiet)
  * UI blip when an on-screen label / timestamp / number box pops in
  * map beats: whoosh into the map + a low bass hit on the zoom
  * hard cuts: whoosh / metal hit on most cuts in the hook (first 60 s), after that only on scene changes
  * radio static on comms words (wingman, radio, tower, call sign ...), radar beep on lock / warning words
  * moments marked by the brain: attack / reveal -> riser that ENDS exactly on the beat + bass hit;
    cliffhanger -> sustained dark chord
  * density: hook up to 16 SFX moments a minute, body about 11; number / explanation beats only get the UI blip
MUSIC (mix -> _mix_music)
  one continuous bed: your tracks in assets/music (chained with crossfades, so a 30-min video does not loop one
  3-min track), else a generated tension bed. Shaped by an intensity curve per beat: calm = quieter + darker,
  action = louder (+ pulse layer on the generated bed); a 2 s swell into every attack / reveal; a dip under
  cliffhanger chords; ducked under the voice (-20 dB) and lifted in pauses (-14 dB); fade-out at the end.
FOOTAGE AUDIO (mix -> _mix_footage)
  every shot keeps its own sound, low and ducked under the voice: action shots (engines, guns, blasts) a little
  up, all other shots as quiet ambience (well under the narrator, a bit up in pauses). Clips that sound like
  speech, or come from packages / interviews / briefings, are skipped - the voiceover always stays clear.

Your own SFX files win: drop .wav/.mp3 into assets/sfx/<category>/ (e.g. assets/sfx/explosion/).
"""
import bisect
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .config import SFX_DIR, MUSIC_DIR

SR = 48000
CTRL = 100                      # control rate (Hz) of the music / ducking curves
CATS = ["ui_click", "whoosh", "metal_hit", "bass_hit", "riser", "drone_chord", "launch", "explosion", "flyby",
        "jet_roar", "rotor", "gunfire", "impact", "flares", "radar", "radio_static", "alarm"]
AUDIO_EXT = (".wav", ".mp3", ".aif", ".aiff", ".flac", ".m4a")
GEN_VERSION = "2"               # bump when the built-in sounds change: old generated files are rebuilt
BED_LOOP = 32.0                 # generated music bed loop length (s); every partial fits a whole number of cycles


# ---------------------------------------------------------------- synthesis

def _t(sec):
    return np.arange(int(sec * SR)) / SR


def _lp(x, k):
    """Moving-average low-pass, same alignment as np.convolve(x, ones(k)/k, 'same'), O(n) for any k."""
    x = np.asarray(x, dtype=np.float64)
    k = max(1, int(k))
    if k == 1 or not len(x):
        return x.copy()
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.clip(i + (k - 1) // 2 - k + 1, 0, len(x))
    hi = np.clip(i + (k - 1) // 2 + 1, 0, len(x))
    return (c[hi] - c[lo]) / k


def _env(n, a, d):
    e = np.ones(n)
    na, nd = int(a * SR), int(d * SR)
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nd:
        e[-nd:] *= np.linspace(1, 0, nd) ** 2
    return e


def _norm(x, peak=0.9):
    m = np.max(np.abs(x)) or 1
    return (x / m * peak).astype(np.float32)


def _burst(x, s, b):
    s = int(s)
    if s < len(x):
        x[s:s + len(b)] += b[:len(x) - s]


def synth(cat, seed=0):
    rng = np.random.default_rng(seed)
    noise = lambda s: rng.standard_normal(int(s * SR))
    if cat == "ui_click":                                   # two-tone data blip
        x = np.zeros(int(0.1 * SR))
        for f, st in ((2400 + 200 * seed, 0.0), (1800 + 150 * seed, 0.045)):
            tt = _t(0.04)
            _burst(x, st * SR, np.sin(2 * np.pi * f * tt) * np.exp(-tt * 90))
    elif cat == "whoosh":
        n = noise(0.9)
        x = np.concatenate([_lp(n[i:i + 4800], 30 - i // 1800) for i in range(0, len(n), 4800)])[:len(n)]
        x *= np.sin(np.linspace(0, np.pi, len(x))) ** 2
    elif cat == "metal_hit":                                # inharmonic clang + thud
        t = _t(1.4)
        parts = [(180, 1.0, 4), (523, 0.6, 6), (1307, 0.4, 9), (2089, 0.3, 12), (3011, 0.2, 16)]
        x = sum(a * np.sin(2 * np.pi * f * (1 + 0.01 * seed) * t) * np.exp(-t * d) for f, a, d in parts)
        x = x + noise(1.4) * np.exp(-t * 60) * 0.8
    elif cat == "bass_hit":
        t = _t(1.8)
        f = 55 * np.exp(-t * 1.5) + 32
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2) + _lp(noise(1.8), 40) * np.exp(-t * 18) * 0.6
    elif cat == "riser":
        L = 2.0
        t = _t(L)
        x = _lp(noise(L), 12) * (t / L) ** 2 + np.sin(2 * np.pi * (200 + 600 * (t / L) ** 2) * t) * (t / L) ** 3 * 0.4
    elif cat == "drone_chord":                              # sustained dark chord for cliffhangers
        t = _t(6.0)
        fs = [55.0, 65.41, 82.41, 110.0, 130.81]
        x = sum(np.sin(2 * np.pi * f * (1 + 0.002 * seed) * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t + 0.3) for f in fs)
        x = x * (1 + 0.15 * np.sin(2 * np.pi * 0.5 * t)) * _env(len(t), 1.2, 2.5)
    elif cat == "launch":
        t = _t(2.6)
        x = _lp(noise(2.6), 25) * _env(len(t), 0.03, 2.2) + _lp(noise(2.6), 200) * np.exp(-t * 1.2) * 2
    elif cat == "explosion":
        t = _t(3.2)
        x = _lp(noise(3.2), 120) * np.exp(-t * 1.6) * 3 + _lp(noise(3.2), 15) * np.exp(-t * 6) + np.sin(2 * np.pi * 40 * t) * np.exp(-t * 2)
    elif cat == "flyby":
        t = _t(3.0)
        x = _lp(noise(3.0), 18) * np.exp(-((t - 1.4) ** 2) / 0.35)
    elif cat == "jet_roar":                                 # afterburner: hiss + rumble + faint whine
        t = _t(3.5)
        body = _lp(noise(3.5), 6) * 0.6 + _lp(noise(3.5), 50) * 1.4
        whine = np.sin(2 * np.pi * (1900 + 500 * (t / 3.5)) * t) * 0.05
        x = (body + whine) * _env(len(t), 0.5, 1.4)
    elif cat == "rotor":                                    # helicopter blade slap
        t = _t(3.0)
        x = np.zeros(len(t))
        rate = 4.6 + 0.3 * seed
        for k in range(int(3.0 * rate)):
            _burst(x, k / rate * SR, _lp(noise(0.08), 25) * np.exp(-_t(0.08) * 45))
        x += np.sin(2 * np.pi * 28 * t) * 0.3 * (0.6 + 0.4 * np.sin(2 * np.pi * rate * t))
        x *= _env(len(t), 0.3, 0.8)
    elif cat == "gunfire":
        x = np.zeros(int(1.2 * SR))
        for k in range(rng.integers(5, 9)):
            b = noise(0.08) * np.exp(-_t(0.08) * 60)
            _burst(x, k * 0.11 * SR + rng.integers(0, 600), b)
    elif cat == "impact":
        t = _t(0.8)
        x = np.sin(2 * np.pi * 420 * t) * np.exp(-t * 9) + noise(0.8) * np.exp(-t * 30)
    elif cat == "flares":
        x = np.zeros(int(1.6 * SR))
        for k in range(6):
            _burst(x, k * 0.22 * SR, _lp(noise(0.25), 6) * np.exp(-_t(0.25) * 20))
    elif cat == "radar":
        x = np.zeros(int(0.9 * SR))
        for k in range(3):
            _burst(x, k * 0.28 * SR, np.sin(2 * np.pi * (1000 + 200 * seed) * _t(0.12)) * _env(int(0.12 * SR), 0.005, 0.03))
    elif cat == "radio_static":
        t = _t(0.7)
        x = _lp(noise(0.7), 3) * 0.5 * _env(len(t), 0.01, 0.1)
        x[:400] += np.sin(2 * np.pi * 2500 * t[:400])
    elif cat == "alarm":
        t = _t(1.6)
        x = np.sign(np.sin(2 * np.pi * np.where((t * 2.5).astype(int) % 2 == 0, 700, 900) * t)) * 0.4
    else:
        raise ValueError(cat)
    return _norm(x)


def synth_bed():
    """Generated tension bed, one 32 s loop: (full, dark, pulse). Loops without a click."""
    t = np.arange(int(BED_LOOP * SR)) / SR

    def saw(f, nh):
        return sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, nh + 1))

    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.125 * t)
    shimmer = np.sin(2 * np.pi * 220 * t) * 0.06 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t))
    full = (saw(55, 6) * 0.5 + saw(55.25, 6) * 0.5 + saw(82.5, 4) * 0.3 + saw(110, 4) * 0.2) * lfo + shimmer
    dark = (saw(55, 2) * 0.5 + saw(55.25, 2) * 0.5 + saw(82.5, 1) * 0.3) * lfo
    pulse = np.zeros(len(t))
    rng = np.random.default_rng(3)
    tt = _t(0.3)
    for k in range(int(BED_LOOP)):                          # 60 bpm heartbeat + soft off-beat tick
        _burst(pulse, k * SR, np.sin(2 * np.pi * 48 * tt) * np.exp(-tt * 14))
        n = rng.standard_normal(int(0.03 * SR))
        _burst(pulse, (k + 0.5) * SR, (n - _lp(n, 8)) * np.exp(-_t(0.03) * 120) * 0.25)
    return _norm(full, 0.7), _norm(dark, 0.7), _norm(pulse, 0.7)


def write_wav(path, x, sr=SR):
    import wave
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def ensure_defaults():
    gen = SFX_DIR / "_generated"
    ver = gen / "VERSION"
    if gen.exists() and (not ver.exists() or ver.read_text().strip() != GEN_VERSION):
        shutil.rmtree(gen, ignore_errors=True)              # built-in sounds changed: rebuild them
    for c in CATS:
        for v in range(3):
            p = gen / c / f"{c}_{v}.wav"
            if not p.exists():
                write_wav(p, synth(c, v))
    bed = gen / "_bed"
    if not (bed / "pulse.wav").exists():
        for name, x in zip(("full", "dark", "pulse"), synth_bed()):
            write_wav(bed / f"{name}.wav", x)
    ver.parent.mkdir(parents=True, exist_ok=True)
    ver.write_text(GEN_VERSION)


def sfx_files(cat):
    user = sorted(p for p in (SFX_DIR / cat).glob("*") if p.suffix.lower() in AUDIO_EXT) if (SFX_DIR / cat).exists() else []
    return user or sorted((SFX_DIR / "_generated" / cat).glob("*.wav"))


def music_tracks(music_file=""):
    if music_file and Path(music_file).exists():
        return [Path(music_file)]
    return sorted(p for p in MUSIC_DIR.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT) if MUSIC_DIR.exists() else []


# ---------------------------------------------------------------- decode helpers

def load_audio(path, sr=SR, af=None, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if dur:
        cmd += ["-t", f"{float(dur):.3f}"]
    if af:
        cmd += ["-af", af]
    raw = subprocess.run(cmd + ["-f", "f32le", "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def db(x):
    return 10 ** (np.asarray(x) / 20) if not np.isscalar(x) else 10 ** (x / 20)


def _add(out, x, s):
    s = int(s)
    if s < 0:
        x, s = x[-s:], 0
    e = min(len(out), s + len(x))
    if e > s:
        out[s:e] += x[:e - s].astype(np.float32)


def _smooth(x, k):
    return _lp(x, k) if k > 1 else np.asarray(x, dtype=np.float64)


def _hold(x, k):
    """Centered running maximum (keeps the duck down through the short gaps between words)."""
    if k <= 1 or len(x) == 0:
        return x
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return sliding_window_view(p, k).max(axis=1)


def talk_curve(vo, n):
    """How much the narrator is talking, 0..1, at CTRL rate, for `n` output samples."""
    hop = SR // CTRL
    m = int(np.ceil(n / hop))
    rms = np.zeros(m)
    full = min(len(vo), m * hop) // hop
    for c in range(0, full, 100000):                     # block RMS without a full-length float64 copy
        e = min(full, c + 100000)
        blk = vo[c * hop:e * hop].reshape(e - c, hop)
        rms[c:e] = np.sqrt(np.einsum("ij,ij->i", blk, blk, dtype=np.float64) / hop)
    live = rms[rms > 1e-5]
    ref = np.percentile(live, 90) if len(live) else 1.0
    act = np.clip(rms / (ref * 0.25 + 1e-9), 0, 1)
    return np.clip(_smooth(_hold(act, int(0.35 * CTRL)), int(0.15 * CTRL)), 0, 1)


def speech_like(x):
    """True when audio sounds like talking: loudness in the voice band goes up and down at syllable rate
    (3-7 Hz) with a wide spread. Engines are steady, blasts are one transient, rotors are one sharp periodic
    peak, gunfire is faster than syllables."""
    if len(x) < SR:
        return False
    band = _lp(x, 14) - _lp(x, 160)                      # ~300 Hz - 3.4 kHz
    hop = SR // CTRL
    m = len(band) // hop
    env = np.sqrt((band[:m * hop].reshape(m, hop) ** 2).mean(axis=1))
    if env.mean() < 1e-6:
        return False
    e = env / env.mean() - 1
    depth = float(e.std())
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    f = np.fft.rfftfreq(len(e), 1 / CTRL)
    syl = spec[(f >= 3) & (f <= 7)]
    tot = spec[(f >= 0.5) & (f <= 20)].sum() + 1e-12
    if not len(syl) or syl.sum() <= 0:
        return False
    periodic = syl.max() / syl.sum() > 0.6 and len(syl) >= 4     # one machine rate, not speech
    return depth > 0.45 and syl.sum() / tot > 0.3 and not periodic


# ---------------------------------------------------------------- what is on screen

HELI = ("helicopter", "black hawk", "blackhawk", "apache", "chinook", "osprey", "seahawk", "sea hawk", "rotor",
        "huey", "venom", "mi-8", "mi-24", "mi-28", "ka-52", "cobra")
DRONE = ("drone", "reaper", "predator", "mq-", "global hawk", "uav", "shahed", "bayraktar", "loitering")
BIG_GUN = ("howitzer", "artillery", "cannon", "tank", "mortar", "naval gun", "main gun", "paladin", "m777")
ROCKET = ("himars", "mlrs", "rocket", "grad", "multiple launch")
NUMBERS = re.compile(r"\d|\b(percent|million|billion|thousand|hundred|dollars|miles|kilometers|kilometres|km|"
                     r"knots|tons|tonnes|feet|meters|metres)\b")


def shot_action(spec, pick=None):
    """What happens in the shot: the library tag of the chosen shot (made from its frames), else the beat's
    visual line."""
    from .library import detect_action
    if pick and pick.get("source") == "library" and pick.get("action"):
        return pick["action"]
    return detect_action(spec.get("visual") or "")


def shot_text(spec, pick=None):
    t = f"{spec.get('visual') or ''} {spec.get('entity') or ''}"
    if pick and pick.get("source") == "library":
        t += " " + str(pick.get("title") or "")
    return t.lower()


def sounds_for(action, text):
    """SFX for what is on screen: [(category, seconds after the cut)]."""
    t = (text or "").lower()
    heli, drone = any(w in t for w in HELI), any(w in t for w in DRONE)
    out = []
    if action in ("takeoff", "landing"):
        out = [("rotor", 0.0)] if heli else [] if drone else [("jet_roar", 0.0)]
    elif action in ("flying", "formation"):
        out = [("rotor", 0.0)] if heli else [] if drone else [("flyby", 0.0)]
    elif action == "hover":
        out = [("rotor", 0.0)]
    elif action == "launch":
        out = [("launch", 0.05)]
    elif action == "explosion":
        out = [("explosion", 0.05), ("bass_hit", 0.0)]
    elif action == "firing":
        if any(w in t for w in ROCKET):
            out = [("launch", 0.05)]
        elif any(w in t for w in BIG_GUN):
            out = [("impact", 0.05), ("bass_hit", 0.0)]
        else:
            out = [("gunfire", 0.05)]
    elif action == "console":
        out = [("radar", 0.15)]
    if ("flare" in t or "chaff" in t) and not any(c == "flares" for c, _ in out):
        out.append(("flares", 0.1))
    return out


def _ov_map(overlays):
    if isinstance(overlays, dict):
        return {int(k): (v or {}) for k, v in overlays.items()}
    return {int(k): {} for k in (overlays or [])}


def is_quiet(beat, spec, ov=None):
    """Numbers / explanation beats: no action SFX, no cut accents, music dips."""
    if spec.get("moment") == "attack":
        return False
    if (ov or {}).get("kind") == "measure":
        return True
    it = spec.get("intensity")
    if it in (1, 2, 3):
        return it == 1
    return bool(NUMBERS.search((beat.get("text") or "").lower()))


def beat_level(beat, spec, action, quiet, hook_s):
    """Music intensity of a beat: 1 calm, 2 tension, 3 action."""
    if spec.get("moment") == "attack":
        lvl = 3
    elif quiet:
        lvl = 1
    elif spec.get("intensity") in (1, 2, 3):
        lvl = spec["intensity"]
    else:
        lvl = 3 if action in ("explosion", "launch", "firing") else 2
    return max(lvl, 2) if beat["start"] < hook_s else lvl


def _words(text, words):
    return any(re.search(r"\b" + re.escape(w), text) for w in words)


# ---------------------------------------------------------------- planning

def plan_events(beats, specs, overlays, style, picks=None):
    """Decide which SFX play where. Returns a list of {t, cat, gain_db, rule, bid[, hit, align]}.
    `rule` says why (action:<name> / overlay / map / cut / comms / warning / moment:<name>)."""
    sc = style["sfx"]
    G = sc["gains_db"]
    spec = {s["id"]: s for s in specs}
    ov = _ov_map(overlays)
    picks = picks or {}
    hook = float(sc.get("hook_seconds", 60))
    cand = []

    def c(t, cat, prio, rule, bid, off=0.0, **kw):
        cand.append(dict(t=round(max(0.0, t), 3), cat=cat, gain_db=G.get(cat, -12) + off, prio=prio, rule=rule,
                         bid=bid, **kw))

    prev_scene, cut_n = None, 0
    for i, b in enumerate(beats):
        bid, t0 = b["id"], float(b["start"])
        s = spec.get(bid, {})
        pick = picks.get(str(bid))
        text = (b.get("text") or "").lower()
        act = shot_action(s, pick)
        moment = "" if b.get("cont") else (s.get("moment") or "")
        quiet = is_quiet(b, s, ov.get(bid))
        in_hook = t0 < hook
        if bid in ov:
            c(t0 + 0.05, "ui_click", 0, "overlay", bid)
        if moment in ("attack", "reveal"):
            c(t0 - 2.0, "riser", 0, f"moment:{moment}", bid, hit=round(t0, 3), align="end")
            c(t0, "bass_hit", 0, f"moment:{moment}", bid)
        elif moment == "cliffhanger":
            c(t0 + 0.1, "drone_chord", 0, "moment:cliffhanger", bid)
        if s.get("footage_class") == "map":
            c(t0 - 0.45, "whoosh", 1, "map", bid)
            c(t0, "bass_hit", 1, "map", bid, off=-4)
        if not quiet:
            for cat, dt in sounds_for(act, shot_text(s, pick)):
                c(t0 + dt, cat, 1, f"action:{act or cat}", bid)
            if _words(text, sc.get("comms_words", [])):
                c(t0 + 0.1, "radio_static", 2, "comms", bid)
            if _words(text, sc.get("warning_words", [])):
                c(t0 + 0.15, "radar", 2, "warning", bid)
        scene = s.get("scene") or ""
        new_scene = i > 0 and bool(scene) and scene != prev_scene
        if i > 0 and not quiet and (in_hook or new_scene):
            cut_n += 1
            cat = "metal_hit" if (new_scene and (not in_hook or cut_n % 3 == 0)) else "whoosh"
            c(t0 - (0.45 if cat == "whoosh" else 0.0), cat, 3, "cut", bid, off=float(sc.get("cut_offset_db", -2)))
        prev_scene = scene or prev_scene
    return _thin(cand, sc)


def _thin(cand, sc):
    """Keep the important sounds first, then fill up to the density budget."""
    cool = sc.get("cooldown_seconds", {})
    cap = sc.get("max_per_minute", {"hook": 16, "body": 11})
    gap = sc.get("min_gap_seconds", {"hook": 1.0, "body": 2.5})
    hook = float(sc.get("hook_seconds", 60))
    kept, times, slots = [], {}, []           # slots: (t, bid) of every kept non-click sound
    slot_t = []                               # sorted time of each beat that already has a sound
    for e in sorted(cand, key=lambda e: (e["prio"], e["t"])):
        t, cat = e["t"], e["cat"]
        zone = "hook" if t < hook else "body"
        cd = float(cool.get(cat, cool.get("default", 3.0)))
        if any(abs(t - u) < cd for u in times.get(cat, [])):
            continue
        if cat != "ui_click" and e["prio"] > 0:
            others = [(u, b) for u, b in slots if b != e["bid"]]
            if e["prio"] >= 2 and any(abs(t - u) < float(gap[zone]) for u, _ in others):
                continue
            if e["rule"] == "cut" and e["bid"] in {b for _, b in slots}:
                continue                          # the cut already has its own sound
            if e["bid"] not in {b for _, b in slots} and _busiest_minute(slot_t, t) >= int(cap[zone]):
                continue
        kept.append(e)
        times.setdefault(cat, []).append(t)
        if cat != "ui_click":
            if e["bid"] not in {b for _, b in slots}:
                bisect.insort(slot_t, t)
            slots.append((t, e["bid"]))
    out = sorted(kept, key=lambda e: e["t"])
    for e in out:
        e.pop("prio", None)
    return out


def needs_replan(events):
    """Events saved by the old keyword / pivot-word rules carry no `rule`: plan those projects again."""
    return bool(events) and not any(isinstance(e, dict) and e.get("rule") for e in events)


def _busiest_minute(ts, t):
    """Most sound moments in any 60 s window that contains t (ts sorted)."""
    best = bisect.bisect_left(ts, t + 60) - bisect.bisect_left(ts, t)
    for u in ts[bisect.bisect_left(ts, t - 60 + 1e-9):bisect.bisect_right(ts, t)]:
        best = max(best, bisect.bisect_left(ts, u + 60) - bisect.bisect_left(ts, u))
    return best


def replan_beat(events, beats, specs, style, picks, bid):
    """After a clip swap: drop that beat's on-screen SFX and add the ones for the new clip."""
    sc = style["sfx"]
    bid = int(bid)
    b = next((x for x in beats if int(x["id"]) == bid), None)
    keep = [e for e in events if not (int(e.get("bid", -1)) == bid and str(e.get("rule", "")).startswith("action:"))]
    if b is None:
        return keep
    s = next((x for x in specs if int(x["id"]) == bid), {})
    pick = (picks or {}).get(str(bid))
    if not is_quiet(b, s):
        act = shot_action(s, pick)
        for cat, dt in sounds_for(act, shot_text(s, pick)):
            keep.append({"t": round(float(b["start"]) + dt, 3), "cat": cat, "gain_db": sc["gains_db"].get(cat, -12),
                         "rule": f"action:{act or cat}", "bid": bid})
    return sorted(keep, key=lambda e: e["t"])


def plan_music(beats, specs, overlays, style, picks=None):
    """Music intensity: [[start_seconds, level 1|2|3], ...]. Levels follow the story, not every cut: after an
    attack the music stays up for `attack_hold_seconds`, and a level shorter than `min_hold_seconds` is folded
    into its neighbour, so the bed never pumps from shot to shot."""
    spec = {s["id"]: s for s in specs}
    ov = _ov_map(overlays)
    picks = picks or {}
    mc = style.get("music", {})
    hook = float(style["sfx"].get("hook_seconds", 60))
    hold, a_hold = float(mc.get("min_hold_seconds", 6)), float(mc.get("attack_hold_seconds", 8))
    if not beats:
        return []
    lv = []
    for b in beats:
        s = spec.get(b["id"], {})
        lv.append(beat_level(b, s, shot_action(s, picks.get(str(b["id"]))), is_quiet(b, s, ov.get(b["id"])), hook))
    for i, b in enumerate(beats):
        if spec.get(b["id"], {}).get("moment") == "attack" and not b.get("cont"):
            for j in range(i, len(beats)):
                if beats[j]["start"] >= b["start"] + a_hold:
                    break
                lv[j] = 3
    segs = []                                      # [start, end, level]
    for i, b in enumerate(beats):
        end = float(beats[i + 1]["start"]) if i + 1 < len(beats) else float(b.get("end", b["start"]))
        if segs and segs[-1][2] == lv[i]:
            segs[-1][1] = end
        else:
            segs.append([float(b["start"]), end, lv[i]])
    while len(segs) > 1:
        k = min(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
        if segs[k][1] - segs[k][0] >= hold:
            break
        left = segs[k - 1] if k > 0 else None
        right = segs[k + 1] if k + 1 < len(segs) else None
        into = left if right is None or (left is not None and left[1] - left[0] >= right[1] - right[0]) else right
        into[0], into[1] = min(into[0], segs[k][0]), max(into[1], segs[k][1])
        segs.pop(k)
        merged = [segs[0]]
        for sg in segs[1:]:
            if sg[2] == merged[-1][2]:
                merged[-1][1] = sg[1]
            else:
                merged.append(sg)
        segs = merged
    out = [[round(sg[0], 3), sg[2]] for sg in segs]
    out[0][0] = 0.0
    return out


def plan_footage_audio(beats, specs, picks, fetched, style, total, settings=None):
    """Every shot keeps its own sound, low: action shots (engines, guns, blasts) a little up, every other shot as
    quiet ambience. Packages / interviews / briefings are left out (talking), numbers beats stay ambient."""
    fa = style.get("footage_audio", {})
    on = (settings or {}).get("footage_audio", fa.get("enabled", True))
    if str(on).lower() in ("false", "0", "no", "off"):
        return []
    every = str(fa.get("all_shots", True)).lower() not in ("false", "0", "no", "off")
    spec = {s["id"]: s for s in specs}
    actions = set(fa.get("actions", []))
    skip = [w.lower() for w in fa.get("skip_title_words", [])]
    out = []
    for i, b in enumerate(beats):
        f = (fetched or {}).get(str(b["id"]))
        if not f or not f.get("file") or not Path(f["file"]).exists() or f.get("source") == "user":
            continue
        s = spec.get(b["id"], {})
        pick = (picks or {}).get(str(b["id"]))
        kind = "action" if shot_action(s, pick) in actions and not is_quiet(b, s) else "ambient"
        if kind == "ambient" and not every:
            continue
        title = f"{f.get('title') or ''} {(pick or {}).get('title') or ''}".lower()
        if any(w in title for w in skip):
            continue
        end = float(total) if i == len(beats) - 1 else float(beats[i + 1]["start"])
        if end - b["start"] >= 0.5:
            out.append({"bid": b["id"], "file": f["file"], "t": round(float(b["start"]), 3),
                        "dur": round(end - float(b["start"]), 3), "kind": kind})
    return out


# ---------------------------------------------------------------- mixing

def mix(vo_wav, events, total, style, out_wav, music_file="", levels=None, footage=None, seed="", log=None, music_plan=None):
    """voice + SFX + footage sound + music bed -> mono 48 kHz wav (loudness is normalized later)."""
    log = log or (lambda *a, **k: None)
    vo = load_audio(vo_wav)
    n = int(total * SR) + SR
    out = np.zeros(n, dtype=np.float32)
    m = min(len(vo), n)
    out[:m] += vo[:m] * db(style["audio"].get("vo_gain_db", 0))
    talk = talk_curve(vo, n)

    rnd = random.Random(f"{seed}:sfx")
    cache = {}
    if clip_audio_only(style):
        events = []                                   # no sound effects in this mode, even old saved ones
    for e in events or []:
        files = sfx_files(e["cat"])
        if not files:
            continue
        f = rnd.choice(files)
        if f not in cache:
            cache[f] = load_audio(f)
        x = cache[f] * db(e["gain_db"])
        if e.get("align") == "end" and e.get("hit") is not None:
            s = int(float(e["hit"]) * SR) - len(x)              # the riser ends exactly on the hit
        else:
            s = int(float(e["t"]) * SR)
        _add(out, x, s)

    _mix_footage(out, footage or [], talk, style.get("footage_audio", {}), log)
    if clip_audio_only(style):
        if music_plan:
            mix_bgm_plan(out, n, talk, music_plan, style.get("bgm", {}), log)
            log("sound: voice + clip audio + music by script - no sound effects")
        elif music_file and Path(music_file).is_file():
            mix_bgm(out, n, talk, music_file, style.get("bgm", {}), log)
            log("sound: voice + clip audio + music - no sound effects")
        else:
            log("sound: voice + clip audio - no music chosen, no sound effects")
    else:
        _mix_music(out, n, float(total), events or [], levels, talk, style.get("music", {}), music_file, seed, log)

    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 0.98:
        out *= 0.98 / peak
    write_wav(out_wav, out)
    return out_wav


def clip_audio_only(style):
    """Sound mode chosen by the user: voice + every clip's own sound (+ the chosen music on loop), nothing generated."""
    return str((style or {}).get("sound_mode", "")).lower() in ("clip_audio_only", "three_sounds")


def mix_bgm(out, n, talk, path, mc, log):
    """The one music file chosen for this video, looped start-to-end until the video ends, under the voice."""
    try:
        a = load_audio(path)
    except Exception as e:
        log(f"music: could not read {Path(path).name} ({str(e)[:80]})")
        return
    if len(a) < SR:
        log(f"music: {Path(path).name} is too short")
        return
    f = min(int(0.03 * SR), len(a) // 4)               # 30 ms fade at each end of the track: no click when it restarts
    a = a.astype(np.float32).copy()
    a[:f] *= np.linspace(0, 1, f, dtype=np.float32)
    a[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    k = -(-n // len(a))
    loop = np.tile(a, k)[:n]
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) or 1e-6
    loop *= db(float(mc.get("level_db", -20))) / rms             # every track lands at the same loudness
    hop = SR // CTRL
    idx = np.arange(n)
    g = _duck(talk, idx, float(mc.get("gap_db", -4)), float(mc.get("under_voice_db", -12)))
    t = idx / SR
    g = g * np.clip(t / max(float(mc.get("fade_in_seconds", 1.0)), 0.01), 0, 1)
    g = g * np.clip((n / SR - t) / max(float(mc.get("end_fade_seconds", 3.0)), 0.01), 0, 1)
    out += (loop * g).astype(np.float32)
    log(f"music: {Path(path).name} looped {k}x under the voice")


MOODS = ["tension", "action", "somber", "triumph", "neutral"]
MOOD_WORDS = {
    "somber": r"\b(killed|died|dead|death|deaths|lost|sank|sunk|casualt\w*|funeral|mourn\w*|tragic|tragedy|wounded|burn(?:ed|ing)|bodies)\b",
    "triumph": r"\b(victory|victorious|won|surrender\w*|liberat\w*|celebrat\w*|freed|heroes?|medal|triumph\w*|returned home)\b",
    "action": r"\b(attack\w*|strik\w*|struck|fired|firing|missiles?|bomb\w*|explo\w*|battle|assault|charge|raid|shot down|hit)\b",
    "tension": r"\b(threat\w*|warn\w*|danger\w*|secret\w*|feared|fear|hunt\w*|could|might|unknown|silence|waiting|approach\w*)\b",
}


def mood_plan(beats, specs, total, min_seg=90.0):
    """Script -> music sections [(start, end, mood)]: every ~90 s window takes the mood its narration leans to most
    (death/loss -> somber, fighting -> action, threat -> tension, victory -> triumph, else neutral)."""
    import re as _re
    sp = {str(s.get("id")): s for s in specs or []}
    weights = []
    for i, b in enumerate(beats):
        end = total if i == len(beats) - 1 else beats[i + 1]["start"]
        d = max(0.1, end - b["start"])
        s = sp.get(str(b["id"]), {})
        w = dict.fromkeys(MOODS, 0.0)
        w["neutral"] = 0.0
        txt = b.get("text", "")
        for m, rx in MOOD_WORDS.items():
            hits = len(_re.findall(rx, txt, _re.I))
            w[m] += hits * d
        if s.get("intensity") == 3 or s.get("moment") in ("attack", "reveal"):
            w["action"] += d
        if s.get("moment") == "cliffhanger":
            w["tension"] += d
        weights.append((b["start"], w))
    avg = dict.fromkeys(MOODS, 0.0)
    for _st, w in weights:
        for m in MOODS:
            avg[m] += w[m] / max(1.0, total)
    secs, t0 = [], 0.0
    starts = [b["start"] for b in beats] + [total]
    while t0 < total - 1:
        t1 = min(total, t0 + min_seg)
        t1 = min((x for x in starts if x >= t1), default=total)       # cut at a sentence start
        if total - t1 < min_seg * 0.6:
            t1 = total
        tot = dict.fromkeys(MOODS, 0.0)
        for st, w in weights:
            if t0 <= st < t1:
                for m in MOODS:
                    tot[m] += w[m]
        # a mood wins a section when its narration leans to it MORE than the video does on average
        # (an absolute count let "neutral" win everything in a long video)
        span = max(1.0, t1 - t0)
        ratio = {m: (tot[m] / span) / (avg[m] or 1e-9) for m in MOODS if m != "neutral" and avg[m] > 0 and tot[m] > 0}
        mood = max(ratio, key=ratio.get) if ratio and max(ratio.values()) >= 1.1 else "neutral"
        if t0 == 0 and mood == "neutral":
            mood = "tension"                                         # the hook never opens on a flat bed
        if secs and secs[-1][2] == mood:
            secs[-1] = (secs[-1][0], t1, mood)
        else:
            secs.append((t0, t1, mood))
        t0 = t1
    return secs


def mood_tracks(music_dir=None, log=None):
    """{mood: [files]}: every file in assets/music (root and sub-folders) listened to and given a mood
    (musicmood.py); files in assets/music/<mood>/ keep their folder's mood."""
    from .musicmood import analyse_library
    d = Path(music_dir or MUSIC_DIR)
    out = {m: [] for m in MOODS}
    for rel, a in analyse_library(d, log or (lambda *_: None)).items():
        out.setdefault(a["mood"], []).append(a["file"])
    out = {m: sorted(v) for m, v in out.items()}
    out["_any"] = sorted(str(p) for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXT) if d.is_dir() else []
    return out


def music_by_script(beats, specs, total, seed="", music_dir=None, log=print):
    """[(start, end, file)] for the whole video: each section's mood -> a track from that mood folder, used in turn,
    never the same track twice in a row."""
    import hashlib
    import random
    secs = mood_plan(beats, specs, total)
    tracks = mood_tracks(music_dir, log)
    rnd = random.Random(hashlib.md5(str(seed).encode()).hexdigest())
    for m in tracks:
        rnd.shuffle(tracks[m])
    log("music library: " + ", ".join(f"{m} {len(tracks.get(m) or [])}" for m in MOODS))
    long_secs = []
    for s, e, m in secs:                      # one track never runs more than ~3.5 min: same mood, next track
        k = max(1, round((e - s) / 210.0))
        cuts = [s + (e - s) * i / k for i in range(k + 1)]
        starts = [b["start"] for b in beats]
        cuts = [cuts[0]] + [min((x for x in starts if x >= c), default=c) for c in cuts[1:-1]] + [cuts[-1]]
        long_secs += [(a, b, m) for a, b in zip(cuts, cuts[1:]) if b > a]
    secs = long_secs
    turn, plan, last = {}, [], None
    fallback = ["neutral", "tension", "action", "somber", "triumph"]
    for s, e, m in secs:
        pool_mood = next((x for x in [m] + fallback if tracks.get(x)), None)
        pool = tracks[pool_mood] if pool_mood else tracks["_any"]
        if not pool:
            continue
        k = turn.get(pool_mood, 0)
        f = pool[k % len(pool)]
        if f == last and len(pool) > 1:
            k += 1
            f = pool[k % len(pool)]
        turn[pool_mood] = k + 1
        if plan and plan[-1][2] == f and len(pool) == 1:
            plan[-1] = (plan[-1][0], e, f)
        else:
            plan.append((s, e, f))
        last = f
    if plan:
        log("music by script: " + " | ".join(f"{int(s // 60)}:{int(s % 60):02d} {m}" for s, _, m in secs) +
            f" -> {len(plan)} section(s), {len(set(f for _, _, f in plan))} track(s)")
    else:
        log("music by script: no files in assets/music - no music")
    return plan


def mix_bgm_plan(out, n, talk, plan, mc, log, xf_s=6.0):
    """Several tracks in turn (music by script): each loops inside its section, 6 s slow crossfades between them, each new track starting at its own beginning."""
    bed = np.zeros(n, dtype=np.float32)
    xf = int(xf_s * SR)
    cache = {}
    for j, (s, e, f) in enumerate(plan):
        if f not in cache:
            try:
                a = load_audio(f).astype(np.float32)
            except Exception as ex:
                log(f"music: could not read {Path(f).name} ({str(ex)[:80]})")
                cache[f] = None
                continue
            r = float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) or 1e-6
            cache[f] = a * (db(float(mc.get("level_db", -20))) / r)
        a = cache[f]
        if a is None or len(a) < SR:
            continue
        i0 = max(0, int(s * SR) - (xf // 2 if j else 0))
        i1 = min(n, int(e * SR) + (xf // 2 if j < len(plan) - 1 else 0))
        L = i1 - i0
        if L <= 0:
            continue
        seg = np.tile(a, -(-L // len(a)))[:L].copy()
        if j and L > xf:
            seg[:xf] *= np.sqrt(np.linspace(0, 1, xf, dtype=np.float32))
        if j < len(plan) - 1 and L > xf:
            seg[-xf:] *= np.sqrt(np.linspace(1, 0, xf, dtype=np.float32))
        bed[i0:i1] += seg
    idx = np.arange(n)
    g = _duck(talk, idx, float(mc.get("gap_db", -4)), float(mc.get("under_voice_db", -12)))
    t = idx / SR
    g = g * np.clip(t / max(float(mc.get("fade_in_seconds", 1.0)), 0.01), 0, 1)
    g = g * np.clip((n / SR - t) / max(float(mc.get("end_fade_seconds", 3.0)), 0.01), 0, 1)
    out += (bed * g).astype(np.float32)
    log(f"music: {len(plan)} section(s) from {len(set(f for _, _, f in plan))} track(s), crossfaded, under the voice")


def _duck(talk, idx, gap_db, under_db):
    hop = SR // CTRL
    tk = np.interp(idx / hop, np.arange(len(talk)), talk)
    return db(gap_db * (1 - tk) + under_db * tk)


def _mix_footage(out, items, talk, fa, log):
    if not items:
        return
    from concurrent.futures import ThreadPoolExecutor

    def dec(it):
        try:
            return it, load_audio(it["file"], dur=it["dur"])
        except Exception:
            return it, None

    with ThreadPoolExecutor(8) as ex:
        decoded = list(ex.map(dec, items))
    kept, kinds, silent, talky = 0, {"action": 0, "ambient": 0}, 0, 0
    fade = max(1, int(float(fa.get("fade_seconds", 0.12)) * SR))
    amb = fa.get("ambient", {})
    boost = db(float(fa.get("max_boost_db", 10)))
    for it, x in decoded:
        if x is None or len(x) < int(0.3 * SR):
            silent += 1
            continue
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if rms < db(-50):
            silent += 1
            continue
        if speech_like(x):
            talky += 1
            continue
        kind = it.get("kind", "action")
        lv = amb if kind == "ambient" else fa
        if fa.get("natural"):
            # the clip's own sound as it is, only a very loud clip (blasts) is brought down to max_rms_db
            lv = fa
            x = (x * min(1.0, db(float(fa.get("max_rms_db", -18))) / rms)).astype(np.float32)
        else:
            # same loudness for every clip, but a quiet clip is never pushed up by more than max_boost_db (hiss)
            x = (x * min(db(float(lv.get("level_db", fa.get("level_db", -20)))) / rms, boost)).astype(np.float32)
        f = min(fade, len(x) // 2)
        x[:f] *= np.linspace(0, 1, f, dtype=np.float32)
        x[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        s = int(float(it["t"]) * SR)
        g = _duck(talk, np.arange(s, s + len(x)), float(lv.get("gap_db", -12)), float(lv.get("under_voice_db", -24)))
        _add(out, x * g, s)
        kept += 1
        kinds[kind] = kinds.get(kind, 0) + 1
    log(f"footage sound: {kept} shot(s) keep their own sound ({kinds.get('action', 0)} action, "
        f"{kinds.get('ambient', 0)} ambient)"
        + (f", {talky} skipped (sounded like speech)" if talky else "")
        + (f", {silent} had no sound" if silent else ""))


def _arrange(tracks, n, seed, xf):
    order = list(tracks)
    random.Random(f"{seed}:music").shuffle(order)
    lengths = {p: len(a) for p, a in order}
    places, pos, k = [], 0, 0
    while pos < n and k < 100000:
        p, _ = order[k % len(order)]
        L = lengths[p]
        places.append((p, pos, L))
        pos += L - xf
        k += 1
    return places


def _mix_music(out, n, total, events, levels, talk, mc, music_file, seed, log):
    hop = SR // CTRL
    mlen = len(talk)
    # intensity per control step
    lvl = np.full(mlen, 2.0)
    lv = [x for x in (levels or []) if isinstance(x, (list, tuple)) and len(x) == 2]
    for (t, L), nxt in zip(lv, lv[1:] + [[total + 10, None]]):
        a, b = max(0, int(float(t) * CTRL)), min(mlen, int(float(nxt[0]) * CTRL))
        if b > a:
            lvl[a:b] = float(L)
    idb = mc.get("intensity_db", {"1": -5, "2": 0, "3": 3})
    ramp = int(float(mc.get("ramp_seconds", 1.5)) * CTRL)
    level_db = _smooth(np.select([lvl <= 1, lvl >= 3], [idb["1"], idb["3"]], idb["2"]).astype(float), ramp)
    bright = np.clip(_smooth((lvl > 1).astype(float), ramp), 0, 1)
    pulse_amt = np.clip(_smooth(np.select([lvl <= 1, lvl >= 3], [0.0, 1.0], 0.35), ramp), 0, 1)

    # swell into every attack / reveal hit (where the riser lands), dip under cliffhanger chords
    add_db = np.zeros(mlen)
    dips = np.zeros(mlen)
    sw_db, sw_s, rel_s = float(mc.get("swell_db", 4)), float(mc.get("swell_seconds", 2.0)), float(mc.get("swell_release_seconds", 1.5))
    for e in events:
        if e.get("cat") == "riser" and e.get("hit") is not None:
            h = int(float(e["hit"]) * CTRL)
            a, r = max(0, h - int(sw_s * CTRL)), min(mlen, h + int(rel_s * CTRL))
            if h > a:
                idx = np.arange(a, min(h, mlen))
                add_db[idx] += sw_db * ((idx - a) / (h - a)) ** 2
            if r > h:
                idx = np.arange(h, r)
                add_db[idx] += sw_db * (1 - (idx - h) / (r - h))
        elif e.get("cat") == "drone_chord":
            a = max(0, int(float(e["t"]) * CTRL))
            dips[a:min(mlen, a + int(float(mc.get("chord_dip_seconds", 5)) * CTRL))] = 1
    add_db += _smooth(dips, int(0.4 * CTRL)) * float(mc.get("chord_dip_db", -6))

    tk = talk[:mlen]
    total_db = float(mc.get("gap_db", -14)) * (1 - tk) + float(mc.get("under_voice_db", -20)) * tk + level_db + add_db
    g = db(total_db)
    tsec = np.arange(mlen) / CTRL
    g *= np.clip(tsec / max(float(mc.get("fade_in_seconds", 1.0)), 0.01), 0, 1)
    fo = float(mc.get("end_fade_seconds", 4.0))
    g *= np.clip((total - tsec) / max(fo, 0.01), 0, 1)

    tracks = music_tracks(music_file)
    CH = 10 * SR
    if tracks:
        lp = int(mc.get("calm_lowpass_hz", 900))
        xf = int(float(mc.get("crossfade_seconds", 3.0)) * SR)
        dec_full, dec_dark = {}, {}
        for p in tracks:
            try:
                a = load_audio(p)
                if len(a) <= 3 * xf:
                    continue
                dec_full[p] = a
                d = load_audio(p, af=f"lowpass=f={lp},lowpass=f={lp}")[:len(a)]
                dec_dark[p] = np.pad(d, (0, len(a) - len(d)))
            except Exception:
                continue
        if not dec_full:
            log("music: no usable track in assets/music (all too short or unreadable)")
            return
        places = _arrange(list(dec_full.items()), n, seed, xf)
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            idx = np.arange(c0, c1)
            gi = np.interp(idx / hop, np.arange(mlen), g)
            bi = np.interp(idx / hop, np.arange(mlen), bright)
            seg = np.zeros(c1 - c0)
            for k, (p, s, L) in enumerate(places):
                a, b = max(c0, s), min(c1, s + L)
                if a >= b:
                    continue
                i = np.arange(a - s, b - s)
                env = np.ones(len(i))
                if s > 0:
                    env *= np.clip(i / xf, 0, 1)
                if k < len(places) - 1:
                    env *= np.clip((L - i) / xf, 0, 1)
                full, dark = dec_full[p][a - s:b - s], dec_dark[p][a - s:b - s]
                seg[a - c0:a - c0 + len(full)] += (full * bi[a - c0:b - c0] + dark * (1 - bi[a - c0:b - c0])) * env
            out[c0:c1] += (seg * gi).astype(np.float32)
        names = ", ".join(p.name for p in dec_full)[:120]
        log(f"music: {len(dec_full)} track(s) chained ({names}), shaped by the intensity curve")
        return
    if str(mc.get("fallback", "generated")).lower() != "generated":
        log("music: none (put tracks into assets/music)")
        return
    bed = SFX_DIR / "_generated" / "_bed"
    try:
        full, dark, pulse = (load_audio(bed / f"{x}.wav") for x in ("full", "dark", "pulse"))
    except Exception:
        full, dark, pulse = synth_bed()
    L = min(len(full), len(dark), len(pulse))
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH)
        idx = np.arange(c0, c1)
        ci = idx / hop
        gi = np.interp(ci, np.arange(mlen), g)
        bi = np.interp(ci, np.arange(mlen), bright)
        pi = np.interp(ci, np.arange(mlen), pulse_amt)
        w = idx % L
        seg = full[w] * bi + dark[w] * (1 - bi) + pulse[w] * pi * 0.8
        out[c0:c1] += (seg * gi).astype(np.float32)
    log("music: generated tension bed (put your own tracks into assets/music for a better score)")
