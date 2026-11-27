"""Paths, API keys and app settings.

Keys never live in the repo: they are stored in ~/.config/warstudio/keys.env (chmod 600).
"""
import json
import os
from pathlib import Path

from . import ROOT

CONFIG_DIR = Path(os.environ.get("WARSTUDIO_CONFIG", Path.home() / ".config" / "warstudio"))
KEYS_FILE = CONFIG_DIR / "keys.env"
DATA_DIR = ROOT / "data"
PROJECTS_DIR = ROOT / "projects"
LIBRARY_DIR = ROOT / "library" / "raw"
STYLES_DIR = ROOT / "styles"
SFX_DIR = ROOT / "assets" / "sfx"
MUSIC_DIR = ROOT / "assets" / "music"
SETTINGS_FILE = DATA_DIR / "settings.json"
FAVORITES_FILE = DATA_DIR / "favorites.json"

KEY_NAMES = ["AI33_API_KEY", "DEEPSEEK_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "DVIDS_API_KEY",
             "HIVE_API_KEY", "CUSTOM_BRAIN_API_KEY"]

# Brain (script -> shot specs). All speak the OpenAI chat-completions format.
BRAIN_PRESETS = {
    "deepseek": {"label": "DeepSeek (official API)", "url": "https://api.deepseek.com/chat/completions",
                 "key": "DEEPSEEK_API_KEY", "model_setting": "deepseek_model"},
    "hive": {"label": "Hive (DeepSeek resold)", "url": "https://api-cdn.thehive.ai/api/v3/chat/completions",
             "key": "HIVE_API_KEY", "model_setting": "hive_model"},
    "custom": {"label": "Custom (any OpenAI-compatible API)", "url": None,
               "key": "CUSTOM_BRAIN_API_KEY", "model_setting": "custom_model"},
}

DEFAULT_SETTINGS = {
    "brain_provider": "deepseek",
    "deepseek_model": "deepseek-flash",
    "hive_model": "deepseek-ai/deepseek-v4.1-flash-1017",
    "custom_base_url": "",
    "custom_model": "",
    "brain_price_in_per_m": 0.30,
    "brain_price_out_per_m": 1.20,
    "gemini_model": "auto",
    "gemini_price_in_per_m": 0.75,    # $ per 1M input tokens, paid tier (Gemini 3.x Flash; $1.50 from 2027-01-01)
    "gemini_price_out_per_m": 3.75,   # $ per 1M output+thinking tokens ($7.50 from 2027-01-01)
    "reranker": "auto",          # auto = Groq if its key is set, else Gemini, else CLIP only; or groq / gemini / clip
    "groq_vision_model": "auto",
    "segment_check": "auto",     # auto = pixels + Apple Vision text/face check on Mac | vlm = also the vision model | off
    "tts_model": "eleven_multilingual_v2",
    "tts_concurrency": 3,
    "parallel_videos": 2,
    "maptest_parallel": 3,
    "auto_retry": 2,                 # a failed video runs again this many times (5 min, then 20 min later)
    "auto_cleanup_days": 3,          # finished videos lose their work files after N days (final.mp4 stays); 0 = never
    "ai33_usd_per_1k_credits": 0,    # your ai33 plan price per 1000 credits (0 = unknown: costs show voice credits)
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",
    "candidates_per_query": 12,
    "rerank_top_k": 5,
    "rerank_concurrency": 8,
    "render_concurrency": 4,
    "clip_max_uses": 3,
    "clip_reuse_gap_seconds": 60,
    "maps_enabled": True,
    "maps_per_video": 0,
    "map_min_seconds": 4.0,
    "map_max_seconds": 5.0,
    "maps_parallel": 3,
    "use_library": True,         # match beats against the local shot library before searching online
    "library_dir": "",           # empty = <WarStudio>/library/shots
    "library_max_gb": 200,
    "library_min_clip": 0.24,    # CLIP similarity a library shot needs to be used without online search
    "library_shots_per_video": 4,
    "library_captions": "auto",  # auto = caption new shots once with the reranker model (Gemini/Groq); off
    "library_rotation_videos": 2,   # a shot is never in two videos in a row (last N videos rest, unless nothing else fits)
    "library_hook_fresh_videos": 20,  # the first minute of a video only uses shots none of the last N videos showed
    "library_hook_seconds": 60,
    "library_caption_model": "lite",   # lite = newest Gemini Flash-Lite for shot captions (~half price); same = reranker model
    "library_auto_resume": True,  # a library build stopped by a crash / app restart continues when the app starts
    "library_learn": True,       # clean online clips of each finished video join the library
    "library_learn_sources": ["dvids"],
    "library_learn_min_score": 0,   # Gemini score a clip needs to be learned (0 = every clean clip)     # video segments encoded at the same time     # beats reranked at the same time (paid Gemini handles this easily)
    "music_file": "",
    "encoder": "auto",
    "hyperframes": True,
}


def load_keys():
    keys = {k: os.environ.get(k, "") for k in KEY_NAMES}
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in keys and v.strip():
                    keys[k.strip()] = v.strip()
    return keys


def save_keys(new):
    keys = load_keys()
    for k, v in new.items():
        if k in KEY_NAMES and v is not None:
            keys[k] = v.strip()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text("".join(f"{k}={v}\n" for k, v in keys.items()))
    os.chmod(KEYS_FILE, 0o600)
    return keys


def masked_keys():
    return {k: (v[:4] + "..." + v[-4:] if len(v) > 10 else ("set" if v else "")) for k, v in load_keys().items()}


# defaults of older versions that the old save wrote into settings.json: they are dropped once, so the
# current defaults apply (the user never chose them - the old save copied every default into the file)
OLD_DEFAULTS = {"tts_concurrency": [4], "library_rotation_videos": [5]}
SETTINGS_V = 3


def _saved():
    if not SETTINGS_FILE.exists():
        return {}
    try:
        d = json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}
    if d.get("_settings_v") != SETTINGS_V:           # migrate: keep only real choices
        d = {k: v for k, v in d.items() if k in DEFAULT_SETTINGS and v != DEFAULT_SETTINGS[k]
             and v not in OLD_DEFAULTS.get(k, [])}
        d["_settings_v"] = SETTINGS_V
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(d, indent=2))
    return d


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    s.update({k: v for k, v in _saved().items() if k in DEFAULT_SETTINGS})
    return s


def save_settings(new):
    """Only what differs from the defaults is written, so a later version's new defaults reach the user."""
    d = _saved()
    for k, v in new.items():
        if k not in DEFAULT_SETTINGS:
            continue
        if v == DEFAULT_SETTINGS[k] or str(v) == str(DEFAULT_SETTINGS[k]):
            d.pop(k, None)
        else:
            d[k] = v
    d["_settings_v"] = SETTINGS_V
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(d, indent=2))
    return load_settings()


def brain_config(settings=None, keys=None, provider=None, key_override=None):
    """Resolve which brain to call: {provider, label, url, model, key, key_name}."""
    settings = settings or load_settings()
    keys = keys or load_keys()
    prov = provider or settings.get("brain_provider") or "deepseek"
    if prov not in BRAIN_PRESETS:
        prov = "deepseek"
    pre = BRAIN_PRESETS[prov]
    url = pre["url"] or settings.get("custom_base_url", "").strip()
    if url and not url.rstrip("/").endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    return {"provider": prov, "label": pre["label"], "url": url, "model": settings.get(pre["model_setting"], ""),
            "key": (key_override if key_override is not None else keys.get(pre["key"], "")), "key_name": pre["key"]}


def list_styles():
    out = []
    for p in sorted(STYLES_DIR.glob("*/style.json")):
        d = json.loads(p.read_text())
        out.append({"id": d.get("id", p.parent.name), "name": d.get("name", p.parent.name)})
    return out


def load_style(style_id):
    p = STYLES_DIR / style_id / "style.json"
    if not p.exists():
        raise FileNotFoundError(f"style '{style_id}' not found in styles/")
    d = json.loads(p.read_text())
    d["_dir"] = str(p.parent)
    return d
