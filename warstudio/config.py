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

KEY_NAMES = ["AI33_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "DVIDS_API_KEY", "PEXELS_API_KEY", "PIXABAY_API_KEY",
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
    "gemini_model": "gemini-2.5-flash",
    "tts_model": "eleven_multilingual_v2",
    "tts_concurrency": 4,
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",
    "candidates_per_query": 12,
    "rerank_top_k": 5,
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


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        s.update(json.loads(SETTINGS_FILE.read_text()))
    return s


def save_settings(new):
    s = load_settings()
    s.update({k: v for k, v in new.items() if k in DEFAULT_SETTINGS})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    return s


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
