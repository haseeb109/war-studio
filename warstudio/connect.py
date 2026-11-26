"""Connection tests for every API: used by the Settings "Test" buttons and by System Check.

Each test can take a key that is not saved yet (what the user just pasted), so a key can be checked
before it is stored. Tests use the smallest real request (a 1-token chat, a 1-result search) and
never use the search cache, so a wrong key can't hide behind an old cached result.
"""
import json
import time

import httpx

from .config import BRAIN_PRESETS, brain_config, load_keys, load_settings
from .util import UA

SERVICES = {
    "AI33_API_KEY": "ai33", "DEEPSEEK_API_KEY": "deepseek", "HIVE_API_KEY": "hive", "CUSTOM_BRAIN_API_KEY": "custom",
    "GROQ_API_KEY": "groq", "GEMINI_API_KEY": "gemini", "DVIDS_API_KEY": "dvids", "PEXELS_API_KEY": "pexels", "PIXABAY_API_KEY": "pixabay",
}


def _get(url, **kw):
    h = {"User-Agent": UA, **kw.pop("headers", {})}
    return httpx.get(url, headers=h, timeout=25, **kw)


def _http_msg(r):
    try:
        d = r.json()
        msg = d.get("error", d.get("message", d)) if isinstance(d, dict) else d
        if isinstance(msg, dict):
            msg = msg.get("message", msg)
    except Exception:
        msg = r.text
    return f"HTTP {r.status_code}: {str(msg)[:180]}"


def test_brain(provider, key=None):
    from .llm import chat, extract_json
    b = brain_config(provider=provider, key_override=key or None)
    if not b["key"]:
        return False, f"{b['key_name']} is empty"
    t0 = time.time()
    try:
        reply, usage = chat([{"role": "user", "content": 'Reply with exactly this JSON: {"ok": true}'}], b, max_tokens=30, timeout=60)
    except Exception as e:
        return False, f"{b['label']}: {str(e)[:220]}"
    try:
        ok = extract_json(reply).get("ok") is True
    except Exception:
        ok = False
    msg = f"{b['label']} / {b['model']} answered in {time.time() - t0:.1f}s" + ("" if ok else f" (unexpected reply: {reply[:60]})")
    if provider == "deepseek":
        try:
            r = _get("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {b['key']}"})
            if r.status_code == 200:
                infos = r.json().get("balance_infos") or []
                if infos:
                    msg += " · balance " + ", ".join(f"{i.get('total_balance')} {i.get('currency')}" for i in infos)
        except Exception:
            pass
    return True, msg


def test_ai33(key):
    if not key:
        return False, "AI33_API_KEY is empty"
    r = _get("https://api.ai33.pro/v1/task/00000000-0000-0000-0000-000000000000", headers={"xi-api-key": key})
    if r.status_code == 401:
        return False, "ai33 rejected the key (401)"
    from . import ai33
    vs, source = ai33.list_voices(key)
    return True, f"key accepted · voice library: {len(vs)} voices ({source})"


def test_groq(key):
    """Real vision call on a tiny image, so a key that can't see any vision model fails here, not mid-run."""
    if not key:
        return False, "GROQ_API_KEY is empty"
    from .llm import _GROQ_MODELS, resolve_groq_model, groq_pick
    _GROQ_MODELS.pop(key, None)
    try:
        model = resolve_groq_model(key, load_settings().get("groq_vision_model", "auto"))
    except Exception as e:
        return False, str(e)[:220]
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        img = f"{d}/t.jpg"
        Image.new("RGB", (256, 144), (255, 196, 0)).save(img)
        t = time.time()
        res, usage = groq_pick(key, model, img, 'What color is this image? Return JSON {"color": "<one word>"}', max_wait=10)
    return True, (f"vision OK with {model} ({time.time() - t:.1f}s, {usage.get('total_tokens', '?')} tokens): {res}. "
                  "Used only for clip rerank.")


def test_gemini(key):
    if not key:
        return False, "GEMINI_API_KEY is empty"
    from .llm import _GEMINI_MODELS, resolve_gemini_model, gemini_pick
    _GEMINI_MODELS.pop(key, None)
    wanted = load_settings().get("gemini_model", "auto")
    try:
        model = resolve_gemini_model(key, wanted)
    except Exception as e:
        return False, str(e)[:220]
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        img = f"{d}/t.jpg"
        Image.new("RGB", (256, 144), (255, 196, 0)).save(img)
        from .llm import ModelGone
        tried = []
        while True:
            try:
                res, usage = gemini_pick(key, model, img, 'What color is this image? Return JSON {"color": "<one word>"}')
                break
            except ModelGone:
                tried.append(model)
                try:
                    model = resolve_gemini_model(key, "auto")
                except Exception as e:
                    return False, f"no working flash model ({', '.join(tried)} gave 404): {str(e)[:120]}"
            except Exception as e:
                from .llm import ProviderBusy, QuotaExhausted
                if isinstance(e, ProviderBusy):
                    return False, (f"model {model} is overloaded at Google right now (503, temporary). "
                                   f"The tool retries and uses backup models during a run. {str(e)[-120:]}")
                if isinstance(e, QuotaExhausted):
                    if getattr(e, "free_tier", False):
                        return False, (f"Google says this key is on the FREE tier and {model}'s free quota is used up. "
                                       f"Billing is not active for this key's project: in AI Studio -> API keys check the "
                                       f"key's project/tier, or create a new key in the billed project. {str(e)[-100:]}")
                    return False, f"quota used up for {model} (paid tier limit). {str(e)[-140:]}"
                return False, f"model {model}: {str(e)[:200]}"
    if tried:
        wanted = "auto"
        from .config import save_settings
        save_settings({"gemini_model": "auto"})
    note = "" if wanted in ("auto", model) else f" ('{wanted}' not available, using this instead)"
    if tried:
        note += f" (retired for this key: {', '.join(tried)})"
    from .llm import _GEMINI_NO_OPT, gemini_fallback_models, gemini_thinking_level
    opts = {"thinking", "media"} - _GEMINI_NO_OPT.get(model, set())
    try:
        backups = ", ".join(gemini_fallback_models(key, model)) or "none"
    except Exception:
        backups = "?"
    return True, (f"key OK · rerank model {model}{note} · backups: {backups} · thinking "
                  f"{gemini_thinking_level(model)}, image "
                  f"{'medium res' if 'media' in opts else 'default res'} · tokens in {usage.get('promptTokenCount', '?')}, "
                  f"thinking {usage.get('thoughtsTokenCount', 0)} · test answer: {res.get('color', res)}")


def test_pexels(key):
    if not key:
        return False, "PEXELS_API_KEY is empty"
    r = _get("https://api.pexels.com/videos/search", params={"query": "fighter jet", "per_page": 1}, headers={"Authorization": key})
    if r.status_code >= 400:
        return False, "Pexels: " + _http_msg(r)
    left = r.headers.get("X-Ratelimit-Remaining")
    return True, f"search OK ({r.json().get('total_results', '?')} results for 'fighter jet')" + (f" · {left} requests left this month" if left else "")


def test_pixabay(key):
    if not key:
        return False, "PIXABAY_API_KEY is empty"
    r = _get("https://pixabay.com/api/videos/", params={"key": key, "q": "fighter jet", "per_page": 3})
    if r.status_code >= 400:
        return False, "Pixabay: " + _http_msg(r)
    return True, f"search OK ({r.json().get('totalHits', '?')} results for 'fighter jet')"


def test_dvids(key):
    if not key:
        return False, "DVIDS_API_KEY is empty"
    r = _get("https://api.dvidshub.net/search", params={"api_key": key, "type": "video", "q": "fighter jet", "max_results": 1})
    if r.status_code >= 400:
        return False, "DVIDS: " + _http_msg(r)
    d = r.json()
    if isinstance(d, dict) and d.get("errors"):
        return False, f"DVIDS: {str(d['errors'])[:180]}"
    total = (d.get("page_info") or {}).get("total_results") if isinstance(d, dict) else None
    # the same search the pipeline runs (with its parameters), so a request the pipeline breaks fails here too
    from .steps.search import search_dvids
    got = search_dvids("F-15E Strike Eagle takeoff", key, 5)
    if not got:
        return False, "DVIDS key works, but the pipeline's search returned 0 clips for 'F-15E Strike Eagle takeoff'"
    small = sum(1 for c in got if "122x92" in (c.get("preview_url") or ""))
    return True, (f"search OK ({total or '?'} videos for 'fighter jet'); pipeline search: {len(got)} clips"
                  + (f", {small} with tiny previews" if small else ", 640px previews"))


def test(key_name, key=None):
    """Test one service. key=None uses the saved key."""
    svc = SERVICES.get(key_name)
    if not svc:
        return False, f"unknown key {key_name}"
    key = key if key else load_keys().get(key_name, "")
    try:
        if svc in BRAIN_PRESETS:
            return test_brain(svc, key)
        return {"ai33": test_ai33, "groq": test_groq, "gemini": test_gemini, "pexels": test_pexels,
                "pixabay": test_pixabay, "dvids": test_dvids}[svc](key)
    except httpx.HTTPError as e:
        return False, f"network error: {str(e)[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"
