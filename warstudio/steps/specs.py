"""Step 3: beats -> shot specs (queries, footage class, entity, label, sound cues) via the brain (DeepSeek official API by default).

Beats go in chunks of 40 with the surrounding script for context. Every reply is validated; a bad
reply gets one repair round, then a deterministic fallback spec so the run never stops.
"""
import json
import re
from pathlib import Path

from .. import ROOT
from ..config import brain_config
from ..llm import chat, extract_json
from ..util import read_json, write_json, num

CLASSES = {"real", "flir", "cockpit", "control_room", "map"}
MOMENTS = {"attack", "reveal", "cliffhanger"}
STOP = set("the a an and or but of to in on at for with from by is are was were be been it its this that his her their "
           "he she they them we you as into over under than then so just still now one two three".split())
CHUNK = 40


CATEGORY_DEFAULT = "military aircraft flying"


def _clean_q(x):
    return re.sub(r"[^\w\s\-]", " ", str(x or "")).strip()


def build_queries(exact, substitute, category):
    """Search order from the master prompt: exact -> substitute -> category. Returns (queries, tiers)."""
    qs, tiers, seen = [], [], set()
    for tier, items in (("exact", exact), ("substitute", substitute), ("category", category)):
        for q in items:
            q = " ".join(_clean_q(q).split()[:6])
            if q and q.lower() not in seen:
                seen.add(q.lower())
                qs.append(q)
                tiers.append(tier)
    if "category" not in tiers:
        qs.append(CATEGORY_DEFAULT)
        tiers.append("category")
    return qs, tiers


def fallback_spec(b):
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", b["text"]) if w.lower() not in STOP]
    caps = [w for w in words if any(c.isdigit() for c in w) or w[:1].isupper()]
    q1 = " ".join((caps or words)[:4]) or "military aircraft"
    qs, tiers = build_queries([q1], [" ".join(words[:3]) or "military"], [CATEGORY_DEFAULT])
    return {"id": b["id"], "queries": qs, "tiers": tiers, "visual": b["text"][:160], "substitute": "",
            "category": CATEGORY_DEFAULT, "scene": "", "footage_class": "real", "entity": "", "avoid": [],
            "label": "", "intensity": None, "moment": "", "fallback": True}


def validate(items, ids):
    errors, out = [], {}
    if not isinstance(items, list):
        return {}, ["'beats' must be a list"]
    for it in items:
        if not isinstance(it, dict) or it.get("id") not in ids:
            errors.append(f"unknown or missing id in {str(it)[:80]}")
            continue
        def _lst(v):
            return [x for x in (v if isinstance(v, list) else [v] if isinstance(v, str) else []) if isinstance(x, str) and x.strip()]
        exact, subq, cat = _lst(it.get("exact_queries"))[:2], _lst(it.get("substitute_queries"))[:2], _lst(it.get("category_query"))[:1]
        legacy = not (exact or subq or cat) and bool(_lst(it.get("queries")))
        if legacy:                                                     # older reply format
            old = _lst(it.get("queries"))
            exact, subq, cat = old[:1], old[1:2], old[2:3] or [CATEGORY_DEFAULT]
        if not cat:
            errors.append(f"beat {it['id']}: category_query is required")
            continue
        visual = str(it.get("visual") or "").strip() or (" ".join(exact[:1] or cat) if legacy else "")
        if not visual:
            errors.append(f"beat {it['id']}: visual is required")
            continue
        q, tiers = build_queries(exact, subq, cat)
        fc = it.get("footage_class", "real")
        if fc not in CLASSES:
            errors.append(f"beat {it['id']}: footage_class '{fc}' not allowed")
            fc = "real"
        try:
            intensity = int(it.get("intensity"))
            intensity = intensity if intensity in (1, 2, 3) else None
        except (TypeError, ValueError):
            intensity = None                   # unknown: the sound planner works it out from the shot and the words
        moment = str(it.get("moment") or "").strip().lower()
        moment = moment if moment in MOMENTS else ""
        out[it["id"]] = {"id": it["id"], "queries": q, "tiers": tiers, "visual": visual[:200],
                         "substitute": str(it.get("substitute") or "").strip()[:80], "category": _clean_q(cat[0]),
                         "scene": str(it.get("scene") or "").strip().lower()[:60], "footage_class": fc,
                         "entity": str(it.get("entity") or "").strip(), "avoid": list(it.get("avoid") or [])[:3],
                         "label": str(it.get("label") or "").strip()[:32],
                         "intensity": intensity, "moment": moment}
    missing = ids - set(out)
    if missing:
        errors.append(f"missing beats: {sorted(missing)[:20]}")
    return out, errors


def run_step(proj, keys, settings, log, dummy=False):
    proj = Path(proj)
    out_path = proj / "specs.json"
    if out_path.exists():
        log("specs already done, skipping")
        return read_json(out_path)
    beats = read_json(proj / "beats.json")
    title = (proj / "title.txt").read_text(encoding="utf-8").strip() if (proj / "title.txt").exists() else ""
    # The master prompt is locked: it is the rulebook every later step (search, rank, reuse) follows.
    system = (ROOT / "prompts" / "master_prompt.md").read_text(encoding="utf-8")
    specs, usage_tot = {}, {"prompt_tokens": 0, "completion_tokens": 0}
    brain = brain_config(settings, keys)
    if not dummy:
        if brain["key"]:
            log(f"brain: {brain['label']} / {brain['model']}")
        else:
            log(f"{brain['key_name']} not set: using simple keyword search terms (weaker clip matches)")
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    done_n = [0]

    def do_chunk(c0):
        chunk = beats[c0:c0 + CHUNK]
        ids = {b["id"] for b in chunk}
        if dummy or not brain["key"]:
            return {b["id"]: fallback_spec(b) for b in chunk}
        ctx = " ".join(b["text"] for b in beats[max(0, c0 - 6):c0])
        user = (f"VIDEO TITLE: {title}\nPREVIOUS CONTEXT: {ctx}\n\nBEATS:\n" +
                "\n".join(json.dumps({"id": b["id"], "text": b["text"]}, ensure_ascii=False) for b in chunk))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        got = {}
        for attempt in range(2):
            try:
                reply, usage = chat(messages, brain)
                with lock:
                    for k in usage_tot:
                        usage_tot[k] += usage.get(k, 0) or 0
                data = extract_json(reply)
                got, errors = validate(data.get("beats") if isinstance(data, dict) else data, ids)
            except Exception as e:
                errors = [str(e)[:300]]
                reply = ""
            if not errors:
                break
            log(f"specs chunk {c0}: {len(errors)} problems, repair round" if attempt == 0 else f"specs chunk {c0}: using fallback for {len(ids - set(got))} beats")
            messages = messages + [{"role": "assistant", "content": reply[:6000]},
                                   {"role": "user", "content": "Fix these problems and return the full JSON again:\n" + "\n".join(errors[:15])}]
        out = {b["id"]: got.get(b["id"]) or fallback_spec(b) for b in chunk}
        with lock:
            done_n[0] += len(chunk)
            log(f"specs {done_n[0]}/{len(beats)}", progress=done_n[0] / len(beats))
        return out

    with ThreadPoolExecutor(6) as ex:
        for part in ex.map(do_chunk, range(0, len(beats), CHUNK)):
            specs.update(part)
    result = [specs[b["id"]] for b in beats]
    write_json(out_path, result)
    cost = (usage_tot["prompt_tokens"] / 1e6 * num(settings.get("brain_price_in_per_m"), 0.3) +
            usage_tot["completion_tokens"] / 1e6 * num(settings.get("brain_price_out_per_m"), 1.2))
    log(f"specs done ({sum(1 for s in result if s.get('fallback'))} fallback), brain ~${cost:.3f}")
    return result
