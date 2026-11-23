"""Step 3: beats -> shot specs (queries, footage class, entity, label, sfx) via the brain (DeepSeek official API by default).

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
SFX = {"launch", "explosion", "flyby", "gunfire", "flares", "radar", "radio_static", "alarm", "impact"}
STOP = set("the a an and or but of to in on at for with from by is are was were be been it its this that his her their "
           "he she they them we you as into over under than then so just still now one two three".split())
CHUNK = 40


def fallback_spec(b):
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", b["text"]) if w.lower() not in STOP]
    caps = [w for w in words if any(c.isdigit() for c in w) or w[:1].isupper()]
    q1 = " ".join((caps or words)[:4]) or "military aircraft"
    return {"id": b["id"], "queries": [q1, " ".join(words[:3]) or "military", "military aircraft flying"],
            "footage_class": "real", "entity": "", "avoid": [], "label": "", "sfx": [], "fallback": True}


def validate(items, ids):
    errors, out = [], {}
    if not isinstance(items, list):
        return {}, ["'beats' must be a list"]
    for it in items:
        if not isinstance(it, dict) or it.get("id") not in ids:
            errors.append(f"unknown or missing id in {str(it)[:80]}")
            continue
        q = it.get("queries")
        if not (isinstance(q, list) and q and all(isinstance(x, str) and x.strip() for x in q)):
            errors.append(f"beat {it['id']}: queries must be a non-empty list of strings")
            continue
        fc = it.get("footage_class", "real")
        if fc not in CLASSES:
            errors.append(f"beat {it['id']}: footage_class '{fc}' not allowed")
            fc = "real"
        sfx = [s for s in (it.get("sfx") or []) if s in SFX]
        out[it["id"]] = {"id": it["id"], "queries": [x.strip() for x in q][:3], "footage_class": fc,
                         "entity": str(it.get("entity") or "").strip(), "avoid": list(it.get("avoid") or [])[:3],
                         "label": str(it.get("label") or "").strip()[:32], "sfx": sfx}
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
    system = (ROOT / "prompts" / "shot_spec.md").read_text(encoding="utf-8")
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

    with ThreadPoolExecutor(3) as ex:
        for part in ex.map(do_chunk, range(0, len(beats), CHUNK)):
            specs.update(part)
    result = [specs[b["id"]] for b in beats]
    write_json(out_path, result)
    cost = (usage_tot["prompt_tokens"] / 1e6 * num(settings.get("brain_price_in_per_m"), 0.3) +
            usage_tot["completion_tokens"] / 1e6 * num(settings.get("brain_price_out_per_m"), 1.2))
    log(f"specs done ({sum(1 for s in result if s.get('fallback'))} fallback), brain ~${cost:.3f}")
    return result
