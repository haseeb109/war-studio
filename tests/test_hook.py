import json
from pathlib import Path

from warstudio import overlays
from warstudio.config import load_style
from warstudio.steps import beats as bt, specs, rank

STYLE = load_style("beyond_military")


def _sents(spec):
    out, t = [], 0.0
    for i, (text, dur) in enumerate(spec):
        out.append({"sid": i, "text": text, "start": t, "end": t + dur})
        t += dur
    return out


def test_hook_zone_has_short_shots_and_ends_after_1000_chars():
    long = "The Strike Eagles push their throttles forward and cross the border at speed, low through the hills."
    ss = _sents([(long, 6.0)] * 14)            # ~102 chars each -> sentences 0..9 start inside the first 1000
    b = bt.make_beats(ss, STYLE)
    hook = [x for x in b if x["hook"]]
    assert {x["sid"] for x in hook} == set(range(10))
    assert max(x["end"] - x["start"] for x in hook) <= STYLE["pacing"]["hook_max_shot_seconds"] + 1e-6
    assert min(x["end"] - x["start"] for x in hook) >= STYLE["pacing"]["hook_min_shot_seconds"] - 1e-6
    after = [x for x in b if not x["hook"]]
    assert after and all(x["end"] - x["start"] > 2.4 for x in after)   # normal pacing again


def test_hook_text_kept_only_in_hook_and_cleaned():
    assert specs.clean_hook_text('  "target: mobile convoy" ') == "TARGET: MOBILE CONVOY"
    assert specs.clean_hook_text("x" * 40) == ""
    out, err = specs.validate([{"id": 1, "visual": "v", "category_query": "fighter jet", "hook_text": "status: tracking",
                                "footage_class": "real"}], {1})
    assert not err and out[1]["hook_text"] == "STATUS: TRACKING"


def test_overlay_plan_uses_hook_captions_with_short_gap():
    beats = [{"id": i, "text": t, "start": i * 2.0, "end": i * 2.0 + 2.0, "hook": True}
             for i, t in enumerate(["Two jets lift off.", "They turn east.", "At 02:15 they climb.", "Silence."])]
    sp = [{"id": 0, "label": "F-15E Strike Eagle", "hook_text": "2 x F-15E"},
          {"id": 1, "label": "", "hook_text": "MISSION: ONE PASS"},
          {"id": 2, "label": "", "hook_text": "IGNORED: TIME WINS"},
          {"id": 3, "label": "", "hook_text": ""}]
    plan = overlays.plan(beats, sp, STYLE)
    assert plan[0]["kind"] == "label" and plan[1]["kind"] == "hook" and plan[1]["line1"] == "MISSION: ONE PASS"
    assert plan[2]["kind"] == "time" and 3 not in plan
    # outside the hook: no hook captions, normal gap
    for b in beats:
        b["hook"] = False
    plan2 = overlays.plan(beats, sp, STYLE)
    assert all(o["kind"] != "hook" for o in plan2.values())


def test_hook_png_fallback(tmp_path):
    out = overlays.render_png({"kind": "hook", "line1": "TARGET: MOBILE CONVOY", "line2": "", "duration": 2},
                              STYLE, tmp_path / "h.png")
    assert Path(out).exists()


def test_rank_hook_never_repeats_a_clip(tmp_path):
    from PIL import Image
    from warstudio.util import write_json
    beats = [{"id": i, "sid": i, "text": f"b{i}", "start": i * 2.0, "end": i * 2 + 2, "hook": True} for i in range(3)]
    sp = [{"id": i, "queries": ["jet"], "tiers": ["category"], "visual": "jets", "footage_class": "real",
           "scene": "jets", "avoid": [], "hook": True} for i in range(3)]
    cands = {str(i): [{"source": "pixabay", "id": str(j), "title": "", "preview_url": "x", "video_url": "v",
                       "duration": 20, "tier": "category"} for j in range(3)] for i in range(3)}
    for f, o in (("beats.json", beats), ("specs.json", sp), ("candidates.json", cands)):
        write_json(tmp_path / f, o)
    (tmp_path / "previews").mkdir()
    for j in range(3):
        Image.new("RGB", (64, 36)).save(tmp_path / "previews" / f"pixabay_{j}.jpg")
    rank.clip_scores = lambda s, t, ps: [0.3 if p.endswith("_0.jpg") else 0.2 for p in ps]
    picks = rank.run_step(tmp_path, {}, {"reranker": "clip"}, {"footage": {"same_source_cooldown_seconds": 90}},
                          lambda m, **k: None)
    assert len({p["id"] for p in picks.values()}) == 3
