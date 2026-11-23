import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warstudio import ai33, overlays, sfx  # noqa: E402
from warstudio.config import load_style  # noqa: E402
from warstudio.steps import beats, search, specs, vo, fetch, rank  # noqa: E402

STYLE = load_style("beyond_military")


def test_split_sentences_keeps_abbreviations_and_numbers():
    s = vo.split_sentences("The U.S. Navy fired at 3.5 miles. Then the F-15E turned! Was it over? No.")
    assert s == ["The U.S. Navy fired at 3.5 miles.", "Then the F-15E turned!", "Was it over?", "No."]


def test_beats_split_intro_and_long_sentences_without_breaking_numbers():
    sents = [{"sid": 0, "text": "Behind it, a pair of F-15E Strike Eagles hold at 28,000 feet, waiting for targets.", "start": 0, "end": 5.0},
             {"sid": 1, "text": "Short one.", "start": 5.0, "end": 6.0},
             {"sid": 2, "text": " ".join(["word"] * 40) + ".", "start": 200, "end": 214}]
    b = beats.make_beats(sents, STYLE)
    assert b[0]["text"].startswith("Behind it")
    assert all("28," not in x["text"] or "28,000" in x["text"] for x in b)
    assert len([x for x in b if x["sid"] == 0]) == 2
    assert len([x for x in b if x["sid"] == 1]) == 1
    long = [x for x in b if x["sid"] == 2]
    assert len(long) >= 3 and all(x["end"] - x["start"] <= STYLE["pacing"]["max_shot_seconds"] for x in long)
    assert b[-1]["end"] == 214


@pytest.mark.parametrize("text,l1,l2", [
    ("At 17:14 local time, a Reaper", "17:14", ""),
    ("holding at 28,000 feet", "28,000 ft", "8,534 m"),
    ("32 miles east of Qeshm", "32 mi", "51.5 km"),
    ("at 450 knots", "450 knots", "833 km/h"),
    ("25 yards off the hull", "25 yd", "22.9 m"),
])
def test_overlay_detect(text, l1, l2):
    ov = overlays.detect(text)
    assert ov["line1"] == l1 and ov["line2"] == l2


def test_overlay_plan_gap_and_labels():
    b = [{"id": 0, "text": "At 17:14 local time", "start": 0, "end": 3},
         {"id": 1, "text": "at 28,000 feet", "start": 3, "end": 5},
         {"id": 2, "text": "the F-15E turns", "start": 6, "end": 8}]
    sp = [{"id": 0}, {"id": 1}, {"id": 2, "label": "F-15E Strike Eagle"}]
    p = overlays.plan(b, sp, STYLE)
    assert 0 in p and 1 not in p and p[2]["kind"] == "label"


def test_specs_validate_and_fallback():
    ok, err = specs.validate([{"id": 1, "queries": ["F-15E flying"], "footage_class": "real", "sfx": ["launch", "bogus"]}], {1})
    assert not err and ok[1]["sfx"] == ["launch"]
    _, err = specs.validate([{"id": 1, "queries": []}], {1, 2})
    assert err
    fb = specs.fallback_spec({"id": 3, "text": "The MQ-9 Reaper turns hard right over the coast."})
    assert fb["queries"][0] and fb["footage_class"] == "real"


def test_sfx_keywords_whole_word():
    assert not sfx.keyword_hit("launch", "watching for missile launchers")
    assert sfx.keyword_hit("detonat*", "the warhead detonated")


def test_sfx_plan_events():
    b = [{"id": 0, "sid": 0, "text": "At 17:14 the missile is fired", "start": 2, "end": 4},
         {"id": 1, "sid": 1, "text": "But unknown to them, it explodes", "start": 10, "end": 12}]
    sp = [{"id": 0, "footage_class": "map", "sfx": []}, {"id": 1, "footage_class": "real", "sfx": []}]
    ev = sfx.plan_events(b, sp, {0}, STYLE)
    cats = [e["cat"] for e in ev]
    for c in ("ui_click", "whoosh", "launch", "riser", "explosion"):
        assert c in cats


def test_sfx_synth_all_categories():
    for c in sfx.CATS:
        x = sfx.synth(c)
        assert len(x) > 1000 and abs(x).max() <= 0.91


def test_ai33_voice_parsing_and_ids():
    assert ai33.normalize_voice_id("abc") == "elevenlabs_abc"
    assert ai33.normalize_voice_id("clone_x") == "clone_x"
    vs = ai33._parse_voices({"data": {"voices": [{"voice_id": "v1", "name": "Bob", "labels": {"accent": "us"}}]}})
    assert vs[0]["voice_id"] == "elevenlabs_v1" and vs[0]["desc"] == "us"


def test_search_parsers(monkeypatch):
    def fake(source, url, params, headers=None, ttl=0):
        if source == "pexels":
            return {"videos": [{"id": 5, "url": "https://www.pexels.com/video/jet-5/", "image": "p.jpg", "duration": 12,
                                "user": {"name": "A"}, "video_files": [
                                    {"file_type": "video/mp4", "width": 3840, "height": 2160, "link": "4k.mp4"},
                                    {"file_type": "video/mp4", "width": 1920, "height": 1080, "link": "hd.mp4"}]}]}
        if source == "pixabay":
            return {"hits": [{"id": 9, "pageURL": "u", "duration": 8, "user": "B", "tags": "jet",
                              "videos": {"large": {"url": "l.mp4"}, "medium": {"url": "m.mp4", "thumbnail": "t.jpg"}}}]}
        return {"results": [{"id": "video:123", "title": "F-15E", "thumbnail": {"url": "d.jpg"}}]}
    monkeypatch.setattr(search, "_cached_get", fake)
    assert search.search_pexels("jet", "k", 5)[0]["video_url"] == "hd.mp4"
    assert search.search_pixabay("jet", "k", 5)[0]["preview_url"] == "t.jpg"
    d = search.search_dvids("jet", "k", 5)[0]
    assert d["id"] == "video:123" and d["preview_url"] == "d.jpg"


def test_choose_start_avoids_reuse():
    a = fetch.choose_start(60, 4, [])
    b = fetch.choose_start(60, 4, [a])
    assert abs(a - b) > 4
    assert fetch.choose_start(0, 4, []) == 0.0


def test_collage(tmp_path):
    from PIL import Image
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 36), (i * 40, 0, 0)).save(p)
        paths.append(p)
    out = rank.make_collage(paths, tmp_path / "c.jpg")
    assert Image.open(out).size == (1536, 576)


def test_time_followed_by_punctuation():
    assert overlays.detect("At 1722, the pin lands")["line1"] == "17:22"
    assert overlays.detect("at 17:14.")["line1"] == "17:14"
    assert overlays.detect("at 1,500 feet")["kind"] == "measure"


def test_report_has_no_keys_and_marks_failures(tmp_path, monkeypatch):
    from warstudio import diagnostics
    monkeypatch.setattr(diagnostics, "REPORT_DIR", tmp_path)
    res = [{"group": "APIs", "name": "x", "status": "fail", "detail": "boom", "seconds": 0.1, "trace": "Traceback\nValueError: boom"},
           {"group": "APIs", "name": "y", "status": "ok", "detail": "fine", "seconds": 0.1}]
    text = diagnostics.write_report(res, False)
    assert "FAIL  x: boom" in text and "PASS  y" in text and "1 pass, 0 warn, 1 fail" in text
    assert (tmp_path / "latest.txt").exists()


def test_project_names_unique(tmp_path, monkeypatch):
    from warstudio import pipeline
    monkeypatch.setattr(pipeline, "PROJECTS_DIR", tmp_path)
    a = pipeline.slugify("Same Title")
    (tmp_path / a).mkdir()
    assert pipeline.slugify("Same Title") != a


def test_brain_config_defaults_to_deepseek(monkeypatch, tmp_path):
    from warstudio import config
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "s.json")
    b = config.brain_config(keys={"DEEPSEEK_API_KEY": "k1"})
    assert b["provider"] == "deepseek" and b["url"] == "https://api.deepseek.com/chat/completions"
    assert b["model"] == "deepseek-flash" and b["key"] == "k1"
    c = config.brain_config(settings=dict(config.DEFAULT_SETTINGS, brain_provider="custom", custom_base_url="https://x.ai/v1",
                                          custom_model="m"), keys={"CUSTOM_BRAIN_API_KEY": "k2"})
    assert c["url"] == "https://x.ai/v1/chat/completions" and c["model"] == "m" and c["key"] == "k2"


def test_connect_brain_and_footage(monkeypatch):
    from warstudio import connect, llm

    class R:
        def __init__(self, code, data, headers=None):
            self.status_code, self._d, self.headers, self.text = code, data, headers or {}, json.dumps(data)

        def json(self):
            return self._d

    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: R(200, {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {}}))
    monkeypatch.setattr(connect, "_get", lambda url, **k: R(200, {"balance_infos": [{"total_balance": "9.50", "currency": "USD"}]})
                        if "balance" in url else R(401, {"error": {"message": "Invalid API key"}}))
    ok, msg = connect.test("DEEPSEEK_API_KEY", "sk-test")
    assert ok and "DeepSeek" in msg and "9.50 USD" in msg
    ok, msg = connect.test("PEXELS_API_KEY", "bad")
    assert not ok and "401" in msg and "Invalid API key" in msg
    ok, msg = connect.test("PIXABAY_API_KEY", "")
    assert not ok and "empty" in msg
