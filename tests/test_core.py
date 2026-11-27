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
             {"sid": 2, "text": " ".join(["word"] * 40) + ".", "start": 6.4, "end": 20.4}]
    b = beats.make_beats(sents, STYLE, total=21.0)
    assert b[0]["text"].startswith("Behind it")
    assert all("28," not in x["text"] or "28,000" in x["text"] for x in b)
    assert len([x for x in b if x["sid"] == 0]) == 2
    assert len([x for x in b if x["sid"] == 1]) == 1
    long = [x for x in b if x["sid"] == 2]
    assert len(long) >= 5
    # LOCKED rule: on screen (until the next beat, pause included) never longer than 3 s
    on = [(b[i + 1]["start"] if i + 1 < len(b) else 21.0) - x["start"] for i, x in enumerate(b)]
    assert max(on) <= STYLE["pacing"]["max_shot_seconds"] + 1e-6 == 3.0 + 1e-6
    assert min(on) >= STYLE["pacing"]["min_shot_seconds"] - 1e-6


def test_long_pause_or_few_words_still_changes_picture():
    sents = [{"sid": 0, "text": "Silence.", "start": 0, "end": 1.0}, {"sid": 1, "text": "Then it came.", "start": 7.0, "end": 8.5}]
    b = beats.make_beats(sents, STYLE, total=9.0)
    first = [x for x in b if x["sid"] == 0]
    assert len(first) >= 3 and all(x["text"] == "Silence." for x in first) and first[1].get("cont")
    on = [(b[i + 1]["start"] if i + 1 < len(b) else 9.0) - x["start"] for i, x in enumerate(b)]
    assert max(on) <= 3.0 + 1e-6


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
    ok, err = specs.validate([{"id": 1, "queries": ["F-15E flying"], "footage_class": "real", "intensity": 3,
                               "moment": "Attack", "sfx": ["launch"]}], {1})
    assert not err and ok[1]["intensity"] == 3 and ok[1]["moment"] == "attack" and "sfx" not in ok[1]
    ok, _ = specs.validate([{"id": 1, "queries": ["F-15E flying"], "intensity": 7, "moment": "boom"}], {1})
    assert ok[1]["intensity"] is None and ok[1]["moment"] == ""
    _, err = specs.validate([{"id": 1, "queries": []}], {1, 2})
    assert err
    fb = specs.fallback_spec({"id": 3, "text": "The MQ-9 Reaper turns hard right over the coast."})
    assert fb["queries"][0] and fb["footage_class"] == "real" and fb["moment"] == ""


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


def test_gemini_model_auto_pick_and_404_switch(monkeypatch, tmp_path):
    from warstudio import llm
    from warstudio.steps import rank
    llm._GEMINI_MODELS.clear(); llm._GONE.clear()
    names = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash", "gemini-3-flash-preview-09", "gemini-3-pro",
             "gemini-2.5-flash-image", "gemini-3-flash-tts"]
    assert llm.pick_flash_model(names) == "gemini-3-flash"
    assert llm.pick_flash_model(names + ["gemini-flash-latest"]) == "gemini-flash-latest"
    llm._GEMINI_MODELS["k"] = names
    assert llm.resolve_gemini_model("k", "gemini-2.5-flash") == "gemini-2.5-flash"

    class R:
        def __init__(self, code, data):
            self.status_code, self._d, self.text = code, data, json.dumps(data)

        def json(self):
            return self._d
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(url)
        if "gemini-2.5-flash:" in url:
            return R(404, {"error": {"code": 404, "message": "This model models/gemini-2.5-flash is no longer available to new users."}})
        return R(200, {"candidates": [{"content": {"parts": [{"text": '{"scores": [3, 9], "best": 2}'}]}}]})
    monkeypatch.setattr(llm.httpx, "post", post)
    img = tmp_path / "c.jpg"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(img)
    with pytest.raises(llm.ModelGone) as e:
        llm.gemini_pick("k", "gemini-2.5-flash", img, "p")
    assert "no longer available" in str(e.value) and "\n" not in str(e.value)
    assert llm.resolve_gemini_model("k", "gemini-2.5-flash") == "gemini-3-flash"
    res, _ = llm.gemini_pick("k", "gemini-3-flash", img, "p")
    assert res["best"] == 2
    llm._GEMINI_MODELS.clear(); llm._GONE.clear()


def test_overlays_parallel_dedupe_and_fallback(tmp_path, monkeypatch):
    calls = []

    def fake_webm(ov, style, out, work):
        calls.append(ov["line1"])
        Path(out).write_bytes(b"webm")
        return out
    monkeypatch.setattr(overlays, "hyperframes_available", lambda: True)
    monkeypatch.setattr(overlays, "render_webm", fake_webm)
    plan = {"1": {"kind": "time", "line1": "17:14", "line2": "", "duration": 2.0},
            "2": {"kind": "time", "line1": "17:14", "line2": "", "duration": 2.0},
            "3": {"kind": "measure", "line1": "32 mi", "line2": "51.5 km", "duration": 2.0}}
    out = overlays.render_all(plan, STYLE, tmp_path / "ov", lambda *a, **k: None)
    assert sorted(calls) == ["17:14", "32 mi"] and all(v["type"] == "webm" for v in out.values())
    monkeypatch.setattr(overlays, "render_webm", lambda *a: None)
    plan2 = {"4": {"kind": "label", "line1": "TOR-M2", "line2": "", "duration": 2.0}}
    out2 = overlays.render_all(plan2, STYLE, tmp_path / "ov2", lambda *a, **k: None)
    assert out2["4"]["type"] == "png" and Path(out2["4"]["path"]).exists()


def test_video_cost_adds_up_and_prices_voice_credits(tmp_path):
    from warstudio import costs
    costs.add(tmp_path, "voice", credits=2155, characters=1938)
    costs.add(tmp_path, "brain", usd=0.052, tokens_in=1000, tokens_out=500)
    costs.add(tmp_path, "rerank", usd=0.0, calls=0)
    costs.add(tmp_path, "brain", usd=0.010)                       # re-run from specs: that money was spent too
    s = costs.summary(tmp_path, {"ai33_usd_per_1k_credits": 0})
    assert s["total_usd"] == 0.062 and not s["voice_priced"]
    assert s["parts"]["brain"]["runs"] == 2 and s["parts"]["voice"]["credits"] == 2155
    line = costs.line(s)
    assert "$0.062 + voice credits" in line and "2155 ai33 credits" in line and "script planner $0.062" in line
    s = costs.summary(tmp_path, {"ai33_usd_per_1k_credits": 0.3})   # price set in Settings
    assert s["voice_priced"] and abs(s["total_usd"] - (0.062 + 2.155 * 0.3)) < 1e-6
    assert "voice $0.646 (2155 ai33 credits)" in costs.line(s)
    assert costs.summary(tmp_path / "none", {})["total_usd"] == 0


def test_more_overlays_money_percent_counts_years():
    from warstudio.overlays import detect
    assert detect("a deal worth $4.5 billion")["line1"] == "$4.5 BILLION"
    assert detect("roughly one fifth of all the oil")["line1"] == "20%"
    assert detect("Iran has 5,000 naval mines")["line2"] == "NAVAL MINES"
    assert detect("In April 1988, after the mine")["line1"] == "APR 1988"
    assert detect("since 2019 the ships")["type"] == "year"
    assert detect("thirty miles off Djibouti") is None or detect("thirty miles off Djibouti")["kind"] != "label"


def test_voice_lock_one_video_at_a_time():
    from warstudio import pipeline
    assert pipeline.VOICE_LOCK.acquire(blocking=False)
    assert not pipeline.VOICE_LOCK.acquire(blocking=False)
    pipeline.VOICE_LOCK.release()


def test_schedule_a_video(tmp_path, monkeypatch):
    import time as _t
    from fastapi.testclient import TestClient
    from warstudio import server, pipeline, config
    monkeypatch.setattr(server, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PROJECTS_DIR", tmp_path)
    put = []
    monkeypatch.setattr(server.JOBS, "put", lambda x: put.append(x))
    c = TestClient(server.app)
    r = c.post("/api/jobs", json={"title": "T", "script": "Hello there. One more line.", "voice_id": "v",
                                  "dummy": True, "sources": ["library"], "start_in_minutes": 60})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["scheduled_for"] > _t.time() + 3500 and not put
    st = json.loads((tmp_path / d["id"] / "status.json").read_text())
    assert st["state"] == "scheduled"


def test_stop_video_and_stop_all(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from warstudio import server, pipeline
    monkeypatch.setattr(server, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PROJECTS_DIR", tmp_path)
    monkeypatch.setattr(server.JOBS, "put", lambda x: None)
    c = TestClient(server.app)
    ids = []
    for mins in (0, 60, 0):
        r = c.post("/api/jobs", json={"title": "T", "script": "One line. Two lines.", "voice_id": "v", "dummy": True,
                                      "sources": ["library"], "start_in_minutes": mins})
        ids.append(r.json()["id"])
    # a running video: the logger stops it at its next line
    log = pipeline.Logger(tmp_path / ids[0])
    log.set(state="running")
    assert c.post(f"/api/jobs/{ids[0]}/stop").status_code == 200
    try:
        log("next line")
        raise AssertionError("should have stopped")
    except pipeline.VideoStopped:
        pass
    # stop all: the scheduled one is cancelled, the queued one gets a stop flag
    assert c.post("/api/jobs/stop_all").json()["stopped"] >= 2
    assert json.loads((tmp_path / ids[1] / "status.json").read_text())["state"] == "stopped"
    assert (tmp_path / ids[2] / "STOP").exists()
