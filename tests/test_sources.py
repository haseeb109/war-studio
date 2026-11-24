from warstudio.steps import search, fetch, rank


def test_commons_license_filter():
    ok = search.commons_license_ok
    assert ok("Public domain") and ok("CC0") and ok("CC BY 4.0") and ok("PD-USGov-Military")
    assert not ok("CC BY-SA 4.0") and ok("CC BY-SA 4.0", allow_sa=True)
    assert not ok("CC BY-NC 2.0") and not ok("CC BY-ND 3.0") and not ok("All rights reserved") and not ok("")


def test_search_commons_parses_and_filters(monkeypatch):
    def fake(source, url, params, headers=None, ttl=0):
        assert params["iiurlwidth"] == 640 and "filetype:video" in params["gsrsearch"]
        mk = lambda pid, idx, lic, mime="video/webm": {"pageid": pid, "index": idx, "title": f"File:Clip {pid}.webm",
              "imageinfo": [{"url": f"https://u/{pid}.webm", "thumburl": f"https://t/{pid}.jpg", "mime": mime,
                             "duration": 42.5, "descriptionurl": f"https://c/{pid}",
                             "extmetadata": {"LicenseShortName": {"value": lic},
                                             "Artist": {"value": "<a href='x'>mil.ru</a>"}}}]}
        return {"query": {"pages": {"1": mk(1, 2, "CC BY 4.0"), "2": mk(2, 1, "CC BY-SA 3.0"),
                                    "3": mk(3, 3, "CC BY-NC 2.0"), "4": mk(4, 4, "CC BY 4.0", "image/jpeg")}}}
    monkeypatch.setattr(search, "_cached_get", fake)
    res = search.search_commons("su-27", None, 10)
    assert [r["id"] for r in res] == ["1"]
    r = res[0]
    assert r["source"] == "commons" and r["video_url"].endswith(".webm") and r["duration"] == 42.5
    assert "mil.ru" in r["credit"] and "CC BY 4.0" in r["credit"] and r["license"] == "CC BY 4.0"
    assert [x["id"] for x in search.search_commons("su-27", None, 10, allow_sa=True)] == ["2", "1"]


def test_dvids_asks_for_big_thumbnails(monkeypatch):
    seen = {}
    def fake(source, url, params, headers=None, ttl=0):
        seen.update(params)
        return {"results": []}
    monkeypatch.setattr(search, "_cached_get", fake)
    search.search_dvids("f-35", "k", 12)
    assert seen["thumb_width"] >= 640 and seen["sort"] == "score"


def test_source_ready():
    assert search.source_ready("commons", {}, {}) is True
    assert search.source_ready("commons", {}, {"use_commons": "false"}) is False
    assert search.source_ready("dvids", {}, {}) is False
    assert search.source_ready("dvids", {"DVIDS_API_KEY": "x"}, {}) is True


def test_choose_start_respects_time_start():
    s = fetch.choose_start(120, 5, [], min_start=20)
    assert s >= 20
    s2 = fetch.choose_start(120, 5, [s], min_start=20)
    assert s2 >= 20 and abs(s2 - s) > 5
    assert fetch.choose_start(0, 5, [], min_start=8) == 8
    assert fetch.choose_start(60, 5, []) == 18.0


def test_diverse_top_keeps_seats_for_each_source():
    c = [{"source": "pixabay", "clip_score": 0.9 - i * 0.01, "id": i} for i in range(8)]
    c += [{"source": "dvids", "clip_score": 0.5, "id": "d1"}, {"source": "dvids", "clip_score": 0.4, "id": "d2"}]
    top = rank.diverse_top(c, 5)
    assert len(top) == 5 and sum(1 for t in top if t["source"] == "dvids") == 2
    assert top[0]["clip_score"] == 0.9


def test_search_uses_only_sources_picked_for_the_video(tmp_path, monkeypatch):
    from warstudio.util import write_json
    write_json(tmp_path / "specs.json", [{"id": 0, "queries": ["f-35 takeoff", "fighter jet"],
                                          "tiers": ["exact", "category"], "footage_class": "real"}])
    called = []

    def fake(src, q, keys, n, settings=None):
        called.append(src)
        return [{"source": src, "id": f"{src}{q}", "preview_url": "x"}]
    monkeypatch.setattr(search, "call_source", fake)
    style = {"footage": {"source_order": ["dvids", "commons", "pexels", "pixabay"]}}
    keys = {"DVIDS_API_KEY": "d", "PIXABAY_API_KEY": "p"}
    logs = []
    out = search.run_step(tmp_path, keys, {"job_sources": ["dvids"]}, style, lambda m, **k: logs.append(m))
    assert set(called) == {"dvids"} and all(c["source"] == "dvids" for c in out["0"])
    assert any("searching footage on: dvids" in m for m in logs)
    # a picked source without a key is reported and skipped
    called.clear()
    (tmp_path / "candidates.json").unlink()
    search.run_step(tmp_path, keys, {"job_sources": ["dvids", "pexels"]}, style, lambda m, **k: logs.append(m))
    assert set(called) == {"dvids"} and any("skipped: pexels" in m for m in logs)
    # nothing picked = every available source (commons needs no key)
    called.clear()
    (tmp_path / "candidates.json").unlink()
    search.run_step(tmp_path, keys, {}, style, lambda m, **k: None)
    assert set(called) == {"dvids", "commons", "pixabay"}
