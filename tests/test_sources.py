from warstudio.steps import search, fetch, rank


def test_only_library_and_dvids_remain():
    assert set(search.SOURCES) == {"dvids"}
    for gone in ("search_pexels", "search_pixabay", "search_commons"):
        assert not hasattr(search, gone)


def test_dvids_asks_for_big_thumbnails(monkeypatch):
    seen = {}
    def fake(source, url, params, headers=None, ttl=0):
        seen.update(params)
        return {"results": []}
    monkeypatch.setattr(search, "_cached_get", fake)
    search.search_dvids("f-35", "k", 12)
    assert seen["thumb_width"] >= 640 and seen["sort"] == "score"


def test_source_ready():
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
    assert set(called) == {"dvids"}                       # old jobs that picked pexels: ignored
    # nothing picked = every available source
    called.clear()
    (tmp_path / "candidates.json").unlink()
    search.run_step(tmp_path, keys, {}, style, lambda m, **k: None)
    assert set(called) == {"dvids"}


def test_dvids_falls_back_to_plain_query_and_never_caches_empty(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(search, "CACHE", tmp_path)
    seen = []

    class R:
        status_code = 200

        def __init__(self, d):
            self.d = d

        def json(self):
            return self.d

        def raise_for_status(self):
            pass

    def get(url, params=None, **kw):
        seen.append(dict(params))
        if "thumb_width" in params and params["q"] == "F-15E Strike Eagle takeoff":
            return R({"results": []})
        return R({"results": [{"id": "video:1", "title": "F-15E B-Roll", "thumbnail": "https://t/1_122x92.jpg",
                               "duration": 120}]})
    monkeypatch.setattr(httpx, "get", get)
    res = search.search_dvids("F-15E Strike Eagle takeoff", "k", 10)
    assert [r["id"] for r in res] == ["video:1"] and len(seen) == 2
    assert "aspect_ratio" not in seen[0] and "thumb_width" not in seen[1]
    assert len(list(tmp_path.glob("dvids_*.json"))) == 1        # only the non-empty answer is cached
    search.search_dvids("F-15E Strike Eagle takeoff", "k", 10)
    assert len(seen) == 3                                         # empty answer asked again, full one from cache


def test_api_error_payload_is_raised(tmp_path, monkeypatch):
    import httpx
    import pytest
    monkeypatch.setattr(search, "CACHE", tmp_path)

    class R:
        status_code = 200

        def json(self):
            return {"errors": ["invalid parameter: aspect_ratio"]}

        def raise_for_status(self):
            pass
    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="aspect_ratio"):
        search.search_dvids("f-35", "k", 5)


def test_search_stops_instead_of_rendering_black(tmp_path, monkeypatch):
    import pytest
    from warstudio.util import write_json
    write_json(tmp_path / "specs.json", [{"id": 0, "queries": ["x"], "tiers": ["category"], "footage_class": "real"}])
    monkeypatch.setattr(search, "call_source", lambda *a, **k: [])
    logs = []
    with pytest.raises(RuntimeError, match="found nothing for any beat"):
        search.run_step(tmp_path, {"DVIDS_API_KEY": "d"}, {"job_sources": ["dvids"]},
                        {"footage": {"source_order": ["dvids"]}}, lambda m, **k: logs.append(m))
    assert any("WARNING: dvids returned no results" in m for m in logs)
