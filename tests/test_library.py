import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from warstudio import library
from warstudio.util import read_json, write_json


def mkvid(path, parts):
    """parts: list of lavfi sources, 3 s each, concatenated -> hard cuts every 3 s."""
    args, fl = [], ""
    for i, src in enumerate(parts):
        args += ["-f", "lavfi", "-i", f"{src}{':' if '=' in src else '='}s=320x180:r=25:d=3"]
        fl += f"[{i}:v]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, "-filter_complex",
                    f"{fl}concat=n={len(parts)}:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast",
                    str(path)], check=True)
    return path


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = {"library_dir": str(tmp_path / "lib"), "encoder": "libx264", "library_captions": "off",
         "clip_model": "x", "clip_pretrained": "x"}
    monkeypatch.setattr(library, "STATUS_FILE", tmp_path / "status.json")

    def fake_img(settings, paths):     # deterministic fake CLIP: vector from the file name's first letter
        out = []
        for p in paths:
            v = np.zeros(512, np.float32)
            v[ord(Path(p).name[0]) % 512] = 1
            out.append(v)
        return np.stack(out)
    monkeypatch.setattr(library, "clip_image_vecs", fake_img)
    return s


def test_plan_shots():
    s = library.plan_shots([3.0, 6.0, 30.0], 40.0, min_start=0.0)
    assert s[0][1] == 8.0                       # 6..30 is one long take -> windows of 8 s
    assert all(3.0 <= L <= 8.0 for _, L in s)
    # opening slate (time_start) and the tail of a long video are never used
    s2 = library.plan_shots([], 300.0, min_start=20.0)
    assert all(st >= 20 and st + L <= 300 - 8 for st, L in s2) and len(s2) == 2


def test_detect_cuts_real_ffmpeg(tmp_path):
    v = mkvid(tmp_path / "v.mp4", ["color=c=red", "testsrc2", "color=c=blue"])
    cuts = library.detect_cuts(v)
    assert any(abs(c - 3) < 0.3 for c in cuts) and any(abs(c - 6) < 0.3 for c in cuts)


def test_key_tokens():
    assert {"f-15e", "f15e"} <= library.key_tokens("F-15E Strike Eagle")
    assert library.key_tokens("USS Carney") == {"carney"}
    assert library.key_tokens("") == set()


def _add(settings, name, entity, caption, origin="build"):
    d = Path(settings["library_dir"])
    d.mkdir(parents=True, exist_ok=True)
    f, kf = d / f"{name}.mp4", d / f"{name}.jpg"
    f.write_bytes(b"x")
    from PIL import Image
    seed = sum(map(ord, name))
    Image.fromarray((np.random.default_rng(seed).random((90, 160, 3)) * 255).astype(np.uint8)).save(kf)
    b = library.Builder({}, settings, lambda m: None)
    return b._insert(f, kf, source="dvids", src_id=name, src_start=0, length=6, title=caption, cat_name="c",
                     cat_entity=entity, credit="DVIDS", license="PD", page_url="", origin=origin,
                     caption={"caption": caption, "entity": entity, "tags": []})


def test_named_system_only_matches_that_system(settings):
    _add(settings, "a_f15", "F-15E Strike Eagle", "F-15E taking off at night")
    _add(settings, "b_f16", "F-16 Fighting Falcon", "F-16 banking left")
    idx = library.LibraryIndex(settings)
    v = np.zeros(512, np.float32)
    v[ord("b")] = 1                              # text vector closest to the F-16 shot...
    res = idx.match(v, "F-15E Strike Eagle", "F-15E takeoff")
    assert [r["id"] for r in res] == ["1"] and res[0]["tier"] == "exact"   # ...but the beat names the F-15E
    gen = idx.match(v, "", "fighter jet banking")
    assert gen[0]["video_url"].endswith("b_f16.mp4") and gen[0]["tier"] == "category"


def test_usage_penalty_and_record(settings, tmp_path):
    _add(settings, "a_x", "", "fighter jet")
    proj = tmp_path / "p1"
    proj.mkdir()
    assert library.record_uses(proj, {"0": {"source": "library", "id": "1"}}, settings) == 1
    idx = library.LibraryIndex(settings)
    v = np.zeros(512, np.float32)
    v[ord("a")] = 1
    r = idx.match(v, "", "jet")[0]
    assert r["lib_penalty"] > 0 and r["lib_score"] < r["clip_sim"]


def test_search_uses_library_first(settings, tmp_path, monkeypatch):
    from warstudio.steps import search
    _add(settings, "a_f15", "F-15E Strike Eagle", "F-15E taking off")
    write_json(tmp_path / "specs.json", [
        {"id": 0, "queries": ["F-15E takeoff"], "tiers": ["exact"], "footage_class": "real", "entity": "F-15E Strike Eagle",
         "visual": "F-15E taking off"},
        {"id": 1, "queries": ["Su-34"], "tiers": ["exact"], "footage_class": "real", "entity": "Su-34", "visual": "Su-34"}])
    monkeypatch.setattr(library, "clip_text_vecs",
                        lambda s, texts: np.stack([np.eye(512, dtype=np.float32)[ord("a")] for _ in texts]))
    asked = []
    monkeypatch.setattr(search, "call_source", lambda src, q, keys, n, settings=None: asked.append(q) or
                        [{"source": src, "id": q, "preview_url": "x"}])
    logs = []
    out = search.run_step(tmp_path, {"DVIDS_API_KEY": "d"}, dict(settings, job_sources=["library", "dvids"]),
                          {"footage": {"source_order": ["dvids"]}}, lambda m, **k: logs.append(m))
    assert out["0"][0]["source"] == "library" and out["1"][0]["source"] == "dvids"
    assert asked == ["Su-34"]                    # only the beat the library could not cover went online
    assert any("library: 1/2 beats matched" in m for m in logs)


def test_rank_skips_rerank_for_library_beats(settings, tmp_path, monkeypatch):
    from warstudio.steps import rank
    _add(settings, "a_x", "", "fighter jet")
    _add(settings, "b_y", "", "fighter jet 2")
    d = Path(settings["library_dir"])
    from PIL import Image
    for n in ("a_x", "b_y"):
        Image.new("RGB", (64, 36)).save(d / f"{n}.jpg")
    cand = [{"source": "library", "id": str(i + 1), "title": "jet", "preview_url": "", "preview_file": str(d / f"{n}.jpg"),
             "video_url": str(d / f"{n}.mp4"), "duration": 6, "tier": "category"} for i, n in enumerate(("a_x", "b_y"))]
    write_json(tmp_path / "beats.json", [{"id": 0, "sid": 0, "text": "t", "start": 0, "end": 2}])
    write_json(tmp_path / "specs.json", [{"id": 0, "queries": ["q"], "visual": "jet", "footage_class": "real", "scene": ""}])
    write_json(tmp_path / "candidates.json", {"0": cand})
    monkeypatch.setattr(rank, "clip_scores", lambda s, t, ps: [0.3, 0.2][:len(ps)])

    class RR:
        active, chain, label, stats = "gemini", ["gemini"], "x", {"ok": 0}

        def pick(self, *a, **k):
            raise AssertionError("no rerank for library beats")

        def summary(self):
            return ""
    monkeypatch.setattr(rank, "Reranker", lambda *a, **k: RR())
    logs = []
    picks = rank.run_step(tmp_path, {}, settings, {"footage": {"same_source_cooldown_seconds": 90}},
                          lambda m, **k: logs.append(m))
    assert picks["0"]["source"] == "library" and picks["0"]["id"] == "1"
    assert any("1 beats come from the library" in m for m in logs)


def test_builder_process_video_end_to_end(settings, tmp_path, monkeypatch):
    src = mkvid(tmp_path / "src.mp4", ["color=c=red", "testsrc2", "testsrc", "color=c=0x305070", "testsrc2"])
    b = library.Builder({"DVIDS_API_KEY": "k"}, dict(settings, library_shots_per_video=2), lambda m: None)
    monkeypatch.setattr(b, "_dvids_asset", lambda i: ({"time_start": 0}, [
        {"src": str(src), "height": 180, "width": 320}, {"src": str(src) + "?hd", "height": 180, "width": 320}]))
    monkeypatch.setattr(library, "pick_files", lambda files: (str(src), str(src)))
    monkeypatch.setattr(library, "download", lambda url, dest, timeout=0: shutil.copy(url, dest))
    monkeypatch.setattr(library, "plan_shots", lambda cuts, dur, ms: [(0.3, 2.4), (3.3, 2.4), (6.3, 2.4), (9.3, 2.4)])
    n = b.process_video(("Test cat", "q", "F-15E"), {"id": "video:1", "title": "F-15E B-Roll"})
    # the solid red shot (0.3 s) is a blank card -> rejected; two clean shots kept
    assert n == 2
    rows = library.LibraryIndex(settings).rows
    assert len(rows) == 2 and all(Path(r["file"]).exists() and r["entity"] == "F-15E" for r in rows)
    assert library.stats(settings)["videos"] == {"done": 1}
    assert b.process_video.__self__._seen("video:1")


def test_learn_from_project(settings, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    clip = mkvid(proj / "seg.mp4", ["testsrc2"])
    clip2 = mkvid(proj / "seg2.mp4", ["testsrc"])
    clip3 = mkvid(proj / "seg3.mp4", ["rgbtestsrc"])
    write_json(proj / "picks.json", {"0": {"gemini_score": 8}, "1": {"gemini_score": 4}, "2": {"gemini_score": 9}})
    write_json(proj / "fetched.json", {
        "0": {"file": str(clip), "source": "dvids", "id": "video:9", "start": 12.0, "length": 3.0, "problems": []},
        "1": {"file": str(clip2), "source": "dvids", "id": "video:8", "start": 1.0, "length": 3.0, "problems": []},
        "2": {"file": str(clip3), "source": "pixabay", "id": "77", "start": 1.0, "length": 3.0, "problems": []}})
    write_json(proj / "specs.json", [{"id": 0, "visual": "F-15E at night", "entity": "F-15E", "queries": ["F-15E"]},
                                     {"id": 1, "visual": "x", "entity": "", "queries": []}])
    b = library.Builder({}, settings, lambda m: None)
    # every clean DVIDS clip (score 8 and 4); the Pixabay one is not learned (DVIDS only)
    assert b.learn_from_project(proj) == 2
    st = library.stats(settings)
    assert st["shots"] == 2 and st["learned_from_videos"] == 2
    assert library.Builder({}, settings, lambda m: None).learn_from_project(proj) == 0     # not added twice


def test_library_api(settings, monkeypatch):
    from fastapi.testclient import TestClient
    from warstudio import server
    monkeypatch.setattr(server, "load_settings", lambda: settings)
    r = TestClient(server.app).get("/api/library")
    assert r.status_code == 200 and "F-15E Strike Eagle" in r.json()["categories"]


def test_duplicates_are_skipped(settings, tmp_path, monkeypatch):
    from PIL import Image
    d = Path(settings["library_dir"])
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    base = (rng.random((180, 320, 3)) * 255).astype(np.uint8)
    other = (rng.random((180, 320, 3)) * 255).astype(np.uint8)

    def add(name, arr, src_id, start, b):
        f, kf = d / f"{name}.mp4", d / f"{name}.jpg"
        f.write_bytes(b"x")
        Image.fromarray(arr).save(kf, quality=95)
        return b._insert(f, kf, source="dvids", src_id=src_id, src_start=start, length=6, title="t", cat_name="c",
                         cat_entity="", credit="", license="", page_url="", origin="build", caption={"caption": "x"})
    b = library.Builder({}, settings, lambda m: None)
    assert add("a1", base, "v1", 10, b)
    # same picture re-posted in another DVIDS package (slightly re-encoded) -> duplicate
    assert not add("a2", np.clip(base.astype(int) + 3, 0, 255).astype(np.uint8), "v2", 50, b)
    # same source video, overlapping time range -> duplicate even with a different picture
    assert not add("a3", other, "v1", 12, b)
    # a different picture from another video -> kept
    assert add("a4", other, "v3", 0, b)
    assert b.dupes == 2 and library.stats(settings)["shots"] == 2
    assert not (d / "a2.mp4").exists()          # duplicate files are removed


def test_candidates_walk_result_pages(settings, monkeypatch):
    b = library.Builder({"DVIDS_API_KEY": "k"}, settings, lambda m: None)
    pages = {1: [{"id": f"video:{i}", "title": "F-15E B-Roll", "duration": 60} for i in range(10)],
             2: [{"id": f"video:{i}", "title": "F-15E", "duration": 60} for i in range(10, 25)], 3: []}
    monkeypatch.setattr(b, "_dvids_search", lambda q, n=50, page=1: pages.get(page, []))
    first = b.candidates("F-15E", want=12)
    assert len(first) == 25 and b.pages["F-15E"] == 3     # needed page 2 to reach 12 new videos
    assert b.candidates("F-15E", want=12) == [] and b.pages["F-15E"] > 3   # no more pages
