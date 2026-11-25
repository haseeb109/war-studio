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
    monkeypatch.setattr(library, "pick_files", lambda files: (str(src), str(src), 1080))
    monkeypatch.setattr(library, "download_resume", lambda url, dest, *a, **k: shutil.copy(url, dest))
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


# ---------------------------------------------------------------- v0.8.2 improvements
def test_detect_action():
    assert library.detect_action("two F-15E Strike Eagles lift off from a desert air base") == "takeoff"
    assert library.detect_action("they meet a KC-135 tanker and the boom operator lowers the boom") == "refueling"
    assert library.detect_action("the first missile leaves the rail") == "launch"
    assert library.detect_action("operators watch the thermal feed") == "console"
    assert library.detect_action("a calm harbor") == ""


def _add_full(settings, name, entity, caption, action="", seed=None):
    from PIL import Image
    d = Path(settings["library_dir"])
    d.mkdir(parents=True, exist_ok=True)
    f, kf = d / f"{name}.mp4", d / f"{name}.jpg"
    f.write_bytes(b"x")
    Image.fromarray((np.random.default_rng(seed or sum(map(ord, name))).random((90, 160, 3)) * 255).astype(np.uint8)).save(kf)
    return library.Builder({}, settings, lambda m: None)._insert(
        f, kf, source="dvids", src_id=name, src_start=0, length=6, title=caption, cat_name="c", cat_entity=entity,
        credit="", license="", page_url="", origin="build", caption={"caption": caption, "entity": entity, "action": action})


def test_action_tag_prefers_the_right_moment(settings):
    _add_full(settings, "a_taxi", "F-15E", "F-15E parked on the ramp", "taxi")
    _add_full(settings, "a_take", "F-15E", "F-15E lifting off with afterburner", "takeoff")
    idx = library.LibraryIndex(settings)
    v = np.zeros(512, np.float32)
    v[ord("a")] = 1                       # both shots have the same CLIP similarity in the fake model
    top = idx.match(v, "F-15E", "two F-15E lift off from the base")
    assert top[0]["video_url"].endswith("a_take.mp4") and top[0]["action"] == "takeoff"


def test_rotation_rests_recent_shots(settings, tmp_path):
    _add_full(settings, "a_one", "", "fighter jet")
    _add_full(settings, "a_two", "", "fighter jet again")
    for i in range(3):
        library.record_uses(tmp_path / f"old{i}", {"0": {"source": "library", "id": "1"}}, settings)
    idx = library.LibraryIndex(dict(settings, library_rotation_videos=5))
    v = np.zeros(512, np.float32)
    v[ord("a")] = 1
    assert 1 in idx.rotated
    assert [r["id"] for r in idx.match(v, "", "jet")] == ["2"]          # shot 1 rests
    library.delete_shot(settings, 2)
    idx2 = library.LibraryIndex(dict(settings, library_rotation_videos=5))
    only = idx2.match(v, "", "jet")
    assert [r["id"] for r in only] == ["1"] and only[0]["lib_penalty"] >= 0.05   # nothing else fits: used, penalised
    # the project being made right now does not rest its own shots (resume / re-run)
    idx3 = library.LibraryIndex(dict(settings, library_rotation_videos=1), exclude_project="old2")
    assert 1 in idx3.rotated            # still used by old1 (the newest other project)


def test_favorite_and_delete_never_readded(settings, tmp_path):
    _add_full(settings, "a_fav", "", "jet")
    library.set_favorite(settings, 1, True)
    lst = library.list_shots(settings, favorites=True)
    assert lst["total"] == 1 and lst["shots"][0]["favorite"] == 1
    assert library.delete_shot(settings, 1)
    assert not (Path(settings["library_dir"]) / "a_fav.mp4").exists()
    # the same picture comes back later (e.g. learned from a video): treated as a duplicate
    assert not _add_full(settings, "a_fav2", "", "jet", seed=sum(map(ord, "a_fav")))
    assert library.stats(settings)["deleted"] == 1


def test_three_frame_vector(settings, tmp_path):
    v = mkvid(tmp_path / "m.mp4", ["testsrc2", "testsrc"])
    vec, n = library.shot_vec(settings, v, 6.0, None)
    assert n == 3 and abs(float(np.linalg.norm(vec)) - 1) < 1e-5


def test_low_resolution_sources_are_skipped(settings, tmp_path, monkeypatch):
    b = library.Builder({"DVIDS_API_KEY": "k"}, settings, lambda m: None)
    monkeypatch.setattr(b, "_dvids_asset", lambda i: ({}, [{"src": "a.mp4", "height": 480}]))
    assert library.pick_files([{"src": "a.mp4", "height": 480}, {"src": "b.mp4", "height": 360}])[2] == 480
    assert b.process_video(("c", "q", ""), {"id": "video:lowres", "title": "t"}) == 0
    assert library.stats(settings)["videos"] == {"skipped": 1}


def test_best_lite_model(monkeypatch):
    import warstudio.llm as llm
    monkeypatch.setattr(llm, "gemini_models", lambda k: ["gemini-3.8-flash", "gemini-3.1-flash-lite",
                                                        "gemini-3.5-flash-lite", "gemini-3.9-flash-lite-preview"])
    assert library.best_lite_model("k") == "gemini-3.5-flash-lite"


def test_backup_index_daily_and_keeps_seven(settings):
    _add_full(settings, "a_b", "", "jet")
    bdir = Path(settings["library_dir"]) / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    for i in range(9):
        (bdir / f"index-2020010{i}.db").write_bytes(b"x")
    dst = library.backup_index(settings)
    assert dst.exists() and dst.stat().st_size > 100
    assert len(list(bdir.glob("index-*.db"))) == 7 and dst in list(bdir.glob("index-*.db"))


def test_review_search_shows_library_first(settings, monkeypatch):
    import warstudio.config as config
    from warstudio import edits
    _add_full(settings, "a_rev", "F-15E", "F-15E takeoff")
    monkeypatch.setattr(config, "load_settings", lambda: dict(settings, use_commons=False))
    monkeypatch.setattr(edits, "load_keys", lambda: {})
    monkeypatch.setattr(library, "clip_text_vecs", lambda s, t: np.stack([np.eye(512, dtype=np.float32)[ord("a")]]))
    res = edits.search("proj", "f-15e takeoff")
    assert res and res[0]["source"] == "library" and res[0]["preview_url"] == "/api/library/thumb/1"


def test_browser_api(settings, monkeypatch):
    from fastapi.testclient import TestClient
    from warstudio import server
    _add_full(settings, "a_api", "", "jet over sea")
    monkeypatch.setattr(server, "load_settings", lambda: settings)
    c = TestClient(server.app)
    assert c.get("/api/library/shots", params={"q": "sea"}).json()["total"] == 1
    assert c.get("/api/library/thumb/1").status_code == 200
    assert c.post("/api/library/shots/1/favorite", json={"on": True}).json()["ok"]
    assert c.post("/api/library/shots/1/delete").json()["ok"]
    assert c.get("/api/library/thumb/1").status_code == 404


def test_library_only_every_beat_gets_a_shot(settings, tmp_path, monkeypatch):
    """Library-only video: weak matches, maps and systems the library lacks still get a library shot (holding the
    previous clip would break the 3 s pacing rule)."""
    from warstudio.steps import search
    _add(settings, "a_f15", "F-15E Strike Eagle", "F-15E taking off")
    write_json(tmp_path / "specs.json", [
        {"id": 0, "queries": ["F-15E takeoff"], "footage_class": "real", "entity": "F-15E Strike Eagle", "visual": "F-15E taking off"},
        {"id": 1, "queries": ["Su-34"], "footage_class": "real", "entity": "Su-34", "visual": "Su-34 bomber"},
        {"id": 2, "queries": ["map"], "footage_class": "map", "entity": "", "visual": "map of the red sea"}])
    v = np.zeros(512, dtype=np.float32)
    v[ord("a")], v[1] = 0.2, 0.98                                # cosine 0.2 to the shot: under 0.24, over 0.15
    v /= np.linalg.norm(v)
    monkeypatch.setattr(library, "clip_text_vecs", lambda s, texts: np.stack([v for _ in texts]))
    monkeypatch.setattr(search, "call_source", lambda *a, **k: pytest.fail("library only must not go online"))
    logs = []
    out = search.run_step(tmp_path, {}, dict(settings, job_sources=["library"]), {"footage": {}},
                          lambda m, **k: logs.append(m))
    assert all(out[str(i)] and out[str(i)][0]["source"] == "library" for i in range(3))
    assert any("stand-in" in m for m in logs) and any("weaker matches" in m for m in logs)


def test_footage_mix_by_screen_time():
    from warstudio.steps import qc
    beats = [{"id": 0, "start": 0.0, "end": 2.0}, {"id": 1, "start": 2.5, "end": 4.0}, {"id": 2, "start": 4.0, "end": 9.0}]
    fetched = {"0": {"source": "library"}, "1": {"source": "dvids"}, "2": {"source": "library"}}
    m = qc.source_mix(beats, fetched, total=10.0)
    assert m["order"] == ["library", "dvids"] and m["shots"] == {"library": 2, "dvids": 1}
    assert m["seconds"] == {"library": 8.5, "dvids": 1.5} and m["percent"] == {"library": 85.0, "dvids": 15.0}
    line = qc.mix_line(m)
    assert "My Library 85% (2 shots, 0m 08s)" in line and "DVIDS 15%" in line


def _git(cwd, *a):
    r = subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(cwd), *a],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_index_recovered_after_update_stashed_it(tmp_path, monkeypatch):
    """v0.8.0-v0.9.3 had library/shots/index.db in git: the Update button stashed the real index and put the
    empty committed one back. The app finds the stashed copy at start and merges it back."""
    import sqlite3
    repo = tmp_path / "war-studio"
    shots = repo / "library" / "shots"
    shots.mkdir(parents=True)
    s = {"library_dir": str(shots)}
    _git(tmp_path, "init", "-q", str(repo))
    library._db(s).close()                                     # the empty index that was committed by mistake
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v0.8.0")
    from PIL import Image
    with sqlite3.connect(shots / "index.db") as con:           # the overnight build fills it
        for i in range(30):
            f, kf = shots / f"dvids_{i}.mp4", shots / f"dvids_{i}.jpg"
            f.write_bytes(b"x")
            Image.new("RGB", (16, 9)).save(kf)
            con.execute("INSERT INTO shots(source, src_id, src_start, file, keyframe, caption, emb, category) "
                        "VALUES('dvids', ?, 0, ?, ?, 'jet', ?, 'F-15E')",
                        (f"v{i}", str(f), str(kf), np.ones(512, np.float32).tobytes()))
        con.execute("INSERT INTO videos(source, src_id, status) VALUES('dvids', 'video:1', 'done')")
    _git(repo, "stash", "push", "-q", "-m", "warstudio auto-update")          # what the Update button did
    assert library.index_health(s) == (0, 30)
    # the next version untracks the file; pulling it deletes the working copy
    _git(repo, "rm", "-q", "--cached", "library/shots/index.db")
    _git(repo, "commit", "-q", "-m", "v0.9.4")
    (shots / "index.db").unlink()
    logs = []
    assert library.recover_index(s, logs.append, root=repo) == 30
    assert library.index_health(s) == (30, 30) and library.stats(s)["videos"] == {"done": 1}
    assert len(library.LibraryIndex(s)) == 30
    assert any("recovered: 30 of 30" in m and "git stash@{0}" in m for m in logs)
    assert library.recover_index(s, logs.append, root=repo) == 0               # healthy now: nothing to do
    assert list((shots / "backups").glob("index-before-recover-*.db"))


def _media(path, sec, tone=None, size="320x180"):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={sec}"]
    if tone:
        cmd += ["-f", "lavfi", "-i", f"sine=f={tone}:d={sec}", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_add_sound_to_old_shots(settings, tmp_path, monkeypatch):
    """Shots cut before v0.9.0 have no sound: the same seconds of the source's audio are muxed in, the picture is
    copied untouched; silent sources are remembered; shots that already have sound are only marked."""
    import sqlite3
    d = Path(settings["library_dir"])
    d.mkdir(parents=True, exist_ok=True)
    src_loud, src_mute = _media(tmp_path / "src1.mp4", 12, tone=440), _media(tmp_path / "src2.mp4", 12)
    shots = {"a": ("video:1", 2.0), "b": ("video:1", 7.0), "c": ("video:2", 3.0), "d": ("video:3", 1.0)}
    for name, (vid, st) in shots.items():
        _add(settings, name, "F-15E", "jet")
        _media(d / f"{name}.mp4", 4, tone=300 if name == "d" else None)       # d was cut by v0.9.0 (has sound)
        with sqlite3.connect(library.db_path(settings)) as con:
            con.execute("UPDATE shots SET src_id = ?, src_start = ?, length = 4 WHERE file = ?",
                        (vid, st, str(d / f"{name}.mp4")))
    with sqlite3.connect(library.db_path(settings)) as con:                    # as an old index looks
        con.execute("UPDATE shots SET audio = NULL")
    asked = []
    files = {"video:1": [{"src": str(src_loud), "height": 360}], "video:2": [{"src": str(src_mute), "height": 360}]}
    b = library.Builder({"DVIDS_API_KEY": "k"}, settings, lambda m: None)
    monkeypatch.setattr(b, "_dvids_asset", lambda vid: (asked.append(vid) or {}, files.get(vid, [])))
    monkeypatch.setattr(library, "pick_files", lambda fs: (fs[0]["src"], fs[0]["src"], 360) if fs else (None, None, 0))
    before = library.duration(d / "a.mp4")
    b.add_sound(workers=2)
    assert sorted(asked) == ["video:1", "video:2"]                            # d needed no download
    with sqlite3.connect(library.db_path(settings)) as con:
        got = {Path(f).stem: a for f, a in con.execute("SELECT file, audio FROM shots")}
    assert got == {"a": 1, "b": 1, "c": 0, "d": 1}
    assert library.has_audio(d / "a.mp4") and not library.has_audio(d / "c.mp4")
    assert abs(library.duration(d / "a.mp4") - before) < 0.2                # picture untouched, same length
    assert not list(d.glob("*.part.*"))
    assert library.stats(settings)["sound"] == {"with": 3, "none": 1, "unknown": 0}
    b.add_sound()                                                             # resumable: nothing left to do
    assert sorted(asked) == ["video:1", "video:2"]


def test_retry_restarts_the_sound_job(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "STATUS_FILE", tmp_path / "st.json")
    calls = []
    monkeypatch.setattr(library, "start_sound", lambda auto=True: calls.append(auto))
    write_json(tmp_path / "st.json", {"state": "stopped", "kind": "sound", "params": {"kind": "sound", "auto_continue": False}})
    assert library.retry_build()["action"] == "restarted" and calls == [False]


def test_added_sound_is_from_the_same_seconds_as_the_picture(settings, tmp_path, monkeypatch):
    import sqlite3
    d = Path(settings["library_dir"])
    d.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "src.mp4"                       # 440 Hz for the first 5 s, then 880 Hz
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=12", "-f", "lavfi",
                    "-i", "aevalsrc='0.5*sin(2*PI*if(lt(t,5),440,880)*t)':s=48000:d=12", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], check=True)
    for name, st in (("early", 1.0), ("late", 7.0)):
        _add(settings, name, "F-15E", "jet")
        _media(d / f"{name}.mp4", 3)
        with sqlite3.connect(library.db_path(settings)) as con:
            con.execute("UPDATE shots SET src_id = 'video:9', src_start = ?, length = 3, audio = NULL WHERE file = ?",
                        (st, str(d / f"{name}.mp4")))
    b = library.Builder({"DVIDS_API_KEY": "k"}, settings, lambda m: None)
    monkeypatch.setattr(b, "_dvids_asset", lambda vid: ({}, [{"src": str(src)}]))
    monkeypatch.setattr(library, "pick_files", lambda fs: (fs[0]["src"], fs[0]["src"], 360))
    b.add_sound()
    from warstudio import sfx
    for name, hz in (("early", 440), ("late", 880)):
        x = sfx.load_audio(d / f"{name}.mp4")
        f = np.fft.rfftfreq(len(x), 1 / sfx.SR)[np.argmax(np.abs(np.fft.rfft(x)))]
        assert abs(f - hz) < 10, (name, f)
