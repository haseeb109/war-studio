import threading
import time

from PIL import Image

from warstudio.steps import rank
from warstudio.util import write_json


class FakeRR:
    """Reranker stand-in: scores by tile order from a table, records concurrency."""
    def __init__(self, table):
        self.table, self.active, self.chain = table, "gemini", ["gemini"]
        self.stats = {"ok": 0}
        self.label = "fake"
        self.live, self.peak, self.lock = 0, 0, threading.Lock()

    def pick(self, make_collage, prompt, tag=""):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
        time.sleep(0.05)
        with self.lock:
            self.live -= 1
            self.stats["ok"] += 1
        bid = int(tag.split()[1].rstrip(":"))
        return self.table(bid), "gemini"

    def summary(self):
        return "fake summary"


def setup(tmp_path, n_beats, n_cands, scene=lambda i: f"s{i}"):
    beats = [{"id": i, "sid": i, "text": f"b{i}", "start": i * 3.0, "end": i * 3 + 3} for i in range(n_beats)]
    specs = [{"id": i, "queries": ["q"], "tiers": ["category"], "visual": "jets", "footage_class": "real",
              "scene": scene(i), "avoid": []} for i in range(n_beats)]
    cands = {str(i): [{"source": "dvids", "id": f"c{j}", "title": "", "preview_url": f"u{j}", "video_url": "",
                       "duration": 20, "tier": "category"} for j in range(n_cands)] for i in range(n_beats)}
    for f, o in (("beats.json", beats), ("specs.json", specs), ("candidates.json", cands)):
        write_json(tmp_path / f, o)
    (tmp_path / "previews").mkdir()
    for j in range(n_cands):
        Image.new("RGB", (64, 36)).save(tmp_path / "previews" / f"dvids_c{j}.jpg")


def test_parallel_rerank_and_cooldown(tmp_path, monkeypatch):
    setup(tmp_path, 12, 6)
    monkeypatch.setattr(rank, "clip_scores", lambda s, t, ps: [0.3 - 0.01 * int(p.rsplit("_c", 1)[1][0]) for p in ps])
    # the reranker always likes tile 1 best (= same clip c0 for every beat)
    fake = FakeRR(lambda bid: {"scores": [9, 7, 5, 3, 1], "best": 1})
    monkeypatch.setattr(rank, "Reranker", lambda *a, **k: fake)
    logs = []
    picks = rank.run_step(tmp_path, {}, {"rerank_concurrency": 6, "clip_max_uses": 99, "clip_reuse_gap_seconds": 0}, {"footage": {"same_source_cooldown_seconds": 9}},
                          lambda m, **k: logs.append(m))
    assert fake.peak > 1                                   # several beats were reranked at the same time
    ids = [picks[str(i)]["id"] for i in range(12)]
    for i in range(12):                                    # cooldown: no clip again within 9 s (3 beats)
        assert ids[i] not in ids[max(0, i - 2):i]
    assert all(picks[str(i)]["gemini_score"] is not None for i in range(12))
    assert any("reranking 12 beats, 6 at a time" in m for m in logs)


def test_scene_reuse_only_when_reranked_clips_are_weak(tmp_path, monkeypatch):
    setup(tmp_path, 3, 3, scene=lambda i: "same")
    monkeypatch.setattr(rank, "clip_scores", lambda s, t, ps: [0.3] * len(ps))
    weak = FakeRR(lambda bid: {"scores": [9, 2, 2], "best": 1} if bid == 0 else {"scores": [4, 3, 2], "best": 1})
    monkeypatch.setattr(rank, "Reranker", lambda *a, **k: weak)
    # beat 0 picks c0; beats 1-2 only get weak scores (<6) -> reuse the strong in-scene clip
    picks = rank.run_step(tmp_path, {}, {"clip_reuse_gap_seconds": 0}, {"footage": {"same_source_cooldown_seconds": 90}},
                          lambda m, **k: None)
    assert picks["0"]["id"] == "c0" and picks["1"].get("scene_reuse") and picks["1"]["id"] == "c0"


def test_scene_reuse_capped_at_three(tmp_path, monkeypatch):
    setup(tmp_path, 5, 3, scene=lambda i: "same")
    monkeypatch.setattr(rank, "clip_scores", lambda s, t, ps: [0.3] * len(ps))
    fake = FakeRR(lambda bid: {"scores": [9, 8, 7], "best": 1})
    monkeypatch.setattr(rank, "Reranker", lambda *a, **k: fake)
    picks = rank.run_step(tmp_path, {}, {"clip_reuse_gap_seconds": 0}, {"footage": {"same_source_cooldown_seconds": 90}},
                          lambda m, **k: None)
    ids = [picks[str(i)]["id"] for i in range(5)]
    assert ids.count("c0") == 3 and ids[3] == "c1"


def test_cap_candidates_keeps_every_tier():
    cl = [{"id": str(i), "tier": "exact"} for i in range(30)] + [{"id": f"k{i}", "tier": "category"} for i in range(5)]
    kept = rank._cap_candidates(cl, 24, 4)
    assert len(kept) == 24 and sum(1 for c in kept if c["tier"] == "category") == 4


def test_one_clip_max_three_times_and_far_apart(tmp_path, monkeypatch):
    setup(tmp_path, 60, 8, scene=lambda i: "same")            # 3 min, one scene, only 8 clips, reranker loves c0
    monkeypatch.setattr(rank, "clip_scores", lambda s, t, ps: [0.3] * len(ps))
    fake = FakeRR(lambda bid: {"scores": [9, 8, 7, 6, 5], "best": 1})
    monkeypatch.setattr(rank, "Reranker", lambda *a, **k: fake)
    picks = rank.run_step(tmp_path, {}, {}, {"footage": {"same_source_cooldown_seconds": 9}}, lambda m, **k: None)
    used = {}
    for i in range(60):
        p = picks[str(i)]
        if p:
            used.setdefault(p["id"], []).append(i * 3.0)
    assert used and all(len(t) <= 3 for t in used.values())
    assert all(b - a >= 9 for t in used.values() for a, b in zip(t, t[1:]))
    # with plenty of fresh clips the full 60 s gap holds
    setup(tmp_path / "x", 20, 24) if (tmp_path / "x").mkdir() is None else None
    picks = rank.run_step(tmp_path / "x", {}, {}, {"footage": {"same_source_cooldown_seconds": 9}}, lambda m, **k: None)
    seen = {}
    for i in range(20):
        p = picks[str(i)]
        if p:
            assert i * 3.0 - seen.get(p["id"], -999) >= 60
            seen[p["id"]] = i * 3.0


def test_two_library_shots_of_the_same_moment_are_not_back_to_back(tmp_path, monkeypatch):
    setup(tmp_path, 4, 3, scene=lambda i: "same")
    import json as _j
    cands = _j.loads((tmp_path / "candidates.json").read_text())
    for i in cands:                                       # c0 and c1 are cut from one source video, same seconds
        for c in cands[i]:
            c["origin"] = "dvids:777" if c["id"] in ("c0", "c1") else "dvids:888"
            c["src_start"], c["src_len"] = (10.0, 3.0) if c["id"] in ("c0", "c1") else (0.0, 3.0)
    write_json(tmp_path / "candidates.json", cands)
    monkeypatch.setattr(rank, "clip_scores", lambda s, t, ps: [0.3] * len(ps))
    fake = FakeRR(lambda bid: {"scores": [9, 8, 1], "best": 1})
    monkeypatch.setattr(rank, "Reranker", lambda *a, **k: fake)
    picks = rank.run_step(tmp_path, {}, {}, {"footage": {"same_source_cooldown_seconds": 0}}, lambda m, **k: None)
    assert picks["0"]["id"] == "c0" and picks["1"]["id"] == "c2"     # c1 would replay c0's moment
