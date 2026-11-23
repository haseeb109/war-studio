"""ai33 TTS under the account queue limit (the 'too many tasks in queue (10/10)' error)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warstudio import ai33  # noqa: E402


class FakeAi33:
    """Queue limit 3; a task finishes 2 polls after it was submitted; finished tasks keep a queue slot
    for one more poll round (like ai33 counting recently finished tasks)."""

    def __init__(self, limit=3, fail_once=None):
        self.limit, self.tasks, self.n, self.submits, self.fail_once = limit, {}, 0, 0, set(fail_once or [])
        self.tick = 0

    def req(self, key, path, data=None, timeout=60):
        self.tick += 1
        for t in self.tasks.values():          # finished tasks hold their queue slot for a few more requests
            if t.get("done_at") is not None and self.tick - t["done_at"] >= 4:
                t["slot"] = False
        if data is not None:
            busy = sum(1 for t in self.tasks.values() if t["slot"])
            if busy >= self.limit:
                raise ai33.Ai33Busy('ai33 busy (HTTP 429): {"error":{"code":"queue_limit_exceeded"}}')
            self.n += 1
            self.submits += 1
            tid = f"t{self.n}"
            status = "failed" if data["text"] in self.fail_once else "doing"
            self.fail_once.discard(data["text"])
            self.tasks[tid] = {"polls": 0, "slot": True, "status": status, "text": data["text"]}
            return {"success": True, "task_id": tid}
        tid = path.rsplit("/", 1)[-1]
        if tid not in self.tasks:
            raise ai33.Ai33Error(f"ai33 {path} HTTP 404: not found")
        t = self.tasks[tid]
        t["polls"] += 1
        if t["status"] == "failed":
            t["slot"] = False
            return {"status": "failed", "error": "boom"}
        if t["polls"] >= 2:
            t.setdefault("done_at", self.tick)
            return {"status": "done", "credit_cost": len(t["text"]), "metadata": {"audio_url": "u://" + tid}}
        return {"status": "doing"}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run(fake, tmp_path, items, monkeypatch, state=None, concurrency=6):
    clock = Clock()
    monkeypatch.setattr(ai33.time, "time", clock.time)
    monkeypatch.setattr(ai33, "_req", fake.req)
    monkeypatch.setattr(ai33, "download", lambda url, out: Path(out).write_text(url))
    logs = []
    c = ai33.tts_many("k", items, "v", concurrency=concurrency, state_file=state, log=logs.append, sleep=clock.sleep)
    return c, logs


def test_queue_full_is_not_fatal(tmp_path, monkeypatch):
    fake = FakeAi33(limit=3)
    items = [(tmp_path / f"s{i}.mp3", f"sentence {i}") for i in range(20)]
    credits, logs = run(fake, tmp_path, items, monkeypatch)
    assert all(p.exists() for p, _ in items)
    assert fake.submits == 20                       # nothing paid twice
    assert any("queue is full" in m for m in logs)


def test_failed_task_is_retried(tmp_path, monkeypatch):
    fake = FakeAi33(limit=5, fail_once={"sentence 3"})
    items = [(tmp_path / f"s{i}.mp3", f"sentence {i}") for i in range(6)]
    run(fake, tmp_path, items, monkeypatch)
    assert all(p.exists() for p, _ in items) and fake.submits == 7


def test_resume_reuses_submitted_tasks(tmp_path, monkeypatch):
    fake = FakeAi33(limit=10)
    items = [(tmp_path / f"s{i}.mp3", f"sentence {i}") for i in range(4)]
    monkeypatch.setattr(ai33, "_req", fake.req)
    # an earlier run submitted 2 tasks and then crashed
    t1 = fake.req("k", "/v3/text-to-speech", {"text": "sentence 0"})["task_id"]
    t2 = fake.req("k", "/v3/text-to-speech", {"text": "sentence 1"})["task_id"]
    state = tmp_path / "pending.json"
    state.write_text(json.dumps({str(items[0][0]): t1, str(items[1][0]): t2, str(tmp_path / "x"): "stale"}))
    run(fake, tmp_path, items, monkeypatch, state=str(state))
    assert fake.submits == 4                        # only the 2 missing sentences were submitted
    assert all(p.exists() for p, _ in items)
    assert json.loads(state.read_text()) == {str(tmp_path / "x"): "stale"} or json.loads(state.read_text()) == {}


def test_stale_task_id_is_resubmitted(tmp_path, monkeypatch):
    fake = FakeAi33(limit=10)
    items = [(tmp_path / "s0.mp3", "sentence 0")]
    state = tmp_path / "pending.json"
    state.write_text(json.dumps({str(items[0][0]): "gone"}))
    run(fake, tmp_path, items, monkeypatch, state=str(state))
    assert items[0][0].exists() and fake.submits == 1


def test_stuck_queue_gives_up_cleanly(tmp_path, monkeypatch):
    fake = FakeAi33(limit=0)                     # queue permanently full (e.g. another app using the account)
    items = [(tmp_path / "s0.mp3", "sentence 0")]
    try:
        run(fake, tmp_path, items, monkeypatch)
        assert False, "should give up"
    except ai33.Ai33Error as e:
        assert "no progress" in str(e) and "Run again" in str(e)


def test_concurrency_recovers_after_queue_clears(tmp_path, monkeypatch):
    fake = FakeAi33(limit=2)
    orig = fake.req

    def req(key, path, data=None, timeout=60):
        if fake.submits >= 6:
            fake.limit = 10                      # the queue frees up after a while
        return orig(key, path, data, timeout)
    fake.req = req
    items = [(tmp_path / f"s{i}.mp3", f"sentence {i}") for i in range(80)]
    credits, logs = run(fake, tmp_path, items, monkeypatch, concurrency=4)
    assert all(p.exists() for p, _ in items) and fake.submits == 80
    assert any("lowering" in m for m in logs)
    assert any("back to 4" in m for m in logs)
