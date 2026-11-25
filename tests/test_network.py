import threading

import httpx
import pytest

from warstudio import util


def test_is_network_error():
    assert util.is_network_error(httpx.ConnectError("boom"))
    assert util.is_network_error(httpx.ReadTimeout("slow"))
    assert util.is_network_error(RuntimeError("command failed ... Connection reset by peer"))
    assert util.is_network_error(RuntimeError("[tls] Error in the pull function. Network is unreachable"))
    assert not util.is_network_error(ValueError("bad json"))
    assert not util.is_network_error(RuntimeError("No such file or directory"))
    assert not util.is_network_error(httpx.UnsupportedProtocol("Request URL is missing an 'http://' or 'https://'"))


def test_download_resume_local_file(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"12345")
    assert util.download_resume(str(src), tmp_path / "b.bin").read_bytes() == b"12345"


def test_waits_every_5s_while_offline_then_continues():
    state = {"online": False, "checks": 0, "calls": 0}
    slept, logs = [], []

    def is_online():
        state["checks"] += 1
        if state["checks"] >= 6:          # internet comes back after a few checks
            state["online"] = True
        return state["online"]

    def fn():
        state["calls"] += 1
        if not state["online"]:
            raise httpx.ConnectError("no route to host")
        return "ok"
    assert util.with_net_retry(fn, logs.append, sleep=slept.append, is_online=is_online) == "ok"
    assert set(slept) == {5} and len(slept) >= 2           # checked every 5 s
    assert any("internet is down" in l for l in logs) and any("internet is back" in l for l in logs)


def test_online_but_failing_gives_up_after_tries():
    slept = []

    def fn():
        raise httpx.ReadError("reset")
    with pytest.raises(httpx.ReadError):
        util.with_net_retry(fn, None, sleep=slept.append, is_online=lambda: True, online_tries=3)
    assert slept == [5, 5]


def test_non_network_error_is_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("bad")
    with pytest.raises(ValueError):
        util.with_net_retry(fn, None, sleep=lambda s: None, is_online=lambda: True)
    assert len(calls) == 1


def test_stop_ends_the_wait():
    stop = threading.Event()
    stop.set()
    assert util.wait_for_internet(None, stop, sleep=lambda s: None, is_online=lambda: False) is False


def test_download_resume_continues_from_saved_bytes(tmp_path, monkeypatch):
    data = bytes(range(256)) * 40
    seen = []

    class Resp:
        def __init__(self, headers):
            self.rng = headers.get("Range")
            self.status_code = 206 if self.rng else 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self, n):
            if not self.rng:                 # first try: the connection drops half way
                yield data[:4000]
                raise httpx.ReadError("connection reset")
            start = int(self.rng.split("=")[1].rstrip("-"))
            yield data[start:]

    def stream(method, url, headers=None, **kw):
        seen.append(headers.get("Range"))
        return Resp(headers)
    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(util, "online", lambda timeout=4: True)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = util.download_resume("https://x/v.mp4", tmp_path / "v.mp4")
    assert out.read_bytes() == data and seen == [None, "bytes=4000-"]
    assert not (tmp_path / "v.mp4.part").exists()


def test_ffmpeg_net_args_only_for_http():
    assert util.ffmpeg_net_args("/local/file.mp4") == []
    a = util.ffmpeg_net_args("https://d1.dvidshub.net/x.mp4")
    assert "-reconnect" in a and "-rw_timeout" in a and "-user_agent" in a


def test_library_requeues_video_after_network_error(tmp_path, monkeypatch):
    import numpy as np
    from warstudio import library
    s = {"library_dir": str(tmp_path / "lib"), "encoder": "libx264", "library_captions": "off"}
    monkeypatch.setattr(library, "STATUS_FILE", tmp_path / "st.json")
    monkeypatch.setattr(library, "wait_for_internet", lambda *a, **k: True)
    b = library.Builder({"DVIDS_API_KEY": "k"}, s, lambda m: None)
    monkeypatch.setattr(b, "candidates", lambda q, want=12, max_page=40: [{"id": "video:1", "title": "t"}]
                        if not b.pages.get(q) else [])
    tries = []

    def process(cat, r):
        tries.append(r["id"])
        b.pages[cat[1]] = 99
        if len(tries) == 1:
            raise httpx.ConnectError("network is unreachable")
        return 0
    monkeypatch.setattr(b, "process_video", process)
    monkeypatch.setattr(b, "upgrade_vectors", lambda: 0)
    monkeypatch.setattr(library, "Captioner", lambda *a, **k: type("C", (), {"rr": None})())
    b.build(1, categories=["F-15E Strike Eagle"])
    assert tries == ["video:1", "video:1"]                    # retried, not marked failed
    assert library.stats(s)["videos"] == {}


def test_resume_if_interrupted(tmp_path, monkeypatch):
    from warstudio import library
    monkeypatch.setattr(library, "STATUS_FILE", tmp_path / "st.json")
    util.write_json(tmp_path / "st.json", {"state": "running", "params": {"target": 5000, "categories": None,
                                                                         "per_video": 4, "max_gb": 100}})
    got = {}
    monkeypatch.setattr(library, "start_build", lambda t, c=None, p=None, g=None: got.update(t=t, p=p, g=g))
    assert library.resume_if_interrupted({}) is True and got == {"t": 5000, "p": 4, "g": 100}
    util.write_json(tmp_path / "st.json", {"state": "done"})
    assert library.resume_if_interrupted({}) is False


def test_online_needs_dns_not_just_an_ip(monkeypatch):
    import httpx
    asked = []

    def head(url, **k):
        asked.append(url)
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    monkeypatch.setattr(httpx, "head", head)
    assert util.online() is False
    assert all("://1.1.1.1" not in u for u in asked)            # every check goes through a host name


def test_dns_errors_get_a_grace_minute_before_counting():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= 8:                                      # DNS lags ~40 s after the connection is back
            raise OSError("[Errno 8] nodename nor servname provided, or not known")
        return "ok"

    logs, slept = [], []
    assert util.with_net_retry(fn, logs.append, sleep=slept.append, is_online=lambda: True, online_tries=3) == "ok"
    assert len(slept) == 8 and sum("DNS" in m for m in logs) == 1


def test_outage_logged_once_for_many_threads():
    import threading
    state = {"up": False}
    logs = []
    gate = threading.Barrier(3)

    def worker():
        gate.wait()
        util.wait_for_internet(logs.append, sleep=lambda s: None, is_online=lambda: state["up"])

    ts = [threading.Thread(target=worker) for _ in range(3)]
    for t in ts:
        t.start()
    import time
    time.sleep(0.2)
    state["up"] = True
    for t in ts:
        t.join(5)
    assert sum("internet is down" in m for m in logs) == 1
    assert sum("internet is back" in m for m in logs) == 1


def test_library_gives_up_on_a_video_after_three_interruptions(tmp_path, monkeypatch):
    from warstudio import library
    s = {"library_dir": str(tmp_path / "lib"), "encoder": "libx264", "library_captions": "off"}
    monkeypatch.setattr(library, "STATUS_FILE", tmp_path / "st.json")
    monkeypatch.setattr(library, "wait_for_internet", lambda *a, **k: True)
    b = library.Builder({"DVIDS_API_KEY": "k"}, s, lambda m: None)
    monkeypatch.setattr(b, "candidates", lambda q, want=12, max_page=40: [{"id": "video:7", "title": "t"}]
                        if not b.pages.get(q) else [])
    tries = []

    def process(cat, r):
        tries.append(r["id"])
        b.pages[cat[1]] = 99
        raise httpx.ConnectError("network is unreachable")
    monkeypatch.setattr(b, "process_video", process)
    monkeypatch.setattr(b, "upgrade_vectors", lambda: 0)
    monkeypatch.setattr(library, "Captioner", lambda *a, **k: type("C", (), {"rr": None})())
    b.build(1, categories=["F-15E Strike Eagle"])
    assert len(tries) == 4                                     # first try + 3 re-queues, then marked failed
    assert library.stats(s)["videos"].get("failed") == 1
