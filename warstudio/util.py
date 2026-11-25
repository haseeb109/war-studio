import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) WarStudio/0.1"


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd[:6]))} ...\n{r.stderr[-1500:]}")
    return r.stdout


def num(x, default=0.0):
    """float() that tolerates None, '', 'N/A' and other junk from APIs/ffprobe."""
    try:
        v = float(str(x).strip())
        return v if v == v and v not in (float("inf"), float("-inf")) else default
    except (TypeError, ValueError):
        return default


def duration(path):
    """Media duration in seconds. Falls back to the stream duration when the container says N/A."""
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=duration", "-of", "json", str(path)])
    d = json.loads(out or "{}")
    vals = [num((d.get("format") or {}).get("duration"))] + [num(s.get("duration")) for s in d.get("streams") or []]
    best = max(vals) if vals else 0.0
    if best <= 0:
        raise RuntimeError(f"cannot read the length of {Path(path).name} (file empty or broken)")
    return best


def download(url, dest, timeout=120):
    """Download with a browser user agent (some CDNs 403 python's default UA)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import os
    import threading
    tmp = p.with_suffix(p.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(p)


def encoder_works(enc):
    """Encode 10 test frames; True if the encoder really works on this machine."""
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=0.34",
                        "-pix_fmt", "yuv420p", *encoder_args(enc), "-f", "null", "-"], capture_output=True, text=True)
    return r.returncode == 0


def video_encoder(setting="auto"):
    """h264_videotoolbox on Apple Silicon (fast) if it really encodes, libx264 otherwise."""
    if setting and setting != "auto":
        return setting
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in enc and encoder_works("h264_videotoolbox"):
        return "h264_videotoolbox"
    return "libx264"


def encoder_args(enc, quality="high"):
    if enc == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", "12M" if quality == "high" else "8M", "-allow_sw", "1"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19" if quality == "high" else "22"]


# ---------------------------------------------------------------- network resilience
NET_ERR_WORDS = ("connection reset", "connection refused", "connection aborted", "timed out", "timeout",
                 "network is unreachable", "network is down", "no route to host", "temporary failure in name resolution",
                 "could not resolve", "name or service not known", "nodename nor servname", "failed to resolve",
                 "i/o error", "broken pipe", "end of file", "server returned 5", "502", "503", "504",
                 "remote end closed", "ssl", "eof occurred", "connection closed", "not connected to the internet",
                 "the internet connection appears to be offline", "errno 8", "errno 50", "errno 51", "errno 60",
                 "errno 61", "errno 64", "errno 65")
RETRY_EVERY = 5          # seconds between retries when the internet drops


def is_network_error(e):
    try:
        import httpx
        if isinstance(e, (httpx.UnsupportedProtocol, httpx.InvalidURL)):
            return False                      # a bad link, not the internet: never wait on it
        if isinstance(e, httpx.TransportError):
            return True
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 500, 502, 503, 504):
            return True
    except Exception:
        pass
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
    import urllib.error
    if isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
        return True
    msg = str(e).lower()
    return any(w in msg for w in NET_ERR_WORDS)


DNS_ERR_WORDS = ("nodename nor servname", "name resolution", "could not resolve", "failed to resolve",
                 "name or service not known", "errno 8", "getaddrinfo")
DNS_GRACE = 12           # DNS errors right after an outage: up to 12 extra tries (~60 s) before they count
_NET_LOCK = threading.Lock()
_NET = {"down_since": None}  # one outage shared by every worker thread: logged once, not once per thread


def is_dns_error(e):
    msg = str(e).lower()
    return any(w in msg for w in DNS_ERR_WORDS)


def online(timeout=4):
    """True when the internet really works: a named host must resolve AND answer. (A raw IP like 1.1.1.1 is not
    enough: after Wi-Fi comes back the router's DNS can lag for a minute, and every download would still fail.)"""
    import httpx
    for url in ("https://api.dvidshub.net", "https://www.google.com/generate_204", "https://one.one.one.one"):
        try:
            httpx.head(url, timeout=timeout, follow_redirects=False)
            return True
        except Exception:
            continue
    return False


def wait_for_internet(log=None, stop=None, every=RETRY_EVERY, status=None, sleep=None, is_online=None):
    """Blocks until the internet is back, checking every `every` seconds. Returns False if `stop` was set.
    Many threads may wait at once: only the first one writes 'down' / 'still down' / 'back' to the log."""
    import time as _t
    sleep = sleep or _t.sleep
    is_online = is_online or online
    if is_online():
        return True
    with _NET_LOCK:
        first = _NET["down_since"] is None
        if first:
            _NET["down_since"] = _t.time()
        t0 = _NET["down_since"]
    if first and log:
        log(f"internet is down - checking again every {every}s, the work continues by itself when it is back")
    if status:
        status(net="waiting for internet")
    last_note = _t.time()
    while not is_online():
        if stop is not None and stop.is_set():
            with _NET_LOCK:
                _NET["down_since"] = None
            return False
        sleep(every)
        if first and log and _t.time() - last_note >= 60:
            log(f"  still no internet ({int(_t.time() - t0)}s) - still checking every {every}s")
            last_note = _t.time()
    with _NET_LOCK:
        was, _NET["down_since"] = _NET["down_since"], None
    if was is not None and log:
        log(f"internet is back after {int(_t.time() - was)}s - continuing")
    if status:
        status(net="")
    return True


class Stopped(RuntimeError):
    pass


class NetGate:
    """One pause switch shared by every worker of a job (the library build).

    After `max_fails` network / DNS errors in a row the WHOLE job pauses: no next file, no next search is tried.
    It continues with the very same request when retry() is pressed (Retry button), or - with `auto` on - when a
    named host really answers again (checked every 5 s; if the internet keeps dropping right after coming back the
    check backs off up to 5 min)."""

    def __init__(self, log=None, status=None, stop=None, max_fails=2, auto=True, every=RETRY_EVERY, is_online=None,
                 clock=None):
        import time as _t
        self.log = log or (lambda *a, **k: None)
        self.status = status or (lambda **k: None)
        self.stop = stop
        self.max_fails, self.auto, self.every = max(1, int(max_fails)), bool(auto), float(every)
        self.is_online = is_online or online
        self.clock = clock or _t.time
        self._go = threading.Event()
        self._go.set()
        self._lock = threading.Lock()
        self.fails = 0
        self.pauses = 0
        self._interval = self.every
        self._next_check = 0.0
        self._resumed_at = None

    @property
    def paused(self):
        return not self._go.is_set()

    def _stopped(self):
        return self.stop is not None and self.stop.is_set()

    def wait(self):
        """Block while paused. False when the job was stopped."""
        while not self._go.is_set():
            if self._stopped():
                return False
            if self.auto:
                check = False
                with self._lock:
                    if self.clock() >= self._next_check:
                        self._next_check = self.clock() + self._interval
                        check = True
                if check and self.is_online():
                    self.resume("internet is back")
                    break
            self._go.wait(min(1.0, self.every))
        return not self._stopped()

    def ok(self):
        with self._lock:
            self.fails = 0
            if self._resumed_at is not None and self.clock() - self._resumed_at > 60:
                self._interval, self._resumed_at = self.every, None      # stable again: back to 5 s checks

    def failed(self, e, what="request"):
        with self._lock:
            self.fails += 1
            n = self.fails
            pause = n >= self.max_fails and self._go.is_set()
            if pause:
                self._go.clear()
                self.pauses += 1
                if self._resumed_at is not None and self.clock() - self._resumed_at < 60:
                    self._interval = min(300.0, self._interval * 2)       # flapping: check less often
                self._next_check = self.clock() + self._interval
        kind = "DNS (name lookup)" if is_dns_error(e) else "internet"
        if pause:
            how = (f"it continues by itself when the internet is really back (checking every {int(self._interval)}s)"
                   if self.auto else "press Retry when the internet is back")
            self.log(f"PAUSED: {kind} problem - {n} failed tries in a row ({what}: {str(e)[:80]}). Nothing new is "
                     f"tried until it is fixed; {how}.")
            self.status(net="paused", net_reason=f"{kind} problem", paused_at=self.clock(), net_auto=self.auto)
        elif self._go.is_set():
            self.log(f"  {what}: {kind} error ({str(e)[:80]}) - try {n}/{self.max_fails}, again in {int(self.every)}s")
            if self.stop is not None:
                self.stop.wait(self.every)
            else:
                import time as _t
                _t.sleep(self.every)

    def retry(self, why="Retry pressed"):
        return self.resume(why)

    def resume(self, why):
        with self._lock:
            was = not self._go.is_set()
            self.fails = 0
            if was:
                self._resumed_at = self.clock()
                self._go.set()
        if was:
            self.log(f"RESUMED ({why}) - continuing from the same file")
            self.status(net="", net_reason="", paused_at=None)
        return was


def with_net_retry(fn, log=None, stop=None, what="download", every=RETRY_EVERY, online_tries=4, status=None,
                   sleep=None, is_online=None, gate=None, gate_tries=6):
    """Run fn(); on a network error: internet down -> wait (checking every 5 s) and retry, as long as it takes;
    internet up but this request keeps failing -> retry every 5 s, give up after `online_tries`.
    DNS errors get DNS_GRACE extra tries first (DNS often comes back a minute after the connection).
    With a `gate` (NetGate) the gate decides instead: 2 failures in a row pause the whole job, and this same
    request is tried again after Retry / when the internet is back (it gives up after `gate_tries` failures, so one
    broken file can never hold the job forever)."""
    import time as _t
    if gate is not None:
        tries = 0
        while True:
            if not gate.wait():
                raise Stopped("stopped")
            try:
                r = fn()
                gate.ok()
                return r
            except Exception as e:
                if not is_network_error(e) or (stop is not None and stop.is_set()):
                    raise
                tries += 1
                if tries >= gate_tries:
                    raise
                gate.failed(e, what)
    sleep = sleep or _t.sleep
    is_online = is_online or online
    fails = dns = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if not is_network_error(e):
                raise
            if stop is not None and stop.is_set():
                raise
            if not is_online():
                if not wait_for_internet(log, stop, every, status, sleep, is_online):
                    raise
                fails = 0                      # an outage does not count as a failed try
                continue
            if is_dns_error(e) and dns < DNS_GRACE:
                dns += 1
                if log and dns == 1:
                    log(f"  {what}: name lookup (DNS) not working yet - retrying every {every}s for up to "
                        f"{DNS_GRACE * every}s")
                sleep(every)
                continue
            fails += 1
            if fails >= online_tries:
                raise
            if log:
                log(f"  {what}: network error ({str(e)[:90]}) - retry {fails}/{online_tries - 1} in {every}s")
            sleep(every)


_FF_NET = None


def ffmpeg_net_args(url):
    """Input options that make ffmpeg reconnect by itself after a dropped connection (http inputs only), and
    give up on a stalled read after 30 s instead of hanging forever."""
    global _FF_NET
    if not str(url).startswith("http"):
        return []
    if _FF_NET is None:
        try:
            h = subprocess.run(["ffmpeg", "-hide_banner", "-h", "protocol=http"], capture_output=True, text=True,
                               timeout=20).stdout
        except Exception:
            h = ""
        args = []
        for opt, val in (("reconnect", "1"), ("reconnect_streamed", "1"), ("reconnect_on_network_error", "1"),
                         ("reconnect_delay_max", "5")):
            if f"-{opt} " in h:
                args += [f"-{opt}", val]
        _FF_NET = args + ["-rw_timeout", "30000000"]
    return ["-user_agent", UA] + list(_FF_NET)


def download_resume(url, dest, log=None, stop=None, timeout=60, status=None, chunk=1 << 20, online_tries=6,
                    gate=None):
    """Download that survives a dropped connection: keeps what it has in dest.part and continues with an HTTP
    Range request (if the server ignores it, starts that file again). Retries every 5 s; waits out outages."""
    import httpx
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not str(url).startswith(("http://", "https://")):
        shutil.copyfile(url, dest)            # a local file (own footage, tests)
        return dest
    part = dest.with_name(dest.name + ".part")

    def attempt():
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as r:
            if r.status_code == 416:            # already complete
                part.replace(dest)
                return dest
            r.raise_for_status()
            mode = "ab" if have and r.status_code == 206 else "wb"
            with open(part, mode) as f:
                for b in r.iter_bytes(chunk):
                    if stop is not None and stop.is_set():
                        raise RuntimeError("stopped")
                    f.write(b)
        part.replace(dest)
        return dest

    return with_net_retry(attempt, log, stop, what=f"download {dest.name}", online_tries=online_tries, status=status,
                          gate=gate)
