"""Vision rerank of the CLIP top-k collage. Providers: Groq (rerank only), Gemini.

How failures are handled (so a busy Google server can't silently turn a whole run into CLIP-only):
- Temporary trouble (503 "high demand", 500, timeout): quick retries inside the call; then, for Gemini,
  the same collage goes to a fallback model (another flash, then flash-lite); if every model is busy the
  provider is PAUSED for 60 s - the next provider (or CLIP) covers those beats - and then tried again.
- Daily quota used up: provider dropped for the rest of the run.
- Real errors (bad request etc.): dropped after 5 in a row.
Every call's token usage is added up, so the log shows the real rerank cost of the run.
"""
import time

from .llm import (LLMError, ModelGone, ProviderBusy, QuotaExhausted, gemini_fallback_models, gemini_pick,
                  groq_pick, resolve_gemini_model, resolve_groq_model)

PAUSE_SECONDS = 60          # provider rests this long after every model of it was busy
MODEL_BUSY_SECONDS = 120    # a busy Gemini model is skipped for this long in favour of its fallback
MAX_HARD_FAILS = 5

# $ per 1M tokens (paid tier); output includes thinking. Overridable in Settings.
DEFAULT_PRICES = {"gemini": (0.75, 3.75), "groq": (0.0, 0.0)}


def chain_order(setting, keys):
    setting = (setting or "auto").lower()
    have = {"groq": bool(keys.get("GROQ_API_KEY")), "gemini": bool(keys.get("GEMINI_API_KEY"))}
    if setting == "clip":
        return []
    first = {"auto": ["groq", "gemini"], "groq": ["groq", "gemini"], "gemini": ["gemini", "groq"]}.get(setting, ["groq", "gemini"])
    return [p for p in first if have[p]]


class Reranker:
    # Groq's free plan counts image tokens against a small per-minute/day token budget: smaller tiles there
    TILE = {"groq": (384, 216), "gemini": (512, 288)}

    def __init__(self, keys, settings, log, save_settings=None, clock=time.monotonic):
        self.keys, self.settings, self.log = keys, settings, log
        self.save_settings = save_settings
        self.clock = clock
        self.models, self.fails, self.chain = {}, {}, []
        self.paused = {}            # provider -> clock time it may be used again
        self.model_busy = {}        # gemini model -> clock time it may be first choice again
        self.fallbacks = []         # gemini fallback models
        self.stats = {"ok": 0, "retried": 0, "fallback_model": 0, "covered_by_other": 0, "clip": 0,
                      "in": 0, "out": 0, "thinking": 0, "by": {}}
        for p in chain_order(settings.get("reranker", "auto"), keys):
            try:
                self.models[p] = self._resolve(p)
                self.chain.append(p)
            except Exception as e:
                log(f"{p} rerank unavailable ({str(e)[:160]})")
        if "gemini" in self.chain:
            try:
                self.fallbacks = gemini_fallback_models(keys["GEMINI_API_KEY"], self.models["gemini"])
            except Exception:
                self.fallbacks = []
        if settings.get("reranker", "auto") in ("groq", "gemini") and settings["reranker"] not in self.chain:
            log(f"reranker is set to {settings['reranker']} but it can't be used; falling back")

    def _resolve(self, p):
        if p == "groq":
            return resolve_groq_model(self.keys["GROQ_API_KEY"], self.settings.get("groq_vision_model", "auto"))
        m = resolve_gemini_model(self.keys["GEMINI_API_KEY"], self.settings.get("gemini_model", "auto"))
        if self.settings.get("gemini_model", "auto") not in ("auto", m):
            self.log(f"Gemini model '{self.settings.get('gemini_model')}' is not available for this key; using {m}")
        return m

    @property
    def label(self):
        if not self.chain:
            return "CLIP only (no Groq/Gemini key, or reranker=clip)"
        s = "CLIP + " + " -> ".join(f"{p} ({self.models[p]})" for p in self.chain)
        if self.fallbacks and "gemini" in self.chain:
            s += f"; Gemini backup models: {', '.join(self.fallbacks)}"
        return s

    @property
    def active(self):
        """First provider that is usable right now (not paused)."""
        now = self.clock()
        for p in self.chain:
            if self.paused.get(p, 0) <= now:
                return p
        return None

    # ------------------------------------------------------------------ calls
    def _gemini_models_now(self):
        now = self.clock()
        primary = self.models["gemini"]
        order = [primary] + [m for m in self.fallbacks if m != primary]
        if self.model_busy.get(primary, 0) > now:          # primary resting: its fallbacks go first
            order = [m for m in order if m != primary] + [primary]
        return order

    def _call(self, p, collage, prompt):
        if p == "groq":
            res, usage = groq_pick(self.keys["GROQ_API_KEY"], self.models[p], collage, prompt)
            self._count(p, usage, groq=True)
            return res
        key = self.keys["GEMINI_API_KEY"]
        last = None
        for i, m in enumerate(self._gemini_models_now()):
            try:
                res, usage = gemini_pick(key, m, collage, prompt,
                                         retry_waits=(2.0, 5.0) if i == 0 else (2.0,))
            except ProviderBusy as e:
                last = e
                if self.model_busy.get(m, 0) <= self.clock():
                    self.log(f"gemini {m} is overloaded right now ({str(e)[-90:]}); trying a backup model")
                self.model_busy[m] = self.clock() + MODEL_BUSY_SECONDS
                continue
            if m != self.models["gemini"]:
                self.stats["fallback_model"] += 1
            self._count(p, usage)
            return res
        raise ProviderBusy(f"every Gemini model is busy ({str(last)[-120:] if last else ''})")

    def _count(self, p, usage, groq=False):
        if groq:
            self.stats["in"] += int(usage.get("prompt_tokens") or 0)
            self.stats["out"] += int(usage.get("completion_tokens") or 0)
        else:
            self.stats["in"] += int(usage.get("promptTokenCount") or 0)
            self.stats["out"] += int(usage.get("candidatesTokenCount") or 0)
            self.stats["thinking"] += int(usage.get("thoughtsTokenCount") or 0)
            if usage.get("_retries"):
                self.stats["retried"] += 1
        self.stats["ok"] += 1
        self.stats["by"][p] = self.stats["by"].get(p, 0) + 1

    def _drop(self, p, why):
        if p in self.chain:
            self.chain.remove(p)
        nxt = self.active or "CLIP only"
        self.log(f"{p} rerank stopped ({why}); using {nxt} for the rest of this run")

    def _pause(self, p, why):
        if self.paused.get(p, 0) <= self.clock():
            self.log(f"{p} rerank paused for {PAUSE_SECONDS}s ({why[-120:]}); other provider/CLIP covers these beats, "
                     f"then {p} is tried again")
        self.paused[p] = self.clock() + PAUSE_SECONDS

    def pick(self, make_collage, prompt, tag=""):
        """make_collage(tile_size) -> image path. Returns (result_dict, provider) or (None, None)."""
        tried_first = None
        while True:
            p = self.active
            if p is None:
                self.stats["clip"] += 1
                return None, None
            tried_first = tried_first or p
            try:
                col = make_collage(self.TILE[p])
                try:
                    res = self._call(p, col, prompt)
                except ModelGone as gone:
                    old = self.models[p]
                    self.settings = dict(self.settings, **{("groq_vision_model" if p == "groq" else "gemini_model"): "auto"})
                    self.models[p] = self._resolve(p)
                    self.log(f"{p} model {old} is gone ({str(gone)[-100:]}); switching to {self.models[p]}")
                    if self.save_settings:
                        self.save_settings({("groq_vision_model" if p == "groq" else "gemini_model"): "auto"})
                    res = self._call(p, col, prompt)
                self.fails[p] = 0
                if p != tried_first:
                    self.stats["covered_by_other"] += 1
                return res, p
            except QuotaExhausted as e:
                self._drop(p, f"quota used up: {str(e)[-140:]}")
            except ProviderBusy as e:
                self._pause(p, str(e))            # temporary: not counted as a failure
            except Exception as e:
                self.fails[p] = self.fails.get(p, 0) + 1
                self.log(f"{tag}{p} rerank failed ({str(e)[:100]})")
                if self.fails[p] >= MAX_HARD_FAILS:
                    self._drop(p, f"failed {MAX_HARD_FAILS} times in a row")
                self.stats["clip"] += 1
                return None, None

    def summary(self):
        st = self.stats
        pin = float(self.settings.get("gemini_price_in_per_m", DEFAULT_PRICES["gemini"][0]))
        pout = float(self.settings.get("gemini_price_out_per_m", DEFAULT_PRICES["gemini"][1]))
        g_share = st["by"].get("gemini", 0) / max(st["ok"], 1)
        cost = g_share * (st["in"] * pin + (st["out"] + st["thinking"]) * pout) / 1e6
        by = ", ".join(f"{k} {v}" for k, v in st["by"].items()) or "none"
        return (f"rerank calls: {by}, CLIP only {st['clip']}; retried {st['retried']}, backup model "
                f"{st['fallback_model']}, covered by other provider {st['covered_by_other']}; tokens in {st['in']}, "
                f"out {st['out']}, thinking {st['thinking']}; Gemini cost ~${cost:.3f} (if billing is on)")
