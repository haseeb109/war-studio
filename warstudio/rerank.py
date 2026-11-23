"""Vision rerank of the CLIP top-k collage. Providers: Groq (rerank only), Gemini.

The chain falls through: a provider that runs out of daily quota, or fails 3 times in a row, is dropped
for the rest of the run and the next one takes over; with none left the CLIP top-1 wins.
"""
from .llm import (LLMError, ModelGone, QuotaExhausted, gemini_pick, groq_pick, resolve_gemini_model,
                  resolve_groq_model)


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

    def __init__(self, keys, settings, log, save_settings=None):
        self.keys, self.settings, self.log = keys, settings, log
        self.save_settings = save_settings
        self.models, self.fails, self.chain = {}, {}, []
        for p in chain_order(settings.get("reranker", "auto"), keys):
            try:
                self.models[p] = self._resolve(p)
                self.chain.append(p)
            except Exception as e:
                log(f"{p} rerank unavailable ({str(e)[:160]})")
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
        return "CLIP + " + " -> ".join(f"{p} ({self.models[p]})" for p in self.chain)

    @property
    def active(self):
        return self.chain[0] if self.chain else None

    def _call(self, p, collage, prompt):
        if p == "groq":
            return groq_pick(self.keys["GROQ_API_KEY"], self.models[p], collage, prompt)[0]
        return gemini_pick(self.keys["GEMINI_API_KEY"], self.models[p], collage, prompt)[0]

    def _drop(self, p, why):
        if p in self.chain:
            self.chain.remove(p)
        nxt = self.chain[0] if self.chain else "CLIP only"
        self.log(f"{p} rerank stopped ({why}); using {nxt} for the rest of this run")

    def pick(self, make_collage, prompt, tag=""):
        """make_collage(tile_size) -> image path. Returns (result_dict, provider) or (None, None)."""
        while self.chain:
            p = self.chain[0]
            try:
                col = make_collage(self.TILE[p])
                try:
                    res = self._call(p, col, prompt)
                except ModelGone as gone:
                    old, self.models[p] = self.models[p], None
                    self.settings = dict(self.settings, **{("groq_vision_model" if p == "groq" else "gemini_model"): "auto"})
                    self.models[p] = self._resolve(p)
                    self.log(f"{p} model {old} is gone ({str(gone)[-100:]}); switching to {self.models[p]}")
                    if self.save_settings:
                        self.save_settings({("groq_vision_model" if p == "groq" else "gemini_model"): "auto"})
                    res = self._call(p, col, prompt)
                self.fails[p] = 0
                return res, p
            except QuotaExhausted as e:
                self._drop(p, f"quota used up: {str(e)[-140:]}")
            except Exception as e:
                self.fails[p] = self.fails.get(p, 0) + 1
                self.log(f"{tag}{p} rerank failed ({str(e)[:100]})")
                if self.fails[p] >= 3:
                    self._drop(p, "failed 3 times in a row")
                return None, None
        return None, None
