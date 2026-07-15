"""Agent memory for the forecasting bot (Cognee-style, lightweight local implementation).

Stores each forecast + its research summary + eventual resolution, and retrieves the most
similar past resolved questions to inform new forecasts. Uses embeddings if OPENAI_API_KEY
is set (semantic recall), else falls back to keyword overlap. Persists to a local JSONL so it
survives across runs and accumulates a track record — the "which reasoning was right/wrong"
signal the bot currently lacks.

Design goals:
  - Zero hard dependency (optional: openai for embeddings, cognee if installed).
  - Safe: never raises into the forecast path; all failures degrade to "no memory".
  - Auditable: plain JSONL you can inspect.
"""
from __future__ import annotations
import os, json, time, math, hashlib
from typing import Any, Dict, List, Optional

_STORE = os.getenv("FORECAST_MEMORY_PATH", "forecast_memory.jsonl")

# ---- optional Cognee backend (used if installed & COGNEE enabled) ----
_USE_COGNEE = os.getenv("USE_COGNEE", "0") == "1"
try:
    import cognee  # type: ignore
    _HAS_COGNEE = True
except Exception:
    _HAS_COGNEE = False


def _cognee_ready() -> bool:
    return _USE_COGNEE and _HAS_COGNEE


async def _cognee_add(text: str) -> None:
    """Feed a resolved forecast into Cognee's knowledge graph (best-effort, async)."""
    try:
        await cognee.add(text)
        await cognee.cognify()
    except Exception:
        pass


async def _cognee_search(query: str) -> str:
    try:
        res = await cognee.search(query_text=query)
        if isinstance(res, list):
            return "\n".join(str(r) for r in res[:5])
        return str(res)
    except Exception:
        return ""


def _embed(texts: List[str]) -> Optional[List[List[float]]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        r = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in r.data]
    except Exception:
        return None


def _cos(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


def _kw_overlap(a: str, b: str) -> float:
    sa = set(a.lower().split()); sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class ForecastMemory:
    def __init__(self, path: str = _STORE):
        self.path = path
        self.records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path) as f:
                    self.records = [json.loads(l) for l in f if l.strip()]
        except Exception:
            self.records = []

    def _qid(self, question_text: str) -> str:
        return hashlib.sha1(question_text.encode()).hexdigest()[:12]

    def remember_forecast(self, question_text: str, forecast: float,
                          research_summary: str, meta: Optional[Dict] = None) -> None:
        """Record a forecast at prediction time (resolution filled later)."""
        try:
            rec = {"qid": self._qid(question_text), "ts": time.time(),
                   "question": question_text[:1000], "forecast": float(forecast),
                   "research": (research_summary or "")[:2000],
                   "resolution": None, "meta": meta or {}}
            emb = _embed([question_text])
            if emb:
                rec["emb"] = emb[0]
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.records.append(rec)
        except Exception:
            pass  # never break the forecast path

    def record_resolution(self, question_text: str, resolved_value: float) -> None:
        """Update a stored forecast with its ground-truth resolution (0/1 or numeric)."""
        try:
            qid = self._qid(question_text)
            updated = False
            for r in self.records:
                if r["qid"] == qid and r.get("resolution") is None:
                    r["resolution"] = float(resolved_value); updated = True
            if updated:
                with open(self.path, "w") as f:
                    for r in self.records:
                        f.write(json.dumps(r) + "\n")
        except Exception:
            pass

    def recall_similar(self, question_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return up to k most-similar RESOLVED past forecasts with hit/miss labels."""
        try:
            resolved = [r for r in self.records if r.get("resolution") is not None]
            if not resolved:
                return []
            q_emb = _embed([question_text])
            q_emb = q_emb[0] if q_emb else None
            scored = []
            for r in resolved:
                if q_emb and "emb" in r:
                    sim = _cos(q_emb, r["emb"])
                else:
                    sim = _kw_overlap(question_text, r["question"])
                scored.append((sim, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            out = []
            for sim, r in scored[:k]:
                res = r["resolution"]; fc = r["forecast"]
                # was the past forecast on the right side?
                correct = (fc >= 0.5) == (res >= 0.5) if res in (0.0, 1.0) else None
                out.append({"similarity": round(sim, 3), "question": r["question"],
                            "past_forecast": fc, "resolution": res, "was_correct": correct})
            return out
        except Exception:
            return []

    def calibration_summary(self) -> Dict[str, Any]:
        """Track-record stats to feed back into prompts / tuning."""
        resolved = [r for r in self.records
                    if r.get("resolution") in (0.0, 1.0)]
        if not resolved:
            return {"n": 0}
        # Brier score + hit rate
        brier = sum((r["forecast"] - r["resolution"]) ** 2 for r in resolved) / len(resolved)
        hits = sum(1 for r in resolved if (r["forecast"] >= 0.5) == (r["resolution"] >= 0.5))
        return {"n": len(resolved), "brier": round(brier, 4),
                "hit_rate": round(hits / len(resolved), 3)}

    def memory_prompt_block(self, question_text: str, k: int = 3) -> str:
        """Formatted block to inject into the forecasting prompt."""
        sims = self.recall_similar(question_text, k)
        if not sims:
            return ""
        lines = ["## Memory: similar past resolved questions (learn from these)"]
        for s in sims:
            verdict = ("✓ correct" if s["was_correct"] else "✗ WRONG") if s["was_correct"] is not None else "resolved"
            lines.append(f"- (sim {s['similarity']}) '{s['question'][:120]}' — you forecast "
                         f"{s['past_forecast']:.2f}, resolved {s['resolution']} [{verdict}]")
        # Cognee knowledge-graph recall (richer cross-question reasoning) when enabled
        if _cognee_ready():
            try:
                import asyncio as _a
                cg = _a.get_event_loop().run_until_complete(_cognee_search(question_text)) \
                    if not _a.get_event_loop().is_running() else ""
                if cg:
                    lines.append("\n## Cognee knowledge-graph recall\n" + cg[:1200])
            except Exception:
                pass
        cal = self.calibration_summary()
        if cal.get("n"):
            lines.append(f"\nYour track record so far: {cal['n']} resolved, "
                         f"hit-rate {cal['hit_rate']}, Brier {cal['brier']}. "
                         f"If your Brier is high, be less confident on this class of question.")
        return "\n".join(lines)
