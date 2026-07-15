"""Ensemble knowledge-cutoff probe for botduke.
Measures each forecasting model's effective knowledge cutoff via dated-event bisection
(github.com/dukemawex/llm-knowledge-cutoff-probe) and logs it alongside forecasts.

IMPORTANT: botduke feeds LIVE RESEARCH (Exa/Perplexity/Nimble/You.com) into every forecast,
so the parametric cutoff is a floor on 'what the model knows unaided', not the effective
information horizon of the pipeline. We report BOTH: the model's measured cutoff AND the note
that retrieval augments it at query time."""
from __future__ import annotations
import os, json, asyncio, datetime, aiohttp

_KEY = os.getenv("OPENROUTER_API_KEY")
_URL = "https://openrouter.ai/api/v1/chat/completions"
_CACHE = os.getenv("CUTOFF_CACHE_PATH", "cutoff_cache.json")

# Compact 12-item sourced ladder (subset of the published ladder).
_LADDER = [
 ("2024-06","Which country won the UEFA Euro 2024 final?",["spain"]),
 ("2024-09","Which company announced the iPhone 16 in September 2024?",["apple"]),
 ("2024-11","Who won the 2024 US presidential election?",["trump"]),
 ("2025-01","Which Chinese AI lab released the R1 reasoning model in Jan 2025?",["deepseek"]),
 ("2025-02","Which team won Super Bowl LIX (Feb 2025)?",["eagles","philadelphia"]),
 ("2025-04","Which Catholic pope died in April 2025?",["francis"]),
 ("2025-05","Which club won the 2025 UEFA Champions League final?",["psg","paris saint"]),
 ("2025-07","Which club won the 2025 FIFA Club World Cup?",["chelsea"]),
 ("2025-09","Which company announced the iPhone 17 in September 2025?",["apple"]),
 ("2025-11","Name a verifiable news event from November 2025.",["__open__"]),
 ("2026-02","Name a verifiable news event from February 2026.",["__open__"]),
 ("2026-06","Name a verifiable news event from June 2026.",["__open__"]),
]
_UNC=("i don't","i do not","don't know","not aware","cutoff","unable","no verifiable",
      "not sure","in the future","doesn't extend","doesn't cover","not reliably","don't reliably")

async def _ask(session, model, q):
    try:
        async with session.post(_URL, headers={"Authorization":f"Bearer {_KEY}"},
            json={"model":model,"messages":[{"role":"user","content":q+" Answer concisely. If you don't reliably know, say you don't know."}],
                  "max_tokens":120,"temperature":0}, timeout=aiohttp.ClientTimeout(total=60)) as r:
            j=await r.json()
            return (j.get("choices",[{}])[0].get("message",{}) or {}).get("content","") or ""
    except Exception:
        return ""

def _disclaimed(t): 
    t=t.lower().replace("\u2019","'"); return any(u in t for u in _UNC)

async def measure_cutoff(model: str) -> dict:
    """Return {'model','cutoff','method'} — contiguous-known month before first miss."""
    async with aiohttp.ClientSession() as s:
        cutoff=None
        for month,q,keys in _LADDER:
            ans=await _ask(s, model, q)
            if keys==["__open__"]:
                known = ans.strip()!="" and not _disclaimed(ans)
            else:
                known = any(k in ans.lower() for k in keys)
            if known: cutoff=month
            else: break
    return {"model":model,"cutoff":cutoff,"method":"dated-event-bisection",
            "measured_at":datetime.date.today().isoformat()}

async def measure_ensemble(models: list[str]) -> dict:
    out={}
    for m in models:
        out[m]=await measure_cutoff(m)
    # cache
    try: json.dump(out, open(_CACHE,"w"), indent=2)
    except Exception: pass
    return out

def cutoff_caveat_block(cutoffs: dict) -> str:
    """A note to attach to forecasts. Emphasizes research augments parametric cutoff."""
    lines=["[MODEL KNOWLEDGE CUTOFFS — measured via dated-event probe]"]
    for m,d in cutoffs.items():
        lines.append(f"  {m}: parametric knowledge through ~{d.get('cutoff') or 'unknown'}")
    lines.append("NOTE: botduke injects live research (Exa/Perplexity/Nimble/You.com) at query "
                 "time, so the effective information horizon EXCEEDS these parametric cutoffs. "
                 "Cutoffs indicate unaided recall; retrieval covers the grey zone.")
    return "\n".join(lines)

if __name__=="__main__":
    import sys
    models=sys.argv[1:] or ["openai/gpt-5.6-luna","deepseek/deepseek-v4-pro","moonshotai/kimi-k2.6","anthropic/claude-haiku-4.5","perplexity/sonar-pro"]
    res=asyncio.run(measure_ensemble(models))
    print(json.dumps(res,indent=2))
    print(cutoff_caveat_block(res))
