"""Azure Foundry v1 client for the harness.

The endpoint is the new Foundry v1 surface
(https://<resource>.services.ai.azure.com/openai/v1/...), which is
OpenAI-compatible. That means a plain OpenAI client with a custom base_url,
not the classic azure `deployments/<name>/chat/completions?api-version=` route.
Deployment name is passed as the model.

Provider is switchable so the harness is not hostage to one account:

    LLM_PROVIDER=azure       AZURE_API_KEY, AZURE_BASE_URL, AZURE_DEPLOYMENT
    LLM_PROVIDER=openrouter  OPENROUTER_API_KEY, OPENROUTER_MODEL

Both speak the same OpenAI-compatible chat/completions shape, so stage 1 does
not care which one is behind it. Keep the provider fixed within a single sweep;
comparing configs across providers measures the provider, not the config.
"""
from __future__ import annotations
import os, re, json, threading, time
from typing import Optional
import requests

_RPM = int(os.environ.get("LLM_RPM", "0"))          # 0 = unlimited
_lock = threading.Lock()
_calls: list[float] = []


def _throttle() -> None:
    """Global requests-per-minute gate. OpenRouter free models cap at 20/min."""
    if _RPM <= 0:
        return
    while True:
        with _lock:
            now = time.time()
            _calls[:] = [t for t in _calls if now - t < 60.0]
            if len(_calls) < _RPM:
                _calls.append(now)
                return
            wait = 60.0 - (now - _calls[0]) + 0.05
        time.sleep(max(wait, 0.1))

DEFAULT_BASE = "https://emmanuelduke243689-6684-resource.services.ai.azure.com/openai/v1"
DEFAULT_DEPLOYMENT = "gpt-5.6-sol"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OR_MODEL = "openai/gpt-5.6-luna"


def _cfg() -> tuple[str, str, str]:
    provider = os.environ.get("LLM_PROVIDER", "azure").lower()
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY not set")
        return (key,
                os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE).rstrip("/"),
                os.environ.get("OPENROUTER_MODEL", DEFAULT_OR_MODEL))
    key = os.environ.get("AZURE_API_KEY")
    if not key:
        raise RuntimeError("AZURE_API_KEY not set (or set LLM_PROVIDER=openrouter)")
    return (key,
            os.environ.get("AZURE_BASE_URL", DEFAULT_BASE).rstrip("/"),
            os.environ.get("AZURE_DEPLOYMENT", DEFAULT_DEPLOYMENT))


def complete(prompt: str, temperature: float = 0.3, max_tokens: int = 1200,
             system: Optional[str] = None) -> str:
    """One completion. Tries chat/completions, falls back to responses."""
    key, base, dep = _cfg()
    headers = {"Authorization": f"Bearer {key}", "api-key": key,
               "Content-Type": "application/json"}
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]

    is_or = "openrouter" in base
    body = {"model": dep, "messages": msgs, "temperature": temperature}
    body["max_tokens" if is_or else "max_completion_tokens"] = max_tokens

    for attempt in range(4):
        _throttle()
        r = requests.post(f"{base}/chat/completions", headers=headers, timeout=180, json=body)
        if r.status_code == 200:
            d = r.json()
            choices = d.get("choices") or []
            if choices:
                return (choices[0].get("message", {}).get("content") or "").strip()
            # OpenRouter can return 200 with an error body or an empty completion
            err = (d.get("error") or {}).get("message", "")
            if "rate" in err.lower() or "429" in err:
                time.sleep(20 * (attempt + 1))
                continue
            raise RuntimeError(f"no choices in 200 response: {json.dumps(d)[:200]}")
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        break
    if is_or:
        raise RuntimeError(f"openrouter {r.status_code}: {r.text[:200]}")

    r2 = requests.post(f"{base}/responses", headers=headers, timeout=180, json={
        "model": dep,
        "input": ([{"role": "system", "content": system}] if system else []) +
                 [{"role": "user", "content": prompt}],
        "max_output_tokens": max_tokens,
    })
    if r2.status_code != 200:
        raise RuntimeError(f"chat {r.status_code}: {r.text[:200]} | responses {r2.status_code}: {r2.text[:200]}")
    d = r2.json()
    if isinstance(d.get("output_text"), str):
        return d["output_text"].strip()
    parts = []
    for item in d.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                parts.append(c["text"])
    return "\n".join(parts).strip()


_PROB = re.compile(r"(?:probability|answer|final)\D{0,20}?(\d{1,3}(?:\.\d+)?)\s*%|(\d{1,3}(?:\.\d+)?)\s*%\s*$",
                   re.IGNORECASE | re.MULTILINE)


def extract_probability(text: str) -> Optional[float]:
    """Last percentage in the text wins; forecasters state the answer last."""
    hits = [float(a or b) for a, b in _PROB.findall(text or "")]
    if not hits:
        hits = [float(x) for x in re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text or "")]
    if not hits:
        return None
    p = hits[-1] / 100.0
    return min(max(p, 0.0), 1.0)
