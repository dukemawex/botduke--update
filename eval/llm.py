"""Azure Foundry v1 client for the harness.

The endpoint is the new Foundry v1 surface
(https://<resource>.services.ai.azure.com/openai/v1/...), which is
OpenAI-compatible. That means a plain OpenAI client with a custom base_url,
not the classic azure `deployments/<name>/chat/completions?api-version=` route.
Deployment name is passed as the model.

Env:
    AZURE_API_KEY     required
    AZURE_BASE_URL    default: the Foundry v1 base for this project
    AZURE_DEPLOYMENT  default: gpt-5.6-sol
"""
from __future__ import annotations
import os, re, json
from typing import Optional
import requests

DEFAULT_BASE = "https://emmanuelduke243689-6684-resource.services.ai.azure.com/openai/v1"
DEFAULT_DEPLOYMENT = "gpt-5.6-sol"


def _cfg() -> tuple[str, str, str]:
    key = os.environ.get("AZURE_API_KEY")
    if not key:
        raise RuntimeError("AZURE_API_KEY not set")
    base = os.environ.get("AZURE_BASE_URL", DEFAULT_BASE).rstrip("/")
    dep = os.environ.get("AZURE_DEPLOYMENT", DEFAULT_DEPLOYMENT)
    return key, base, dep


def complete(prompt: str, temperature: float = 0.3, max_tokens: int = 1200,
             system: Optional[str] = None) -> str:
    """One completion. Tries chat/completions, falls back to responses."""
    key, base, dep = _cfg()
    headers = {"Authorization": f"Bearer {key}", "api-key": key,
               "Content-Type": "application/json"}
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]

    r = requests.post(f"{base}/chat/completions", headers=headers, timeout=180, json={
        "model": dep, "messages": msgs,
        "temperature": temperature, "max_completion_tokens": max_tokens,
    })
    if r.status_code == 200:
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

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
