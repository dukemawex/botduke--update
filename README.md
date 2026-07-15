# my Spring Advanced Forecasting Bot
from metac template code
A general-purpose forecasting bot that pulls evidence from multiple news/search providers (Tavily, Exa, AskNews), ensembles multiple LLM forecasters, applies critique + red-teaming, and optionally publishes forecasts/reports to Metaculus tournaments.

It’s built on top of `forecasting_tools` (e.g., `ForecastBot`, `MetaculusClient`, question/prediction schemas) and adds:

- **Multi-source research**: Tavily + Exa + AskNews + AskNews OS (all optional, enabled via env vars)
- **Query optimization**: turns a Metaculus question into 3 targeted search queries
- **LLM ensemble**: multiple models generate forecasts, then a critic model produces the final call
- **Median protocol**: ensemble aggregation uses median (with GPT-5.1 added) for stronger outlier robustness
- **Safety + robustness**:
  - JSON sanitization for messy LLM outputs (e.g., `1_000`, `"percentile": "10%"`)
  - Pydantic-safe coercion into typed schemas
⁹  - disagreement-based shrinkage toward 50%
  - time-decay smoothing toward 50% for far-future close dates
  - calibration to reduce overconfidence in extremes
- **Brief “comment-style” reasoning**: outputs concise debug-style metadata (median/spread/search footprint)

---

## Features

### Research pipeline
For each question the bot:

1. **Question Classification**: Domain detection (geopolitics, economics, science, tech, business, health, sports)
2. Generates up to **3 optimized search queries** via an LLM (parallel decomposition + optimization)
3. Runs searches concurrently:
    - Tavily (advanced depth)
    - Exa (neural news search)
    - AskNews (news summaries)
    - AskNews OS (parallel news/search stream)
    - Perplexity Sonar (long-context reasoning)
    - GPT-5 search via OpenRouter
4. **Synthesis Node (GPT-5.5 Online)**: merges all parallel source outputs into:
   - qualitative context summary
   - `Signal Strength` coefficient (0.0–1.0) based on cross-source consistency
5. **Fact Verification**: Extracts claims and validates against sources; flags contradictions
6. **Regulatory Event Tracking**: Identifies SEC filings, policy changes, legal actions
7. **Market Data Integration**: Real-time crypto/stock prices for relevant keywords
8. Prepends baseline forecasting guidance + domain-specific analytical frameworks (AI Safety / Geopolitics / Finance):
   - Base rate reminder
   - Fermi decomposition guidance

### Forecasting pipeline (Binary / Multiple Choice / Numeric)
Depending on question type:

- **Binary**: 
  - Returns probability 0–1 with advanced augmentation
  - **NEW**: Scenario analysis (bull/base/bear blended 30%)
  - **NEW**: Uncertainty quantification (Monte Carlo, 90% credible intervals)
  - Critic + red-team models for robustness
  - Shrinkage + time-decay + calibration

- **Multiple Choice**: 
  - Returns normalized probabilities across options
  - Ensemble from 3 models (GPT-5.5, Claude Opus 4.6 + 4.7)

- **Numeric**:
  - Returns percentile-based distribution
  - **NEW**: Time-series momentum detection (acceleration/deceleration)
  - Regime detection + bounded delta estimation

### Advanced Features Overview

| Feature | What It Does | Where It's Used |
|---------|-------------|-----------------|
| **Fact Verification** | Extracts & validates claims against sources; flags contradictions | Research block enrichment |
| **Time-Series Momentum** | Detects trend acceleration/deceleration; projects future values | Numeric level-series forecasts |
| **Market Data** | Real-time crypto/stock prices from CoinGecko & Yahoo Finance | Research context for market-sensitive Qs |
| **Regulatory Tracker** | Identifies SEC filings, policy changes, legal actions | Research block enrichment |
| **Scenario Analysis** | Bull/base/bear scenarios with LLM generation | Binary forecasts (30% weight blend) |
| **Uncertainty Quant** | Monte Carlo posteriors, 90% credible intervals | Binary forecast reasoning & confidence |
| **Question Classifier** | Domain detection (geo, econ, science, tech, business, health, sports) | Research strategy routing |
| **Domain Router** | Routes to AI Safety / Geopolitics / Finance mental models | Research + forecast framing |
| **Synthesis + Signal Strength** | GPT-5.5 Online consolidation + consistency coefficient (0.0–1.0) | Post-research calibration |

---

## Tournament Configuration

The bot is configured to forecast on:

- **Current AI Competition** (ID: **33022**) — Primary tournament
- **MiniBench** (ID: **33022**, slug: `minibench`) — Aggressive extremization (strength=1.8 for 0.45-0.55 zone)
- **Market Pulse 26Q2** (slug: `market-pulse-26q2`) — Additional predictions

All three tournaments run in parallel during forecasting batch mode.

---

## Model Configuration

**LLM Ensemble** (as of April 2026):

| Role | Model | Purpose |
|------|-------|---------|
| default | openrouter/openai/gpt-5.5 | General reasoning |
| ensemble member | openrouter/openai/gpt-5.1 | Additional diversity + median robustness |
| parser | openrouter/openai/gpt-5-mini | Structured output |
| summarizer | openrouter/openai/gpt-5-mini | Fast summarization |
| researcher | openrouter/openai/gpt-5.5 | Long-context web search |
| query_optimizer | openrouter/openai/gpt-5-mini | Query optimization |
| critic | openrouter/anthropic/claude-opus-4.6 | Adversarial reasoning |
| red_team | openrouter/anthropic/claude-sonnet-4.6 | Challenge forecasts |
| decomposer | openrouter/openai/gpt-5-mini | Question decomposition |

---

## Requirements

- Python 3.10+ (async + typing + modern dependencies)
- `forecasting_tools` package/module available in your environment
- API keys (optional but recommended for best performance)

---

## Installation

1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

---

## Update (2026-07): calibration, memory, Nimble, minibench profile

**Motivation:** a negative tournament score is a calibration problem first — the bot was too confident on genuine toss-ups.

1. **Recalibrated extremization** — the old minibench curve pushed near-50/50 forecasts with strength 1.8 (confident-wrong). New policy: leave toss-ups (0.45–0.55) unextremized, gentle for mild leanings, moderate only for clear signals. *This is the highest-impact change for the negative score.*
2. **Agent memory (`forecast_memory.py`, Cognee-style)** — stores every forecast + research + eventual resolution; recalls the most similar past resolved questions (semantic if `OPENAI_API_KEY` set, else keyword) and injects them plus a live Brier/hit-rate track record into the forecasting prompt. Optional Cognee backend via `USE_COGNEE=1`. Learns hit/miss over time.
3. **`sync_resolutions.py`** — pulls resolved questions back into memory so calibration accumulates (run after each scoring window).
4. **Nimble research tier** — `research_nimble()` added to the parallel fan-out (enabled by `NIMBLE_API_KEY`); degrades safely to empty on failure.
5. **`--minibench-biweekly` flag** — profile hook for topping the biweekly minibench.

**Honest note:** calibration + memory are the real accuracy levers; Nimble is supplementary (already 6 sources). Score recovery depends most on the recalibration and on accumulating resolved-question memory over several cycles.

## Update (2026-07): research sources — Nimble + You.com

Run-log verification showed **Tavily** (bad call signature), **AskNews** (SDK incompatibility), and **AskNews OS** (401 auth) were silently failing every run — the bot was effectively forecasting on ~2 sources. Per decision, these are **replaced** by:
- **Nimble** (`NIMBLE_API_KEY`)
- **You.com Search** (`YOUCOM_API_KEY`)

Active research fan-out is now: Exa · Perplexity (OpenRouter) · GPT-5.1 web search · GPT-5.1 web_search tool · **Nimble** · **You.com**. Add `YOUCOM_API_KEY` in repo Settings → Secrets → Actions.

## Update (2026-07): cost-optimized OpenRouter models

Swapped the ensemble and role models to cheap-but-top-tier verified OpenRouter IDs:
- **Forecast ensemble:** gpt-5.6-luna ($1/$6) · deepseek-v4-pro ($0.44/$0.87) · kimi-k2.6 ($0.66/$3.41) · claude-haiku-4.5 · perplexity/sonar-pro
- **Synthesis / critic / default:** gpt-5.6-luna (replaces pricier gpt-5.5)
- **Red-team:** deepseek-v4-pro (replaces o3)
- **Utility (parser/summarizer/query/decomposer):** gpt-5-mini · deepseek-v4-flash · gpt-4o-mini

Weights unchanged (0.30/0.25/0.20/0.15/0.10, luna highest). All IDs verified live on OpenRouter to avoid the silent-failure trap that hit gpt-5.5-online.
