# my Spring Advanced Forecasting Bot
from metac template code
A general-purpose forecasting bot that pulls evidence from multiple news/search providers (Tavily, Exa, AskNews), ensembles multiple LLM forecasters, applies critique + red-teaming, and optionally publishes forecasts/reports to Metaculus tournaments.

It’s built on top of `forecasting_tools` (e.g., `ForecastBot`, `MetaculusClient`, question/prediction schemas) and adds:

- **Multi-source research**: Tavily + Exa + AskNews (all optional, enabled via env vars)
- **Query optimization**: turns a Metaculus question into 3 targeted search queries
- **LLM ensemble**: multiple models generate forecasts, then a critic model produces the final call
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
   - Perplexity Sonar (long-context reasoning)
   - GPT-5 search via OpenRouter
4. **Fact Verification**: Extracts claims and validates against sources; flags contradictions
5. **Regulatory Event Tracking**: Identifies SEC filings, policy changes, legal actions
6. **Market Data Integration**: Real-time crypto/stock prices for relevant keywords
7. Prepends baseline forecasting guidance:
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
