# my Spring Advanced Forecasting Bot

A general-purpose forecasting bot that pulls evidence from multiple news/search providers (Tavily, Exa, AskNews), ensembles multiple LLM forecasters, applies critique + red-teaming, and optionally publishes forecasts/reports to Metaculus tournaments.

It’s built on top of `forecasting_tools` (e.g., `ForecastBot`, `MetaculusClient`, question/prediction schemas) and adds:

- **Multi-source research**: Tavily + Exa + AskNews (all optional, enabled via env vars)
- **Query optimization**: turns a Metaculus question into 3 targeted search queries
- **LLM ensemble**: multiple models generate forecasts, then a critic model produces the final call
- **Safety + robustness**:
  - JSON sanitization for messy LLM outputs (e.g., `1_000`, `"percentile": "10%"`)
  - Pydantic-safe coercion into typed schemas
  - disagreement-based shrinkage toward 50%
  - time-decay smoothing toward 50% for far-future close dates
  - calibration to reduce overconfidence in extremes
- **Brief “comment-style” reasoning**: outputs concise debug-style metadata (median/spread/search footprint)

---

## Features

### Research pipeline
For each question the bot:

1. Generates up to **3 optimized search queries** via an LLM.
2. Runs the searches concurrently:
   - Tavily (advanced depth)
   - Exa (neural news search)
   - AskNews (news summaries)
3. Prepends baseline forecasting guidance:
   - Base rate reminder
   - Fermi decomposition guidance

### Forecasting pipeline (Binary / Multiple Choice / Numeric)
Depending on question type:

- **Binary**: returns a probability in decimal (0–1)
- **Multiple choice**: returns normalized probabilities across the provided options
- **Numeric**: returns a percentile-based distribution converted into `NumericDistribution`

Binary questions include additional steps:
- critic model produces a refined forecast from the ensemble
- red team model challenges the critic’s probability and proposes a revision
- shrinkage toward 0.5 when models disagree a lot
- optional internal claim verification (lightweight; does not inflate final reasoning output)
- consistency check vs recent predictions
- optional blend with Metaculus community prediction (weighted by research footprint)
- time decay + calibration + “near-impossible” cap

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
