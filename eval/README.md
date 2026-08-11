# eval — offline forecast harness

Replaces vibes-tuning with measurement. Nothing here touches the live tournament.

## Why two stages

Research + the LLM ensemble is the only expensive part. It runs **once** per
question and is cached. Calibration then reads the cache, so sweeping hundreds
of aggregation configs costs zero tokens.

    stage 1  research + ensemble  -> stage1.jsonl   (Azure credits go here)
    stage 2  calibration + score  -> free, unlimited

## Files

- `fetch_corpus.py` — pulls resolved Metaculus questions. The list endpoint does
  not serialize `resolution`; only the detail endpoint does, so this is
  list-then-hydrate. The API rate-limits hard: keep workers <= 3 and expect 429s.
- `calibration.py` — `legacy()` reproduces `main.py@main` exactly (six sequential
  pulls toward 0.5). `v2()` replaces it with one bounded shrink in log-odds space
  toward a base-rate prior. Every constant in `V2Config` is a sweep axis.
- `score.py` — log score, Brier, calibration error, and a baseline-relative
  score used as a peer-score proxy. Metaculus peer score compares you to other
  forecasters, which we cannot see offline; this ranks variants, it is not the
  real number.
- `replay.py` — `compare` runs legacy vs v2; `sweep` grid-searches `V2Config`.

## Measured facts (corpus of 190 resolved binary questions, May-Aug 2026)

- Base rate: **60 YES / 130 NO = 31.6% YES**. Shrinking toward 0.5 is therefore
  a systematic bias toward YES, not a neutral safety measure.
- Bot API tokens do **not** receive community-prediction aggregates
  (`community_latest` was null on all 190). The `community-blend` branch in
  `main.py` is dead code.

## Usage

    export METACULUS_TOKEN=...
    python eval/fetch_corpus.py --months 18 --limit 900 --workers 3
    python eval/replay.py compare --stage1 eval/stage1.jsonl
    python eval/replay.py sweep   --stage1 eval/stage1.jsonl
