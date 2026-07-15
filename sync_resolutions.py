"""Sync resolved Metaculus questions into forecast memory so the bot learns hit/miss over time.
Run periodically (e.g. after each tournament scoring window):

    python sync_resolutions.py

It reads recorded forecasts from forecast_memory.jsonl, checks each question's resolution via
forecasting_tools' MetaculusClient, and writes resolutions back into memory. Safe to re-run."""
from __future__ import annotations
import os, json, asyncio
from forecast_memory import ForecastMemory

async def main():
    mem = ForecastMemory()
    pending = [r for r in mem.records if r.get("resolution") is None]
    if not pending:
        print("No pending forecasts to resolve."); print(mem.calibration_summary()); return
    print(f"{len(pending)} pending forecasts. Attempting resolution lookup...")
    try:
        from forecasting_tools import MetaculusApi  # type: ignore
    except Exception:
        print("forecasting_tools not available; cannot auto-resolve. "
              "You can manually call mem.record_resolution(question_text, 0/1).")
        return
    resolved_n = 0
    for r in pending:
        try:
            # Best-effort: search by question text is unreliable; if you stored a URL/id in meta,
            # use it. This is a template — adapt to your MetaculusApi version.
            qid = r.get("meta", {}).get("metaculus_id")
            if not qid:
                continue
            q = MetaculusApi.get_question_by_id(qid)
            res = getattr(q, "resolution", None)
            if res is None:
                continue
            val = 1.0 if str(res).lower() in ("yes", "true", "1") else 0.0
            mem.record_resolution(r["question"], val); resolved_n += 1
        except Exception as e:
            print(f"  skip {r['qid']}: {e}")
    print(f"Resolved {resolved_n} new. Track record now:")
    print(json.dumps(mem.calibration_summary(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
