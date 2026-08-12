"""OracleDeckv1 MiniBench-only runner.

MiniBench is deliberately a separate process from main.py. This keeps its
schedule, state, and calibration profile out of the seasonal/Market Pulse run.
The initial MiniBench policy disables generic extremization until a complete
round proves that extremization improves peer score.
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from forecasting_tools import MetaculusClient

from main import BotFeatureFlags, SpringAdvancedForecastingBot


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run OracleDeckv1 on MiniBench only")
    parser.add_argument("--bot-name", type=str, default="oracledeckv1")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-meta-forecast", action="store_true")
    parser.add_argument("--no-numeric-regimes", action="store_true")
    parser.add_argument("--no-detailed-reasoning", action="store_true")
    args = parser.parse_args()

    flags = BotFeatureFlags(
        # Do not apply the generic extremization curve to MiniBench until it is
        # validated on a complete resolved round.
        enable_extremize=False,
        enable_decomposition=not args.no_decomposition,
        enable_meta_forecast=not args.no_meta_forecast,
        enable_numeric_regimes=not args.no_numeric_regimes,
        enable_detailed_reasoning=not args.no_detailed_reasoning,
    )

    bot = SpringAdvancedForecastingBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        bot_name=args.bot_name,
        flags=flags,
    )
    client = MetaculusClient()

    async def run_minibench():
        return await bot.forecast_on_tournament(
            client.CURRENT_MINIBENCH_ID,
            return_exceptions=True,
        )

    reports = asyncio.run(run_minibench())
    bot.log_report_summary(reports)
