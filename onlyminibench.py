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
from collections.abc import Sequence

from forecasting_tools import MetaculusClient

from main import BotFeatureFlags, SpringAdvancedForecastingBot


class SingleModelMiniBenchBot(SpringAdvancedForecastingBot):
    def __init__(self, *args, single_model: str, **kwargs):
        self.single_model = single_model
        super().__init__(*args, **kwargs)

    def _llm_config_defaults(self):
        model = self.single_model
        return {role: model for role in (
            "default", "parser", "summarizer", "researcher",
            "query_optimizer", "critic", "red_team", "decomposer",
        )}

    def get_synthesis_model(self):
        return self.single_model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run OracleDeckv1 on MiniBench only")
    parser.add_argument("--bot-name", type=str, default="oracledeckv1")
    parser.add_argument("--model", type=str, default="openrouter/openai/gpt-5.6-luna",
                        help="One model used for research synthesis and forecasting")
    parser.add_argument("--no-decomposition", action="store_true")
    parser.add_argument("--no-meta-forecast", action="store_true")
    parser.add_argument("--no-numeric-regimes", action="store_true")
    parser.add_argument("--no-detailed-reasoning", action="store_true")
    args = parser.parse_args()

    flags = BotFeatureFlags(
        # Do not apply the generic extremization curve to MiniBench until it is
        # validated on a complete resolved round.
        enable_extremize=False,
        forecast_model_names=(args.model,),
        enable_decomposition=not args.no_decomposition,
        enable_meta_forecast=not args.no_meta_forecast,
        enable_numeric_regimes=not args.no_numeric_regimes,
        enable_detailed_reasoning=not args.no_detailed_reasoning,
    )

    bot = SingleModelMiniBenchBot(
        single_model=args.model,
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
        """Forecast every currently open question, then reconcile failures.

        The upstream helper makes one concurrent pass. MiniBench coverage is
        too important to accept that as success: provider timeouts and a newly
        opened question otherwise look identical to a completed run. We batch
        the first pass, retry only failures, then refetch the tournament and
        repeat until there are no eligible questions or the retry budget ends.
        """
        tournament_id = client.CURRENT_MINIBENCH_ID
        max_passes = 4
        batch_size = 8
        total_success = 0
        last_open = 0

        for pass_no in range(1, max_passes + 1):
            questions = client.get_all_open_questions_from_tournament(tournament_id)
            eligible = [q for q in questions if not getattr(q, "already_forecasted", False)]
            last_open = len(questions)
            logger.info(
                "MiniBench reconciliation pass %d: open=%d eligible=%d",
                pass_no, len(questions), len(eligible),
            )
            if not eligible:
                logger.info("MiniBench coverage complete: no eligible questions remain")
                break

            failures = []
            for start in range(0, len(eligible), batch_size):
                batch = eligible[start:start + batch_size]
                reports = await bot.forecast_questions(batch, return_exceptions=True)
                for question, report in zip(batch, reports):
                    if isinstance(report, BaseException):
                        failures.append(question)
                        logger.warning(
                            "MiniBench forecast failed (pass=%d, question=%s): %s",
                            pass_no, getattr(question, "id", "unknown"), report,
                        )
                    else:
                        total_success += 1

            if not failures:
                # Refetch once anyway: this catches questions opened while the
                # pass was running and confirms the server-side forecast state.
                remaining = client.get_all_open_questions_from_tournament(tournament_id)
                remaining = [q for q in remaining if not getattr(q, "already_forecasted", False)]
                if not remaining:
                    logger.info("MiniBench coverage complete: submitted=%d", total_success)
                    break
            if pass_no < max_passes:
                delay = 5 * pass_no
                logger.info("MiniBench retrying failures/new questions after %ds", delay)
                await asyncio.sleep(delay)

        remaining = client.get_all_open_questions_from_tournament(tournament_id)
        unresolved = [q for q in remaining if not getattr(q, "already_forecasted", False)]
        if unresolved:
            logger.error(
                "MiniBench coverage incomplete: open=%d unresolved=%d ids=%s",
                last_open, len(unresolved),
                ",".join(str(getattr(q, "id", "unknown")) for q in unresolved),
            )
        else:
            logger.info("MiniBench final reconciliation: all open questions forecasted")
        return reports if "reports" in locals() else []

    reports = asyncio.run(run_minibench())
    bot.log_report_summary(reports)
