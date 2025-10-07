import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal, List, Any, Dict, Union, Tuple
import numpy as np

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


class ConfidenceWeightedEnsembleBot2025(ForecastBot):
    """
    Multi-model ensemble bot with:
    - Researcher: gpt-5 (OpenRouter)
    - Forecasters: gpt-5, gpt-4o-mini, claude-sonnet-4.5
    - Confidence-weighted aggregation
    - Dynamic model weighting per question type
    - Targets tournaments 32813, 32831, and Minibench
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    FORECAST_MODELS = {
        "gpt-5": "openrouter/openai/gpt-5",
        "gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
        "claude-sonnet-4.5": "openrouter/anthropic/claude-sonnet-4.5",
    }

    BASE_WEIGHTS = {
        "gpt-5": 1.0,
        "gpt-4o-mini": 0.9,
        "claude-sonnet-4.5": 1.0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_question = None

    async def run_research(self, question: MetaculusQuestion) -> str:
        self._current_question = question
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher")
            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                Generate a concise but detailed rundown of the most relevant news, trends, and contextual factors.
                Do not produce forecasts.

                Question:
                {question.question_text}

                Resolution Criteria:
                {question.resolution_criteria}

                Fine Print:
                {question.fine_print}
                """
            )
            research = await researcher.invoke(prompt)
            logger.info(f"Research for {question.page_url}:\n{research}")
            return research

    def get_llm(self, role: str, return_type: str = "model_name") -> Any:
        if role == "default":
            raise RuntimeError("Do not call get_llm('default') directly in this bot.")
        return super().get_llm(role, return_type)

    async def _run_forecast_with_confidence(
        self, question: MetaculusQuestion, research: str, model_key: str
    ) -> Tuple[ReasonedPrediction, float]:
        model_name = self.FORECAST_MODELS[model_key]
        llm = GeneralLlm(model=model_name, temperature=0.3, timeout=45, allowed_tries=2)

        common_prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Research Summary: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}
            """
        )

        if isinstance(question, BinaryQuestion):
            prompt = common_prompt + clean_indents(
                """
                Before answering:
                (a) Time until resolution
                (b) Status quo outcome
                (c) Scenario for No
                (d) Scenario for Yes

                End with: "Probability: ZZ%" and "Confidence: WW%" (0–100)
                """
            )
            reasoning = await llm.invoke(prompt)
            pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction, model=llm)
            value = max(0.01, min(0.99, pred.prediction_in_decimal))
            confidence = self._extract_confidence(reasoning)

        elif isinstance(question, MultipleChoiceQuestion):
            prompt = common_prompt + clean_indents(
                f"""
                Options: {question.options}

                Before answering:
                (a) Time until resolution
                (b) Status quo option
                (c) Unexpected scenario

                End with probabilities for each option in order, then "Confidence: WW%"
                """
            )
            parsing_instructions = f"Valid options: {question.options}"
            reasoning = await llm.invoke(prompt)
            pred: PredictedOptionList = await structure_output(
                reasoning, PredictedOptionList, model=llm, additional_instructions=parsing_instructions
            )
            value = pred
            confidence = self._extract_confidence(reasoning)

        elif isinstance(question, NumericQuestion):
            upper, lower = self._create_upper_and_lower_bound_messages(question)
            prompt = common_prompt + clean_indents(
                f"""
                Units: {question.unit_of_measure or 'inferred'}
                {lower}
                {upper}

                Before answering:
                (a) Time until resolution
                (b) Status quo value
                (c) Trend continuation
                (d) Expert expectations
                (e) Low-outcome scenario
                (f) High-outcome scenario

                End with percentiles (10,20,40,60,80,90) and "Confidence: WW%"
                """
            )
            reasoning = await llm.invoke(prompt)
            percentiles: List[Percentile] = await structure_output(reasoning, list[Percentile], model=llm)
            value = NumericDistribution.from_question(percentiles, question)
            confidence = self._extract_confidence(reasoning)

        else:
            raise TypeError(f"Unsupported question type: {type(question)}")

        return ReasonedPrediction(prediction_value=value, reasoning=reasoning), confidence

    def _extract_confidence(self, text: str) -> float:
        import re
        match = re.search(r"Confidence:\s*([\d.]+)%?", text, re.IGNORECASE)
        if match:
            conf = float(match.group(1)) / 100.0
            return min(1.0, max(0.1, conf))
        return 0.7

    def _get_dynamic_weights(self, question: MetaculusQuestion) -> Dict[str, float]:
        weights = self.BASE_WEIGHTS.copy()
        q_text = question.question_text.lower()

        # Favor Claude for labor/social dynamics (e.g., strikes)
        if any(kw in q_text for kw in ["strike", "labor", "union", "workforce", "social"]):
            weights["claude-sonnet-4.5"] *= 1.2

        # Favor GPT-5 for long-range tech/existential questions
        if any(kw in q_text for kw in ["extinction", "ai catastrophe", "by 2100", "leading labs"]):
            weights["gpt-5"] *= 1.3

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        return await self._run_generic_forecast(question, research)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        return await self._run_generic_forecast(question, research)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_generic_forecast(question, research)

    async def _run_generic_forecast(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction:
        weights = self._get_dynamic_weights(question)
        results = []

        for model_key in self.FORECAST_MODELS:
            pred, conf = await self._run_forecast_with_confidence(question, research, model_key)
            effective_weight = weights[model_key] * conf
            results.append((pred, effective_weight, model_key))
            logger.info(f"Model {model_key}: confidence={conf:.2f}, weight={effective_weight:.2f}")

        if isinstance(question, BinaryQuestion):
            values = [r[0].prediction_value for r in results]
            weighted_median_val = self._weighted_median(values, [r[1] for r in results])
            combined_reasoning = "\n\n".join(f"[{r[2]}] {r[0].reasoning}" for r in results)
            return ReasonedPrediction(prediction_value=weighted_median_val, reasoning=combined_reasoning)

        elif isinstance(question, MultipleChoiceQuestion):
            options = question.options
            weighted_probs = np.zeros(len(options))
            total_weight = sum(r[1] for r in results)
            for pred, weight, _ in results:
                for i, opt in enumerate(options):
                    weighted_probs[i] += pred.get_probability_for_option(opt) * weight
            if total_weight > 0:
                weighted_probs /= total_weight
            final_pred = PredictedOptionList.from_option_probabilities(options, weighted_probs.tolist())
            combined_reasoning = "\n\n".join(f"[{r[2]}] {r[0].reasoning}" for r in results)
            return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

        elif isinstance(question, NumericQuestion):
            target_percentiles = [10, 20, 40, 60, 80, 90]
            median_percentiles = []
            for p in target_percentiles:
                vals = []
                weights_list = []
                for pred, weight, _ in results:
                    dist: NumericDistribution = pred.prediction_value
                    val = dist.get_percentile_value(p)
                    if val is not None:
                        vals.append(val)
                        weights_list.append(weight)
                if vals:
                    med_val = self._weighted_median(vals, weights_list)
                    median_percentiles.append(Percentile(percentile=p, value=med_val))
                else:
                    median_percentiles.append(Percentile(percentile=p, value=0))
            final_dist = NumericDistribution.from_question(median_percentiles, question)
            combined_reasoning = "\n\n".join(f"[{r[2]}] {r[0].reasoning}" for r in results)
            return ReasonedPrediction(prediction_value=final_dist, reasoning=combined_reasoning)

        else:
            raise TypeError("Unsupported question type")

    def _weighted_median(self, values: List[float], weights: List[float]) -> float:
        sorted_pairs = sorted(zip(values, weights), key=lambda x: x[0])
        total_weight = sum(weights)
        cum_weight = 0
        for val, w in sorted_pairs:
            cum_weight += w
            if cum_weight >= total_weight / 2:
                return val
        return sorted_pairs[-1][0] if sorted_pairs else 0.5

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        ub = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lb = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound

        upper_msg = (
            f"The outcome can not be higher than {ub}."
            if not question.open_upper_bound
            else f"The question creator thinks the number is likely not higher than {ub}."
        )
        lower_msg = (
            f"The outcome can not be lower than {lb}."
            if not question.open_lower_bound
            else f"The question creator thinks the number is likely not lower than {lb}."
        )
        return upper_msg, lower_msg


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["32813", "32831", "minibench", "both"],
        default="both",
        help="Tournament to forecast on: 32813 (Market Pulse 25Q4), 32831, minibench, or both"
    )
    args = parser.parse_args()

    bot = ConfidenceWeightedEnsembleBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "researcher": GeneralLlm(
                model="openrouter/openai/gpt-5",
                temperature=0.3,
                timeout=60,
                allowed_tries=2,
            ),
            "parser": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.0,
                timeout=30,
                allowed_tries=2,
            ),
        },
    )

    async def run():
        reports = []
        if args.mode in ("32813", "both"):
            r1 = await bot.forecast_on_tournament(32813, return_exceptions=True)
            reports.extend(r1)
        if args.mode in ("32831", "both"):
            r2 = await bot.forecast_on_tournament(32831, return_exceptions=True)
            reports.extend(r2)
        if args.mode in ("minibench", "both"):
            r3 = await bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True)
            reports.extend(r3)
        bot.log_report_summary(reports)

    asyncio.run(run())
