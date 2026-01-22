import argparse
import asyncio
import logging
import os
import re
import json
import time
import random
import numpy as np
import pandas as pd
import dotenv
from datetime import datetime, timezone
from typing import Literal, List, Any, Dict, Union, Tuple, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# ==========================================
# 🛡️ PYDANTIC V2 SAFETY HELPERS
# ==========================================

T = TypeVar("T", bound=BaseModel)

def safe_model(model_cls: Type[T], data: Any) -> T:
    """Universal factory for Pydantic v2 models to prevent positional arg errors."""
    try:
        if isinstance(data, model_cls): return data
        if isinstance(data, (str, bytes)):
            clean_data = data.replace("```json", "").replace("```", "").strip()
            return model_cls.model_validate_json(clean_data)
        if isinstance(data, dict): return model_cls.model_validate(data)
        if hasattr(data, '__dict__'): return model_cls.model_validate(data, from_attributes=True)
        return model_cls(**data)
    except (ValidationError, TypeError, ValueError) as e:
        logger.error(f"❌ MODEL INSTANTIATION FAILED: {model_cls.__name__}")
        raise ValueError(f"Failed to create {model_cls.__name__}: {e}") from e

# ==========================================
# 🤖 GENERAL-PURPOSE ENSEMBLE BOT
# ==========================================

class SpringAdvancedForecastingBot(ForecastBot):
    """
    Universal Ensemble Bot for Spring 2026.
    Uses 3 top-tier models + Critic Layer.
    Logic: General Structural Constraints, Friction Analysis, and Bayesian Calibration.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fallback_active = False
        
        # Configure Forecaster Ensemble
        self.forecasters = {
            "gpt-5.1": "openrouter/openai/gpt-5.1",
            "gpt-5": "openrouter/openai/gpt-5",
            "claude-4.5": "openrouter/anthropic/claude-4.5-sonnet"
        }
        
        # Configure the Critic
        self.critic_model = "openrouter/openai/gpt-5"

    # ##################################### UNIVERSAL ADJUSTMENT ENGINE #####################################

    def apply_bayesian_calibration(self, estimate_pct: float) -> float:
        """
        Logit Contraction for Bias Correction.
        Calibrates model output to account for overconfidence in tail events.
        """
        p = np.clip(estimate_pct / 100.0, 0.005, 0.995)
        
        alpha = 0.92  # Contraction factor
        logit_p = np.log(p / (1 - p))
        
        # 0.08 shift handles the requested logic in a probabilistic curve
        adjusted_logit = (logit_p * alpha) + 0.08
        adjusted_p = 1 / (1 + np.exp(-adjusted_logit))
        
        return round(float(np.clip(adjusted_p * 100, 1.0, 99.0)), 2)

    def _apply_structural_constraints(self, raw_p: float, research: str) -> float:
        """
        Evaluates 'Friction' and 'Structural Constraints'.
        """
        block_triggers = ["out of reach", "impossible", "unprecedented", "structural barrier"]
        if any(t in research.lower() for t in block_triggers):
            anchor_p = min(raw_p, 1.5)
        else:
            anchor_p = raw_p

        return self.apply_bayesian_calibration(anchor_p)

    # ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Universal Multi-Source Research Workflow."""
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher")
            prompt = clean_indents(f"""
                Target Question: {question.question_text}
                
                Analyze the Structural Context:
                1. Identify the Benchmark/Target: What value or event must occur?
                2. Current Status Quo: Where do we stand currently relative to the target?
                3. Path to Success: What specific actions/events are required to hit the target?
                4. Friction/Constraints: What procedural, physical, or legal barriers exist?
                5. Expert Consensus: What do major analytical groups or markets predict?
                6. Tail Risks: Identify rare events that could force a deviation from the status quo.
            """)

            try:
                if researcher == "asknews/news-summaries":
                    return await AskNewsSearcher().call_preconfigured_version(researcher, prompt)
                elif researcher.startswith("smart-searcher"):
                    model_name = researcher.removeprefix("smart-searcher/")
                    searcher = SmartSearcher(model=model_name, num_searches_to_run=3)
                    return await searcher.invoke(prompt)
                else:
                    return await self.get_llm("researcher", "llm").invoke(prompt)
            except Exception as e:
                self._is_fallback_active = True
                logger.warning(f"Research failed: {e}")
                return "FALLBACK: Structural data unavailable. Applying prior base rates and status quo bias."

    # ##################################### CRITIC LAYER #####################################

    async def _run_critic_layer(self, question: MetaculusQuestion, research: str, forecasts: Dict[str, Any]) -> Any:
        """
        General-Purpose Logic Reviewer.
        """
        logger.info(f"[*] Engaging Universal Critic: {self.critic_model}")
        llm = GeneralLlm(model=self.critic_model, temperature=0.0)
        
        ensemble_data = json.dumps(forecasts, indent=2)
        
        prompt = clean_indents(f"""
            You are a Meta-Critic for a superforecasting ensemble.
            
            QUESTION: {question.question_text}
            CRITERIA: {question.resolution_criteria}
            RESEARCH: {research}
            ENSEMBLE OUTPUTS: {ensemble_data}
            
            CRITICAL TASKS:
            1. Bias Detection: Is the ensemble too optimistic about change, or too anchored to the past?
            2. Friction Check: Does the consensus ignore structural constraints?
            3. Convergence: If models vary wildly, identify the most logically sound outlier.
            
            Based on your analysis, provide a final refined forecast output.
        """)
        
        output_type = BinaryPrediction if isinstance(question, BinaryQuestion) else PredictedOptionList
        if isinstance(question, NumericQuestion):
            output_type = list[Percentile]

        try:
            critique = await llm.invoke(prompt)
            return await structure_output(critique, output_type, model=self.get_llm("parser", "llm"))
        except Exception as e:
            logger.error(f"Critic Layer failed: {e}")
            return list(forecasts.values())[0]

    # ##################################### FORECASTING LOGIC #####################################

    async def _get_model_forecast(self, model_id: str, question: MetaculusQuestion, research: str) -> Any:
        """Generic model invoker."""
        llm = GeneralLlm(model=model_id, temperature=0.1)
        prompt = clean_indents(f"""
            You are a professional Superforecaster. Use Structural Constraint Analysis.
            
            QUESTION: {question.question_text}
            CONTEXT: {question.background_info}
            RESEARCH: {research}
            
            INSTRUCTIONS:
            - Evaluate 'Friction': If required change is unprecedented, favor status quo.
            - Identify Scenario: What event triggers each outcome?
            - Output your reasoning and final values.
        """)
        
        output_type = BinaryPrediction if isinstance(question, BinaryQuestion) else PredictedOptionList
        if isinstance(question, NumericQuestion):
            output_type = list[Percentile]

        try:
            reasoning = await llm.invoke(prompt)
            return await structure_output(reasoning, output_type, model=self.get_llm("parser", "llm"))
        except Exception:
            return None

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        tasks = [self._get_model_forecast(m, question, research) for m in self.forecasters.values()]
        results = await asyncio.gather(*tasks)
        forecast_map = {name: (r.prediction_in_decimal if r else 0.5) for name, r in zip(self.forecasters.keys(), results)}
        
        critic_pred = await self._run_critic_layer(question, research, forecast_map)
        raw_p = critic_pred.prediction_in_decimal * 100
        
        final_p = self._apply_structural_constraints(raw_p, research)
        return ReasonedPrediction(prediction_value=final_p/100.0, reasoning=f"Ensemble: {forecast_map} | Final: {final_p}%")

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        tasks = [self._get_model_forecast(m, question, research) for m in self.forecasters.values()]
        results = await asyncio.gather(*tasks)
        forecast_map = {name: (r.model_dump() if r else {}) for name, r in zip(self.forecasters.keys(), results)}
        final_list = await self._run_critic_layer(question, research, forecast_map)
        return ReasonedPrediction(prediction_value=final_list, reasoning="Ensemble MCQ synthesis.")

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        tasks = [self._get_model_forecast(m, question, research) for m in self.forecasters.values()]
        results = await asyncio.gather(*tasks)
        forecast_map = {name: ([p.model_dump() for p in r] if r else []) for name, r in zip(self.forecasters.keys(), results)}
        final_pcts = await self._run_critic_layer(question, research, forecast_map)
        dist = NumericDistribution.from_question(final_pcts, question)
        return ReasonedPrediction(prediction_value=dist, reasoning="Ensemble Numeric synthesis.")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    bot = SpringAdvancedForecastingBot(
        publish_reports_to_metaculus=True,
        llms={
            "researcher": "smart-searcher/openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4o-mini",
        }
    )

    client = MetaculusClient()
    
    # Target Tournaments
    TARGET_TOURNAMENTS = [
        "ACX2026",
        "market-pulse-26q1",
        "32916",
        "minibench"
    ]

    async def run():
        for tournament_id in TARGET_TOURNAMENTS:
            logger.info(f"▶ Starting forecast for Tournament ID: {tournament_id}")
            try:
                # Metaculus IDs can be numeric strings or slugs
                # forecast_on_tournament handles both slug and ID
                await bot.forecast_on_tournament(tournament_id, return_exceptions=True)
            except Exception as e:
                logger.error(f"Critical error in tournament {tournament_id}: {e}")

    asyncio.run(run())
