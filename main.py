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
from tavily import TavilyClient
import httpx

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

# Load environment variables
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 🛡️ DATA SANITIZATION UTILITIES
# ==========================================

def sanitize_llm_json(text: str) -> str:
    """Fixes common LLM JSON syntax errors (underscores, % signs, etc.)"""
    text = re.sub(r'(?<=\d)_(?=\d)', '', text)
    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f'"{match.group(1)}": {nums[0]}' if nums else match.group(0)
    text = re.sub(r'"(value|percentile)":\s*"([^"]+)"', clean_num, text)
    # Remove any trailing/leading non-JSON
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return text

T = TypeVar("T", bound=BaseModel)

def safe_model(model_cls: Type[T], data: Any) -> T:
    try:
        if isinstance(data, model_cls): 
            return data
        if isinstance(data, (str, bytes)):
            clean_data = sanitize_llm_json(data)
            return model_cls.model_validate_json(clean_data)
        if isinstance(data, dict): 
            return model_cls.model_validate(data)
        return model_cls(**data)
    except Exception as e:
        logger.error(f"❌ MODEL INSTANTIATION FAILED for {model_cls.__name__}: {e}")
        raise

# ==========================================
# 🔍 EXA SEARCH CLIENT
# ==========================================

class ExaSearcher:
    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY is required for Exa search.")
        self.base_url = "https://api.exa.ai/search"

    async def search(self, query: str, num_results: int = 5) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "numResults": num_results,
            "type": "neural",
            "useAutoprompt": True,
            "category": "news"
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                results = []
                for r in data.get("results", []):
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    snippet = r.get("text", "")[:500]
                    results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
                return "[Exa Search Results]\n" + "\n\n".join(results)
        except Exception as e:
            logger.error(f"Exa search failed: {e}")
            return "[Exa search failed]"

# ==========================================
# 🤖 ADVANCED ENSEMBLE BOT (Spring 2026)
# ==========================================

class SpringAdvancedForecastingBot(ForecastBot):
    """
    Bot leveraging Exa, AskNews, and Tavily for triple-engine research.
    Ensemble: GPT-5.1, GPT-5, Claude 4.5.
    Critic: GPT-5.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        required_keys = ["EXA_API_KEY", "TAVILY_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_CLIENT_SECRET"]
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            logger.warning(f"⚠️ Missing API keys: {missing}")
        
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.exa_searcher = ExaSearcher() if os.getenv("EXA_API_KEY") else None
        self.asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")
        
        # 🔒 PRESERVING YOUR MODEL NAMES EXACTLY AS REQUESTED
        self.forecasters = {
            "gpt-5.1": "openrouter/openai/gpt-5.1",
            "gpt-5": "openrouter/openai/gpt-5",
            "claude-4.5": "openrouter/anthropic/claude-4.5-sonnet"
        }
        self.critic_model = "openrouter/openai/gpt-5"

    def apply_bayesian_calibration(self, estimate_pct: float) -> float:
        p = np.clip(estimate_pct / 100.0, 0.005, 0.995)
        alpha = 0.92 
        logit_p = np.log(p / (1 - p))
        adjusted_logit = (logit_p * alpha) + 0.08
        adjusted_p = 1 / (1 + np.exp(-adjusted_logit))
        return round(float(np.clip(adjusted_p * 100, 1.0, 99.0)), 2)

    async def _run_tavily_search(self, query: str) -> str:
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.tavily.search(query=query, search_depth="advanced", max_results=5)
            )
            context = "\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in response.get('results', [])])
            return f"[Tavily Data]\n{context}"
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return "[Tavily search failed]"

    async def _run_exa_search(self, query: str) -> str:
        if not self.exa_searcher:
            return "[Exa not configured]"
        return await self.exa_searcher.search(query, num_results=5)

    async def _run_asknews_search(self, query: str) -> str:
        try:
            searcher = AskNewsSearcher(
                client_id=self.asknews_client_id,
                client_secret=self.asknews_client_secret
            )
            result = await searcher.call_preconfigured_version("asknews/news-summaries", query)
            return f"[AskNews Data]\n{result}"
        except Exception as e:
            logger.error(f"AskNews search failed: {e}")
            return "[AskNews search failed]"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            query = question.question_text[:200]
            tasks = [
                self._run_tavily_search(query),
                self._run_exa_search(query),
                self._run_asknews_search(query)
            ]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                cleaned = []
                for res in results:
                    if isinstance(res, Exception):
                        cleaned.append(f"[Search failed: {str(res)}]")
                    else:
                        cleaned.append(res)
                return "\n\n".join(cleaned)
            except Exception as e:
                logger.error(f"Research failure (Check API keys): {e}")
                return "Research partially failed."

    async def _run_critic_layer(self, question: MetaculusQuestion, research: str, forecasts: Dict[str, Any]) -> Any:
        llm = GeneralLlm(model=self.critic_model, temperature=0.0)
        
        # 🔧 CRITICAL: Enforce JSON-only output
        if isinstance(question, BinaryQuestion):
            schema_example = '{"prediction_in_decimal": 0.75}'
            out_type = BinaryPrediction
        elif isinstance(question, MultipleChoiceQuestion):
            example_opts = [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]
            schema_example = json.dumps({"predicted_options": example_opts})
            out_type = PredictedOptionList
        else:
            schema_example = '[{"percentile": 10, "value": 5}, {"percentile": 50, "value": 10}, {"percentile": 90, "value": 20}]'
            out_type = list[Percentile]

        prompt = clean_indents(f"""
            Question: {question.question_text}
            Research: {research}
            Ensemble Forecasts: {json.dumps(forecasts)}

            You are a superforecaster synthesizing final predictions.
            OUTPUT ONLY A VALID JSON OBJECT THAT MATCHES THIS SCHEMA EXAMPLE:
            {schema_example}

            RULES:
            - NO MARKDOWN. NO EXPLANATION. NO PREFIX/SUFFIX TEXT.
            - ONLY OUTPUT THE JSON.
            - Ensure probabilities sum to 1.0 for MCQ.
            - For binary, prediction_in_decimal must be between 0.0 and 1.0.
        """)
        
        critique = await llm.invoke(prompt)
        self._last_critique = critique
        return await structure_output(sanitize_llm_json(critique), out_type, model=self.get_llm("parser", "llm"))

    async def _get_model_forecast(self, model_id: str, question: MetaculusQuestion, research: str) -> Any:
        llm = GeneralLlm(model=model_id, temperature=0.1)
        
        # 🔧 CRITICAL: Force JSON-only response
        if isinstance(question, BinaryQuestion):
            schema_example = '{"prediction_in_decimal": 0.35}'
            out_type = BinaryPrediction
        elif isinstance(question, MultipleChoiceQuestion):
            example_opts = [{"option_name": opt, "probability": 0.5} for opt in question.options[:2]]
            schema_example = json.dumps({"predicted_options": example_opts})
            out_type = PredictedOptionList
        else:
            schema_example = '[{"percentile": 10, "value": 100}, {"percentile": 50, "value": 200}, {"percentile": 90, "value": 500}]'
            out_type = list[Percentile]

        prompt = clean_indents(f"""
            Question: {question.question_text}
            Research: {research}

            Apply 'Negative Knowledge': Why might change be impossible? Favor status quo.

            OUTPUT ONLY A VALID JSON OBJECT THAT MATCHES THIS SCHEMA EXAMPLE:
            {schema_example}

            RULES:
            - DO NOT include reasoning, markdown, or extra text.
            - ONLY output the JSON.
            - For binary: use key "prediction_in_decimal" with value between 0.0 and 1.0.
        """)
        
        raw = await llm.invoke(prompt)
        return await structure_output(sanitize_llm_json(raw), out_type, model=self.get_llm("parser", "llm"))

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        tasks = [self._get_model_forecast(m, question, research) for m in self.forecasters.values()]
        results = await asyncio.gather(*tasks)
        forecast_map = {n: (r.prediction_in_decimal if r else 0.5) for n, r in zip(self.forecasters.keys(), results)}
        
        critic_out = await self._run_critic_layer(question, research, forecast_map)
        raw_p = critic_out.prediction_in_decimal * 100
        
        if any(x in research.lower() for x in ["out of reach", "impossible", "unprecedented"]):
            raw_p = min(raw_p, 1.5)
            
        final_p = self.apply_bayesian_calibration(raw_p)
        comment = f"### Ensemble Analysis\n**Models:** {forecast_map}\n**Critic:** {self._last_critique[:1000]}..."
        return ReasonedPrediction(prediction_value=final_p/100.0, reasoning=comment)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        tasks = [self._get_model_forecast(m, question, research) for m in self.forecasters.values()]
        results = await asyncio.gather(*tasks)
        forecast_map = {n: (r.model_dump() if r else {}) for n, r in zip(self.forecasters.keys(), results)}
        
        final_list: PredictedOptionList = await self._run_critic_layer(question, research, forecast_map)
        
        option_names = question.options
        current_options = {o.option_name: o.probability for o in final_list.predicted_options}
        aligned_options = [{"option_name": name, "probability": current_options.get(name, 0.0)} for name in option_names]
        
        total = sum(o["probability"] for o in aligned_options)
        for o in aligned_options: 
            o["probability"] /= (total if total > 0 else 1.0)
        
        final_val = safe_model(PredictedOptionList, {"predicted_options": aligned_options})
        return ReasonedPrediction(prediction_value=final_val, reasoning=f"### MCQ Synthesis\n{self._last_critique[:1000]}")

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        tasks = [self._get_model_forecast(m, question, research) for m in self.forecasters.values()]
        results = await asyncio.gather(*tasks)
        forecast_map = {n: ([p.model_dump() for p in r] if r else []) for n, r in zip(self.forecasters.keys(), results)}
        
        final_pcts: list[Percentile] = await self._run_critic_layer(question, research, forecast_map)
        final_pcts.sort(key=lambda x: x.percentile)
        
        for i in range(1, len(final_pcts)):
            if final_pcts[i].value <= final_pcts[i-1].value:
                final_pcts[i].value = final_pcts[i-1].value + 1e-6
        
        dist = NumericDistribution.from_question(final_pcts, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=f"### Numeric Synthesis\n{self._last_critique[:1000]}")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    bot = SpringAdvancedForecastingBot(
        publish_reports_to_metaculus=True,
        llms={
            "researcher": "smart-searcher/openrouter/openai/gpt-4o",
            "parser": "openrouter/openai/gpt-4o-mini",
        }
    )

    TARGETS = ["32916", "minibench"]

    async def run():
        for tid in TARGETS:
            logger.info(f"▶ Tournament: {tid}")
            try:
                await bot.forecast_on_tournament(tid, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error in {tid}: {e}")

    asyncio.run(run())
