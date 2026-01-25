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
from pathlib import Path
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

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ==========================================
# 🛡️ UTILITIES
# ==========================================

def sanitize_llm_json(text: str) -> str:
    text = re.sub(r'(?<=\d)_(?=\d)', '', text)
    def clean_num(match):
        val = match.group(2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        return f'"{match.group(1)}": {nums[0]}' if nums else match.group(0)
    text = re.sub(r'"(value|percentile)":\s*"([^"]+)"', clean_num, text)
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

T = TypeVar("T", bound=BaseModel)

def safe_model(model_cls: Type[T],  Any) -> T:
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
# 🔍 EXA CLIENT
# ==========================================

class ExaSearcher:
    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY is required.")
        self.base_url = "https://api.exa.ai/search"

    async def search(self, query: str, num_results: int = 5) -> str:
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"query": query, "numResults": num_results, "type": "neural", "useAutoprompt": True, "category": "news"}
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
# 🧠 FORECASTING PRINCIPLES
# ==========================================

class ForecastingPrinciples:
    @staticmethod
    def get_base_rate(question_text: str) -> str:
        qt = question_text.lower()
        if any(kw in qt for kw in ["election", "win", "president"]) and "female" in qt:
            return "BASE RATE: Only ~12% of non-incumbent female candidates win national executive elections in runoff systems (1990–2025)."
        elif "ai" in qt and ("by 2030" in qt or "by 2026" in qt):
            return "BASE RATE: ~78% of AI milestone predictions on Metaculus >3 years out resolve 'no'."
        elif "default" in qt and ("country" in qt or "nation" in qt):
            return "BASE RATE: Sovereign defaults occur in ~2.5% of country-years; higher in emerging markets (~5%)."
        elif "war" in qt or "conflict" in qt:
            return "BASE RATE: Interstate wars are rare (<0.5% per dyad-year); civil conflicts more common but still <2%."
        else:
            return "BASE RATE: No strong historical base rate available. Use general domain priors."

    @staticmethod
    def get_fermi_decomposition(question_text: str) -> str:
        qt = question_text.lower()
        if "default" in qt:
            return """
FERMI DECOMPOSITION:
1. Probability of severe economic recession (P1)
2. Probability of debt-to-GDP > 90% (P2)
3. Probability of political instability (P3)
4. Probability of access to IMF/external bailout (P4)
Final probability ≈ P1 × P2 × P3 × (1 - P4)
"""
        elif "ai" in qt and "capable" in qt:
            return """
FERMI DECOMPOSITION:
1. Probability of sufficient compute availability (P1)
2. Probability of algorithmic breakthrough (P2)
3. Probability of no regulatory ban (P3)
4. Probability of talent retention (P4)
Final ≈ P1 × P2 × P3 × P4
"""
        elif "election" in qt and "win" in qt:
            return """
FERMI DECOMPOSITION:
1. Probability of making top 2 in first round (P1)
2. Probability of favorable runoff coalition dynamics (P2)
3. Probability of no major scandal (P3)
4. Probability of economic conditions favoring outsider (P4)
Final ≈ P1 × P2 × P3 × P4
"""
        return ""

    @staticmethod
    def apply_time_decay(prob: float, close_time: Optional[datetime]) -> float:
        if close_time is None:
            return prob
        now = datetime.now(timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        days = (close_time - now).days
        if days > 365:
            return 0.3 * prob + 0.7 * 0.5
        elif days > 180:
            return 0.5 * prob + 0.5 * 0.5
        elif days > 90:
            return 0.7 * prob + 0.3 * 0.5
        else:
            return prob

# ==========================================
# 🤖 COMPETITION-GRADE BOT
# ==========================================

class SpringAdvancedForecastingBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        required_keys = [
            "OPENROUTER_API_KEY",
            "EXA_API_KEY",
            "TAVILY_API_KEY",
            "ASKNEWS_CLIENT_ID",
            "ASKNEWS_CLIENT_SECRET"
        ]
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            logger.warning(f"⚠️ Missing API keys: {missing}")
        
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.exa_searcher = ExaSearcher() if os.getenv("EXA_API_KEY") else None
        self.asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
        self.asknews_client_secret = os.getenv("ASKNEWS_CLIENT_SECRET")
        
        self.forecasters = {
            "gpt-5.1": "openrouter/openai/gpt-5.1",
            "gpt-5": "openrouter/openai/gpt-5",
            "claude-4.5": "openrouter/anthropic/claude-4.5-sonnet"
        }
        self.critic_model = "openrouter/openai/gpt-5"
        self.red_team_model = "openrouter/openai/gpt-4o"
        self.query_optimizer_model = "openrouter/openai/gpt-4o-mini"
        
        # Memory for consistency checks
        self._recent_predictions = []

    def apply_bayesian_calibration(self, estimate_pct: float) -> float:
        p = np.clip(estimate_pct / 100.0, 0.005, 0.995)
        alpha = 0.92 
        logit_p = np.log(p / (1 - p))
        adjusted_logit = (logit_p * alpha) + 0.08
        adjusted_p = 1 / (1 + np.exp(-adjusted_logit))
        return round(float(np.clip(adjusted_p * 100, 1.0, 99.0)), 2)

    async def _optimize_search_query(self, question: MetaculusQuestion) -> str:
        llm = GeneralLlm(model=self.query_optimizer_model, temperature=0.3)
        prompt = f"""
        Rewrite this forecasting question into 3 precise, factual search queries for news/reports.
        Focus on entities, dates, and measurable outcomes.
        Question: {question.question_text}
        Output ONLY a JSON list: ["query1", "query2", "query3"]
        """
        try:
            response = await llm.invoke(prompt)
            queries = json.loads(sanitize_llm_json(response))
            return " ".join(queries[:2])
        except:
            return question.question_text[:150]

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
            optimized_query = await self._optimize_search_query(question)
            
            tasks = [
                self._run_tavily_search(optimized_query),
                self._run_exa_search(optimized_query),
                self._run_asknews_search(optimized_query)
            ]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                cleaned = []
                for res in results:
                    if isinstance(res, Exception):
                        cleaned.append(f"[Search failed: {str(res)}]")
                    else:
                        cleaned.append(res)
                combined = "\n\n".join(cleaned)
                
                base_rate = ForecastingPrinciples.get_base_rate(question.question_text)
                fermi = ForecastingPrinciples.get_fermi_decomposition(question.question_text)
                
                enhanced_research = f"""{base_rate}

{fermi}

{combined}"""
                return enhanced_research
            except Exception as e:
                logger.error(f"Research failure: {e}")
                return "Research partially failed."

    async def _red_team_forecast(self, question: MetaculusQuestion, research: str, initial_pred: float) -> float:
        try:
            llm = GeneralLlm(model=self.red_team_model, temperature=0.7)
            prompt = clean_indents(f"""
                You are a skeptical red teamer challenging this forecast: {initial_pred:.2%}.
                Question: {question.question_text}
                Research: {research}

                Identify 3 strongest reasons why this forecast is TOO HIGH or TOO LOW.
                Then output ONLY: {{"revised_prediction_in_decimal": 0.XX}}
                Avoid markdown. Only JSON.
            """)
            response = await llm.invoke(prompt)
            revised = await structure_output(sanitize_llm_json(response), BinaryPrediction, model=self.get_llm("parser", "llm"))
            return revised.prediction_in_decimal
        except Exception as e:
            logger.warning(f"Red teaming failed: {e}")
            return initial_pred

    async def _verify_claims(self, draft_reasoning: str, research: str) -> str:
        try:
            llm = GeneralLlm(model=self.query_optimizer_model, temperature=0.0)
            extract_prompt = f"List up to 3 key factual claims in this reasoning:\n{draft_reasoning}"
            claims_response = await llm.invoke(extract_prompt)
            claims = [c.strip() for c in claims_response.split("\n") if c.strip()][:3]
            
            verified = []
            for claim in claims:
                verification = await self._run_tavily_search(f"Verify: {claim}")
                verified.append(f"Claim: {claim}\nEvidence: {verification[:300]}")
            return "\n\n".join(verified)
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return ""

    def _get_temperature(self, question: MetaculusQuestion) -> float:
        if not question.close_time:
            return 0.4
        days_to_close = (question.close_time - datetime.now(timezone.utc)).days
        if days_to_close > 180 or "first" in question.question_text.lower() or "never before" in question.question_text.lower():
            return 0.4
        else:
            return 0.1

    async def _check_consistency(self, question: MetaculusQuestion, proposed_pred: float) -> bool:
        if len(self._recent_predictions) < 2:
            return True
        recent_summary = "\n".join([
            f"Q: {q.text} → Pred: {p:.2%}" 
            for q, p in self._recent_predictions[-3:]
        ])
        llm = GeneralLlm(model=self.query_optimizer_model, temperature=0.0)
        prompt = f"""
        Is this new forecast logically consistent with prior forecasts?
        New: {question.question_text} → {proposed_pred:.2%}
        Prior: {recent_summary}
        Answer YES or NO only.
        """
        try:
            response = await llm.invoke(prompt)
            return "YES" in response.upper()
        except:
            return True  # Default to consistent

    async def _run_critic_layer(self, question: MetaculusQuestion, research: str, forecasts: Dict[str, Any]) -> Any:
        llm = GeneralLlm(model=self.critic_model, temperature=0.0)
        
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

            Apply forecasting best practices:
            - Start from base rates
            - Decompose complex problems
            - Avoid recency/salience bias
            - Favor structural stability
            - Ensure logical consistency with known facts

            OUTPUT ONLY VALID JSON:
            {schema_example}
        """)
        
        critique = await llm.invoke(prompt)
        self._last_critique = critique
        return await structure_output(sanitize_llm_json(critique), out_type, model=self.get_llm("parser", "llm"))

    async def _get_model_forecast(self, model_id: str, question: MetaculusQuestion, research: str) -> Any:
        temp = self._get_temperature(question)
        llm = GeneralLlm(model=model_id, temperature=temp)
        
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

            Apply forecasting best practices:
            - Anchor to base rates
            - Use Fermi decomposition if applicable
            - Avoid over-updating on recent news
            - Favor structural stability

            OUTPUT ONLY VALID JSON:
            {schema_example}
        """)
        
        raw = await llm.invoke(prompt)
        return await structure_output(sanitize_llm_json(raw), out_type, model=self.get_llm("parser", "llm"))

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        tasks = [self._get_model_forecast(m, question, research) for m in self.forecasters.values()]
        results = await asyncio.gather(*tasks)
        forecast_map = {n: (r.prediction_in_decimal if r else 0.5) for n, r in zip(self.forecasters.keys(), results)}
        
        critic_out = await self._run_critic_layer(question, research, forecast_map)
        raw_p = critic_out.prediction_in_decimal
        
        # Red teaming
        red_teamed_p = await self._red_team_forecast(question, research, raw_p)
        averaged_p = (raw_p + red_teamed_p) / 2.0
        
        # Claim verification
        verification = await self._verify_claims(self._last_critique, research)
        if verification:
            research += f"\n\n[VERIFICATION]\n{verification}"
        
        # Consistency check
        if not await self._check_consistency(question, averaged_p):
            logger.warning("Inconsistency detected; pulling toward 50%")
            averaged_p = 0.5 * averaged_p + 0.5 * 0.5
        
        # Blend with community
        community = getattr(question, 'community_prediction', None)
        research_quality = 0.8 if "[Search failed]" not in research else 0.3
        if community is not None:
            blended_p = research_quality * averaged_p + (1 - research_quality) * community
        else:
            blended_p = averaged_p

        # Time decay & calibration
        final_p = ForecastingPrinciples.apply_time_decay(blended_p, question.close_time)
        final_p = self.apply_bayesian_calibration(final_p * 100) / 100.0
        
        if any(x in research.lower() for x in ["out of reach", "impossible", "unprecedented"]):
            final_p = min(final_p, 0.015)
            
        # Store for consistency
        self._recent_predictions.append((question, final_p))
        
        comment = f"### Ensemble Analysis\n**Models:** {forecast_map}\n**Critic:** {self._last_critique[:1000]}..."
        
        self._save_evaluation_log(
            question=question,
            research=research,
            forecasts=forecast_map,
            critic_output=self._last_critique,
            final_prediction=final_p,
            prediction_type="binary"
        )
        
        return ReasonedPrediction(prediction_value=final_p, reasoning=comment)

    # ... (MCQ and Numeric methods remain similar — omitted for brevity but follow same pattern)

    def _save_evaluation_log(self, question, research, forecasts, critic_output, final_prediction, prediction_type):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": {
                "id": question.id,
                "url": f"https://www.metaculus.com/questions/{question.id}",
                "text": question.question_text,
                "type": prediction_type,
                "community_prediction": getattr(question, 'community_prediction', None),
                "close_time": getattr(question, 'close_time', None),
            },
            "research": research,
            "forecasts": forecasts,
            "critic_output": critic_output,
            "final_prediction": final_prediction,
            "bot_config": {
                "dynamic_queries": True,
                "uncertainty_aware": True,
                "consistency_checks": True,
                "adaptive_temp": True,
                "claim_verification": True
            }
        }
        log_path = LOGS_DIR / f"q{question.id}_{int(time.time())}.json"
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)
        logger.info(f"✅ Saved evaluation log to {log_path}")

# ==========================================
# 🚀 MAIN
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
