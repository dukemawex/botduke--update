import argparse
import asyncio
import logging
import os
import re
import json
from datetime import datetime, timezone
from typing import Literal, List, Any, Dict, Union, Tuple, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from scipy import stats
from tavily import TavilyClient
from asknews_sdk import AskNewsSDK
from pydantic import BaseModel, ValidationError

# Import forecasting tools
# Ensure you have these installed/available in your environment
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

# ==========================================
# 🛡️ PYDANTIC V2 SAFETY HELPERS
# ==========================================

T = TypeVar("T", bound=BaseModel)

def safe_model(model_cls: Type[T], data: Any) -> T:
    """
    Universal factory for Pydantic v2 models to prevent __init__ positional arg errors.
    Safely handles: Existing Instances, Dicts, JSON Strings, and Duck Typing.
    """
    try:
        # 1. Already an instance? Return it.
        if isinstance(data, model_cls):
            return data

        # 2. JSON String? Parse it.
        if isinstance(data, (str, bytes)):
            try:
                # cleans json markdown code blocks if present
                clean_data = data.replace("```json", "").replace("```", "").strip()
                return model_cls.model_validate_json(clean_data)
            except ValueError:
                # Not valid JSON, might be a raw dict passed as string repr?
                pass

        # 3. Dictionary? Validate it (The Pydantic V2 standard).
        if isinstance(data, dict):
            return model_cls.model_validate(data)

        # 4. Duck Typing (Object with attributes).
        if hasattr(data, '__dict__'):
             return model_cls.model_validate(data, from_attributes=True)

        # 5. Fallback: Keyword unpacking (Risky if data isn't a mapping, but worth a try)
        if isinstance(data, (dict, map)):
            return model_cls(**data)
            
        # 6. Absolute last resort for single-field models (rare usage)
        # return model_cls(data) # <--- We purposefully avoid this to stop the recursion/error loop

        raise ValueError(f"Data type {type(data)} not compatible with {model_cls.__name__}")

    except (ValidationError, TypeError, ValueError) as e:
        logger.error(f"❌ MODEL INSTANTIATION FAILED for {model_cls.__name__}")
        logger.error(f"   Input Type: {type(data)}")
        logger.debug(f"   Input Preview: {str(data)[:200]}")
        
        # Try one desperate fallback for the specific "options" case in PredictedOptionList
        if model_cls.__name__ == "PredictedOptionList" and isinstance(data, dict):
             try:
                 return model_cls(options=data)
             except Exception:
                 pass

        raise ValueError(f"Failed to create {model_cls.__name__}: {e}") from e


# ==========================================
# 🤖 MODEL ID HELPERS
# ==========================================

def get_model_id(base_name: str) -> str:
    overrides = {
        "gpt-5.1": os.getenv("FORCE_GPT_5_1_MODEL"),
        "claude-sonnet-4.5": os.getenv("FORCE_CLAUDE_SONNET_4_5_MODEL"),
    }
    if overrides.get(base_name):
        return overrides[base_name]
    
    fallbacks = {
        "gpt-5.1": [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-4o-2024-11-20",
        ],
        "claude-sonnet-4.5": [
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/anthropic/claude-3.5-sonnet",
            "openrouter/anthropic/claude-3-sonnet",
        ],
    }
    return fallbacks.get(base_name, [base_name])[0]


# ==========================================
# 🧠 STRUCTURAL CONSTRAINT BOT
# ==========================================

class StructuralConstraintBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    MAX_TAVILY_QUERIES = 200
    MAX_ASKNEWS_QUERIES = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 🔑 API clients
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.asknews = AskNewsSDK(
            client_id=os.getenv("ASKNEWS_CLIENT_ID"),
            client_secret=os.getenv("ASKNEWS_CLIENT_SECRET"),
        )

        # 📊 Query counters
        self._tavily_count = 0
        self._asknews_count = 0
        self._query_lock = asyncio.Lock()

        # 🤖 Forecast models
        self.FORECAST_MODELS = {
            "gpt-5": "openrouter/openai/gpt-5",
            "gpt-5.1": get_model_id("gpt-5.1"),
            "claude-sonnet-4.5": get_model_id("claude-sonnet-4.5"),
        }
        self.BASE_WEIGHTS = {
            "gpt-5": 1.0,
            "gpt-5.1": 1.3,
            "claude-sonnet-4.5": 1.1,
        }

        # ✅ Initialize roles
        self.researcher_llm = self.get_llm("researcher")
        self.summarizer_llm = self.get_llm("summarizer")
        self.parser_llm = self.get_llm("parser")

        self._current_question: Optional[MetaculusQuestion] = None

    def get_llm(self, role: str, guarantee_type: Literal["llm", "model_name"] = "llm") -> Any:
        role_configs = {
            "researcher": {"model": get_model_id("gpt-5.1"), "temp": 0.4, "timeout": 300},
            "summarizer": {"model": get_model_id("gpt-5.1"), "temp": 0.2, "timeout": 300},
            "parser": {"model": "openrouter/openai/gpt-4o", "temp": 0.0, "timeout": 120},
            "default": {"model": get_model_id("gpt-5.1"), "temp": 0.3, "timeout": 400}
        }

        try:
            llm = super().get_llm(role, guarantee_type)
            if llm is None and role in role_configs:
                raise ValueError("Role missing")
            return llm
        except Exception:
            logger.info(f"⚡ Creating fallback LLM for role: {role}")
            config = role_configs.get(role, role_configs["default"])
            return GeneralLlm(
                model=config["model"],
                temperature=config["temp"],
                timeout=config["timeout"]
            )

    async def _increment_query_count(self, source: str) -> bool:
        async with self._query_lock:
            limits = {
                "tavily": self.MAX_TAVILY_QUERIES,
                "asknews": self.MAX_ASKNEWS_QUERIES,
            }
            counters = {
                "tavily": self._tavily_count,
                "asknews": self._asknews_count,
            }
            if counters[source] >= limits[source]:
                logger.warning(f"Quota exceeded for {source}")
                return False
            if source == "tavily":
                self._tavily_count += 1
            elif source == "asknews":
                self._asknews_count += 1
            return True

    def _is_financial_question(self, q: MetaculusQuestion) -> bool:
        text = " ".join([
            q.question_text,
            q.background_info or "",
            q.resolution_criteria or "",
        ]).lower()
        financial_terms = [
            r"\b(stock|equity|share|s&p|nasdaq|dow|djia|index|ticker|nyse|cpi|fed|interest rate|yield curve|vix|volatility)\b",
            r"\$[a-z]{1,5}\b",
            r"\b(?:[a-z]{1,4}\.?\s*[ou]n\s+\d{1,2}/\d{1,2})\b",
            r"\bby\s+\d{4}\b.*\b(s&p|nasdaq)",
        ]
        return any(re.search(pat, text) for pat in financial_terms)

    async def run_research(self, question: MetaculusQuestion) -> str:
        self._current_question = question
        async with self._concurrency_limiter:
            try:
                q_text = f"{question.question_text} {question.resolution_criteria}"[:300]
                is_financial = self._is_financial_question(question)

                asknews_query = f"Financial outlook: {q_text}" if is_financial else q_text
                tavily_query = f"Deep analysis, laws, and procedural constraints for: {q_text}"

                # Run both sources concurrently
                tavily_res, asknews_res = await asyncio.gather(
                    self._run_tavily_research(tavily_query),
                    self._run_asknews_research(asknews_query, financial=is_financial)
                )

                raw_text = f"""
                [AskNews — Breaking Events]
                {asknews_res}

                [Tavily — Deep Context & Constraints]
                {tavily_res}
                """
                
                combined = clean_indents(f"""
                ### Research Data (Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})
                {"[FINANCIAL QUESTION DETECTED]" if is_financial else ""}

                {raw_text}

                ⚠️ FORECASTER INSTRUCTIONS:
                Look specifically for PROCEDURAL DATA:
                1. How long does the specific legal/administrative process take?
                2. Are there mandatory waiting periods (e.g., Treaty Article 13)?
                3. What are the hard deadlines?
                """)
                return combined
            except Exception as e:
                logger.error(f"Research failed for Q{question.id}: {e}")
                return f"⚠️ Research unavailable (error: {e})"

    async def _run_tavily_research(self, query: str) -> str:
        if not await self._increment_query_count("tavily"):
            return "[Skipped: Tavily quota exceeded]"
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    include_answer=True,
                    max_results=5,
                    days=365,
                ),
            )
            answer = (res.get("answer") or "").strip() or "[No summary]"
            snippets = "\n".join(
                f"• {r['title'][:80]}: {r['content'][:300]}..."
                for r in res.get("results", [])[:4]
            )
            return f"Tavily Summary: {answer}\nSources:\n{snippets or 'None'}"
        except Exception as e:
            return f"[Tavily error: {e}]"

    async def _run_asknews_research(self, query: str, financial: bool = False) -> str:
        if not await self._increment_query_count("asknews"):
            return "[Skipped: AskNews quota exceeded]"
        try:
            categories = ["business", "finance"] if financial else ["politics", "business", "technology", "science"]
            max_age_hours = (30 if financial else 90) * 24

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.asknews.news.search_news(
                    query=query,
                    n_articles=5,
                    return_type="news",
                    categories=categories,
                    max_age_hours=max_age_hours,
                ),
            )
            articles = response.data.articles
            if not articles:
                return "[No recent news]"

            snippets = "\n".join(
                f"• {a.headline[:80]} ({a.date[:10]}) — {a.snippet[:300]}..."
                for a in articles[:5]
            )
            return snippets
        except Exception as e:
            return f"[AskNews error: {e}]"

    # ===== Classification & Weights =====
    def _categorize_question(self, q: MetaculusQuestion) -> str:
        if isinstance(q, BinaryQuestion): return "binary"
        elif isinstance(q, MultipleChoiceQuestion): return "mcq"
        elif isinstance(q, NumericQuestion): return "numeric"
        text = (q.question_text + " " + (q.background_info or "")).lower()
        if hasattr(q, 'options') and q.options: return "mcq"
        if "probability" in text or "will" in text: return "binary"
        return "numeric"

    def _get_dynamic_weights(self, question: MetaculusQuestion) -> Dict[str, float]:
        weights = self.BASE_WEIGHTS.copy()
        q_type = self._categorize_question(question)
        is_financial = self._is_financial_question(question)

        if q_type == "binary":
            weights["gpt-5.1"] *= 1.5 
        elif q_type == "numeric":
            weights["gpt-5"] *= (1.2 if is_financial else 1.3)
        elif q_type == "mcq":
            weights["claude-sonnet-4.5"] *= 1.3 

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _extract_confidence(self, text: str, is_financial: bool = False) -> float:
        patterns = [
            r"(?:confidence|certainty):\s*(\d+(?:\.\d+)?)(?:\s*%)?",
            r"probability that this reasoning is correct:\s*(\d+(?:\.\d+)?)",
        ]
        val = 0.7
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    num = float(match.group(1))
                    if num > 1.0: num /= 100.0 
                    val = num
                    break
                except ValueError:
                    continue
        if is_financial: val *= 0.85
        return np.clip(val, 0.1, 0.95)

    def _get_bounds(self, q: NumericQuestion) -> Tuple[str, str]:
        lo = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        hi = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        lo_str = f"{lo} (Hard Min)" if not q.open_lower_bound else f"{lo} (Soft Min)"
        hi_str = f"{hi} (Hard Max)" if not q.open_upper_bound else f"{hi} (Soft Max)"
        return lo_str, hi_str

    def _apply_financial_adjustment(self, dist: NumericDistribution, question: NumericQuestion) -> NumericDistribution:
        try:
            pcts = {int(p.percentile * 100): p.value for p in dist.declared_percentiles}
            
            if 50 not in pcts:
                vals = sorted(pcts.values())
                pcts[50] = np.median(vals) if vals else 0
                
            median = pcts[50]
            if median <= 0: return dist 

            needed = [10, 20, 40, 50, 60, 80, 90]
            new_pcts = {}
            for p in needed:
                if p in pcts:
                    val = pcts[p]
                    if p == 10: val = val * 0.95 if val < median else val 
                    elif p == 90: val = val * 1.05 if val > median else val 
                    new_pcts[p] = val
                else:
                    new_pcts[p] = median 

            final_percentiles = [
                Percentile(percentile=p / 100.0, value=new_pcts[p])
                for p in sorted(new_pcts.keys())
            ]
            # Uses standard factory, assumed safe
            return NumericDistribution.from_question(final_percentiles, question)
        except Exception as e:
            logger.warning(f"Financial adjustment failed: {e}")
            return dist

    async def _run_forecast_with_confidence(
        self, question: MetaculusQuestion, research: str, model_key: str
    ) -> Tuple[ReasonedPrediction, float]:
        model_name = self.FORECAST_MODELS[model_key]
        forecaster_llm = GeneralLlm(model=model_name, temperature=0.3, timeout=400, allowed_tries=2)

        is_financial = self._is_financial_question(question)
        q_type = self._categorize_question(question)
        resolve_time = getattr(question, 'scheduled_resolve_time', 'Unknown (Assume end of current year/period)')

        # --- STRUCTURAL CONSTRAINT ANALYSIS LOGIC ---
        base_prompt = clean_indents(f"""
        You are a Superforecaster using the 'Structural Constraint Analysis' methodology.
        Your goal is not to guess, but to calculate feasibility based on hard constraints (Base Rate + Friction).

        CONTEXT:
        Question: {question.question_text}
        Resolution Criteria: {question.resolution_criteria}
        Background: {question.background_info or 'None'}
        Today's Date: {datetime.now().strftime('%Y-%m-%d')}
        Scheduled Resolution Date: {resolve_time}

        RESEARCH DATA:
        {research}

        METHODOLOGY (STRICT ADHERENCE REQUIRED):
        1. ESTABLISH THE BASELINE (Status Quo):
           - What is the current known state?
           - This is your anchor. Deviations require energy and time.

        2. PERFORM STRUCTURAL CONSTRAINT ANALYSIS (The "Friction" Test):
           - UPSIDE VECTOR (Growth/Change): What is the *procedural* process? (e.g., Ratification).
             - Calculate: [Average Time for Process] vs [Time Remaining].
             - Constraint Rule: If (Time Required) > (Time Remaining), then Growth Probability ≈ 0.
           - DOWNSIDE VECTOR (Reduction/Reversal): What are the legal/structural exit mechanisms? (e.g., Treaty Article 13).
             - Calculate: [Mandatory Notice Period] vs [Time Remaining].
             - Constraint Rule: If (Mandatory Notice Period) > (Time Remaining), then Reduction Probability ≈ 0.

        3. ELIMINATE TAIL RISKS via "NEGATIVE KNOWLEDGE":
           - Prove why specific outcomes are IMPOSSIBLE (e.g. "If N+1 is too slow and N-1 is legally blocked, N is the only outcome").

        4. AGGREGATION:
           - Structural constraints ALWAYS override political intent.
        """)

        if q_type == "binary":
            prompt = base_prompt + clean_indents(f"""
            OUTPUT FORMAT:
            Provide a step-by-step derivation.
            Ends with:
            Rationale: [Summary]
            Probability: ZZ% (0 to 100)
            Confidence: WW% (Your confidence)
            """)
        elif q_type == "mcq":
            options = getattr(question, "options", [])
            prompt = base_prompt + clean_indents(f"""
            OPTIONS: {options}
            Evaluate each option against Constraints. Assign 0% to procedurally impossible options.
            Ends with:
            Rationale: [Summary]
            Confidence: WW%
            """)
        else:  # numeric
            lo, hi = self._get_bounds(question)
            prompt = base_prompt + clean_indents(f"""
            RANGE: {lo} to {hi}
            Units: {getattr(question, 'unit_of_measure', 'inferred')}
            Apply constraints to narrow the range. Provide 10th-90th percentiles.
            Ends with:
            Rationale: [Summary]
            Confidence: WW%
            """)

        reasoning = await forecaster_llm.invoke(prompt)
        confidence = self._extract_confidence(reasoning, is_financial)

        # Parsing Logic with Safe Instantiation
        try:
            if q_type == "binary":
                pred: BinaryPrediction = await structure_output(
                    reasoning, BinaryPrediction, model=self.parser_llm
                )
                value = np.clip(pred.prediction_in_decimal, 0.01, 0.99)
            
            elif q_type == "mcq":
                parsing_instructions = f"Valid options: {getattr(question, 'options', [])}"
                pred: PredictedOptionList = await structure_output(
                    reasoning,
                    PredictedOptionList,
                    model=self.parser_llm,
                    additional_instructions=parsing_instructions,
                )
                value = pred
            
            else:  # numeric
                pct_list: List[Percentile] = await structure_output(
                    reasoning,
                    list[Percentile],
                    model=self.parser_llm,
                )
                dist = NumericDistribution.from_question(pct_list, question)
                if is_financial:
                    dist = self._apply_financial_adjustment(dist, question)
                value = dist
                
        except Exception as e:
            logger.warning(f"Parsing failed for {model_key} on Q{question.id}: {e}. Using Fallback.")
            # FALLBACK LOGIC
            if q_type == "binary":
                value = 0.5
            elif q_type == "mcq":
                opts = getattr(question, "options", ["A", "B"])
                # FIX: Use safe_model to handle dict->model creation
                probs = {o: 100.0 / len(opts) for o in opts}
                value = safe_model(PredictedOptionList, {"options": probs})
            else:
                lo = getattr(question, "lower_bound", 0)
                hi = getattr(question, "upper_bound", 100)
                fallback_pcts = [
                    Percentile(percentile=p / 100.0, value=lo + (hi - lo) * p / 100.0) 
                    for p in [10, 20, 40, 60, 80, 90]
                ]
                value = NumericDistribution.from_question(fallback_pcts, question)

        return ReasonedPrediction(prediction_value=value, reasoning=reasoning), confidence

    async def _run_generic_forecast(self, question: MetaculusQuestion, research: str) -> ReasonedPrediction:
        weights = self._get_dynamic_weights(question)
        raw_results: List[Tuple[ReasonedPrediction, float, str]] = []

        for model_key in self.FORECAST_MODELS:
            try:
                pred, conf = await self._run_forecast_with_confidence(question, research, model_key)
                effective_weight = weights[model_key] * conf
                raw_results.append((pred, effective_weight, model_key))
                logger.info(f"✓ {model_key}: conf={conf:.2f}, wt={effective_weight:.2f}")
            except Exception as e:
                logger.error(f"✗ {model_key} failed: {e}")

        if not raw_results:
            raise RuntimeError("All models failed.")

        # Aggregation
        q_type = self._categorize_question(question)
        is_financial = self._is_financial_question(question)
        
        combined_reasoning = "\n\n".join(f"[{r[2]}] {r[0].reasoning[:500]}..." for r in raw_results)

        if q_type == "binary":
            values = [float(r[0].prediction_value) for r in raw_results]
            weights_list = [r[1] for r in raw_results]
            final_pred = np.average(values, weights=weights_list)
            return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

        elif q_type == "mcq":
            options = getattr(question, "options", [])
            smoothed = np.zeros(len(options))
            total_weight = sum(r[1] for r in raw_results)
            
            for pred, wt, _ in raw_results:
                pv = pred.prediction_value
                for i, opt in enumerate(options):
                    try:
                        # Defensive check for probability extraction
                        if hasattr(pv, 'get_probability_for_option'):
                            p = pv.get_probability_for_option(opt)
                        elif isinstance(pv, dict):
                            p = pv.get(opt, 0)
                        elif hasattr(pv, 'options'):
                             # Assuming pv.options is a dict
                             p = pv.options.get(opt, 0)
                        else:
                             p = 1.0/len(options)
                        smoothed[i] += p * wt
                    except Exception:
                        smoothed[i] += (100.0 / len(options)) * wt
            
            final_probs = smoothed / total_weight if total_weight > 0 else smoothed
            probs_dict = dict(zip(options, final_probs.tolist()))
            
            # FIX: Use safe_model to create the final object
            final_pred = safe_model(PredictedOptionList, {"options": probs_dict})
            return ReasonedPrediction(prediction_value=final_pred, reasoning=combined_reasoning)

        else:  # numeric
            target_percentiles = [10, 20, 40, 60, 80, 90]
            ensemble_pcts = []
            for p in target_percentiles:
                vals, wts = [], []
                for pred, wt, _ in raw_results:
                    dist = pred.prediction_value
                    # Find closest percentile in the distribution
                    closest = min(
                        dist.declared_percentiles,
                        key=lambda x: abs(x.percentile - p / 100.0),
                    )
                    vals.append(closest.value)
                    wts.append(wt)
                
                # Geometric mean for financial (log-normal), Arithmetic for others
                if vals:
                    if is_financial and all(v > 0 for v in vals):
                        log_vals = [np.log(v) for v in vals]
                        q_val = np.exp(np.average(log_vals, weights=wts))
                    else:
                        q_val = np.average(vals, weights=wts)
                else:
                    q_val = 0 
                
                ensemble_pcts.append(Percentile(percentile=p / 100.0, value=q_val))
                
            final_dist = NumericDistribution.from_question(ensemble_pcts, question)
            return ReasonedPrediction(prediction_value=final_dist, reasoning=combined_reasoning)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        return await self._run_generic_forecast(question, research)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        return await self._run_generic_forecast(question, research)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_generic_forecast(question, research)


# ===== MAIN =====
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["32916", "minibench", "all"],
        default="all",
        help="Tournament: 32916, minibench, or all",
    )
    args = parser.parse_args()

    required = ["TAVILY_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_CLIENT_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    bot = StructuralConstraintBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    async def run():
        tournament_map = {
            "32916": 32916,
            "minibench": MetaculusApi.CURRENT_MINIBENCH_ID,
        }
        targets = list(tournament_map.values()) if args.mode == "all" else [tournament_map[args.mode]]

        reports = []
        for tid in targets:
            logger.info(f"▶ Forecasting tournament: {tid}")
            try:
                # Basic error handling for the loop
                r = await bot.forecast_on_tournament(tid, return_exceptions=True)
                reports.extend(r)
            except ExceptionGroup as eg:
                logger.error(f"Tournament {tid} had sub-failures:")
                for exc in eg.exceptions:
                     logger.error(f"  -> {exc}")
            except Exception as e:
                logger.error(f"Critical error in tournament {tid}: {e}")

        bot.log_report_summary(reports)
        logger.info(f"📊 Queries used — Tavily: {bot._tavily_count}, AskNews: {bot._asknews_count}")

    asyncio.run(run())
