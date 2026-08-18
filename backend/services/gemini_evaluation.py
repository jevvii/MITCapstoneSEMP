"""
Gemini AI Evaluation Service for Call Simulation
Evaluates trainee performance based on transcript, script accuracy, grammar, soft skills, and KPIs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

# Lazy-load google.genai to avoid importing heavy SDK at startup
_genai = None
_genai_types = None

def _ensure_genai():
    global _genai, _genai_types
    if _genai is not None or _genai_types is not None:
        return _genai, _genai_types
    try:
        import importlib

        _genai = importlib.import_module('google.genai')
        _genai_types = importlib.import_module('google.genai').types
        return _genai, _genai_types
    except Exception as exc:
        logger.info('Gemini genai not available: %s', exc)
        _genai = None
        _genai_types = None
        return None, None

from ..config_validation import normalize_env_value, resolve_gemini_api_key

logger = logging.getLogger(__name__)
DEFAULT_GEMINI_EVALUATION_MODEL = "gemini-2.5-flash"


class GeminiEvaluationEngine:
    """Gemini-powered evaluation engine for call simulation feedback."""

    def __init__(self) -> None:
        self.api_key = resolve_gemini_api_key(os.getenv)
        self.model_name = (
            normalize_env_value(os.getenv("GEMINI_EVALUATION_MODEL"))
            or normalize_env_value(os.getenv("GEMINI_CALL_SIM_MODEL"))
            or DEFAULT_GEMINI_EVALUATION_MODEL
        )
        self.client = None
        genai, _types = _ensure_genai()
        if genai and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                logger.warning("Failed to initialize Gemini client: %s", exc)
                self.client = None

    def is_available(self) -> bool:
        return bool(GEMINI_AVAILABLE and self.client and self.api_key)

    def evaluate(
        self,
        transcript: str,
        script_flow: list[dict[str, Any]],
        target_kpis: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Evaluate trainee performance using Gemini API.

        Args:
            transcript: Full conversation transcript
            script_flow: The expected script flow with suggested CSR scripts
            target_kpis: Target KPIs for evaluation

        Returns:
            Evaluation result dictionary with scores and feedback
        """
        if not self.is_available():
            return self._fallback_evaluation(transcript, script_flow, target_kpis)

        # Build the evaluation prompt
        prompt = self._build_evaluation_prompt(transcript, script_flow, target_kpis)

        try:
            genai, types = _ensure_genai()
            if not genai or not types:
                logger.warning('Gemini genai SDK not available; falling back to local evaluation.')
                return self._fallback_evaluation(transcript, script_flow, target_kpis)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                    system_instruction=self._get_system_instruction(),
                ),
            )

            if response and getattr(response, 'text', None):
                return json.loads(response.text)

        except Exception as e:
            logger.error(f"Gemini evaluation failed: {e}")

        return self._fallback_evaluation(transcript, script_flow, target_kpis)

    def _get_system_instruction(self) -> str:
        """Get the system instruction for the evaluation prompt."""
        return """You are an expert BPO call quality evaluator. Your role is to analyze call transcripts 
and provide detailed, actionable feedback for trainee development.

Evaluate the following aspects:
1. SCRIPT ACCURACY: How well did the trainee follow the suggested CSR script?
2. GRAMMAR & PRONUNCIATION: Identify any grammar issues or pronunciation errors (from STT output)
3. SOFT SKILLS: Evaluate pacing, empathy, tone, and professional demeanor
4. PACING & AHT: Analyze speaking pace and average handle time

Provide your evaluation in JSON format with the following structure:
{
    "overallSummary": "Brief overall summary of performance",
    "totalScore": 0-100,
    "passingScore": 80,
    "passed": true/false,
    "scriptAccuracy": {
        "score": 0-100,
        "strengths": ["list of what the trainee did well"],
        "misses": ["list of what was missed or incorrect"]
    },
    "grammarAndPronunciation": {
        "score": 0-100,
        "notes": ["list of specific issues found"]
    },
    "softSkills": {
        "score": 0-100,
        "notes": ["observations about tone, empathy, pacing"]
    },
    "pacingAndAht": {
        "ahtSeconds": actual handle time in seconds,
        "notes": ["observations about pace"]
    },
    "coachingTips": ["specific actionable tips for improvement"]
}

Be objective, constructive, and specific in your feedback."""

    def _build_evaluation_prompt(
        self,
        transcript: str,
        script_flow: list[dict[str, Any]],
        target_kpis: dict[str, Any],
    ) -> str:
        """Build the evaluation prompt with transcript and context."""

        # Format script flow for the prompt
        script_steps = "\n".join([
            f"Step {i+1}: CSR should say: '{step.get('suggested_csr_script', '')}' | Member responds: '{step.get('member_response_text', '')}' (Points: {step.get('point_value', 0)})"
            for i, step in enumerate(script_flow)
        ])

        # Format target KPIs
        kpi_text = "\n".join([f"- {k}: {v}" for k, v in target_kpis.items()]) if target_kpis else "No specific KPIs defined"

        prompt = f"""Please evaluate this BPO call simulation transcript.

## Expected Script Flow:
{script_steps}

## Target KPIs:
{kpi_text}

## Full Transcript:
{transcript}

## Task:
Analyze the transcript above and provide a detailed evaluation following the JSON structure specified in your instructions. Focus on:
1. How closely the trainee followed the suggested CSR scripts
2. Grammar and pronunciation issues (note: this is from speech-to-text, so some errors may be STT artifacts)
3. Soft skills like empathy, tone, and professionalism
4. Pacing and overall call duration

Provide specific, actionable coaching tips that will help the trainee improve."""

        return prompt

    def _fallback_evaluation(
        self,
        transcript: str,
        script_flow: list[dict[str, Any]],
        target_kpis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Fallback evaluation when Gemini is not available.
        Performs basic keyword matching and scoring.
        """
        # Basic keyword matching for script accuracy
        csr_steps = [s for s in script_flow if s.get("suggested_csr_script")]
        matched_keywords = 0
        total_keywords = 0

        transcript_lower = transcript.lower()

        for step in csr_steps:
            keywords = step.get("expected_keywords", [])
            if keywords:
                total_keywords += len(keywords)
                for kw in keywords:
                    if kw.lower() in transcript_lower:
                        matched_keywords += 1

        keyword_score = (matched_keywords / total_keywords * 100) if total_keywords > 0 else 0

        # Calculate total possible points
        total_points = sum(step.get("point_value", 0) for step in script_flow)

        return {
            "overallSummary": "Evaluation completed with basic scoring. For detailed feedback, please enable Gemini API.",
            "totalScore": round(keyword_score, 2),
            "passingScore": 80,
            "passed": keyword_score >= 80,
            "scriptAccuracy": {
                "score": round(keyword_score, 2),
                "strengths": ["Completed the call simulation"],
                "misses": ["Full evaluation requires Gemini API"],
            },
            "grammarAndPronunciation": {
                "score": 75,
                "notes": ["Basic scoring only - enable Gemini for detailed analysis"],
            },
            "softSkills": {
                "score": 75,
                "notes": ["Basic scoring only - enable Gemini for detailed analysis"],
            },
            "pacingAndAht": {
                "ahtSeconds": 0,
                "notes": ["Timing data not available in fallback mode"],
            },
            "coachingTips": [
                "Enable Gemini API for comprehensive evaluation",
                "Practice speaking clearly and at a moderate pace",
                "Follow the suggested script more closely",
            ],
            "provider": "fallback",
        }

    def score_microlearning_open_response(
        self,
        *,
        prompt: str,
        sample_answer: str,
        trainee_response: str,
        required_keywords: Optional[list[str]] = None,
        max_points: int = 10,
    ) -> dict[str, Any]:
        """Score an open-ended microlearning response against the trainer sample answer."""
        normalized_response = (trainee_response or "").strip()
        normalized_sample_answer = (sample_answer or "").strip()
        normalized_prompt = (prompt or "").strip()
        cleaned_keywords = [
            str(keyword or "").strip()
            for keyword in (required_keywords or [])
            if str(keyword or "").strip()
        ]

        if not normalized_response:
            return {
                "score": 0,
                "feedback": "No answer was submitted.",
                "matched_keywords": [],
                "missing_keywords": cleaned_keywords,
                "provider": "fallback",
            }

        if not self.is_available():
            return self._fallback_microlearning_open_response(
                prompt=normalized_prompt,
                sample_answer=normalized_sample_answer,
                trainee_response=normalized_response,
                required_keywords=cleaned_keywords,
                max_points=max_points,
            )

        prompt_text = self._build_microlearning_open_response_prompt(
            prompt=normalized_prompt,
            sample_answer=normalized_sample_answer,
            trainee_response=normalized_response,
            required_keywords=cleaned_keywords,
            max_points=max_points,
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    system_instruction=self._get_microlearning_open_response_instruction(max_points=max_points),
                ),
            )

            if response.text:
                parsed = json.loads(response.text)
                score = parsed.get("score")
                try:
                    numeric_score = int(round(float(score)))
                except (TypeError, ValueError):
                    numeric_score = max_points
                numeric_score = max(1, min(max_points, numeric_score))

                matched_keywords = [
                    str(keyword or "").strip()
                    for keyword in (parsed.get("matched_keywords") or [])
                    if str(keyword or "").strip()
                ]
                missing_keywords = [
                    str(keyword or "").strip()
                    for keyword in (parsed.get("missing_keywords") or [])
                    if str(keyword or "").strip()
                ]
                feedback = str(parsed.get("feedback") or "").strip()
                if not feedback:
                    feedback = "Gemini reviewed your response against the trainer sample answer."

                return {
                    "score": numeric_score,
                    "feedback": feedback,
                    "matched_keywords": matched_keywords,
                    "missing_keywords": missing_keywords,
                    "provider": "gemini",
                }
        except Exception as exc:
            logger.error("Gemini microlearning open-response scoring failed: %s", exc)

        return self._fallback_microlearning_open_response(
            prompt=normalized_prompt,
            sample_answer=normalized_sample_answer,
            trainee_response=normalized_response,
            required_keywords=cleaned_keywords,
            max_points=max_points,
        )

    def summarize_microlearning_assignment_performance(
        self,
        *,
        module_title: str,
        module_type: str,
        percentage_score: float,
        passing_score: float,
        passed: bool,
        breakdown: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarize overall microlearning assignment performance from the question breakdown."""
        cleaned_breakdown = [
            {
                "question_number": int(item.get("question_number") or 0),
                "title": str(item.get("title") or "").strip(),
                "prompt": str(item.get("prompt") or "").strip(),
                "question_result": str(item.get("question_result") or "").strip().lower(),
                "score": float(item.get("score") or 0.0),
                "points_earned": float(item.get("points_earned") or 0.0),
                "points_possible": float(item.get("points_possible") or 0.0),
                "trainee_answer": str(item.get("trainee_answer") or "").strip(),
                "correct_answer": str(item.get("correct_answer") or "").strip(),
                "matched_keywords": [str(keyword or "").strip() for keyword in (item.get("matched_keywords") or []) if str(keyword or "").strip()],
                "missing_keywords": [str(keyword or "").strip() for keyword in (item.get("missing_keywords") or []) if str(keyword or "").strip()],
                "feedback": str(item.get("feedback") or "").strip(),
            }
            for item in (breakdown or [])
        ]

        if not cleaned_breakdown:
            return self._fallback_microlearning_assignment_summary(
                module_title=module_title,
                module_type=module_type,
                percentage_score=percentage_score,
                passing_score=passing_score,
                passed=passed,
                breakdown=[],
            )

        if not self.is_available():
            return self._fallback_microlearning_assignment_summary(
                module_title=module_title,
                module_type=module_type,
                percentage_score=percentage_score,
                passing_score=passing_score,
                passed=passed,
                breakdown=cleaned_breakdown,
            )

        prompt = self._build_microlearning_assignment_summary_prompt(
            module_title=module_title,
            module_type=module_type,
            percentage_score=percentage_score,
            passing_score=passing_score,
            passed=passed,
            breakdown=cleaned_breakdown,
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    system_instruction=self._get_microlearning_assignment_summary_instruction(),
                ),
            )

            if response.text:
                parsed = json.loads(response.text)
                if isinstance(parsed, dict):
                    parsed["provider"] = "gemini"
                    return parsed
        except Exception as exc:
            logger.error("Gemini microlearning assignment summary failed: %s", exc)

        return self._fallback_microlearning_assignment_summary(
            module_title=module_title,
            module_type=module_type,
            percentage_score=percentage_score,
            passing_score=passing_score,
            passed=passed,
            breakdown=cleaned_breakdown,
        )

    def _get_microlearning_assignment_summary_instruction(self) -> str:
        return """You are an expert BPO microlearning coach.

Review the finished module score and question breakdown, then produce a JSON object with:
{
  "overallSummary": "2-4 sentence performance summary",
  "strengths": ["specific strengths"],
  "weakAreas": ["specific weak areas"],
  "improvementOpportunities": ["specific improvement opportunities"],
  "recommendedNextSteps": ["practical next steps"],
  "explanation": "short explanation grounded in the question-by-question breakdown"
}

Rules:
- Base every point on the supplied question breakdown.
- Do not mention hidden system behavior or model limitations.
- Keep feedback constructive, specific, and trainee-facing.
- If the trainee passed, still include the most important next improvement area.
- If open-ended answers were AI-reviewed, explain the outcome using the observed answer quality and missing keywords when available."""

    def _build_microlearning_assignment_summary_prompt(
        self,
        *,
        module_title: str,
        module_type: str,
        percentage_score: float,
        passing_score: float,
        passed: bool,
        breakdown: list[dict[str, Any]],
    ) -> str:
        breakdown_lines = []
        for item in breakdown:
            breakdown_lines.append(
                "\n".join([
                    f"Question {item.get('question_number')}: {item.get('title') or 'Untitled Question'}",
                    f"Prompt: {item.get('prompt') or 'No prompt provided'}",
                    f"Result: {item.get('question_result') or 'unknown'}",
                    f"Score: {float(item.get('score') or 0.0):.2f}%",
                    f"Points: {float(item.get('points_earned') or 0.0):.2f}/{float(item.get('points_possible') or 0.0):.2f}",
                    f"Trainee Answer: {item.get('trainee_answer') or 'No answer submitted'}",
                    f"Correct Answer: {item.get('correct_answer') or 'Not available'}",
                    f"Matched Keywords: {', '.join(item.get('matched_keywords') or []) or 'None'}",
                    f"Missing Keywords: {', '.join(item.get('missing_keywords') or []) or 'None'}",
                    f"Feedback: {item.get('feedback') or 'No feedback available'}",
                ]),
            )

        status_label = "passed" if passed else "failed"
        return f"""Summarize this completed trainee microlearning module.

Module Title: {module_title or 'Untitled Module'}
Module Type: {module_type or 'unknown'}
Percentage Score: {float(percentage_score or 0.0):.2f}%
Passing Score: {float(passing_score or 0.0):.2f}%
Final Status: {status_label}

Question Breakdown:
{chr(10).join(breakdown_lines)}

Return only valid JSON using the structure from the instructions."""

    def _fallback_microlearning_assignment_summary(
        self,
        *,
        module_title: str,
        module_type: str,
        percentage_score: float,
        passing_score: float,
        passed: bool,
        breakdown: list[dict[str, Any]],
    ) -> dict[str, Any]:
        correct_items = [item for item in breakdown if item.get("question_result") == "correct"]
        incorrect_items = [item for item in breakdown if item.get("question_result") == "incorrect"]
        review_items = [item for item in breakdown if item.get("question_result") == "needs_review"]

        def _item_label(it: dict[str, Any]) -> str:
            return str(it.get("title") or f"Question {it.get('question_number', '')}").strip()

        strengths = [
            f"Answered {_item_label(item)} correctly."
            for item in correct_items[:3]
        ]
        if not strengths and passed:
            strengths.append("Maintained a passing performance across the module.")
        elif not strengths:
            strengths.append("Completed the module and submitted all required answers.")

        weak_areas = [
            f"Needs improvement on {_item_label(item)}."
            for item in incorrect_items[:3]
        ]
        weak_areas.extend(
            f"Open-ended response quality should be reviewed for {_item_label(item)}."
            for item in review_items[:2]
        )

        improvement_opportunities: list[str] = []
        for item in breakdown:
            missing_keywords = [str(keyword or "").strip() for keyword in (item.get("missing_keywords") or []) if str(keyword or "").strip()]
            if missing_keywords:
                improvement_opportunities.append(
                    f"Include key ideas such as {', '.join(missing_keywords[:3])} in {_item_label(item)}."
                )
            elif item.get("question_result") == "incorrect":
                improvement_opportunities.append(
                    f"Review the expected answer pattern for {_item_label(item)}."
                )

        if not improvement_opportunities:
            improvement_opportunities.append("Keep reinforcing the same response structure to make strong answers more consistent.")

        recommended_next_steps = [
            "Retake the module if the score is still below the passing target.",
            "Review the missed questions and compare your answer with the correct answer before the next attempt.",
            "Practice short, structured responses that include the required keywords or action steps.",
        ]
        if passed:
            recommended_next_steps[0] = "Review the weakest questions once more to turn a passing result into a stronger repeat performance."

        performance_gap = max(0.0, float(passing_score or 0.0) - float(percentage_score or 0.0))
        overall_summary = (
            f"{module_title or 'This module'} ended with a score of {float(percentage_score or 0.0):.2f}%, "
            f"which {'meets' if passed else 'does not meet'} the passing target of {float(passing_score or 0.0):.2f}%."
        )
        if not passed and performance_gap > 0:
            overall_summary += f" The score is short by {performance_gap:.2f} percentage points."
        if correct_items:
            overall_summary += f" Stronger performance appeared in {len(correct_items)} question(s)."
        if incorrect_items or review_items:
            overall_summary += f" The biggest gains will come from the {len(incorrect_items) + len(review_items)} item(s) that were missed or need review."

        explanation = (
            "This summary is based on the saved question breakdown, including correct and incorrect answers, "
            "open-ended feedback, and any missing keyword patterns."
        )

        return {
            "overallSummary": overall_summary,
            "strengths": strengths,
            "weakAreas": weak_areas or ["Review the lowest-scoring questions for the clearest next improvement area."],
            "improvementOpportunities": improvement_opportunities[:4],
            "recommendedNextSteps": recommended_next_steps,
            "explanation": explanation,
            "provider": "fallback",
        }

    def _get_microlearning_open_response_instruction(self, *, max_points: int) -> str:
        return f"""You are grading a trainee's open-ended microlearning response for a BPO training platform.

Score the trainee response against the trainer's sample answer and prompt intent.
- Use semantic meaning, not just exact wording.
- Treat the required keywords as helpful signals, not a strict checklist.
- Give an integer score from 1 to {max_points} when the trainee submitted a non-empty answer.
- Give higher scores when the response is clearly relevant, accurate, and aligned with the trainer sample answer.
- Give lower scores when the response is vague, off-topic, or misses the main idea.
- Do not invent facts not supported by the trainee response.

Return JSON only in this exact shape:
{{
  "score": 1,
  "feedback": "Short constructive feedback for the trainee",
  "matched_keywords": ["keyword 1"],
  "missing_keywords": ["keyword 2"]
}}"""

    def _build_microlearning_open_response_prompt(
        self,
        *,
        prompt: str,
        sample_answer: str,
        trainee_response: str,
        required_keywords: list[str],
        max_points: int,
    ) -> str:
        keyword_text = ", ".join(required_keywords) if required_keywords else "None provided"
        return f"""Grade this microlearning open-ended response.

Prompt:
{prompt or "No prompt provided."}

Trainer sample answer:
{sample_answer or "No sample answer provided."}

Required keywords:
{keyword_text}

Trainee response:
{trainee_response}

Score the response from 1 to {max_points} based on how closely it matches the trainer's intended meaning and key ideas."""

    def _fallback_microlearning_open_response(
        self,
        *,
        prompt: str,
        sample_answer: str,
        trainee_response: str,
        required_keywords: list[str],
        max_points: int,
    ) -> dict[str, Any]:
        del prompt

        response_tokens = set(_tokenize_for_microlearning(trainee_response))
        sample_tokens = set(_tokenize_for_microlearning(sample_answer))
        matched_keywords = [
            keyword for keyword in required_keywords if keyword.lower() in trainee_response.lower()
        ]
        missing_keywords = [
            keyword for keyword in required_keywords if keyword not in matched_keywords
        ]

        keyword_coverage = (
            len(matched_keywords) / len(required_keywords)
            if required_keywords
            else 0.0
        )
        sample_overlap = (
            len(response_tokens & sample_tokens) / len(sample_tokens)
            if sample_tokens
            else 0.0
        )

        if required_keywords and sample_tokens:
            blended_score = keyword_coverage * 0.45 + sample_overlap * 0.55
        elif required_keywords:
            blended_score = keyword_coverage
        elif sample_tokens:
            blended_score = sample_overlap
        else:
            blended_score = 1.0 if trainee_response.strip() else 0.0

        if not trainee_response.strip():
            points = 0
        elif blended_score <= 0:
            points = 1
        else:
            points = max(1, min(max_points, int(round(blended_score * max_points))))

        if points >= max_points - 1:
            feedback = "Strong response. It closely matches the trainer sample answer."
        elif points >= max(6, max_points - 4):
            feedback = "Mostly correct. A few trainer ideas could be stated more clearly."
        elif points >= 4:
            feedback = "Partially related. Review the trainer sample answer and strengthen the main idea."
        else:
            feedback = "The response is weak or off-target. Review the trainer sample answer and key ideas."

        return {
            "score": points,
            "feedback": feedback,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
            "provider": "fallback",
        }


# Singleton instance
_evaluation_engine: Optional[GeminiEvaluationEngine] = None


def get_evaluation_engine() -> GeminiEvaluationEngine:
    """Get the singleton evaluation engine instance."""
    global _evaluation_engine
    if _evaluation_engine is None:
        _evaluation_engine = GeminiEvaluationEngine()
    return _evaluation_engine


async def generate_evaluation_feedback(
    evaluation_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate evaluation feedback for a call simulation session.

    Args:
        evaluation_data: Dictionary containing transcript, script_flow, and target_kpis

    Returns:
        Evaluation result with scores and feedback
    """
    engine = get_evaluation_engine()

    transcript = evaluation_data.get("transcript", "")
    script_flow = evaluation_data.get("script_flow", [])
    target_kpis = evaluation_data.get("target_kpis", {})

    return engine.evaluate(transcript, script_flow, target_kpis)


def _tokenize_for_microlearning(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (value or "").lower())


def score_microlearning_open_response(
    *,
    prompt: str,
    sample_answer: str,
    trainee_response: str,
    required_keywords: Optional[list[str]] = None,
    max_points: int = 10,
) -> dict[str, Any]:
    """Score a microlearning open-ended response with Gemini when available."""
    engine = get_evaluation_engine()
    return engine.score_microlearning_open_response(
        prompt=prompt,
        sample_answer=sample_answer,
        trainee_response=trainee_response,
        required_keywords=required_keywords,
        max_points=max_points,
    )


def summarize_microlearning_assignment_performance(
    *,
    module_title: str,
    module_type: str,
    percentage_score: float,
    passing_score: float,
    passed: bool,
    breakdown: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize a finished microlearning module using Gemini when available."""
    engine = get_evaluation_engine()
    return engine.summarize_microlearning_assignment_performance(
        module_title=module_title,
        module_type=module_type,
        percentage_score=percentage_score,
        passing_score=passing_score,
        passed=passed,
        breakdown=breakdown,
    )
