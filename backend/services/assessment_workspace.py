"""Database-backed assessment workspace helpers.

These helpers intentionally use the primary PostgreSQL connection instead of the
Supabase REST client so the assessment workspace can still function when the
configured Supabase API keys do not match the database project.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import random
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Sequence

from fastapi import HTTPException
from sqlalchemy import bindparam, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.models import User, UserRole
from backend.services.notifications import notify_assessment_completion

logger = logging.getLogger(__name__)

QUESTION_TEMPLATE_HEADER = [
    "Question Number",
    "Assessment Title",
    "Category",
    "Question",
    "Choice 1",
    "Choice 2",
    "Choice 3",
    "Choice 4",
    "Correct Answer",
    "Difficulty Level",
    "Points",
    "Explanation",
]

CSV_COLUMN_ALIASES = {
    "Question Number": ["Question Number", "Question No", "QuestionNo"],
    "Assessment Title": ["Assessment Title", "Assessment", "Assessment Name"],
    "Category": ["Category", "Category Name"],
    "Question": ["Question", "Question Text", "Question Prompt"],
    "Choice 1": ["Choice 1", "Choice1", "Option 1", "Option1", "Option A", "Choice A"],
    "Choice 2": ["Choice 2", "Choice2", "Option 2", "Option2", "Option B", "Choice B"],
    "Choice 3": ["Choice 3", "Choice3", "Option 3", "Option3", "Option C", "Choice C"],
    "Choice 4": ["Choice 4", "Choice4", "Option 4", "Option4", "Option D", "Choice D"],
    "Correct Answer": ["Correct Answer", "CorrectAnswer", "Answer Key", "Answer"],
    "Difficulty Level": ["Difficulty Level", "Difficulty", "DifficultyLevel"],
    "Points": ["Points", "Point Value", "PointValue"],
    "Explanation": ["Explanation", "Rationale", "Notes"],
}

REQUIRED_TEMPLATE_COLUMNS = [column for column in QUESTION_TEMPLATE_HEADER if column != "Explanation"]
DEFAULT_PASSING_SCORE = 90
DEFAULT_ANALYSIS_SUMMARY = (
    "Passing score achieved. Your certificate has been unlocked for this assessment category."
)
DEFAULT_ANALYSIS_RETRY = (
    "Passing score not reached yet. Review the summary, study the missed items, and retake the assessment."
)
ALLOWED_ASSESSMENT_TYPES = {"multiple_choice", "fill_blank", "mixed"}
ALLOWED_QUESTION_TYPES = {"multiple_choice", "fill_blank"}
ALLOWED_DIFFICULTY_LEVELS = {"easy", "medium", "hard"}
ALLOWED_ASSIGNMENT_TARGET_TYPES = {"batch", "wave", "trainee"}
ALLOWED_ASSIGNMENT_MODES = {"selected_questions", "entire_category", "random_subset"}


def _normalize_value(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _sanitize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", "").replace("\r\n", "\n").split()).strip()


def _normalize_csv_column_name(value: str) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _normalize_datetime(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()
    if value is None:
        return None
    return str(value)


def _normalize_float(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_int(value: Any, fallback: int = 0) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _truthy(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    return bool(value)


def _question_point_value(metadata: Any) -> int:
    if isinstance(metadata, dict):
        raw_value = metadata.get("points")
        parsed = _normalize_int(raw_value, fallback=1)
        return parsed if parsed > 0 else 1
    return 1


def _to_csv_cell(value: Any) -> str:
    normalized = "" if value is None else str(value)
    if not any(character in normalized for character in [",", "\"", "\n", "\r"]):
        return normalized
    escaped = normalized.replace('"', '""')
    return f'"{escaped}"'


def build_assessment_csv_template() -> str:
    sample_row = [
        "1",
        "Product Knowledge Readiness Check",
        "Product Knowledge",
        "Which statement best describes the product escalation path?",
        "Transfer to Tier 2 after validating the account details.",
        "End the call and ask the customer to email support.",
        "Skip verification if the customer sounds upset.",
        "Promise a refund immediately.",
        "A",
        "medium",
        "5",
        "Validate the account before escalating to protect customer data and route the issue correctly.",
    ]
    return "\n".join(
        [
            ",".join(_to_csv_cell(value) for value in QUESTION_TEMPLATE_HEADER),
            ",".join(_to_csv_cell(value) for value in sample_row),
        ]
    )


def _build_error_csv(errors: Sequence[dict[str, Any]]) -> str | None:
    if not errors:
        return None
    rows = [["Row Number", "Category", "Question Number", "Question", "Error"]]
    for error in errors:
        rows.append(
            [
                error.get("rowNumber", ""),
                error.get("category", ""),
                error.get("questionNumber", ""),
                error.get("question", ""),
                error.get("error", ""),
            ]
        )
    return "\n".join(",".join(_to_csv_cell(value) for value in row) for row in rows)


def _fetch_rows(
    db: Session,
    sql: str,
    params: dict[str, Any] | None = None,
    *,
    expanding: Iterable[str] = (),
) -> list[dict[str, Any]]:
    statement = text(sql)
    for key in expanding:
        statement = statement.bindparams(bindparam(key, expanding=True))
    return [dict(row) for row in db.execute(statement, params or {}).mappings().all()]


def _fetch_one(
    db: Session,
    sql: str,
    params: dict[str, Any] | None = None,
    *,
    expanding: Iterable[str] = (),
) -> dict[str, Any] | None:
    rows = _fetch_rows(db, sql, params, expanding=expanding)
    return rows[0] if rows else None


def _execute_returning_one(db: Session, sql: str, params: dict[str, Any]) -> dict[str, Any]:
    statement = text(sql)
    row = db.execute(statement, params).mappings().first()
    if not row:
        raise HTTPException(status_code=500, detail="Database operation did not return a row.")
    return dict(row)


def _format_batch_label(batch: dict[str, Any] | None) -> str:
    if not batch:
        return "Direct Assignment"
    if batch.get("waveNumber") is not None:
        return f"{batch['name']} | Wave {batch['waveNumber']}"
    return batch["name"]


def _build_choice_validation(options: Sequence[str], correct_answer: str) -> tuple[list[str], str]:
    choices = [_sanitize_text(option) for option in options]
    if len(choices) != 4 or any(not choice for choice in choices):
        raise HTTPException(
            status_code=400,
            detail="Exactly four answer choices are required for each multiple-choice question.",
        )

    normalized_choices = {_normalize_value(choice) for choice in choices}
    if len(normalized_choices) != len(choices):
        raise HTTPException(status_code=400, detail="Each answer choice must be unique.")

    normalized_correct = _normalize_value(correct_answer)
    matched_choice = next((choice for choice in choices if _normalize_value(choice) == normalized_correct), None)
    if not matched_choice:
        answer_keys = {"a": 0, "b": 1, "c": 2, "d": 3, "choice1": 0, "choice2": 1, "choice3": 2, "choice4": 3}
        answer_index = answer_keys.get(normalized_correct.replace(" ", ""))
        if answer_index is not None:
            matched_choice = choices[answer_index]

    if not matched_choice:
        raise HTTPException(
            status_code=400,
            detail=(
                "Correct answer must be A, B, C, D, Choice 1-4, "
                "or exactly match one of the four answer choices."
            ),
        )

    return choices, matched_choice


def _build_canonical_column_index(header: Sequence[str]) -> dict[str, int]:
    normalized_header = [_normalize_csv_column_name(column) for column in header]
    result: dict[str, int] = {}
    for canonical_column, aliases in CSV_COLUMN_ALIASES.items():
        normalized_aliases = {_normalize_csv_column_name(alias) for alias in aliases}
        for index, column_name in enumerate(normalized_header):
            if column_name in normalized_aliases:
                result[canonical_column] = index
                break
    return result


def _question_record(
    question: dict[str, Any],
    category_name: str | None,
    assessment_title: str | None,
    usage_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    options = question.get("options") if isinstance(question.get("options"), list) else []
    answer_count = _normalize_int(usage_stats.get("answerCount") if usage_stats else 0)
    correct_count = _normalize_int(usage_stats.get("correctCount") if usage_stats else 0)
    incorrect_count = _normalize_int(usage_stats.get("incorrectCount") if usage_stats else 0)
    accuracy_rate = round((correct_count / answer_count) * 100, 2) if answer_count else 0.0
    metadata = question.get("metadata") if isinstance(question.get("metadata"), dict) else {}
    return {
        "id": str(question["id"]),
        "assessmentId": str(question["assessment_id"]),
        "assessmentTitle": assessment_title,
        "categoryId": str(question["category_id"]),
        "categoryName": category_name,
        "trainerId": question.get("created_by"),
        "questionNumber": _normalize_int(question.get("question_number")),
        "questionText": question.get("question_text") or "",
        "questionType": question.get("question_type") or "multiple_choice",
        "options": options,
        "choices": options,
        "correctAnswer": question.get("correct_answer") or "",
        "difficulty": question.get("difficulty"),
        "explanation": question.get("explanation"),
        "pointValue": _question_point_value(metadata),
        "orderIndex": _normalize_int(question.get("order_index")),
        "activeStatus": _truthy(question.get("active_status"), True),
        "createdAt": _normalize_datetime(question.get("created_at")),
        "updatedAt": _normalize_datetime(question.get("updated_at")),
        "metadata": metadata,
        "usageCount": answer_count,
        "answerCount": answer_count,
        "correctCount": correct_count,
        "incorrectCount": incorrect_count,
        "accuracyRate": accuracy_rate,
        "missRate": _normalize_float(usage_stats.get("missRate") if usage_stats else 0),
    }


def _assessment_record(assessment: dict[str, Any], questions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": str(assessment["id"]),
        "categoryId": str(assessment["category_id"]),
        "title": assessment.get("title") or "Assessment",
        "description": assessment.get("description"),
        "type": assessment.get("type") or "multiple_choice",
        "isPublished": _truthy(assessment.get("is_published"), True),
        "instantFeedback": _truthy(assessment.get("instant_feedback"), False),
        "sortOrder": _normalize_int(assessment.get("sort_order")),
        "createdAt": _normalize_datetime(assessment.get("created_at")),
        "updatedAt": _normalize_datetime(assessment.get("updated_at")),
        "questionCount": len(questions),
        "questions": list(questions),
    }


def _fallback_analysis(score: float, passing_score: float, category_id: str, category_title: str) -> dict[str, Any]:
    passed = score >= passing_score
    return {
        "source": "rules",
        "summary": DEFAULT_ANALYSIS_SUMMARY if passed else DEFAULT_ANALYSIS_RETRY,
        "strengths": ["Passing threshold met."] if passed else ["Assessment submitted successfully."],
        "improvements": [] if passed else ["Review the missed items and retry the assessment."],
        "recommendations": (
            ["Keep practicing higher-difficulty questions to maintain readiness."]
            if passed
            else ["Review the missed questions, then retake the assessment once coaching is complete."]
        ),
        "categoryBreakdown": [
            {
                "categoryId": category_id,
                "categoryTitle": category_title,
                "totalQuestions": 0,
                "correctAnswers": 0,
                "score": score,
            }
        ],
    }


def _normalize_question_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    results: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "questionId": item.get("questionId") or item.get("question_id") or "",
                "questionNumber": _normalize_int(item.get("questionNumber") or item.get("question_number")),
                "questionText": item.get("questionText") or item.get("question_text") or "",
                "questionType": item.get("questionType") or item.get("question_type") or "multiple_choice",
                "difficulty": item.get("difficulty"),
                "options": item.get("options") if isinstance(item.get("options"), list) else [],
                "choiceOrder": item.get("choiceOrder") or item.get("choice_order") or [],
                "userAnswer": item.get("userAnswer") or item.get("user_answer") or "",
                "correctAnswer": item.get("correctAnswer") or item.get("correct_answer") or "",
                "isCorrect": bool(item.get("isCorrect") if "isCorrect" in item else item.get("is_correct")),
                "explanation": item.get("explanation"),
                "points": _normalize_int(item.get("points"), fallback=0),
                "earnedPoints": _normalize_int(item.get("earnedPoints") or item.get("earned_points"), fallback=0),
            }
        )
    return results


def _normalize_analysis(
    raw_analysis: Any,
    score: float,
    passing_score: float,
    category_id: str,
    category_title: str,
) -> dict[str, Any]:
    fallback = _fallback_analysis(score, passing_score, category_id, category_title)
    if not isinstance(raw_analysis, dict):
        return fallback

    strengths = raw_analysis.get("strengths")
    improvements = raw_analysis.get("improvements")
    recommendations = raw_analysis.get("recommendations")
    category_breakdown = raw_analysis.get("categoryBreakdown") or raw_analysis.get("category_breakdown")
    return {
        "source": "ai" if raw_analysis.get("source") == "ai" else "rules",
        "summary": raw_analysis.get("summary") or fallback["summary"],
        "strengths": strengths if isinstance(strengths, list) else fallback["strengths"],
        "improvements": improvements if isinstance(improvements, list) else fallback["improvements"],
        "recommendations": recommendations if isinstance(recommendations, list) else fallback["recommendations"],
        "earnedPoints": _normalize_int(raw_analysis.get("earnedPoints") or raw_analysis.get("earned_points"), fallback=0),
        "totalPoints": _normalize_int(raw_analysis.get("totalPoints") or raw_analysis.get("total_points"), fallback=0),
        "categoryBreakdown": category_breakdown if isinstance(category_breakdown, list) else fallback["categoryBreakdown"],
    }


def _attempt_record(attempt: dict[str, Any]) -> dict[str, Any]:
    question_results = _normalize_question_results(attempt.get("question_results"))
    score = _normalize_float(attempt.get("score"))
    passing_score = _normalize_float(attempt.get("passing_score"), fallback=90.0)
    return {
        "id": str(attempt["id"]),
        "assignmentId": str(attempt["assignment_id"]) if attempt.get("assignment_id") else None,
        "assessmentId": str(attempt["assessment_id"]),
        "categoryId": str(attempt["category_id"]),
        "assignmentTitle": attempt.get("assignment_title") or attempt.get("assessment_title"),
        "assessmentTitle": attempt.get("assessment_title") or "Assessment",
        "categoryTitle": attempt.get("category_title") or "Assessment Category",
        "traineeId": attempt.get("trainee_id") or "",
        "traineeName": attempt.get("trainee_name") or "Trainee",
        "traineeEmail": attempt.get("trainee_email"),
        "batchId": attempt.get("batch_id"),
        "batchName": attempt.get("batch_name"),
        "waveNumber": _normalize_int(attempt.get("wave_number"), fallback=0) if attempt.get("wave_number") is not None else None,
        "attemptNo": _normalize_int(attempt.get("attempt_no"), fallback=1),
        "score": score,
        "passingScore": passing_score,
        "status": "pass" if str(attempt.get("status") or "").lower() == "pass" else "fail",
        "feedback": attempt.get("feedback"),
        "trainerNote": attempt.get("trainer_note"),
        "submittedAt": _normalize_datetime(attempt.get("submitted_at")),
        "startedAt": _normalize_datetime(attempt.get("started_at")),
        "completedAt": _normalize_datetime(attempt.get("completed_at") or attempt.get("submitted_at")),
        "timeSpentSeconds": _normalize_int(attempt.get("time_spent_seconds"), fallback=0),
        "correctAnswers": _normalize_int(attempt.get("correct_answers"), fallback=sum(1 for result in question_results if result["isCorrect"])),
        "incorrectAnswers": _normalize_int(attempt.get("incorrect_answers"), fallback=sum(1 for result in question_results if not result["isCorrect"])),
        "totalQuestions": _normalize_int(attempt.get("total_questions"), fallback=len(question_results)),
        "certificateId": str(attempt["certificate_id"]) if attempt.get("certificate_id") else None,
        "certificateCode": attempt.get("certificate_code"),
        "certificateStatus": attempt.get("certificate_status") or "not_issued",
        "certificateUrl": attempt.get("certificate_url"),
        "questionResults": question_results,
        "analysis": _normalize_analysis(
            attempt.get("analysis_summary"),
            score,
            passing_score,
            str(attempt["category_id"]),
            attempt.get("category_title") or "Assessment Category",
        ),
    }


def _certificate_record(
    certificate: dict[str, Any],
    categories_by_id: dict[str, dict[str, Any]],
    assignments_by_id: dict[str, dict[str, Any]],
    assessments_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    category = categories_by_id.get(str(certificate["category_id"]))
    assignment = assignments_by_id.get(str(certificate["assignment_id"])) if certificate.get("assignment_id") else None
    assessment = assessments_by_id.get(str(certificate["assessment_id"]))
    return {
        "id": str(certificate["id"]),
        "traineeId": certificate.get("trainee_id") or "",
        "categoryId": str(certificate["category_id"]),
        "assignmentId": str(certificate["assignment_id"]) if certificate.get("assignment_id") else None,
        "assessmentId": str(certificate["assessment_id"]),
        "attemptId": str(certificate["attempt_id"]),
        "categoryTitle": category.get("title") if category else "Assessment Category",
        "assignmentTitle": certificate.get("assignment_title") or (assignment.get("title") if assignment else None) or (assessment.get("title") if assessment else None) or "Assessment",
        "assessmentTitle": (assessment.get("title") if assessment else None) or (assignment.get("title") if assignment else None) or "Assessment",
        "certificateCode": certificate.get("certificate_code") or "",
        "certificateStatus": certificate.get("certificate_status") or "issued",
        "certificateUrl": certificate.get("certificate_url"),
        "earnedAt": _normalize_datetime(certificate.get("earned_at")),
    }


def _fetch_visible_categories(db: Session, current_user: User, *, include_inactive: bool = False) -> list[dict[str, Any]]:
    sql = """
        select *
        from training_assessment_categories
        where created_by = :user_id
    """
    params = {"user_id": current_user.id}
    if current_user.role == UserRole.ADMIN:
        sql = "select * from training_assessment_categories where 1=1"
        params = {}
    if not include_inactive:
        sql += " and coalesce(is_archived, false) = false and coalesce(active_status, true) = true"
    sql += " order by updated_at desc nulls last, created_at desc nulls last"
    return _fetch_rows(db, sql, params)


def _fetch_assessments_for_categories(db: Session, category_ids: Sequence[str], *, include_inactive: bool = False) -> list[dict[str, Any]]:
    if not category_ids:
        return []
    sql = """
        select *
        from training_assessments
        where category_id in :category_ids
    """
    if not include_inactive:
        sql += " and coalesce(active_status, true) = true"
    sql += " order by coalesce(is_primary, false) desc, coalesce(sort_order, 0) asc, created_at asc"
    return _fetch_rows(db, sql, {"category_ids": list(category_ids)}, expanding=("category_ids",))


def _fetch_questions_for_categories(db: Session, category_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not category_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_questions
        where category_id in :category_ids
        order by coalesce(question_number, 0) asc, created_at asc
        """,
        {"category_ids": list(category_ids)},
        expanding=("category_ids",),
    )


def _fetch_batch_context(db: Session, current_user: User) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    batch_sql = """
        select id, name, description, wave_number, created_by, is_active
        from batch
        where coalesce(is_active, true) = true
    """
    params: dict[str, Any] = {}
    if current_user.role != UserRole.ADMIN:
        batch_sql += " and created_by = :user_id"
        params["user_id"] = current_user.id
    batch_sql += " order by wave_number asc nulls last, name asc"

    batch_rows = _fetch_rows(db, batch_sql, params)
    if not batch_rows:
        return [], [], []

    batch_ids = [row["id"] for row in batch_rows]
    membership_rows = _fetch_rows(
        db,
        """
        select batch_id, user_id
        from batch_user
        where batch_id in :batch_ids
        """,
        {"batch_ids": batch_ids},
        expanding=("batch_ids",),
    )
    trainee_ids = sorted({row["user_id"] for row in membership_rows if row.get("user_id")})
    trainee_rows = (
        _fetch_rows(
            db,
            """
            select id, email, full_name, role
            from "user"
            where id in :trainee_ids and lower(role::text) = 'trainee'
            """,
            {"trainee_ids": trainee_ids},
            expanding=("trainee_ids",),
        )
        if trainee_ids
        else []
    )
    return batch_rows, membership_rows, trainee_rows


def _fetch_assignments(db: Session, category_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not category_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_assignments
        where category_id in :category_ids
        order by assigned_at desc
        """,
        {"category_ids": list(category_ids)},
        expanding=("category_ids",),
    )


def _fetch_assignment_question_rows(db: Session, assignment_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not assignment_ids:
        return []
    return _fetch_rows(
        db,
        """
        select assignment_id, question_id, question_order
        from training_assessment_assignment_questions
        where assignment_id in :assignment_ids
        order by question_order asc, created_at asc
        """,
        {"assignment_ids": list(assignment_ids)},
        expanding=("assignment_ids",),
    )


def _fetch_attempts(db: Session, category_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not category_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_attempt_feed
        where category_id in :category_ids
        order by submitted_at desc
        """,
        {"category_ids": list(category_ids)},
        expanding=("category_ids",),
    )


def _fetch_certificates(db: Session, category_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not category_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_certificates
        where category_id in :category_ids
        order by earned_at desc
        """,
        {"category_ids": list(category_ids)},
        expanding=("category_ids",),
    )


def _fetch_question_reports(db: Session, category_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not category_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_question_report
        where category_id in :category_ids
        order by miss_rate desc nulls last, question_number asc nulls last
        """,
        {"category_ids": list(category_ids)},
        expanding=("category_ids",),
    )


def _ensure_primary_assessment(db: Session, category: dict[str, Any]) -> dict[str, Any]:
    category_id = str(category["id"])
    primary = _fetch_one(
        db,
        """
        select *
        from training_assessments
        where category_id = :category_id
          and coalesce(is_primary, false) = true
        order by created_at asc
        limit 1
        """,
        {"category_id": category_id},
    )
    if primary:
        if not _truthy(primary.get("active_status"), True):
            primary = _execute_returning_one(
                db,
                """
                update training_assessments
                set active_status = true,
                    title = :title,
                    description = :description,
                    updated_at = timezone('utc', now())
                where id = :assessment_id
                returning *
                """,
                {
                    "assessment_id": primary["id"],
                    "title": category["title"],
                    "description": category.get("description"),
                },
            )
            db.commit()
        return primary

    existing = _fetch_one(
        db,
        """
        select *
        from training_assessments
        where category_id = :category_id
        order by coalesce(sort_order, 0) asc, created_at asc
        limit 1
        """,
        {"category_id": category_id},
    )
    if existing:
        updated = _execute_returning_one(
            db,
            """
            update training_assessments
            set title = :title,
                description = :description,
                is_primary = true,
                active_status = true,
                updated_at = timezone('utc', now())
            where id = :assessment_id
            returning *
            """,
            {
                "assessment_id": existing["id"],
                "title": category["title"],
                "description": category.get("description"),
            },
        )
        db.commit()
        return updated

    inserted = _execute_returning_one(
        db,
        """
        insert into training_assessments (
            category_id,
            title,
            description,
            type,
            is_published,
            instant_feedback,
            sort_order,
            is_primary,
            active_status
        )
        values (
            :category_id,
            :title,
            :description,
            'multiple_choice',
            true,
            false,
            0,
            true,
            true
        )
        returning *
        """,
        {
            "category_id": category_id,
            "title": category["title"],
            "description": category.get("description"),
        },
    )
    db.commit()
    return inserted


def _get_or_create_category(
    db: Session,
    current_user: User,
    category_name: str,
    category_cache: dict[str, dict[str, Any]],
    created_categories: list[str],
) -> dict[str, Any]:
    normalized_name = _normalize_value(category_name)
    existing = category_cache.get(normalized_name)
    if existing:
        return existing

    owned_categories = _fetch_visible_categories(db, current_user, include_inactive=True)
    archived = next((row for row in owned_categories if _normalize_value(row.get("title")) == normalized_name), None)
    if archived:
        reactivated = _execute_returning_one(
            db,
            """
            update training_assessment_categories
            set title = :title,
                active_status = true,
                is_archived = false,
                updated_at = timezone('utc', now())
            where id = :category_id
            returning *
            """,
            {"title": category_name, "category_id": archived["id"]},
        )
        db.commit()
        category_cache[normalized_name] = reactivated
        if reactivated["title"] not in created_categories:
            created_categories.append(reactivated["title"])
        return reactivated

    inserted = _execute_returning_one(
        db,
        """
        insert into training_assessment_categories (
            title,
            description,
            passing_score,
            created_by,
            is_archived,
            active_status
        )
        values (
            :title,
            :description,
            :passing_score,
            :created_by,
            false,
            true
        )
        returning *
        """,
        {
            "title": category_name,
            "description": None,
            "passing_score": DEFAULT_PASSING_SCORE,
            "created_by": current_user.id,
        },
    )
    db.commit()
    category_cache[normalized_name] = inserted
    created_categories.append(inserted["title"])
    return inserted


def _get_or_create_assessment(
    db: Session,
    category: dict[str, Any],
    assessment_title: str,
    assessment_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    category_id = str(category["id"])
    normalized_title = _normalize_value(assessment_title)
    cache_key = (category_id, normalized_title)
    existing = assessment_cache.get(cache_key)
    if existing:
        if normalized_title == _normalize_value(category["title"]) and not _truthy(existing.get("is_primary"), False):
            existing = _ensure_primary_assessment(db, category)
            assessment_cache[(category_id, _normalize_value(existing["title"]))] = existing
        return existing

    assessments = _fetch_assessments_for_categories(db, [category_id], include_inactive=True)
    matching = next((row for row in assessments if _normalize_value(row.get("title")) == normalized_title), None)
    if matching:
        if not _truthy(matching.get("active_status"), True):
            matching = _execute_returning_one(
                db,
                """
                update training_assessments
                set title = :title,
                    description = :description,
                    active_status = true,
                    updated_at = timezone('utc', now())
                where id = :assessment_id
                returning *
                """,
                {
                    "assessment_id": matching["id"],
                    "title": assessment_title,
                    "description": category.get("description"),
                },
            )
            db.commit()
        if normalized_title == _normalize_value(category["title"]):
            matching = _ensure_primary_assessment(db, category)
        assessment_cache[(category_id, _normalize_value(matching["title"]))] = matching
        return matching

    if normalized_title == _normalize_value(category["title"]):
        primary = _ensure_primary_assessment(db, category)
        assessment_cache[(category_id, _normalize_value(primary["title"]))] = primary
        return primary

    inserted = _execute_returning_one(
        db,
        """
        insert into training_assessments (
            category_id,
            title,
            description,
            type,
            is_published,
            instant_feedback,
            sort_order,
            is_primary,
            active_status
        )
        values (
            :category_id,
            :title,
            :description,
            'multiple_choice',
            true,
            false,
            0,
            false,
            true
        )
        returning *
        """,
        {
            "category_id": category_id,
            "title": assessment_title,
            "description": category.get("description"),
        },
    )
    db.commit()
    assessment_cache[cache_key] = inserted
    return inserted


def _ensure_assignment_workspace_schema(db: Session) -> None:
    dialect_name = getattr(getattr(db, "bind", None), "dialect", None)
    dialect_name = str(getattr(dialect_name, "name", "") or "").lower()

    if dialect_name and dialect_name != "postgresql":
        raise RuntimeError("Assessment workspaces require the Supabase PostgreSQL database.")

    db.execute(
        text(
            """
            alter table training_assessment_assignments
                add column if not exists target_scope text,
                add column if not exists wave_number integer
            """
        )
    )
    blank_target_predicate = "target_scope is null or btrim(target_scope) = ''"

    db.execute(
        text(
            f"""
            update training_assessment_assignments
            set target_scope = case
                when trainee_id is not null then 'trainee'
                when wave_number is not null and batch_id is null then 'wave'
                else 'batch'
            end
            where {blank_target_predicate}
            """
        )
    )
    db.commit()


def _require_owned_category(
    db: Session,
    current_user: User,
    category_id: str,
) -> dict[str, Any]:
    sql = """
        select *
        from training_assessment_categories
        where id = :category_id
    """
    params: dict[str, Any] = {"category_id": category_id}
    if current_user.role != UserRole.ADMIN:
        sql += " and created_by = :user_id"
        params["user_id"] = current_user.id

    category = _fetch_one(db, sql, params)
    if not category:
        raise HTTPException(status_code=404, detail="Assessment category not found.")
    return category


def _require_owned_assessment(
    db: Session,
    current_user: User,
    assessment_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assessment = _fetch_one(
        db,
        """
        select *
        from training_assessments
        where id = :assessment_id
        """,
        {"assessment_id": assessment_id},
    )
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment definition not found.")

    category = _require_owned_category(db, current_user, str(assessment["category_id"]))
    return assessment, category


def _require_owned_question(
    db: Session,
    current_user: User,
    question_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    question = _fetch_one(
        db,
        """
        select *
        from training_assessment_questions
        where id = :question_id
        """,
        {"question_id": question_id},
    )
    if not question:
        raise HTTPException(status_code=404, detail="Assessment question not found.")

    category = _require_owned_category(db, current_user, str(question["category_id"]))
    assessment = None
    if question.get("assessment_id"):
        assessment = _fetch_one(
            db,
            """
            select *
            from training_assessments
            where id = :assessment_id
            """,
            {"assessment_id": question["assessment_id"]},
        )
    return question, category, assessment


def _require_owned_assignment(
    db: Session,
    current_user: User,
    assignment_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _ensure_assignment_workspace_schema(db)
    assignment = _fetch_one(
        db,
        """
        select *
        from training_assessment_assignments
        where id = :assignment_id
        """,
        {"assignment_id": assignment_id},
    )
    if not assignment:
        raise HTTPException(status_code=404, detail="Assessment assignment not found.")

    category = _require_owned_category(db, current_user, str(assignment["category_id"]))
    return assignment, category


def _normalize_nullable_text(value: Any) -> str | None:
    sanitized = _sanitize_text(value)
    return sanitized or None


def _parse_passing_score(value: Any) -> int:
    passing_score = _normalize_int(value, fallback=-1)
    if passing_score < 0 or passing_score > 100:
        raise HTTPException(status_code=400, detail="Passing score must be between 0 and 100.")
    return passing_score


def _parse_optional_positive_int(value: Any, field_label: str) -> int | None:
    if value in (None, "", 0):
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_label} must be a whole number.") from exc

    if parsed < 1:
        raise HTTPException(status_code=400, detail=f"{field_label} must be greater than zero.")
    return parsed


def _validate_assessment_type(value: Any) -> str:
    assessment_type = _sanitize_text(value or "multiple_choice").lower()
    if assessment_type not in ALLOWED_ASSESSMENT_TYPES:
        raise HTTPException(status_code=400, detail="Assessment type is invalid.")
    return assessment_type


def _validate_question_type(value: Any) -> str:
    question_type = _sanitize_text(value or "multiple_choice").lower()
    if question_type not in ALLOWED_QUESTION_TYPES:
        raise HTTPException(status_code=400, detail="Question type is invalid.")
    return question_type


def _validate_difficulty(value: Any) -> str | None:
    difficulty = _normalize_value(value)
    if not difficulty:
        return None
    if difficulty not in ALLOWED_DIFFICULTY_LEVELS:
        raise HTTPException(status_code=400, detail="Difficulty must be easy, medium, or hard.")
    return difficulty


def _serialize_category(category: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(category["id"]),
        "title": category.get("title") or "Assessment Category",
        "description": category.get("description"),
        "passingScore": _normalize_int(category.get("passing_score"), fallback=DEFAULT_PASSING_SCORE),
        "createdBy": category.get("created_by") or "",
        "isArchived": _truthy(category.get("is_archived"), False),
        "activeStatus": _truthy(category.get("active_status"), True),
        "createdAt": _normalize_datetime(category.get("created_at")),
        "updatedAt": _normalize_datetime(category.get("updated_at")),
    }


def _serialize_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(assessment["id"]),
        "categoryId": str(assessment["category_id"]),
        "title": assessment.get("title") or "Assessment",
        "description": assessment.get("description"),
        "type": assessment.get("type") or "multiple_choice",
        "isPublished": _truthy(assessment.get("is_published"), True),
        "instantFeedback": _truthy(assessment.get("instant_feedback"), True),
        "sortOrder": _normalize_int(assessment.get("sort_order")),
        "isPrimary": _truthy(assessment.get("is_primary"), False),
        "activeStatus": _truthy(assessment.get("active_status"), True),
        "createdAt": _normalize_datetime(assessment.get("created_at")),
        "updatedAt": _normalize_datetime(assessment.get("updated_at")),
    }


def _serialize_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(assignment["id"]),
        "categoryId": str(assignment["category_id"]),
        "assessmentId": str(assignment["assessment_id"]) if assignment.get("assessment_id") else None,
        "batchId": assignment.get("batch_id"),
        "waveNumber": _normalize_int(assignment.get("wave_number")) if assignment.get("wave_number") is not None else None,
        "traineeId": assignment.get("trainee_id"),
        "targetType": assignment.get("target_scope") or ("trainee" if assignment.get("trainee_id") else "batch"),
        "title": assignment.get("title") or "Assessment Assignment",
        "description": assignment.get("description"),
        "assignedBy": assignment.get("assigned_by") or "",
        "assignedAt": _normalize_datetime(assignment.get("assigned_at")),
        "dueAt": _normalize_datetime(assignment.get("due_at")),
        "isActive": _truthy(assignment.get("is_active"), True),
        "assignmentMode": assignment.get("assignment_mode") or "entire_category",
        "questionCount": _normalize_int(assignment.get("question_count")),
        "passingScore": _normalize_int(assignment.get("passing_score"), fallback=DEFAULT_PASSING_SCORE),
        "maximumAttempts": _normalize_int(assignment.get("maximum_attempts")) if assignment.get("maximum_attempts") is not None else None,
        "timeLimitMinutes": _normalize_int(assignment.get("time_limit_minutes")) if assignment.get("time_limit_minutes") is not None else None,
        "shuffleChoices": _truthy(assignment.get("shuffle_choices"), True),
        "shuffleQuestions": _truthy(assignment.get("shuffle_questions"), False),
    }


def _next_assessment_sort_order(db: Session, category_id: str) -> int:
    rows = _fetch_rows(
        db,
        """
        select coalesce(sort_order, 0) as sort_order
        from training_assessments
        where category_id = :category_id
          and coalesce(active_status, true) = true
        order by coalesce(sort_order, 0) desc
        limit 1
        """,
        {"category_id": category_id},
    )
    return _normalize_int(rows[0]["sort_order"], fallback=-1) + 1 if rows else 0


def _next_question_number(db: Session, category_id: str) -> int:
    row = _fetch_one(
        db,
        """
        select coalesce(max(question_number), 0) as max_question_number
        from training_assessment_questions
        where category_id = :category_id
        """,
        {"category_id": category_id},
    )
    return _normalize_int((row or {}).get("max_question_number"), fallback=0) + 1


def _resolve_question_scope(
    db: Session,
    current_user: User,
    *,
    category_id: str | None,
    assessment_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if assessment_id:
        assessment, category = _require_owned_assessment(db, current_user, assessment_id)
        if category_id and str(assessment["category_id"]) != category_id:
            raise HTTPException(
                status_code=400,
                detail="The selected assessment does not belong to the selected category.",
            )
        return category, assessment

    if category_id:
        category = _require_owned_category(db, current_user, category_id)
        assessment = _ensure_primary_assessment(db, category)
        return category, assessment

    raise HTTPException(status_code=400, detail="Category is required before saving a question.")


def _validate_question_number_availability(
    db: Session,
    category_id: str,
    question_number: int,
    *,
    exclude_question_id: str | None = None,
) -> None:
    sql = """
        select id
        from training_assessment_questions
        where category_id = :category_id
          and question_number = :question_number
    """
    params: dict[str, Any] = {
        "category_id": category_id,
        "question_number": question_number,
    }
    if exclude_question_id:
        sql += " and id <> :question_id"
        params["question_id"] = exclude_question_id

    duplicate = _fetch_one(db, sql, params)
    if duplicate:
        raise HTTPException(status_code=400, detail="Question Number already exists for this category.")


def _validate_question_text_availability(
    db: Session,
    category_id: str,
    question_text: str,
    *,
    exclude_question_id: str | None = None,
) -> None:
    rows = _fetch_rows(
        db,
        """
        select id, question_text
        from training_assessment_questions
        where category_id = :category_id
        """,
        {"category_id": category_id},
    )
    normalized_text = _normalize_value(question_text)
    for row in rows:
        if exclude_question_id and str(row["id"]) == exclude_question_id:
            continue
        if _normalize_value(row.get("question_text")) == normalized_text:
            raise HTTPException(status_code=400, detail="Duplicate question text detected for this category.")


def _resolve_assignment_target(
    *,
    target_type: str,
    batch_id: str | None,
    wave_number: int | None,
    trainee_id: str | None,
    batch_options: Sequence[dict[str, Any]],
    trainee_options: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if target_type not in ALLOWED_ASSIGNMENT_TARGET_TYPES:
        raise HTTPException(status_code=400, detail="Assignment target type is invalid.")

    if target_type == "batch":
        if not batch_id:
            raise HTTPException(status_code=400, detail="Select a batch before saving the assignment.")
        selected_batch = next((batch for batch in batch_options if batch["id"] == batch_id), None)
        if not selected_batch:
            raise HTTPException(status_code=404, detail="The selected batch is not available in this workspace.")
        return {
            "targetType": "batch",
            "batchId": selected_batch["id"],
            "waveNumber": selected_batch.get("waveNumber"),
            "traineeId": None,
        }

    if target_type == "wave":
        if wave_number is None:
            raise HTTPException(status_code=400, detail="Select a wave before saving the assignment.")
        matching_batches = [batch for batch in batch_options if batch.get("waveNumber") == wave_number]
        if not matching_batches:
            raise HTTPException(status_code=404, detail="The selected wave is not available in this workspace.")
        return {
            "targetType": "wave",
            "batchId": None,
            "waveNumber": wave_number,
            "traineeId": None,
        }

    if not trainee_id:
        raise HTTPException(status_code=400, detail="Select a trainee before saving the assignment.")
    selected_trainee = next((trainee for trainee in trainee_options if trainee["id"] == trainee_id), None)
    if not selected_trainee:
        raise HTTPException(status_code=404, detail="The selected trainee is not available in this workspace.")
    return {
        "targetType": "trainee",
        "batchId": None,
        "waveNumber": None,
        "traineeId": selected_trainee["id"],
    }


def _validate_assignment_mode(value: Any) -> str:
    assignment_mode = _sanitize_text(value or "entire_category").lower()
    if assignment_mode not in ALLOWED_ASSIGNMENT_MODES:
        raise HTTPException(status_code=400, detail="Assignment mode is invalid.")
    return assignment_mode


def _ensure_assignment_title_available(
    db: Session,
    *,
    category_id: str,
    assessment_id: str | None,
    title: str,
    target_type: str,
    batch_id: str | None,
    wave_number: int | None,
    trainee_id: str | None,
    exclude_assignment_id: str | None = None,
) -> None:
    sql = """
        select id
        from training_assessment_assignments
        where category_id = :category_id
          and coalesce(is_active, true) = true
          and lower(btrim(title)) = lower(btrim(:title))
          and coalesce(target_scope, case when trainee_id is not null then 'trainee' when wave_number is not null and batch_id is null then 'wave' else 'batch' end) = :target_scope
          and ((:assessment_id is null and assessment_id is null) or assessment_id = :assessment_id)
          and ((:batch_id is null and batch_id is null) or batch_id = :batch_id)
          and ((:wave_number is null and wave_number is null) or wave_number = :wave_number)
          and ((:trainee_id is null and trainee_id is null) or trainee_id = :trainee_id)
    """
    params: dict[str, Any] = {
        "category_id": category_id,
        "assessment_id": assessment_id,
        "title": title,
        "target_scope": target_type,
        "batch_id": batch_id,
        "wave_number": wave_number,
        "trainee_id": trainee_id,
    }
    if exclude_assignment_id:
        sql += " and id <> :assignment_id"
        params["assignment_id"] = exclude_assignment_id

    duplicate = _fetch_one(db, sql, params)
    if duplicate:
        raise HTTPException(
            status_code=400,
            detail="An active assignment with the same title already exists for this target.",
        )


def _refresh_selected_assignment_counts(db: Session, assignment_ids: Sequence[str]) -> None:
    if not assignment_ids:
        return

    count_rows = _fetch_rows(
        db,
        """
        select assignment_id, count(*)::int as question_count
        from training_assessment_assignment_questions
        where assignment_id in :assignment_ids
        group by assignment_id
        """,
        {"assignment_ids": list(assignment_ids)},
        expanding=("assignment_ids",),
    )
    count_map = {str(row["assignment_id"]): _normalize_int(row.get("question_count")) for row in count_rows}
    for assignment_id in assignment_ids:
        db.execute(
            text(
                """
                update training_assessment_assignments
                set question_count = :question_count,
                    updated_at = timezone('utc', now())
                where id = :assignment_id
                  and assignment_mode = 'selected_questions'
                """
            ),
            {
                "assignment_id": assignment_id,
                "question_count": count_map.get(str(assignment_id), 0),
            },
        )


def create_category_record(
    db: Session,
    current_user: User,
    *,
    title: str,
    description: str | None,
    passing_score: int,
) -> dict[str, Any]:
    normalized_title = _sanitize_text(title)
    if not normalized_title:
        raise HTTPException(status_code=400, detail="Category title is required.")

    owned_categories = _fetch_visible_categories(db, current_user, include_inactive=True)
    normalized_lookup = _normalize_value(normalized_title)
    existing = next(
        (row for row in owned_categories if _normalize_value(row.get("title")) == normalized_lookup),
        None,
    )

    try:
        if existing and _truthy(existing.get("active_status"), True) and not _truthy(existing.get("is_archived"), False):
            raise HTTPException(
                status_code=400,
                detail="An assessment category with this title already exists in your workspace.",
            )

        if existing:
            category = _execute_returning_one(
                db,
                """
                update training_assessment_categories
                set title = :title,
                    description = :description,
                    passing_score = :passing_score,
                    is_archived = false,
                    active_status = true,
                    updated_at = timezone('utc', now())
                where id = :category_id
                returning *
                """,
                {
                    "category_id": existing["id"],
                    "title": normalized_title,
                    "description": description,
                    "passing_score": passing_score,
                },
            )
        else:
            category = _execute_returning_one(
                db,
                """
                insert into training_assessment_categories (
                    title,
                    description,
                    passing_score,
                    created_by,
                    is_archived,
                    active_status
                )
                values (
                    :title,
                    :description,
                    :passing_score,
                    :created_by,
                    false,
                    true
                )
                returning *
                """,
                {
                    "title": normalized_title,
                    "description": description,
                    "passing_score": passing_score,
                    "created_by": current_user.id,
                },
            )

        db.commit()
        _ensure_primary_assessment(db, category)
        return _serialize_category(category)
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save the assessment category.") from exc


def update_category_record(
    db: Session,
    current_user: User,
    category_id: str,
    *,
    title: str,
    description: str | None,
    passing_score: int,
) -> dict[str, Any]:
    category = _require_owned_category(db, current_user, category_id)
    normalized_title = _sanitize_text(title)
    if not normalized_title:
        raise HTTPException(status_code=400, detail="Category title is required.")

    sibling_categories = _fetch_visible_categories(db, current_user, include_inactive=True)
    for sibling in sibling_categories:
        if str(sibling["id"]) == category_id:
            continue
        if _normalize_value(sibling.get("title")) == _normalize_value(normalized_title):
            raise HTTPException(
                status_code=400,
                detail="Another assessment category in your workspace already uses that title.",
            )

    try:
        updated = _execute_returning_one(
            db,
            """
            update training_assessment_categories
            set title = :title,
                description = :description,
                passing_score = :passing_score,
                updated_at = timezone('utc', now())
            where id = :category_id
            returning *
            """,
            {
                "category_id": category_id,
                "title": normalized_title,
                "description": description,
                "passing_score": passing_score,
            },
        )
        db.execute(
            text(
                """
                update training_assessments
                set title = :title,
                    description = :description,
                    updated_at = timezone('utc', now())
                where category_id = :category_id
                  and coalesce(is_primary, false) = true
                """
            ),
            {
                "category_id": category_id,
                "title": normalized_title,
                "description": description,
            },
        )
        db.commit()
        return _serialize_category(updated)
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to update the assessment category.") from exc


def archive_category_record(
    db: Session,
    current_user: User,
    category_id: str,
) -> None:
    _require_owned_category(db, current_user, category_id)
    _ensure_assignment_workspace_schema(db)
    try:
        db.execute(
            text(
                """
                update training_assessment_questions
                set active_status = false,
                    updated_at = timezone('utc', now())
                where category_id = :category_id
                """
            ),
            {"category_id": category_id},
        )
        db.execute(
            text(
                """
                update training_assessments
                set active_status = false,
                    updated_at = timezone('utc', now())
                where category_id = :category_id
                """
            ),
            {"category_id": category_id},
        )
        db.execute(
            text(
                """
                update training_assessment_assignments
                set is_active = false,
                    updated_at = timezone('utc', now())
                where category_id = :category_id
                """
            ),
            {"category_id": category_id},
        )
        db.execute(
            text(
                """
                update training_assessment_categories
                set is_archived = true,
                    active_status = false,
                    updated_at = timezone('utc', now())
                where id = :category_id
                """
            ),
            {"category_id": category_id},
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to archive the assessment category.") from exc


def create_assessment_record(
    db: Session,
    current_user: User,
    *,
    category_id: str,
    title: str,
    description: str | None,
    assessment_type: str,
    is_published: bool,
) -> dict[str, Any]:
    _require_owned_category(db, current_user, category_id)
    normalized_title = _sanitize_text(title)
    if not normalized_title:
        raise HTTPException(status_code=400, detail="Assessment title is required.")

    sibling_assessments = _fetch_assessments_for_categories(db, [category_id], include_inactive=True)
    if any(_normalize_value(row.get("title")) == _normalize_value(normalized_title) for row in sibling_assessments):
        raise HTTPException(
            status_code=400,
            detail="An assessment with this title already exists in the selected category.",
        )

    try:
        inserted = _execute_returning_one(
            db,
            """
            insert into training_assessments (
                category_id,
                title,
                description,
                type,
                is_published,
                instant_feedback,
                sort_order,
                is_primary,
                active_status
            )
            values (
                :category_id,
                :title,
                :description,
                :assessment_type,
                :is_published,
                true,
                :sort_order,
                false,
                true
            )
            returning *
            """,
            {
                "category_id": category_id,
                "title": normalized_title,
                "description": description,
                "assessment_type": assessment_type,
                "is_published": is_published,
                "sort_order": _next_assessment_sort_order(db, category_id),
            },
        )
        db.commit()
        return _serialize_assessment(inserted)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to create the assessment definition.") from exc


def update_assessment_record(
    db: Session,
    current_user: User,
    assessment_id: str,
    *,
    title: str,
    description: str | None,
    assessment_type: str,
    is_published: bool,
) -> dict[str, Any]:
    assessment, category = _require_owned_assessment(db, current_user, assessment_id)
    normalized_title = _sanitize_text(title)
    if not normalized_title:
        raise HTTPException(status_code=400, detail="Assessment title is required.")

    sibling_assessments = _fetch_assessments_for_categories(db, [str(category["id"])], include_inactive=True)
    for sibling in sibling_assessments:
        if str(sibling["id"]) == assessment_id:
            continue
        if _normalize_value(sibling.get("title")) == _normalize_value(normalized_title):
            raise HTTPException(
                status_code=400,
                detail="Another assessment in this category already uses that title.",
            )

    try:
        if _truthy(assessment.get("is_primary"), False):
            db.execute(
                text(
                    """
                    update training_assessment_categories
                    set title = :title,
                        description = :description,
                        updated_at = timezone('utc', now())
                    where id = :category_id
                    """
                ),
                {
                    "category_id": category["id"],
                    "title": normalized_title,
                    "description": description,
                },
            )

        updated = _execute_returning_one(
            db,
            """
            update training_assessments
            set title = :title,
                description = :description,
                type = :assessment_type,
                is_published = :is_published,
                active_status = true,
                updated_at = timezone('utc', now())
            where id = :assessment_id
            returning *
            """,
            {
                "assessment_id": assessment_id,
                "title": normalized_title,
                "description": description,
                "assessment_type": assessment_type,
                "is_published": is_published,
            },
        )
        db.commit()
        return _serialize_assessment(updated)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to update the assessment definition.") from exc


def delete_assessment_record(
    db: Session,
    current_user: User,
    assessment_id: str,
) -> None:
    assessment, category = _require_owned_assessment(db, current_user, assessment_id)
    _ensure_assignment_workspace_schema(db)
    category_id = str(category["id"])

    active_assignments = _fetch_rows(
        db,
        """
        select id
        from training_assessment_assignments
        where assessment_id = :assessment_id
          and coalesce(is_active, true) = true
        """,
        {"assessment_id": assessment_id},
    )
    if active_assignments:
        raise HTTPException(
            status_code=400,
            detail="Remove or deactivate assessment assignments before deleting this assessment.",
        )

    attempt_history = _fetch_rows(
        db,
        """
        select id
        from training_assessment_attempts
        where assessment_id = :assessment_id
        limit 1
        """,
        {"assessment_id": assessment_id},
    )
    if attempt_history:
        raise HTTPException(
            status_code=400,
            detail="This assessment already has recorded trainee attempts and can no longer be deleted.",
        )

    category_assessments = _fetch_assessments_for_categories(db, [category_id], include_inactive=True)
    if _truthy(assessment.get("is_primary"), False) and len(category_assessments) <= 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Each category requires at least one assessment definition. "
                "Create another assessment first or archive the category instead."
            ),
        )

    try:
        assignment_rows = _fetch_rows(
            db,
            """
            select id
            from training_assessment_assignments
            where assessment_id = :assessment_id
            """,
            {"assessment_id": assessment_id},
        )
        assignment_ids = [str(row["id"]) for row in assignment_rows]
        if assignment_ids:
            db.execute(
                text(
                    """
                    delete from training_assessment_assignment_questions
                    where assignment_id in :assignment_ids
                    """
                ).bindparams(bindparam("assignment_ids", expanding=True)),
                {"assignment_ids": assignment_ids},
            )
            db.execute(
                text(
                    """
                    delete from training_assessment_assignments
                    where id in :assignment_ids
                    """
                ).bindparams(bindparam("assignment_ids", expanding=True)),
                {"assignment_ids": assignment_ids},
            )

        db.execute(
            text(
                """
                delete from training_assessment_questions
                where assessment_id = :assessment_id
                """
            ),
            {"assessment_id": assessment_id},
        )
        db.execute(
            text(
                """
                delete from training_assessments
                where id = :assessment_id
                """
            ),
            {"assessment_id": assessment_id},
        )

        if _truthy(assessment.get("is_primary"), False):
            fallback = _fetch_one(
                db,
                """
                select *
                from training_assessments
                where category_id = :category_id
                order by coalesce(sort_order, 0) asc, created_at asc
                limit 1
                """,
                {"category_id": category_id},
            )
            if fallback:
                db.execute(
                    text(
                        """
                        update training_assessments
                        set title = :title,
                            description = :description,
                            is_primary = true,
                            active_status = true,
                            updated_at = timezone('utc', now())
                        where id = :assessment_id
                        """
                    ),
                    {
                        "assessment_id": fallback["id"],
                        "title": category.get("title") or "Assessment",
                        "description": category.get("description"),
                    },
                )

        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to delete the assessment definition.") from exc


def create_question_record(
    db: Session,
    current_user: User,
    *,
    category_id: str | None,
    assessment_id: str | None,
    question_number: int | None,
    question_text: str,
    question_type: str,
    options: Sequence[str],
    correct_answer: str,
    difficulty: str | None,
    explanation: str | None,
    points: int | None,
    order_index: int | None,
) -> dict[str, Any]:
    category, assessment = _resolve_question_scope(
        db,
        current_user,
        category_id=category_id,
        assessment_id=assessment_id,
    )
    normalized_question_text = _sanitize_text(question_text)
    if not normalized_question_text:
        raise HTTPException(status_code=400, detail="Question text is required.")

    next_question_number = question_number or _next_question_number(db, str(category["id"]))
    if next_question_number < 1:
        raise HTTPException(status_code=400, detail="Question Number must be a positive whole number.")

    _validate_question_number_availability(db, str(category["id"]), next_question_number)
    _validate_question_text_availability(db, str(category["id"]), normalized_question_text)

    validated_options, validated_correct_answer = (
        _build_choice_validation(options, correct_answer)
        if question_type == "multiple_choice"
        else ([], _sanitize_text(correct_answer))
    )
    if question_type != "multiple_choice" and not validated_correct_answer:
        raise HTTPException(status_code=400, detail="Correct answer is required.")

    point_value = max(_normalize_int(points, fallback=1), 1)
    next_order_index = order_index if order_index is not None and order_index >= 0 else max(next_question_number - 1, 0)

    try:
        inserted = _execute_returning_one(
            db,
            """
            insert into training_assessment_questions (
                assessment_id,
                category_id,
                question_number,
                question_text,
                question_type,
                options,
                correct_answer,
                difficulty,
                explanation,
                order_index,
                active_status,
                created_by,
                metadata
            )
            values (
                :assessment_id,
                :category_id,
                :question_number,
                :question_text,
                :question_type,
                cast(:options as jsonb),
                :correct_answer,
                :difficulty,
                :explanation,
                :order_index,
                true,
                :created_by,
                cast(:metadata as jsonb)
            )
            returning *
            """,
            {
                "assessment_id": assessment["id"],
                "category_id": category["id"],
                "question_number": next_question_number,
                "question_text": normalized_question_text,
                "question_type": question_type,
                "options": json.dumps(validated_options),
                "correct_answer": validated_correct_answer,
                "difficulty": difficulty,
                "explanation": explanation,
                "order_index": next_order_index,
                "created_by": current_user.id,
                "metadata": json.dumps({"points": point_value}),
            },
        )
        db.commit()
        return _question_record(inserted, category.get("title"), assessment.get("title"), None)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to create the assessment question.") from exc


def update_question_record(
    db: Session,
    current_user: User,
    question_id: str,
    *,
    category_id: str | None,
    assessment_id: str | None,
    question_number: int | None,
    question_text: str,
    question_type: str,
    options: Sequence[str],
    correct_answer: str,
    difficulty: str | None,
    explanation: str | None,
    points: int | None,
    order_index: int | None,
) -> dict[str, Any]:
    current_question, _, _ = _require_owned_question(db, current_user, question_id)
    category, assessment = _resolve_question_scope(
        db,
        current_user,
        category_id=category_id or str(current_question["category_id"]),
        assessment_id=assessment_id,
    )
    normalized_question_text = _sanitize_text(question_text)
    if not normalized_question_text:
        raise HTTPException(status_code=400, detail="Question text is required.")

    next_question_number = question_number or _normalize_int(current_question.get("question_number"), fallback=1)
    if next_question_number < 1:
        raise HTTPException(status_code=400, detail="Question Number must be a positive whole number.")

    _validate_question_number_availability(
        db,
        str(category["id"]),
        next_question_number,
        exclude_question_id=question_id,
    )
    _validate_question_text_availability(
        db,
        str(category["id"]),
        normalized_question_text,
        exclude_question_id=question_id,
    )

    validated_options, validated_correct_answer = (
        _build_choice_validation(options, correct_answer)
        if question_type == "multiple_choice"
        else ([], _sanitize_text(correct_answer))
    )
    if question_type != "multiple_choice" and not validated_correct_answer:
        raise HTTPException(status_code=400, detail="Correct answer is required.")

    existing_metadata = current_question.get("metadata") if isinstance(current_question.get("metadata"), dict) else {}
    point_value = max(_normalize_int(points, fallback=_question_point_value(existing_metadata)), 1)
    next_order_index = (
        order_index
        if order_index is not None and order_index >= 0
        else _normalize_int(current_question.get("order_index"), fallback=max(next_question_number - 1, 0))
    )

    try:
        updated = _execute_returning_one(
            db,
            """
            update training_assessment_questions
            set assessment_id = :assessment_id,
                category_id = :category_id,
                question_number = :question_number,
                question_text = :question_text,
                question_type = :question_type,
                options = cast(:options as jsonb),
                correct_answer = :correct_answer,
                difficulty = :difficulty,
                explanation = :explanation,
                order_index = :order_index,
                metadata = cast(:metadata as jsonb),
                active_status = true,
                updated_at = timezone('utc', now())
            where id = :question_id
            returning *
            """,
            {
                "question_id": question_id,
                "assessment_id": assessment["id"],
                "category_id": category["id"],
                "question_number": next_question_number,
                "question_text": normalized_question_text,
                "question_type": question_type,
                "options": json.dumps(validated_options),
                "correct_answer": validated_correct_answer,
                "difficulty": difficulty,
                "explanation": explanation,
                "order_index": next_order_index,
                "metadata": json.dumps({**existing_metadata, "points": point_value}),
            },
        )
        db.commit()
        return _question_record(updated, category.get("title"), assessment.get("title"), None)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to update the assessment question.") from exc


def delete_question_record(
    db: Session,
    current_user: User,
    question_id: str,
) -> None:
    _require_owned_question(db, current_user, question_id)
    try:
        assignment_rows = _fetch_rows(
            db,
            """
            select assignment_id
            from training_assessment_assignment_questions
            where question_id = :question_id
            """,
            {"question_id": question_id},
        )
        assignment_ids = [str(row["assignment_id"]) for row in assignment_rows]
        db.execute(
            text(
                """
                delete from training_assessment_assignment_questions
                where question_id = :question_id
                """
            ),
            {"question_id": question_id},
        )
        _refresh_selected_assignment_counts(db, assignment_ids)
        db.execute(
            text(
                """
                delete from training_assessment_questions
                where id = :question_id
                """
            ),
            {"question_id": question_id},
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to delete the assessment question.") from exc


def create_assignment_record(
    db: Session,
    current_user: User,
    *,
    category_id: str,
    assessment_id: str | None,
    target_type: str,
    batch_id: str | None,
    wave_number: int | None,
    trainee_id: str | None,
    due_at: str | None,
    title: str,
    description: str | None,
    assignment_mode: str,
    question_ids: Sequence[str],
    random_question_count: int | None,
    passing_score: int,
    maximum_attempts: int | None,
    time_limit_minutes: int | None,
    shuffle_choices: bool,
    shuffle_questions: bool,
) -> dict[str, Any]:
    _ensure_assignment_workspace_schema(db)
    category = _require_owned_category(db, current_user, category_id)
    assessment = None
    if assessment_id:
        assessment, assessment_category = _require_owned_assessment(db, current_user, assessment_id)
        if str(assessment_category["id"]) != category_id:
            raise HTTPException(
                status_code=400,
                detail="The selected assessment does not belong to the selected category.",
            )

    batch_rows, membership_rows, trainee_rows = _fetch_batch_context(db, current_user)
    batch_options = [
        {
            "id": batch["id"],
            "name": batch["name"],
            "waveNumber": batch.get("wave_number"),
        }
        for batch in batch_rows
    ]
    trainee_options = [
        {
            "id": trainee["id"],
            "email": trainee.get("email"),
        }
        for trainee in trainee_rows
    ]
    target = _resolve_assignment_target(
        target_type=target_type,
        batch_id=batch_id,
        wave_number=wave_number,
        trainee_id=trainee_id,
        batch_options=batch_options,
        trainee_options=trainee_options,
    )

    selected_question_ids = [question_id for question_id in dict.fromkeys(question_ids or []) if question_id]
    category_questions = [
        question
        for question in _fetch_questions_for_categories(db, [category_id])
        if _truthy(question.get("active_status"), True)
        and (not assessment_id or str(question.get("assessment_id")) == assessment_id)
    ]
    question_pool_ids = {str(question["id"]) for question in category_questions}
    if assignment_mode == "selected_questions" and not selected_question_ids:
        raise HTTPException(
            status_code=400,
            detail="Select at least one question when using selected question mode.",
        )
    if any(question_id not in question_pool_ids for question_id in selected_question_ids):
        raise HTTPException(
            status_code=400,
            detail="One or more selected questions do not belong to the chosen category.",
        )

    if assignment_mode == "random_subset":
        available_pool_size = len(selected_question_ids) or len(question_pool_ids)
        if not random_question_count or random_question_count < 1:
            raise HTTPException(
                status_code=400,
                detail="Provide how many questions should be drawn in random subset mode.",
            )
        if random_question_count > available_pool_size:
            raise HTTPException(
                status_code=400,
                detail="Random subset count cannot exceed the available question pool.",
            )

    normalized_title = _sanitize_text(title)
    if not normalized_title:
        raise HTTPException(status_code=400, detail="Assignment title is required.")

    _ensure_assignment_title_available(
        db,
        category_id=category_id,
        assessment_id=assessment_id,
        title=normalized_title,
        target_type=target["targetType"],
        batch_id=target["batchId"],
        wave_number=target["waveNumber"],
        trainee_id=target["traineeId"],
    )

    question_count = (
        random_question_count
        if assignment_mode == "random_subset"
        else len(selected_question_ids) if assignment_mode == "selected_questions"
        else len(category_questions)
    )

    try:
        inserted = _execute_returning_one(
            db,
            """
            insert into training_assessment_assignments (
                category_id,
                assessment_id,
                batch_id,
                trainee_id,
                assigned_by,
                due_at,
                is_active,
                title,
                description,
                assignment_mode,
                question_count,
                passing_score,
                maximum_attempts,
                time_limit_minutes,
                shuffle_choices,
                shuffle_questions,
                target_scope,
                wave_number
            )
            values (
                :category_id,
                :assessment_id,
                :batch_id,
                :trainee_id,
                :assigned_by,
                :due_at,
                true,
                :title,
                :description,
                :assignment_mode,
                :question_count,
                :passing_score,
                :maximum_attempts,
                :time_limit_minutes,
                :shuffle_choices,
                :shuffle_questions,
                :target_scope,
                :wave_number
            )
            returning *
            """,
            {
                "category_id": category_id,
                "assessment_id": assessment_id,
                "batch_id": target["batchId"],
                "trainee_id": target["traineeId"],
                "assigned_by": current_user.id,
                "due_at": due_at,
                "title": normalized_title,
                "description": description,
                "assignment_mode": assignment_mode,
                "question_count": question_count,
                "passing_score": passing_score,
                "maximum_attempts": maximum_attempts,
                "time_limit_minutes": time_limit_minutes,
                "shuffle_choices": shuffle_choices,
                "shuffle_questions": shuffle_questions,
                "target_scope": target["targetType"],
                "wave_number": target["waveNumber"],
            },
        )

        if selected_question_ids:
            db.execute(
                text(
                    """
                    insert into training_assessment_assignment_questions (
                        assignment_id,
                        question_id,
                        question_order
                    )
                    values (
                        :assignment_id,
                        :question_id,
                        :question_order
                    )
                    """
                ),
                [
                    {
                        "assignment_id": inserted["id"],
                        "question_id": question_id,
                        "question_order": index,
                    }
                    for index, question_id in enumerate(selected_question_ids)
                ],
            )

        db.commit()
        return _serialize_assignment(inserted)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to create the assessment assignment.") from exc


def update_assignment_record(
    db: Session,
    current_user: User,
    assignment_id: str,
    *,
    category_id: str,
    assessment_id: str | None,
    target_type: str,
    batch_id: str | None,
    wave_number: int | None,
    trainee_id: str | None,
    due_at: str | None,
    title: str,
    description: str | None,
    assignment_mode: str,
    question_ids: Sequence[str],
    random_question_count: int | None,
    passing_score: int,
    maximum_attempts: int | None,
    time_limit_minutes: int | None,
    shuffle_choices: bool,
    shuffle_questions: bool,
) -> dict[str, Any]:
    _ensure_assignment_workspace_schema(db)
    existing, _ = _require_owned_assignment(db, current_user, assignment_id)
    category = _require_owned_category(db, current_user, category_id)
    if assessment_id:
        assessment, assessment_category = _require_owned_assessment(db, current_user, assessment_id)
        if str(assessment_category["id"]) != category_id:
            raise HTTPException(
                status_code=400,
                detail="The selected assessment does not belong to the selected category.",
            )

    batch_rows, _, trainee_rows = _fetch_batch_context(db, current_user)
    batch_options = [
        {
            "id": batch["id"],
            "name": batch["name"],
            "waveNumber": batch.get("wave_number"),
        }
        for batch in batch_rows
    ]
    trainee_options = [
        {
            "id": trainee["id"],
            "email": trainee.get("email"),
        }
        for trainee in trainee_rows
    ]
    target = _resolve_assignment_target(
        target_type=target_type,
        batch_id=batch_id,
        wave_number=wave_number,
        trainee_id=trainee_id,
        batch_options=batch_options,
        trainee_options=trainee_options,
    )

    normalized_title = _sanitize_text(title)
    if not normalized_title:
        raise HTTPException(status_code=400, detail="Assignment title is required.")

    selected_question_ids = [question_id for question_id in dict.fromkeys(question_ids or []) if question_id]
    category_questions = [
        question
        for question in _fetch_questions_for_categories(db, [category_id])
        if _truthy(question.get("active_status"), True)
        and (not assessment_id or str(question.get("assessment_id")) == assessment_id)
    ]
    question_pool_ids = {str(question["id"]) for question in category_questions}
    if assignment_mode == "selected_questions" and not selected_question_ids:
        raise HTTPException(status_code=400, detail="Select at least one question for selected question mode.")
    if any(question_id not in question_pool_ids for question_id in selected_question_ids):
        raise HTTPException(
            status_code=400,
            detail="One or more selected questions do not belong to the chosen category.",
        )
    if assignment_mode == "random_subset":
        available_pool_size = len(selected_question_ids) or len(question_pool_ids)
        if not random_question_count or random_question_count < 1:
            raise HTTPException(status_code=400, detail="Provide a valid random subset count.")
        if random_question_count > available_pool_size:
            raise HTTPException(
                status_code=400,
                detail="Random subset count cannot exceed the available question pool.",
            )

    _ensure_assignment_title_available(
        db,
        category_id=category_id,
        assessment_id=assessment_id,
        title=normalized_title,
        target_type=target["targetType"],
        batch_id=target["batchId"],
        wave_number=target["waveNumber"],
        trainee_id=target["traineeId"],
        exclude_assignment_id=assignment_id,
    )

    question_count = (
        random_question_count
        if assignment_mode == "random_subset"
        else len(selected_question_ids) if assignment_mode == "selected_questions"
        else len(category_questions)
    )

    try:
        updated = _execute_returning_one(
            db,
            """
            update training_assessment_assignments
            set category_id = :category_id,
                assessment_id = :assessment_id,
                batch_id = :batch_id,
                trainee_id = :trainee_id,
                due_at = :due_at,
                title = :title,
                description = :description,
                assignment_mode = :assignment_mode,
                question_count = :question_count,
                passing_score = :passing_score,
                maximum_attempts = :maximum_attempts,
                time_limit_minutes = :time_limit_minutes,
                shuffle_choices = :shuffle_choices,
                shuffle_questions = :shuffle_questions,
                target_scope = :target_scope,
                wave_number = :wave_number,
                updated_at = timezone('utc', now())
            where id = :assignment_id
            returning *
            """,
            {
                "assignment_id": assignment_id,
                "category_id": category_id,
                "assessment_id": assessment_id,
                "batch_id": target["batchId"],
                "trainee_id": target["traineeId"],
                "due_at": due_at,
                "title": normalized_title,
                "description": description,
                "assignment_mode": assignment_mode,
                "question_count": question_count,
                "passing_score": passing_score,
                "maximum_attempts": maximum_attempts,
                "time_limit_minutes": time_limit_minutes,
                "shuffle_choices": shuffle_choices,
                "shuffle_questions": shuffle_questions,
                "target_scope": target["targetType"],
                "wave_number": target["waveNumber"],
            },
        )

        db.execute(
            text(
                """
                delete from training_assessment_assignment_questions
                where assignment_id = :assignment_id
                """
            ),
            {"assignment_id": assignment_id},
        )

        if selected_question_ids:
            db.execute(
                text(
                    """
                    insert into training_assessment_assignment_questions (
                        assignment_id,
                        question_id,
                        question_order
                    )
                    values (
                        :assignment_id,
                        :question_id,
                        :question_order
                    )
                    """
                ),
                [
                    {
                        "assignment_id": assignment_id,
                        "question_id": question_id,
                        "question_order": index,
                    }
                    for index, question_id in enumerate(selected_question_ids)
                ],
            )

        db.commit()
        return _serialize_assignment(updated)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to update the assessment assignment.") from exc


def delete_assignment_record(
    db: Session,
    current_user: User,
    assignment_id: str,
) -> None:
    _require_owned_assignment(db, current_user, assignment_id)
    try:
        db.execute(
            text(
                """
                update training_assessment_assignments
                set is_active = false,
                    updated_at = timezone('utc', now())
                where id = :assignment_id
                """
            ),
            {"assignment_id": assignment_id},
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to deactivate the assessment assignment.") from exc


def get_trainer_workspace_bootstrap(db: Session, current_user: User) -> dict[str, Any]:
    _ensure_assignment_workspace_schema(db)
    categories_raw = _fetch_visible_categories(db, current_user)
    category_ids = [str(category["id"]) for category in categories_raw]
    assessments_raw = _fetch_assessments_for_categories(db, category_ids)
    questions_raw = _fetch_questions_for_categories(db, category_ids)
    question_reports_raw = _fetch_question_reports(db, category_ids)
    batch_rows, membership_rows, trainee_rows = _fetch_batch_context(db, current_user)
    assignment_rows = _fetch_assignments(db, category_ids)
    assignment_ids = [str(row["id"]) for row in assignment_rows]
    assignment_question_rows = _fetch_assignment_question_rows(db, assignment_ids)
    attempt_rows = _fetch_attempts(db, category_ids)
    certificate_rows = _fetch_certificates(db, category_ids)

    batch_options: list[dict[str, Any]] = []
    trainee_options: list[dict[str, Any]] = []
    trainee_row_map = {str(row["id"]): row for row in trainee_rows}

    for batch in batch_rows:
        trainee_count = sum(
            1
            for membership in membership_rows
            if membership["batch_id"] == batch["id"] and membership["user_id"] in trainee_row_map
        )
        batch_options.append(
            {
                "id": batch["id"],
                "name": batch["name"],
                "description": batch.get("description"),
                "waveNumber": batch.get("wave_number"),
                "traineeCount": trainee_count,
                "createdBy": batch.get("created_by"),
            }
        )

    batch_option_map = {batch["id"]: batch for batch in batch_options}
    memberships_by_trainee: dict[str, list[str]] = defaultdict(list)
    for membership in membership_rows:
        if membership.get("user_id"):
            memberships_by_trainee[membership["user_id"]].append(membership["batch_id"])

    for trainee in trainee_rows:
        trainee_batch_ids = memberships_by_trainee.get(trainee["id"], [])
        trainee_batches = [batch_option_map[batch_id] for batch_id in trainee_batch_ids if batch_id in batch_option_map]
        trainee_options.append(
            {
                "id": trainee["id"],
                "fullName": trainee.get("full_name") or trainee.get("email") or "Trainee",
                "email": trainee.get("email") or "",
                "batchIds": [batch["id"] for batch in trainee_batches],
                "batchNames": [_format_batch_label(batch) for batch in trainee_batches],
            }
        )

    wave_map: dict[int, dict[str, Any]] = {}
    for batch in batch_options:
        wave_number = batch.get("waveNumber")
        if wave_number is None:
            continue
        current = wave_map.setdefault(
            int(wave_number),
            {"waveNumber": int(wave_number), "label": f"Wave {int(wave_number)}", "batchCount": 0, "traineeIds": set()},
        )
        current["batchCount"] += 1
        for membership in membership_rows:
            if membership["batch_id"] == batch["id"]:
                current["traineeIds"].add(membership["user_id"])

    waves = [
        {
            "waveNumber": wave_number,
            "label": value["label"],
            "batchCount": value["batchCount"],
            "traineeCount": len(value["traineeIds"]),
        }
        for wave_number, value in sorted(wave_map.items(), key=lambda item: item[0])
    ]

    category_title_map = {str(category["id"]): category.get("title") or "Assessment Category" for category in categories_raw}
    assessment_title_map = {str(assessment["id"]): assessment.get("title") or "Assessment" for assessment in assessments_raw}
    question_report_map: dict[str, dict[str, Any]] = {}
    for report in question_reports_raw:
        question_report_map[str(report["question_id"])] = {
            "answerCount": _normalize_int(report.get("answer_count")),
            "correctCount": _normalize_int(report.get("correct_count")),
            "incorrectCount": _normalize_int(report.get("incorrect_count")),
            "missRate": _normalize_float(report.get("miss_rate")),
        }

    question_records = [
        _question_record(
            question,
            category_title_map.get(str(question["category_id"])),
            assessment_title_map.get(str(question["assessment_id"])),
            question_report_map.get(str(question["id"])),
        )
        for question in questions_raw
    ]
    questions_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    questions_by_assessment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    question_record_map = {question["id"]: question for question in question_records}
    for question in question_records:
        questions_by_category[question["categoryId"]].append(question)
        questions_by_assessment[question["assessmentId"]].append(question)

    assessment_records = [
        _assessment_record(assessment, questions_by_assessment.get(str(assessment["id"]), []))
        for assessment in assessments_raw
    ]
    assessments_by_id = {assessment["id"]: assessment for assessment in assessment_records}
    assessments_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assessment in assessment_records:
        assessments_by_category[assessment["categoryId"]].append(assessment)

    categories = []
    for category in categories_raw:
        category_id = str(category["id"])
        categories.append(
            {
                "id": category_id,
                "title": category.get("title") or "Assessment Category",
                "categoryName": category.get("title") or "Assessment Category",
                "description": category.get("description"),
                "passingScore": _normalize_int(category.get("passing_score"), fallback=DEFAULT_PASSING_SCORE),
                "createdBy": category.get("created_by") or "",
                "trainerId": category.get("created_by") or "",
                "activeStatus": _truthy(category.get("active_status"), True),
                "isArchived": _truthy(category.get("is_archived"), False),
                "createdAt": _normalize_datetime(category.get("created_at")),
                "updatedAt": _normalize_datetime(category.get("updated_at")),
                "questionCount": len(questions_by_category.get(category_id, [])),
                "assignmentCount": 0,
                "activeAssignmentCount": 0,
                "attemptCount": 0,
                "passRate": 0.0,
                "averageScore": 0.0,
                "completionRate": 0.0,
                "retakeRate": 0.0,
                "highestScore": 0.0,
                "lowestScore": 0.0,
                "assessments": assessments_by_category.get(category_id, []),
            }
        )
    categories_by_id = {category["id"]: category for category in categories}

    attempts = [_attempt_record(attempt) for attempt in attempt_rows]
    latest_attempts_by_assignment: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for attempt in attempts:
        assignment_id = attempt.get("assignmentId")
        trainee_id = attempt.get("traineeId")
        if not assignment_id or not trainee_id:
            continue
        current = latest_attempts_by_assignment[assignment_id].get(trainee_id)
        if not current:
            latest_attempts_by_assignment[assignment_id][trainee_id] = attempt

    selected_question_ids_by_assignment: dict[str, list[str]] = defaultdict(list)
    for row in assignment_question_rows:
        selected_question_ids_by_assignment[str(row["assignment_id"])].append(str(row["question_id"]))

    trainee_option_map = {trainee["id"]: trainee for trainee in trainee_options}
    assignments: list[dict[str, Any]] = []
    for assignment in assignment_rows:
        assignment_id = str(assignment["id"])
        category_id = str(assignment["category_id"])
        assessment_id = str(assignment["assessment_id"]) if assignment.get("assessment_id") else None
        batch_id = assignment.get("batch_id")
        trainee_id = assignment.get("trainee_id")
        batch = batch_option_map.get(batch_id) if batch_id else None
        target_type = assignment.get("target_scope") or ("trainee" if trainee_id else "batch")
        assignment_wave_number = (
            _normalize_int(assignment.get("wave_number"))
            if assignment.get("wave_number") is not None
            else batch.get("waveNumber") if batch else None
        )
        wave_batch_ids = [
            candidate["id"]
            for candidate in batch_options
            if assignment_wave_number is not None and candidate.get("waveNumber") == assignment_wave_number
        ]
        target_label = (
            trainee_option_map.get(trainee_id, {}).get("fullName")
            if target_type == "trainee"
            else f"Wave {assignment_wave_number}" if target_type == "wave" and assignment_wave_number is not None
            else _format_batch_label(batch) if batch else "Assessment Assignment"
        )
        if target_type == "trainee":
            assigned_trainees = 1
        elif target_type == "wave" and assignment_wave_number is not None:
            assigned_trainees = len(
                {
                    row["user_id"]
                    for row in membership_rows
                    if row["batch_id"] in wave_batch_ids and row["user_id"] in trainee_row_map
                }
            )
        else:
            assigned_trainees = len(
                [
                    row
                    for row in membership_rows
                    if batch_id and row["batch_id"] == batch_id and row["user_id"] in trainee_row_map
                ]
            )
        latest_attempts = list(latest_attempts_by_assignment.get(assignment_id, {}).values())
        passed_trainees = sum(1 for attempt in latest_attempts if attempt["status"] == "pass")
        failed_trainees = sum(1 for attempt in latest_attempts if attempt["status"] == "fail")
        completed_trainees = len(latest_attempts)
        average_score = round(
            sum(float(attempt["score"]) for attempt in latest_attempts) / completed_trainees,
            2,
        ) if completed_trainees else 0.0
        highest_score = max((float(attempt["score"]) for attempt in latest_attempts), default=0.0)
        lowest_score = min((float(attempt["score"]) for attempt in latest_attempts), default=0.0)
        retake_rate = round(
            (sum(1 for attempt in latest_attempts if _normalize_int(attempt.get("attemptNo"), fallback=1) > 1) / completed_trainees) * 100,
            2,
        ) if completed_trainees else 0.0
        selected_question_ids = selected_question_ids_by_assignment.get(assignment_id, [])
        question_count = _normalize_int(assignment.get("question_count"))
        if question_count <= 0:
            if assignment.get("assignment_mode") == "selected_questions" and selected_question_ids:
                question_count = len(selected_question_ids)
            elif assessment_id and assessments_by_id.get(assessment_id):
                question_count = assessments_by_id[assessment_id]["questionCount"]
            else:
                question_count = len(questions_by_category.get(category_id, []))
        if assigned_trainees <= 0:
            status_label = "Assigned"
        elif passed_trainees >= assigned_trainees:
            status_label = "Passed"
        elif failed_trainees >= assigned_trainees:
            status_label = "Failed"
        elif completed_trainees >= assigned_trainees:
            status_label = "Completed"
        elif completed_trainees > 0:
            status_label = "In Progress"
        else:
            status_label = "Assigned"

        assignments.append(
            {
                "id": assignment_id,
                "categoryId": category_id,
                "assessmentId": assessment_id,
                "batchId": batch_id,
                "traineeId": trainee_id,
                "assignedBy": assignment.get("assigned_by") or "",
                "assignedAt": _normalize_datetime(assignment.get("assigned_at")),
                "dueAt": _normalize_datetime(assignment.get("due_at")),
                "isActive": _truthy(assignment.get("is_active"), True),
                "categoryTitle": categories_by_id.get(category_id, {}).get("title", "Assessment Category"),
                "categoryName": categories_by_id.get(category_id, {}).get("title", "Assessment Category"),
                "assessmentTitle": assessments_by_id.get(assessment_id, {}).get("title") if assessment_id else None,
                "title": assignment.get("title") or assessments_by_id.get(assessment_id, {}).get("title") or categories_by_id.get(category_id, {}).get("title", "Assessment"),
                "description": assignment.get("description"),
                "targetLabel": target_label or "Assessment Assignment",
                "targetType": target_type,
                "waveNumber": assignment_wave_number,
                "assignmentMode": assignment.get("assignment_mode") or "entire_category",
                "questionCount": question_count,
                "randomQuestionCount": question_count if assignment.get("assignment_mode") == "random_subset" else None,
                "passingScore": _normalize_int(assignment.get("passing_score"), fallback=categories_by_id.get(category_id, {}).get("passingScore", DEFAULT_PASSING_SCORE)),
                "maximumAttempts": _normalize_int(assignment.get("maximum_attempts")) if assignment.get("maximum_attempts") is not None else None,
                "timeLimitMinutes": _normalize_int(assignment.get("time_limit_minutes")) if assignment.get("time_limit_minutes") is not None else None,
                "shuffleChoices": _truthy(assignment.get("shuffle_choices"), True),
                "shuffleQuestions": _truthy(assignment.get("shuffle_questions"), False),
                "selectedQuestionIds": selected_question_ids,
                "assignedTrainees": assigned_trainees,
                "completedTrainees": completed_trainees,
                "passedTrainees": passed_trainees,
                "failedTrainees": failed_trainees,
                "certificateCount": sum(1 for certificate in certificate_rows if str(certificate.get("assignment_id")) == assignment_id),
                "averageScore": average_score,
                "highestScore": highest_score,
                "lowestScore": lowest_score,
                "retakeRate": retake_rate,
                "statusLabel": status_label,
            }
        )

    assignments_by_id = {assignment["id"]: assignment for assignment in assignments}
    certificates = [
        _certificate_record(certificate, categories_by_id, assignments_by_id, assessments_by_id)
        for certificate in certificate_rows
    ]

    for category in categories:
        category_id = category["id"]
        category_assignments = [assignment for assignment in assignments if assignment["categoryId"] == category_id]
        category_attempts = [attempt for attempt in attempts if attempt["categoryId"] == category_id]
        attempt_count = len(category_attempts)
        pass_count = sum(1 for attempt in category_attempts if attempt["status"] == "pass")
        assigned_total = sum(_normalize_int(assignment.get("assignedTrainees")) for assignment in category_assignments)
        completed_total = sum(_normalize_int(assignment.get("completedTrainees")) for assignment in category_assignments)

        category["assignmentCount"] = len(category_assignments)
        category["activeAssignmentCount"] = sum(1 for assignment in category_assignments if assignment["isActive"])
        category["attemptCount"] = attempt_count
        category["passRate"] = round((pass_count / attempt_count) * 100, 2) if attempt_count else 0.0
        category["averageScore"] = round(sum(float(attempt["score"]) for attempt in category_attempts) / attempt_count, 2) if attempt_count else 0.0
        category["completionRate"] = round((completed_total / assigned_total) * 100, 2) if assigned_total else 0.0
        category["retakeRate"] = round(
            (sum(1 for attempt in category_attempts if _normalize_int(attempt.get("attemptNo"), fallback=1) > 1) / attempt_count) * 100,
            2,
        ) if attempt_count else 0.0
        category["highestScore"] = max((float(attempt["score"]) for attempt in category_attempts), default=0.0)
        category["lowestScore"] = min((float(attempt["score"]) for attempt in category_attempts), default=0.0)

    question_reports = []
    for report in question_reports_raw:
        question_id = str(report["question_id"])
        question_reference = question_record_map.get(question_id)
        question_reports.append(
            {
                "questionId": question_id,
                "categoryId": str(report["category_id"]),
                "categoryTitle": report.get("category_title") or categories_by_id.get(str(report["category_id"]), {}).get("title"),
                "questionNumber": _normalize_int(report.get("question_number")),
                "questionText": report.get("question_text") or "",
                "questionType": (question_reference or {}).get("questionType") or "multiple_choice",
                "difficulty": report.get("difficulty") or (question_reference or {}).get("difficulty"),
                "answerCount": _normalize_int(report.get("answer_count")),
                "correctCount": _normalize_int(report.get("correct_count")),
                "incorrectCount": _normalize_int(report.get("incorrect_count")),
                "missRate": _normalize_float(report.get("miss_rate")),
            }
        )

    category_reports = []
    for category in categories:
        category_attempts = [attempt for attempt in attempts if attempt["categoryId"] == category["id"]]
        category_assignments = [assignment for assignment in assignments if assignment["categoryId"] == category["id"]]
        pass_count = sum(1 for attempt in category_attempts if attempt["status"] == "pass")
        fail_count = sum(1 for attempt in category_attempts if attempt["status"] == "fail")
        category_reports.append(
            {
                "categoryId": category["id"],
                "categoryTitle": category["title"],
                "passingScore": category["passingScore"],
                "questionCount": category["questionCount"],
                "assignmentCount": len(category_assignments),
                "assignedTraineeCount": sum(_normalize_int(assignment.get("assignedTrainees")) for assignment in category_assignments),
                "completedTraineeCount": sum(_normalize_int(assignment.get("completedTrainees")) for assignment in category_assignments),
                "attemptCount": len(category_attempts),
                "passCount": pass_count,
                "failCount": fail_count,
                "averageScore": category["averageScore"],
                "passRate": category["passRate"],
                "failRate": round((fail_count / len(category_attempts)) * 100, 2) if category_attempts else 0.0,
                "retakeRate": category["retakeRate"],
                "highestScore": category["highestScore"],
                "lowestScore": category["lowestScore"],
                "completionRate": category["completionRate"],
            }
        )

    total_attempts = len(attempts)
    total_passed = sum(1 for attempt in attempts if attempt["status"] == "pass")
    total_failed = sum(1 for attempt in attempts if attempt["status"] == "fail")
    retake_attempts = sum(1 for attempt in attempts if _normalize_int(attempt.get("attemptNo"), fallback=1) > 1)
    return {
        "categories": categories,
        "questions": question_records,
        "batches": batch_options,
        "waves": waves,
        "trainees": trainee_options,
        "assignments": assignments,
        "attempts": attempts,
        "certificates": certificates,
        "reports": {
            "categories": category_reports,
            "batches": [],
            "waves": [],
            "trainees": [],
            "trainers": [],
            "questions": question_reports,
        },
        "analytics": {
            "totalQuestions": len(question_records),
            "totalAssignments": len(assignments),
            "activeAssignments": sum(1 for assignment in assignments if assignment["isActive"]),
            "totalAttempts": total_attempts,
            "passRate": round((total_passed / total_attempts) * 100, 2) if total_attempts else 0.0,
            "failRate": round((total_failed / total_attempts) * 100, 2) if total_attempts else 0.0,
            "retakeRate": round((retake_attempts / total_attempts) * 100, 2) if total_attempts else 0.0,
            "averageScore": round(sum(float(attempt["score"]) for attempt in attempts) / total_attempts, 2) if total_attempts else 0.0,
            "highestScore": max((float(attempt["score"]) for attempt in attempts), default=0.0),
            "lowestScore": min((float(attempt["score"]) for attempt in attempts), default=0.0),
            "certificatesIssued": len(certificates),
        },
    }


def _fetch_categories_by_ids(db: Session, category_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not category_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_categories
        where id in :category_ids
          and coalesce(is_archived, false) = false
          and coalesce(active_status, true) = true
        order by updated_at desc nulls last, created_at desc nulls last
        """,
        {"category_ids": list(category_ids)},
        expanding=("category_ids",),
    )


def _fetch_trainee_batch_rows(db: Session, trainee_id: str) -> list[dict[str, Any]]:
    return _fetch_rows(
        db,
        """
        select
            b.id,
            b.name,
            b.description,
            b.wave_number,
            b.created_by
        from batch_user bu
        join batch b on b.id = bu.batch_id
        where bu.user_id = :trainee_id
          and coalesce(b.is_active, true) = true
        order by b.wave_number asc nulls last, b.name asc
        """,
        {"trainee_id": trainee_id},
    )


def _fetch_trainee_assignments(
    db: Session,
    *,
    trainee_id: str,
    batch_ids: Sequence[str],
    wave_numbers: Sequence[int],
) -> list[dict[str, Any]]:
    expanded_batch_ids = list(batch_ids) or ["__none__"]
    expanded_wave_numbers = list(wave_numbers) or [-1]
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_assignments
        where coalesce(is_active, true) = true
          and (
              trainee_id = :trainee_id
              or batch_id in :batch_ids
              or (
                  coalesce(target_scope, '') = 'wave'
                  and wave_number in :wave_numbers
              )
          )
        order by due_at asc nulls last, assigned_at desc nulls last
        """,
        {
            "trainee_id": trainee_id,
            "batch_ids": expanded_batch_ids,
            "wave_numbers": expanded_wave_numbers,
        },
        expanding=("batch_ids", "wave_numbers"),
    )


def _fetch_attempts_for_trainee(
    db: Session,
    *,
    trainee_id: str,
    assignment_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not assignment_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_attempt_feed
        where trainee_id = :trainee_id
          and assignment_id in :assignment_ids
        order by completed_at desc nulls last, submitted_at desc nulls last, attempt_no desc
        """,
        {
            "trainee_id": trainee_id,
            "assignment_ids": list(assignment_ids),
        },
        expanding=("assignment_ids",),
    )


def _fetch_certificates_for_trainee(
    db: Session,
    *,
    trainee_id: str,
    assignment_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not assignment_ids:
        return []
    return _fetch_rows(
        db,
        """
        select *
        from training_assessment_certificates
        where trainee_id = :trainee_id
          and assignment_id in :assignment_ids
        order by earned_at desc nulls last, created_at desc nulls last
        """,
        {
            "trainee_id": trainee_id,
            "assignment_ids": list(assignment_ids),
        },
        expanding=("assignment_ids",),
    )


def _build_question_lookup(
    questions_raw: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    question_map: dict[str, dict[str, Any]] = {}
    questions_by_assessment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    questions_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for question in questions_raw:
        if not _truthy(question.get("active_status"), True):
            continue
        question_id = str(question["id"])
        assessment_id = str(question["assessment_id"])
        category_id = str(question["category_id"])
        question_map[question_id] = question
        questions_by_assessment[assessment_id].append(question)
        questions_by_category[category_id].append(question)

    for question_list in questions_by_assessment.values():
        question_list.sort(key=lambda row: (_normalize_int(row.get("question_number")), _normalize_int(row.get("order_index"))))
    for question_list in questions_by_category.values():
        question_list.sort(key=lambda row: (_normalize_int(row.get("question_number")), _normalize_int(row.get("order_index"))))

    return question_map, questions_by_assessment, questions_by_category


def _resolve_trainee_target_label(
    assignment: dict[str, Any],
    batch_lookup: dict[str, dict[str, Any]],
) -> str:
    target_scope = assignment.get("target_scope") or ("trainee" if assignment.get("trainee_id") else "batch")
    if target_scope == "trainee":
        return "Direct Assignment"
    if target_scope == "wave" and assignment.get("wave_number") is not None:
        return f"Wave {assignment['wave_number']}"
    batch = batch_lookup.get(str(assignment.get("batch_id") or ""))
    if not batch:
        return "Batch Assignment"
    return _format_batch_label(
        {
            "id": batch["id"],
            "name": batch.get("name") or "Batch",
            "waveNumber": batch.get("wave_number"),
        }
    )


def _shuffle_question_choices(
    options: Sequence[str],
    *,
    shuffle_choices: bool,
) -> list[str]:
    normalized_options = [str(option) for option in options]
    if not shuffle_choices:
        return normalized_options
    shuffled = list(normalized_options)
    random.shuffle(shuffled)
    return shuffled


def _build_assignment_question_pool(
    assignment: dict[str, Any],
    *,
    question_map: dict[str, dict[str, Any]],
    questions_by_assessment: dict[str, list[dict[str, Any]]],
    questions_by_category: dict[str, list[dict[str, Any]]],
    selected_question_ids: Sequence[str],
    randomize_subset: bool,
) -> list[dict[str, Any]]:
    category_id = str(assignment["category_id"])
    assessment_id = str(assignment["assessment_id"]) if assignment.get("assessment_id") else None
    assignment_mode = assignment.get("assignment_mode") or "entire_category"

    if selected_question_ids:
        ordered_selected = [
            question_map[question_id]
            for question_id in selected_question_ids
            if question_id in question_map
        ]
    else:
        ordered_selected = []

    if ordered_selected:
        base_pool = ordered_selected
    elif assessment_id and assessment_id in questions_by_assessment:
        base_pool = list(questions_by_assessment[assessment_id])
    else:
        base_pool = list(questions_by_category.get(category_id, []))

    if not base_pool:
        return []

    prepared_pool = list(base_pool)
    if assignment_mode == "random_subset":
        subset_count = min(
            max(_normalize_int(assignment.get("question_count"), fallback=len(prepared_pool)), 1),
            len(prepared_pool),
        )
        if randomize_subset:
            prepared_pool = random.sample(prepared_pool, subset_count)
        else:
            prepared_pool = prepared_pool[:subset_count]
    elif _truthy(assignment.get("shuffle_questions"), False):
        prepared_pool = list(prepared_pool)
        random.shuffle(prepared_pool)

    return prepared_pool


def _serialize_trainee_session_question(
    question: dict[str, Any],
    *,
    question_number: int,
    shuffle_choices: bool,
) -> dict[str, Any]:
    metadata = question.get("metadata") if isinstance(question.get("metadata"), dict) else {}
    choices = question.get("options") if isinstance(question.get("options"), list) else []
    return {
        "id": str(question["id"]),
        "questionNumber": question_number,
        "questionText": question.get("question_text") or "",
        "questionType": question.get("question_type") or "multiple_choice",
        "difficulty": question.get("difficulty"),
        "choices": _shuffle_question_choices(choices, shuffle_choices=shuffle_choices),
        "pointValue": _question_point_value(metadata),
    }


def _build_trainee_status(
    latest_attempt: dict[str, Any] | None,
    attempt_count: int,
    maximum_attempts: int | None,
) -> dict[str, Any]:
    attempts_remaining = None if maximum_attempts is None else max(maximum_attempts - attempt_count, 0)
    passed = bool(latest_attempt and latest_attempt.get("status") == "pass")
    failed = bool(latest_attempt and latest_attempt.get("status") == "fail")
    can_retake = bool(failed and (maximum_attempts is None or attempt_count < maximum_attempts))
    can_start = bool(not passed and (attempt_count == 0 or can_retake))
    is_completed = attempt_count > 0

    if passed:
        status_label = "Passed"
    elif failed and not can_retake:
        status_label = "Failed"
    elif failed:
        status_label = "Completed"
    else:
        status_label = "Assigned"

    return {
        "attemptsRemaining": attempts_remaining,
        "canRetake": can_retake,
        "canStart": can_start,
        "isCompleted": is_completed,
        "statusLabel": status_label,
    }


def _require_trainee_assignment_access(
    db: Session,
    current_user: User,
    assignment_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _ensure_assignment_workspace_schema(db)
    batch_rows = _fetch_trainee_batch_rows(db, current_user.id)
    batch_ids = {str(batch["id"]) for batch in batch_rows}
    wave_numbers = {
        _normalize_int(batch.get("wave_number"))
        for batch in batch_rows
        if batch.get("wave_number") is not None
    }

    assignment = _fetch_one(
        db,
        """
        select *
        from training_assessment_assignments
        where id = :assignment_id
          and coalesce(is_active, true) = true
        """,
        {"assignment_id": assignment_id},
    )
    if not assignment:
        raise HTTPException(status_code=404, detail="Assigned assessment not found.")

    target_scope = assignment.get("target_scope") or ("trainee" if assignment.get("trainee_id") else "batch")
    allowed = False
    if target_scope == "trainee":
        allowed = str(assignment.get("trainee_id") or "") == current_user.id
    elif target_scope == "wave":
        assignment_wave = _normalize_int(assignment.get("wave_number"), fallback=-1)
        allowed = assignment_wave in wave_numbers
    else:
        allowed = str(assignment.get("batch_id") or "") in batch_ids

    if not allowed:
        raise HTTPException(status_code=403, detail="You do not have access to this assigned assessment.")

    return assignment, batch_rows


def _fetch_attempt_feed_row(db: Session, attempt_id: str) -> dict[str, Any]:
    attempt = _fetch_one(
        db,
        """
        select *
        from training_assessment_attempt_feed
        where id = :attempt_id
        """,
        {"attempt_id": attempt_id},
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Saved assessment attempt not found.")
    return attempt


def get_trainee_workspace_dashboard(db: Session, current_user: User) -> dict[str, Any]:
    _ensure_assignment_workspace_schema(db)

    batch_rows = _fetch_trainee_batch_rows(db, current_user.id)
    batch_lookup = {str(batch["id"]): batch for batch in batch_rows}
    batch_ids = [str(batch["id"]) for batch in batch_rows]
    wave_numbers = [
        _normalize_int(batch.get("wave_number"))
        for batch in batch_rows
        if batch.get("wave_number") is not None
    ]

    assignment_rows = _fetch_trainee_assignments(
        db,
        trainee_id=current_user.id,
        batch_ids=batch_ids,
        wave_numbers=wave_numbers,
    )
    assignment_ids = [str(row["id"]) for row in assignment_rows]
    category_ids = list({str(row["category_id"]) for row in assignment_rows if row.get("category_id")})

    categories_raw = _fetch_categories_by_ids(db, category_ids)
    assessments_raw = _fetch_assessments_for_categories(db, category_ids)
    questions_raw = _fetch_questions_for_categories(db, category_ids)
    assignment_question_rows = _fetch_assignment_question_rows(db, assignment_ids)
    attempt_rows = _fetch_attempts_for_trainee(
        db,
        trainee_id=current_user.id,
        assignment_ids=assignment_ids,
    )
    certificate_rows = _fetch_certificates_for_trainee(
        db,
        trainee_id=current_user.id,
        assignment_ids=assignment_ids,
    )

    categories_by_id = {str(category["id"]): category for category in categories_raw}
    assessments_by_id = {str(assessment["id"]): assessment for assessment in assessments_raw}
    question_map, questions_by_assessment, questions_by_category = _build_question_lookup(questions_raw)

    selected_question_ids_by_assignment: dict[str, list[str]] = defaultdict(list)
    for row in assignment_question_rows:
        selected_question_ids_by_assignment[str(row["assignment_id"])].append(str(row["question_id"]))

    attempts = [_attempt_record(row) for row in attempt_rows]
    attempts_by_assignment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        assignment_id = attempt.get("assignmentId")
        if assignment_id:
            attempts_by_assignment[assignment_id].append(attempt)

    certificates_by_assignment: dict[str, dict[str, Any]] = {}
    certificates = []
    assignments_by_id = {
        str(assignment["id"]): {
            "id": str(assignment["id"]),
            "title": assignment.get("title"),
        }
        for assignment in assignment_rows
    }
    for certificate in certificate_rows:
        certificate_record = _certificate_record(certificate, categories_by_id, assignments_by_id, assessments_by_id)
        certificates.append(certificate_record)
        assignment_id = certificate_record.get("assignmentId")
        if assignment_id and assignment_id not in certificates_by_assignment:
            certificates_by_assignment[assignment_id] = certificate_record

    available_assessments: list[dict[str, Any]] = []
    completed_count = 0
    passed_count = 0

    for assignment in assignment_rows:
        assignment_id = str(assignment["id"])
        category_id = str(assignment["category_id"])
        assessment_id = str(assignment["assessment_id"]) if assignment.get("assessment_id") else None
        selected_question_ids = selected_question_ids_by_assignment.get(assignment_id, [])
        question_pool = _build_assignment_question_pool(
            assignment,
            question_map=question_map,
            questions_by_assessment=questions_by_assessment,
            questions_by_category=questions_by_category,
            selected_question_ids=selected_question_ids,
            randomize_subset=False,
        )
        assignment_attempts = sorted(
            attempts_by_assignment.get(assignment_id, []),
            key=lambda row: (
                _normalize_int(row.get("attemptNo"), fallback=0),
                row.get("completedAt") or row.get("submittedAt") or "",
            ),
            reverse=True,
        )
        latest_attempt = assignment_attempts[0] if assignment_attempts else None
        maximum_attempts = _normalize_int(assignment.get("maximum_attempts")) if assignment.get("maximum_attempts") is not None else None
        status = _build_trainee_status(latest_attempt, len(assignment_attempts), maximum_attempts)

        if latest_attempt:
            completed_count += 1
            if latest_attempt["status"] == "pass":
                passed_count += 1

        available_assessments.append(
            {
                "assignmentId": assignment_id,
                "assessmentId": assessment_id or assignment_id,
                "categoryId": category_id,
                "categoryTitle": categories_by_id.get(category_id, {}).get("title", "Assessment Category"),
                "targetType": assignment.get("target_scope") or ("trainee" if assignment.get("trainee_id") else "batch"),
                "waveNumber": _normalize_int(assignment.get("wave_number")) if assignment.get("wave_number") is not None else None,
                "assignmentTitle": assignment.get("title") or assessments_by_id.get(assessment_id or "", {}).get("title"),
                "assessmentTitle": assessments_by_id.get(assessment_id or "", {}).get("title") or assignment.get("title") or categories_by_id.get(category_id, {}).get("title", "Assessment"),
                "assessmentDescription": assignment.get("description") or assessments_by_id.get(assessment_id or "", {}).get("description"),
                "type": assessments_by_id.get(assessment_id or "", {}).get("type") or "multiple_choice",
                "passingScore": _normalize_int(
                    assignment.get("passing_score"),
                    fallback=_normalize_int(categories_by_id.get(category_id, {}).get("passing_score"), fallback=DEFAULT_PASSING_SCORE),
                ),
                "targetDueAt": _normalize_datetime(assignment.get("due_at")),
                "targetLabel": _resolve_trainee_target_label(assignment, batch_lookup),
                "questionCount": len(question_pool),
                "questionTypes": sorted({question.get("question_type") or "multiple_choice" for question in question_pool}),
                "latestAttempt": latest_attempt,
                "attemptCount": len(assignment_attempts),
                "attemptsRemaining": status["attemptsRemaining"],
                "canStart": status["canStart"],
                "canRetake": status["canRetake"],
                "isCompleted": status["isCompleted"],
                "maximumAttempts": maximum_attempts,
                "timeLimitMinutes": _normalize_int(assignment.get("time_limit_minutes")) if assignment.get("time_limit_minutes") is not None else None,
                "certificate": certificates_by_assignment.get(assignment_id),
                "questions": [],
                "statusLabel": status["statusLabel"],
            }
        )

    average_score = round(
        sum(float(attempt["score"]) for attempt in attempts) / len(attempts),
        2,
    ) if attempts else 0.0

    return {
        "availableAssessments": available_assessments,
        "attempts": attempts,
        "coachingNotes": [],
        "certificates": certificates,
        "stats": {
            "assignedCount": len(available_assessments),
            "completedCount": completed_count,
            "passedCount": passed_count,
            "averageScore": average_score,
            "retakeCount": sum(1 for assessment in available_assessments if assessment["canRetake"]),
            "certificateCount": len(certificates),
        },
    }


def get_trainee_workspace_session(
    db: Session,
    current_user: User,
    assignment_id: str,
) -> dict[str, Any]:
    assignment, batch_rows = _require_trainee_assignment_access(db, current_user, assignment_id)
    category_id = str(assignment["category_id"])
    assessment_id = str(assignment["assessment_id"]) if assignment.get("assessment_id") else None

    categories_by_id = {str(row["id"]): row for row in _fetch_categories_by_ids(db, [category_id])}
    assessments_raw = _fetch_assessments_for_categories(db, [category_id])
    assessments_by_id = {str(row["id"]): row for row in assessments_raw}
    questions_raw = _fetch_questions_for_categories(db, [category_id])
    question_map, questions_by_assessment, questions_by_category = _build_question_lookup(questions_raw)
    selected_question_ids = [
        str(row["question_id"])
        for row in _fetch_assignment_question_rows(db, [assignment_id])
    ]
    attempt_rows = _fetch_attempts_for_trainee(
        db,
        trainee_id=current_user.id,
        assignment_ids=[assignment_id],
    )
    certificate_rows = _fetch_certificates_for_trainee(
        db,
        trainee_id=current_user.id,
        assignment_ids=[assignment_id],
    )

    attempts = [_attempt_record(row) for row in attempt_rows]
    latest_attempt = attempts[0] if attempts else None
    maximum_attempts = _normalize_int(assignment.get("maximum_attempts")) if assignment.get("maximum_attempts") is not None else None
    status = _build_trainee_status(latest_attempt, len(attempts), maximum_attempts)

    question_pool = _build_assignment_question_pool(
        assignment,
        question_map=question_map,
        questions_by_assessment=questions_by_assessment,
        questions_by_category=questions_by_category,
        selected_question_ids=selected_question_ids,
        randomize_subset=True,
    )
    if not question_pool and not latest_attempt:
        raise HTTPException(status_code=400, detail="This assigned assessment does not have any active questions yet.")

    session_questions = [
        _serialize_trainee_session_question(
            question,
            question_number=index + 1,
            shuffle_choices=_truthy(assignment.get("shuffle_choices"), True),
        )
        for index, question in enumerate(question_pool)
    ]

    certificate = None
    if certificate_rows:
        certificate = _certificate_record(
            certificate_rows[0],
            categories_by_id,
            {assignment_id: assignment},
            assessments_by_id,
        )

    batch_lookup = {str(batch["id"]): batch for batch in batch_rows}
    return {
        "assignmentId": assignment_id,
        "assessmentId": assessment_id or assignment_id,
        "categoryId": category_id,
        "categoryTitle": categories_by_id.get(category_id, {}).get("title", "Assessment Category"),
        "targetType": assignment.get("target_scope") or ("trainee" if assignment.get("trainee_id") else "batch"),
        "waveNumber": _normalize_int(assignment.get("wave_number")) if assignment.get("wave_number") is not None else None,
        "assignmentTitle": assignment.get("title") or assessments_by_id.get(assessment_id or "", {}).get("title"),
        "assessmentTitle": assessments_by_id.get(assessment_id or "", {}).get("title") or assignment.get("title") or categories_by_id.get(category_id, {}).get("title", "Assessment"),
        "description": assignment.get("description") or assessments_by_id.get(assessment_id or "", {}).get("description"),
        "passingScore": _normalize_int(
            assignment.get("passing_score"),
            fallback=_normalize_int(categories_by_id.get(category_id, {}).get("passing_score"), fallback=DEFAULT_PASSING_SCORE),
        ),
        "targetDueAt": _normalize_datetime(assignment.get("due_at")),
        "targetLabel": _resolve_trainee_target_label(assignment, batch_lookup),
        "questionCount": len(session_questions),
        "attemptCount": len(attempts),
        "attemptsRemaining": status["attemptsRemaining"],
        "maximumAttempts": maximum_attempts,
        "timeLimitMinutes": _normalize_int(assignment.get("time_limit_minutes")) if assignment.get("time_limit_minutes") is not None else None,
        "canStart": status["canStart"],
        "canRetake": status["canRetake"],
        "isCompleted": status["isCompleted"],
        "latestAttempt": latest_attempt,
        "certificate": certificate,
        "questions": session_questions,
        "statusLabel": status["statusLabel"],
    }


def _build_attempt_feedback(score: float, passing_score: float, passed: bool) -> str:
    if passed:
        return f"Passing score achieved at {score:.2f}% against the {passing_score:.2f}% target."
    return f"Score saved at {score:.2f}%. Review the missed questions before the next attempt."


def _build_attempt_analysis_summary(
    *,
    category_id: str,
    category_title: str,
    earned_points: int,
    total_points: int,
    correct_answers: int,
    total_questions: int,
    score: float,
    passing_score: float,
    passed: bool,
) -> dict[str, Any]:
    return {
        "source": "rules",
        "summary": _build_attempt_feedback(score, passing_score, passed),
        "strengths": ["Passing threshold met."] if passed else [f"{correct_answers} of {total_questions} questions were answered correctly."],
        "improvements": [] if passed else ["Focus on the missed questions before using another attempt."],
        "recommendations": (
            ["This category is complete. No retake is available after passing."]
            if passed
            else ["Use the saved review details to correct the weakest questions, then retake if attempts remain."]
        ),
        "earnedPoints": earned_points,
        "totalPoints": total_points,
        "categoryBreakdown": [
            {
                "categoryId": category_id,
                "categoryTitle": category_title,
                "totalQuestions": total_questions,
                "correctAnswers": correct_answers,
                "score": score,
            }
        ],
    }


def submit_trainee_workspace_attempt(
    db: Session,
    current_user: User,
    payload: dict[str, Any],
) -> dict[str, Any]:
    assignment_id = _sanitize_text(payload.get("assignmentId") or payload.get("assignment_id"))
    if not assignment_id:
        raise HTTPException(status_code=400, detail="Assigned assessment is required.")

    assignment, batch_rows = _require_trainee_assignment_access(db, current_user, assignment_id)
    category_id = str(assignment["category_id"])
    assessment_id = str(assignment["assessment_id"]) if assignment.get("assessment_id") else assignment_id
    batch_id = str(assignment["batch_id"]) if assignment.get("batch_id") else None
    maximum_attempts = _normalize_int(assignment.get("maximum_attempts")) if assignment.get("maximum_attempts") is not None else None

    prior_attempt_rows = _fetch_attempts_for_trainee(
        db,
        trainee_id=current_user.id,
        assignment_ids=[assignment_id],
    )
    prior_attempts = [_attempt_record(row) for row in prior_attempt_rows]
    latest_attempt = prior_attempts[0] if prior_attempts else None

    if latest_attempt and latest_attempt["status"] == "pass":
        raise HTTPException(status_code=400, detail="Passed assessments cannot be retaken.")
    if maximum_attempts is not None and len(prior_attempts) >= maximum_attempts:
        raise HTTPException(status_code=400, detail="No assessment attempts remain for this assignment.")

    answers = payload.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise HTTPException(status_code=400, detail="At least one assessment answer is required.")

    categories_by_id = {str(row["id"]): row for row in _fetch_categories_by_ids(db, [category_id])}
    category = categories_by_id.get(category_id)
    assessments_raw = _fetch_assessments_for_categories(db, [category_id])
    assessments_by_id = {str(row["id"]): row for row in assessments_raw}
    questions_raw = _fetch_questions_for_categories(db, [category_id])
    question_map, questions_by_assessment, questions_by_category = _build_question_lookup(questions_raw)
    selected_question_ids = [
        str(row["question_id"])
        for row in _fetch_assignment_question_rows(db, [assignment_id])
    ]

    served_question_ids = [
        _sanitize_text(question_id)
        for question_id in payload.get("questionIds") or payload.get("question_ids") or []
        if _sanitize_text(question_id)
    ]
    if not served_question_ids:
        served_question_ids = [
            str(question["id"])
            for question in _build_assignment_question_pool(
                assignment,
                question_map=question_map,
                questions_by_assessment=questions_by_assessment,
                questions_by_category=questions_by_category,
                selected_question_ids=selected_question_ids,
                randomize_subset=False,
            )
        ]

    if not served_question_ids:
        raise HTTPException(status_code=400, detail="This assigned assessment does not have any active questions yet.")

    if selected_question_ids:
        allowed_question_ids = {
            question_id
            for question_id in selected_question_ids
            if question_id in question_map
        }
    elif assignment.get("assessment_id"):
        allowed_question_ids = {
            str(question["id"])
            for question in questions_by_assessment.get(str(assignment["assessment_id"]), [])
        }
    else:
        allowed_question_ids = {
            str(question["id"])
            for question in questions_by_category.get(category_id, [])
        }
    if any(question_id not in allowed_question_ids for question_id in served_question_ids):
        raise HTTPException(status_code=400, detail="The submitted assessment question set is no longer valid. Start the assignment again.")

    choice_map = payload.get("choiceMap") or payload.get("choice_map") or {}
    if not isinstance(choice_map, dict):
        choice_map = {}

    question_results: list[dict[str, Any]] = []
    question_snapshot: list[dict[str, Any]] = []
    choice_snapshot: dict[str, list[str]] = {}
    correct_answers = 0
    earned_points = 0
    total_points = 0

    for index, question_id in enumerate(served_question_ids, start=1):
        question = question_map.get(question_id)
        if not question:
            raise HTTPException(status_code=400, detail="One or more submitted questions could not be found.")

        metadata = question.get("metadata") if isinstance(question.get("metadata"), dict) else {}
        points = _question_point_value(metadata)
        total_points += points

        user_answer = str(answers.get(question_id) or "").strip()
        correct_answer = str(question.get("correct_answer") or "").strip()
        is_correct = _normalize_value(user_answer) == _normalize_value(correct_answer)
        if is_correct:
            correct_answers += 1
            earned_points += points

        rendered_choices = choice_map.get(question_id)
        if not isinstance(rendered_choices, list) or not rendered_choices:
            rendered_choices = [str(option) for option in (question.get("options") or [])]
        else:
            rendered_choices = [str(option) for option in rendered_choices]
        choice_snapshot[question_id] = rendered_choices

        question_results.append(
            {
                "questionId": question_id,
                "questionNumber": index,
                "questionText": question.get("question_text") or "",
                "questionType": question.get("question_type") or "multiple_choice",
                "difficulty": question.get("difficulty"),
                "options": rendered_choices,
                "choiceOrder": rendered_choices,
                "userAnswer": user_answer,
                "correctAnswer": correct_answer,
                "isCorrect": is_correct,
                "explanation": question.get("explanation"),
                "points": points,
                "earnedPoints": points if is_correct else 0,
            }
        )
        question_snapshot.append(
            {
                "id": question_id,
                "questionText": question.get("question_text") or "",
                "questionType": question.get("question_type") or "multiple_choice",
                "difficulty": question.get("difficulty"),
                "choices": rendered_choices,
                "correctAnswer": correct_answer,
                "points": points,
                "explanation": question.get("explanation"),
            }
        )

    total_questions = len(served_question_ids)
    incorrect_answers = max(total_questions - correct_answers, 0)
    score = round((earned_points / total_points) * 100, 2) if total_points else 0.0
    passing_score = _normalize_float(
        assignment.get("passing_score"),
        fallback=_normalize_float((category or {}).get("passing_score"), fallback=float(DEFAULT_PASSING_SCORE)),
    )
    passed = score >= passing_score
    attempt_no = len(prior_attempts) + 1
    attempts_remaining = None if maximum_attempts is None else max(maximum_attempts - attempt_no, 0)
    now_iso = datetime.utcnow().isoformat()
    started_at = _normalize_datetime(payload.get("startedAt") or payload.get("started_at")) or now_iso
    time_spent_seconds = _normalize_int(payload.get("timeSpentSeconds") or payload.get("time_spent_seconds"), fallback=0)
    category_title = (category or {}).get("title") or "Assessment Category"
    feedback = _build_attempt_feedback(score, passing_score, passed)
    analysis_summary = _build_attempt_analysis_summary(
        category_id=category_id,
        category_title=category_title,
        earned_points=earned_points,
        total_points=total_points,
        correct_answers=correct_answers,
        total_questions=total_questions,
        score=score,
        passing_score=passing_score,
        passed=passed,
    )
    category_breakdown = analysis_summary["categoryBreakdown"]
    assignment_title = assignment.get("title") or assessments_by_id.get(assessment_id, {}).get("title") or category_title
    certificate_status = "issued" if passed else "not_issued"
    batch_row = next(
        (row for row in batch_rows if str(row.get("id")) == str(batch_id))
        if batch_id
        else [],
        None,
    )
    batch_name = _format_batch_label(batch_row)

    certificate_row = None
    try:
        inserted_attempt = _execute_returning_one(
            db,
            """
            insert into training_assessment_attempts (
                assignment_id,
                assessment_id,
                category_id,
                trainee_id,
                batch_id,
                attempt_no,
                answers,
                question_results,
                total_questions,
                correct_answers,
                incorrect_answers,
                score,
                status,
                feedback,
                submitted_at,
                question_snapshot,
                choice_snapshot,
                analysis_summary,
                category_breakdown,
                time_spent_seconds,
                passing_score,
                assignment_title,
                started_at,
                completed_at,
                certificate_status
            )
            values (
                :assignment_id,
                :assessment_id,
                :category_id,
                :trainee_id,
                :batch_id,
                :attempt_no,
                cast(:answers as jsonb),
                cast(:question_results as jsonb),
                :total_questions,
                :correct_answers,
                :incorrect_answers,
                :score,
                :status,
                :feedback,
                :submitted_at,
                cast(:question_snapshot as jsonb),
                cast(:choice_snapshot as jsonb),
                cast(:analysis_summary as jsonb),
                cast(:category_breakdown as jsonb),
                :time_spent_seconds,
                :passing_score,
                :assignment_title,
                :started_at,
                :completed_at,
                :certificate_status
            )
            returning *
            """,
            {
                "assignment_id": assignment_id,
                "assessment_id": assessment_id,
                "category_id": category_id,
                "trainee_id": current_user.id,
                "batch_id": batch_id,
                "attempt_no": attempt_no,
                "answers": json.dumps(answers),
                "question_results": json.dumps(question_results),
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "score": score,
                "status": "pass" if passed else "fail",
                "feedback": feedback,
                "submitted_at": now_iso,
                "question_snapshot": json.dumps(question_snapshot),
                "choice_snapshot": json.dumps(choice_snapshot),
                "analysis_summary": json.dumps(analysis_summary),
                "category_breakdown": json.dumps(category_breakdown),
                "time_spent_seconds": time_spent_seconds,
                "passing_score": passing_score,
                "assignment_title": assignment_title,
                "started_at": started_at,
                "completed_at": now_iso,
                "certificate_status": certificate_status,
            },
        )

        if passed:
            existing_certificate = _fetch_one(
                db,
                """
                select *
                from training_assessment_certificates
                where attempt_id = :attempt_id
                   or (
                       trainee_id = :trainee_id
                       and assignment_id = :assignment_id
                   )
                order by earned_at desc nulls last, created_at desc nulls last
                limit 1
                """,
                {
                    "attempt_id": inserted_attempt["id"],
                    "trainee_id": current_user.id,
                    "assignment_id": assignment_id,
                },
            )
            if existing_certificate:
                certificate_row = existing_certificate
            else:
                certificate_code = f"CERT-{current_user.id[:6].upper()}-{category_id[:6].upper()}-{attempt_no:02d}-{int(datetime.utcnow().timestamp())}"
                certificate_row = _execute_returning_one(
                    db,
                    """
                    insert into training_assessment_certificates (
                        trainee_id,
                        category_id,
                        assessment_id,
                        attempt_id,
                        certificate_code,
                        assignment_id,
                        assignment_title,
                        certificate_status,
                        certificate_url
                    )
                    values (
                        :trainee_id,
                        :category_id,
                        :assessment_id,
                        :attempt_id,
                        :certificate_code,
                        :assignment_id,
                        :assignment_title,
                        'issued',
                        '/trainee/certificates'
                    )
                    returning *
                    """,
                    {
                        "trainee_id": current_user.id,
                        "category_id": category_id,
                        "assessment_id": assessment_id,
                        "attempt_id": inserted_attempt["id"],
                        "certificate_code": certificate_code,
                        "assignment_id": assignment_id,
                        "assignment_title": assignment_title,
                    },
                )
        notify_assessment_completion(
            db,
            trainee=current_user,
            assignment_title=assignment_title,
            category_title=category_title,
            batch_name=batch_name,
            score=score,
            passing_score=passing_score,
            passed=passed,
            attempt_no=attempt_no,
            completed_at=now_iso,
        )
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save the trainee assessment attempt.") from exc

    feed_row = _fetch_attempt_feed_row(db, str(inserted_attempt["id"]))
    attempt_record = _attempt_record(feed_row)
    attempt_record["attemptsRemaining"] = attempts_remaining
    attempt_record["canRetake"] = bool(not passed and (maximum_attempts is None or attempt_no < maximum_attempts))
    attempt_record["statusLabel"] = "Passed" if passed else "Failed" if attempts_remaining == 0 else "Completed"

    certificate_payload = None
    if certificate_row:
        certificate_payload = _certificate_record(
            certificate_row,
            categories_by_id,
            {assignment_id: assignment},
            assessments_by_id,
        )

    return {
        "attempt": attempt_record,
        "certificate": certificate_payload,
    }


def bulk_upload_questions_from_csv(db: Session, current_user: User, csv_text: str) -> dict[str, Any]:
    reader = csv.reader(io.StringIO(csv_text))
    rows = [row for row in reader if any(str(cell or "").strip() for cell in row)]
    if not rows:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    header = [str(value or "").replace("\ufeff", "").strip() for value in rows[0]]
    column_index = _build_canonical_column_index(header)
    missing_columns = [column for column in REQUIRED_TEMPLATE_COLUMNS if column not in column_index]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"The CSV file is missing required columns: {', '.join(missing_columns)}.",
        )

    categories_raw = _fetch_visible_categories(db, current_user, include_inactive=True)
    category_cache = {_normalize_value(category.get("title")): category for category in categories_raw}
    assessments_raw = _fetch_assessments_for_categories(db, [str(category["id"]) for category in categories_raw], include_inactive=True)
    assessment_cache = {
        (str(assessment["category_id"]), _normalize_value(assessment.get("title"))): assessment
        for assessment in assessments_raw
    }
    questions_raw = _fetch_questions_for_categories(db, [str(category["id"]) for category in categories_raw])
    existing_numbers: dict[str, set[int]] = defaultdict(set)
    existing_texts: dict[str, set[str]] = defaultdict(set)
    for question in questions_raw:
        category_id = str(question["category_id"])
        existing_numbers[category_id].add(_normalize_int(question.get("question_number")))
        existing_texts[category_id].add(_normalize_value(question.get("question_text")))

    errors: list[dict[str, Any]] = []
    imported_questions: list[dict[str, Any]] = []
    created_categories: list[str] = []
    pending_numbers: dict[str, set[int]] = defaultdict(set)
    pending_texts: dict[str, set[str]] = defaultdict(set)

    for row_index in range(1, len(rows)):
        row = rows[row_index]

        def get_value(column: str) -> str:
            index = column_index.get(column)
            return str(row[index]).strip() if index is not None and index < len(row) else ""

        assessment_title = get_value("Assessment Title")
        category_name = get_value("Category")
        question_number_value = get_value("Question Number")
        question_text = get_value("Question")
        choices = [
            get_value("Choice 1"),
            get_value("Choice 2"),
            get_value("Choice 3"),
            get_value("Choice 4"),
        ]
        correct_answer = get_value("Correct Answer")
        difficulty = _normalize_value(get_value("Difficulty Level"))
        points_value = get_value("Points")
        explanation = get_value("Explanation")
        row_number = row_index + 1

        if not assessment_title:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Assessment Title is required."})
            continue
        if not category_name:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Category is required."})
            continue

        try:
            parsed_question_number = int(question_number_value)
            if parsed_question_number < 1:
                raise ValueError
        except ValueError:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Question Number must be a positive whole number."})
            continue

        sanitized_question_text = _sanitize_text(question_text)
        if not sanitized_question_text:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Question text is required."})
            continue

        try:
            parsed_points = float(points_value)
            if parsed_points <= 0:
                raise ValueError
        except ValueError:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Points must be a positive number."})
            continue

        if difficulty and difficulty not in {"easy", "medium", "hard"}:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Difficulty must be easy, medium, or hard when provided."})
            continue

        try:
            validated_choices, validated_correct_answer = _build_choice_validation(choices, correct_answer)
        except HTTPException as exc:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": exc.detail})
            continue

        try:
            category = _get_or_create_category(db, current_user, category_name, category_cache, created_categories)
            assessment = _get_or_create_assessment(db, category, assessment_title, assessment_cache)
        except HTTPException as exc:
            db.rollback()
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": exc.detail})
            continue
        except SQLAlchemyError:
            db.rollback()
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Unable to create the linked category or assessment."})
            continue

        category_id = str(category["id"])
        normalized_question_text = _normalize_value(sanitized_question_text)
        if parsed_question_number in existing_numbers[category_id] or parsed_question_number in pending_numbers[category_id]:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Question Number already exists for this category."})
            continue
        if normalized_question_text in existing_texts[category_id] or normalized_question_text in pending_texts[category_id]:
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Duplicate question text detected for this category."})
            continue

        try:
            inserted = _execute_returning_one(
                db,
                """
                insert into training_assessment_questions (
                    assessment_id,
                    category_id,
                    question_number,
                    question_text,
                    question_type,
                    options,
                    correct_answer,
                    difficulty,
                    explanation,
                    order_index,
                    active_status,
                    created_by,
                    metadata
                )
                values (
                    :assessment_id,
                    :category_id,
                    :question_number,
                    :question_text,
                    'multiple_choice',
                    cast(:options as jsonb),
                    :correct_answer,
                    :difficulty,
                    :explanation,
                    :order_index,
                    true,
                    :created_by,
                    cast(:metadata as jsonb)
                )
                returning *
                """,
                {
                    "assessment_id": assessment["id"],
                    "category_id": category["id"],
                    "question_number": parsed_question_number,
                    "question_text": sanitized_question_text,
                    "options": json.dumps(validated_choices),
                    "correct_answer": validated_correct_answer,
                    "difficulty": difficulty or None,
                    "explanation": _sanitize_text(explanation) or None,
                    "order_index": max(parsed_question_number - 1, 0),
                    "created_by": current_user.id,
                    "metadata": json.dumps({
                        "points": int(parsed_points) if float(parsed_points).is_integer() else parsed_points,
                        "imported_from_csv": True,
                        "assessment_title": assessment.get("title"),
                    }),
                },
            )
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            logger.exception("Assessment CSV insert failed for category %s / assessment %s", category_name, assessment_title)
            errors.append({"rowNumber": row_number, "category": category_name, "questionNumber": question_number_value, "question": question_text, "error": "Unable to save this question to the assessment database."})
            continue

        existing_numbers[category_id].add(parsed_question_number)
        existing_texts[category_id].add(normalized_question_text)
        pending_numbers[category_id].add(parsed_question_number)
        pending_texts[category_id].add(normalized_question_text)
        imported_questions.append(
            _question_record(
                inserted,
                category.get("title"),
                assessment.get("title"),
                None,
            )
        )

    return {
        "totalRows": max(len(rows) - 1, 0),
        "successfulImports": len(imported_questions),
        "failedRows": len(errors),
        "importedQuestions": imported_questions,
        "errors": errors,
        "createdCategories": created_categories,
        "errorCsv": _build_error_csv(errors),
    }
