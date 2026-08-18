"""
Call Simulation routes.
Handles trainer scenario management, KPI configuration, trainee sessions,
Supabase-backed audio uploads, coaching notes, and analytics.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import uuid
import zipfile
from base64 import b64decode, b64encode
from collections import defaultdict
from csv import reader as csv_reader
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

# Lazy pandas loader to avoid importing heavy dependency at startup
_pd = None

def _ensure_pandas():
    global _pd
    if _pd is not None:
        return _pd
    try:
        import importlib

        _pd = importlib.import_module('pandas')
        return _pd
    except Exception as exc:
        logger.warning('Pandas not available: %s', exc)
        _pd = None
        return None
import requests
from fastapi import APIRouter, Body, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse, FileResponse
from openpyxl import Workbook
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from .. import auth_utils
from ..database import get_db
from ..models import (
    Batch,
    BatchKPIConfig,
    BatchScenarioMapping,
    CertificateRecord,
    CoachingLog,
    CallSimulationEvent,
    CallSimulationAudioAsset,
    Scenario,
    ScenarioFlow,
    CallSimulationAssignment,
    SessionResponseRecord,
    ScenarioVariation,
    SimSession,
    User,
    UserRole,
)
from ..schemas import (
    BatchKPIConfigCreate,
    BatchKPIConfigResponse,
    CallSimulationScenarioAssignmentSummary,
    CallSimulationAssignmentCreate,
    CallSimulationAssignmentResponse,
    CallSimulationAssignmentTargetResponse,
    CallSimulationAudioSettingsResponse,
    CallSimulationAudioSettingsUpdate,
    CallSimulationAudioAssetResponse,
    CallSimulationAudioAssetUploadResponse,
    CallSimulationScenarioRowInput,
    CallSimulationScenarioStepResponse,
    BatchKPIConfigUpdate,
    BatchScenarioMappingCreate,
    BulkUploadResponse,
    CallSimulationScenarioCreate,
    CallSimulationScenarioResponse,
    CallSimulationScenarioUpdate,
    SimSessionCompleteRequest,
    SimSessionCoachingNoteUpdate,
    SimSessionCreate,
    SimSessionResponse,
    SimSessionStartResponse,
    SimSessionTrainerVerdictUpdate,
    SimSessionTurnResponse,
    ScenarioVariationCreate,
    ScenarioVariationResponse,
    ScenarioVariationUpdate,
    SuccessResponse,
)
from ..services.certificate_awards import (
    award_certificate,
    prune_trainee_activity_certificates,
    sync_trainee_completion_certificates,
)
from ..services.coaching import generate_coaching_id, normalize_competency_status
from ..services.gemini_evaluation import generate_evaluation_feedback
from ..services.gemini_tts import gemini_tts_engine
from ..services.live_updates import live_update_manager
from ..services.notifications import notify_call_simulation_completion
from ..services.speech_assessment import assess_audio_submission, normalize_text, tokenize_text
from ..services.supabase_auth_service import filter_to_supabase_active_users
from ..services.tts_service import get_tts_service, text_to_speech, text_to_speech_for_persistence
from ..services.audit import create_audit_log
from ..supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/call-simulation", tags=["call-simulation"])

DEFAULT_PASSING_SCORE = 90.0
DEFAULT_MAX_ATTEMPTS = 99
AUTO_CERTIFICATE_PASSING_SCORE = 80.0
TURN_REPEAT_PROMPT = "Repeat, I can't understand what you're saying."
MIN_SCRIPT_SIMILARITY_FOR_PROGRESS = 0.58
MIN_STT_ACCURACY_FOR_PROGRESS = 55.0
MIN_KEYWORD_COVERAGE_FOR_PROGRESS = 0.34
CONTENT_SCORE_WEIGHT = 0.5
KPI_SCORE_WEIGHT = 0.5
CALL_SIMULATION_AUDIO_SETTINGS_KEY = "call_simulation_audio"
CALL_SIMULATION_AUDIO_MAX_BYTES = 50 * 1024 * 1024
CALL_SIMULATION_ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}
CALL_SIMULATION_AUDIO_MIME_ALIASES = {
    "audio/mpeg": "audio/mpeg",
    "audio/mp3": "audio/mpeg",
    "audio/wav": "audio/wav",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/ogg": "audio/ogg",
    "audio/vorbis": "audio/ogg",
    "audio/mp4": "audio/mp4",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
    "audio/aac": "audio/mp4",
}
_UNSET = object()
_LEGACY_CALL_SIMULATION_RELATION_CACHE: dict[str, bool] = {}
_LEGACY_CALL_SIMULATION_SKIP_LOG_CACHE: set[tuple[str, ...]] = set()


def _require_trainer(current_user: User) -> None:
    if current_user.role not in [UserRole.ADMIN, UserRole.TRAINER]:
        raise HTTPException(status_code=403, detail="Trainer access required")


def _require_trainee(current_user: User) -> None:
    if current_user.role != UserRole.TRAINEE:
        raise HTTPException(status_code=403, detail="Trainee access required")


async def _broadcast_call_simulation_completion(
    *,
    trainee: User,
    session: SimSession,
    scenario: Optional[Scenario],
) -> None:
    payload = {
        "type": "call_simulation_completed",
        "details": {
            "session_id": session.id,
            "scenario_id": session.scenario_id,
            "scenario_title": getattr(scenario, "title", None) or "Call Simulation",
            "trainee_id": trainee.id,
            "trainee_name": trainee.full_name,
            "batch_id": session.batch_id,
            "score": float(getattr(session, "overall_score", 0.0) or 0.0),
            "passed": bool(getattr(session, "pass_fail", False)),
            "attempt_no": int(getattr(session, "attempt_number", 1) or 1),
            "completed_at": session.updated_at.isoformat() if session.updated_at else None,
        },
    }
    await live_update_manager.broadcast_training_update(payload)
    await live_update_manager.broadcast(
        f"trainee:{trainee.id}",
        payload,
    )


def _require_supabase_storage(detail: str) -> Any:
    supabase = get_supabase_client()
    if not supabase.is_available:
        raise HTTPException(status_code=503, detail=detail)
    return supabase


def _require_server_side_tts(detail: str) -> None:
    if not get_tts_service().is_available():
        raise HTTPException(status_code=503, detail=detail)


def _log_call_simulation_audio_action(
    db: Session,
    *,
    user: User,
    action_type: str,
    scenario_id: Optional[str] = None,
    step_number: Optional[int] = None,
    file_name: Optional[str] = None,
    status_value: str = "success",
    error_detail: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    create_audit_log(
        db,
        user=user,
        action_type=action_type,
        module_name="Call Simulation",
        entity_type="call_simulation_audio",
        entity_id=scenario_id,
        description=f"Call Simulation audio {action_type.replace('_', ' ')} {status_value}.",
        status=status_value,
        severity="warning" if status_value == "failed" else "info",
        trainer_id=user.id if user.role in [UserRole.TRAINER, UserRole.ADMIN] else None,
        trainee_id=user.id if user.role == UserRole.TRAINEE else None,
        metadata={
            "scenario_id": scenario_id,
            "step_number": step_number,
            "file_name": file_name,
            "error_detail": error_detail,
            **(metadata or {}),
        },
    )


def _is_missing_supabase_table_error(error: Exception) -> bool:
    normalized_error = str(error or "").strip().lower()
    if not normalized_error:
        return False

    return (
        "pgrst205" in normalized_error
        or "schema cache" in normalized_error
        or "could not find the table" in normalized_error
    )


def _log_legacy_supabase_mirror_skip(context: str, error: Exception) -> None:
    logger.info(
        "Skipping legacy Call Simulation Supabase mirror for %s because the optional PostgREST tables are not present in this project. Core data is already saved through the primary Supabase Postgres connection. Details: %s",
        context,
        error,
    )


def _legacy_call_simulation_relation_exists(db: Session, relation_name: str) -> bool:
    normalized_name = str(relation_name or "").strip().lower()
    if not normalized_name:
        return False

    cached_value = _LEGACY_CALL_SIMULATION_RELATION_CACHE.get(normalized_name)
    if cached_value is not None:
        return cached_value

    try:
        exists = bool(
            db.execute(
                text("select to_regclass(:qualified_name) is not null"),
                {"qualified_name": f"public.{normalized_name}"},
            ).scalar()
        )
    except Exception as exc:
        logger.warning(
            "Unable to inspect optional legacy Call Simulation relation %s. Treating it as unavailable: %s",
            normalized_name,
            exc,
        )
        exists = False

    _LEGACY_CALL_SIMULATION_RELATION_CACHE[normalized_name] = exists
    return exists


def _legacy_call_simulation_mirror_relations_ready(
    db: Session,
    relation_names: list[str] | tuple[str, ...],
    *,
    context: str,
) -> bool:
    normalized_names = tuple(
        sorted(
            {
                str(relation_name or "").strip().lower()
                for relation_name in relation_names
                if str(relation_name or "").strip()
            }
        )
    )
    if not normalized_names:
        return False

    missing_relations = [
        relation_name
        for relation_name in normalized_names
        if not _legacy_call_simulation_relation_exists(db, relation_name)
    ]
    if not missing_relations:
        return True

    missing_key = tuple(missing_relations)
    if missing_key not in _LEGACY_CALL_SIMULATION_SKIP_LOG_CACHE:
        logger.info(
            "Skipping legacy Call Simulation Supabase mirror for %s because these optional relations are not present in this project: %s. Core data is already saved through the primary Supabase Postgres connection.",
            context,
            ", ".join(missing_relations),
        )
        _LEGACY_CALL_SIMULATION_SKIP_LOG_CACHE.add(missing_key)
    return False


def _normalize_keyword_list(keywords: Optional[list[str]]) -> list[str]:
    return [keyword.strip() for keyword in (keywords or []) if keyword and keyword.strip()]


def _coerce_audio_bytes(payload: Any) -> Optional[bytes]:
    if payload is None:
        return None
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if hasattr(payload, "tobytes"):
        try:
            return payload.tobytes()
        except Exception:
            return None
    if isinstance(payload, str) and payload.strip():
        text_payload = payload.strip()
        if text_payload.startswith("data:"):
            try:
                _, base64_part = text_payload.split(",", 1)
                return b64decode(base64_part)
            except Exception:
                pass
        try:
            return b64decode(text_payload)
        except Exception:
            return None
    return None


def _build_audio_data_url(audio_bytes: bytes, content_type: str) -> str:
    encoded_audio = b64encode(audio_bytes).decode("ascii")
    normalized_content_type = (content_type or "audio/wav").strip() or "audio/wav"
    return f"data:{normalized_content_type};base64,{encoded_audio}"


def _normalize_actor_role(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _is_csr_actor(value: Optional[str]) -> bool:
    normalized = _normalize_actor_role(value)
    return normalized in {"csr", "agent", "trainee"}


def _group_script_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    for row in rows:
        scenario_key = str(row.get("scenario") or "").strip()
        if not scenario_key:
            continue
        if scenario_key not in grouped:
            grouped[scenario_key] = {
                "scenario_key": scenario_key,
                "csr_variants": [],
                "member_rows": [],
            }
            ordered_keys.append(scenario_key)

        if _is_csr_actor(str(row.get("actor") or row.get("actor_name") or "")):
            grouped[scenario_key]["csr_variants"].append(row)
        else:
            grouped[scenario_key]["member_rows"].append(row)

    return [grouped[key] for key in ordered_keys if grouped[key]["csr_variants"]]


def _coerce_nonnegative_float(value: Any, *, fallback: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = fallback
    return round(max(0.0, resolved), 2)


def _coerce_positive_int(value: Any, *, fallback: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = fallback
    return max(1, resolved)


def _normalize_uploaded_document_title(filename: Optional[str]) -> str:
    raw_name = os.path.splitext(os.path.basename(filename or ""))[0]
    cleaned = re.sub(r"[_\-]+", " ", raw_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "Call Simulation Scenario"


def _parse_uploaded_document_metadata(filename: Optional[str]) -> dict[str, str]:
    raw_name = os.path.splitext(os.path.basename(filename or ""))[0]
    collapsed_name = re.sub(r"\s+", " ", raw_name).strip()
    fallback = _normalize_uploaded_document_title(filename)
    if not collapsed_name:
        return {
            "title": fallback,
            "topic": fallback,
            "description": fallback,
        }

    underscore_parts = [
        re.sub(r"[-]+", " ", segment).strip()
        for segment in collapsed_name.split("_")
        if segment and segment.strip()
    ]
    if len(underscore_parts) >= 3:
        title = re.sub(r"\s+", " ", underscore_parts[0]).strip() or fallback
        topic = re.sub(r"\s+", " ", underscore_parts[1]).strip() or title
        description = re.sub(r"\s+", " ", " ".join(underscore_parts[2:])).strip() or topic
        return {
            "title": title,
            "topic": topic,
            "description": description,
        }

    dashed_parts = [
        re.sub(r"_+", " ", segment).strip()
        for segment in re.split(r"\s+-\s+", collapsed_name)
        if segment and segment.strip()
    ]
    if len(dashed_parts) >= 2:
        title = re.sub(r"\s+", " ", dashed_parts[0]).strip() or fallback
        topic = re.sub(r"\s+", " ", dashed_parts[1]).strip() or title
        description = re.sub(r"\s+", " ", " ".join(dashed_parts[2:])).strip() or fallback
        return {
            "title": title,
            "topic": topic,
            "description": description,
        }

    return {
        "title": fallback,
        "topic": fallback,
        "description": fallback,
    }


def _extract_docx_paragraphs(file_bytes: bytes) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            document_xml = archive.read("word/document.xml")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read DOCX file: {exc}") from exc

    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail=f"DOCX content could not be parsed: {exc}") from exc

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        chunks = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        text = "".join(chunks).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _parse_document_row_columns(raw_line: str) -> Optional[list[str]]:
    line = str(raw_line or "").strip()
    if not line:
        return None

    for delimiter in ("\t", "|"):
        if delimiter in line:
            columns = [part.strip() for part in line.split(delimiter, 3)]
            return columns if len(columns) >= 4 else None

    try:
        parsed = next(csv_reader([line]))
    except Exception:
        parsed = []
    cleaned = [part.strip() for part in parsed if part is not None]
    if len(cleaned) < 4:
        return None
    if len(cleaned) == 4:
        return cleaned
    actor = cleaned[0]
    scenario = cleaned[-1]
    score = cleaned[-2]
    script = ",".join(cleaned[1:-2]).strip()
    return [actor, script, score, scenario]


def _looks_like_bulk_upload_header(columns: list[str]) -> bool:
    if len(columns) < 4:
        return False
    normalized = [str(value or "").strip().lower() for value in columns[:4]]
    return normalized == ["actor", "script", "score", "scenario"]


def _normalize_bulk_upload_columns(dataframe: Any, required_columns: list[str]) -> Any:
    pd = _ensure_pandas()
    if pd is None:
        raise HTTPException(status_code=503, detail='Pandas is required for spreadsheet uploads but is not installed.')

    normalized_header_map: dict[str, str] = {
        str(column).strip().lower(): column for column in dataframe.columns
    }
    mapped_columns: dict[str, Any] = {}

    for expected in required_columns:
        normalized_key = expected.strip().lower()
        if normalized_key in normalized_header_map:
            mapped_columns[normalized_header_map[normalized_key]] = expected

    if len(mapped_columns) != len(required_columns):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required columns: {', '.join(required_columns)}. "
                "Use exact column names or any case-insensitive variant."
            ),
        )

    return dataframe.rename(columns=mapped_columns)


def _build_bulk_upload_rows_from_dataframe(dataframe: Any) -> tuple[list[dict[str, Any]], int, list[str]]:
    pd = _ensure_pandas()
    if pd is None:
        raise HTTPException(status_code=503, detail='Pandas is required for spreadsheet uploads but is not installed.')
    required_columns = ["Actor", "Script", "Score", "Scenario"]
    try:
        dataframe = _normalize_bulk_upload_columns(dataframe, required_columns)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Error parsing spreadsheet upload. Ensure the first row contains Actor, Script, Score, and Scenario columns. "
                f"{exc}"
            ),
        ) from exc

    script_rows: list[dict[str, Any]] = []
    failed_rows = 0
    errors: list[str] = []

    for index, row in dataframe.iterrows():
        actor = str(row.get("Actor", "")).strip()
        script = str(row.get("Script", "")).strip()
        scenario_text = str(row.get("Scenario", "")).strip()
        score_value = row.get("Score", 0)

        if not actor or not script or not scenario_text:
            failed_rows += 1
            errors.append(f"Row {index + 2}: Actor, Script, and Scenario are required.")
            continue

        score = 0.0
        if _is_csr_actor(actor):
            if pd.isna(score_value):
                failed_rows += 1
                errors.append(f"Row {index + 2}: CSR rows need a numeric Score.")
                continue
            try:
                score = float(score_value)
            except Exception:
                failed_rows += 1
                errors.append(f"Row {index + 2}: CSR rows need a numeric Score.")
                continue

            if score < 0:
                failed_rows += 1
                errors.append(f"Row {index + 2}: Score must be zero or greater.")
                continue
        else:
            try:
                score = 0.0 if pd.isna(score_value) or score_value in (None, "") else float(score_value)
            except Exception:
                score = 0.0

        script_rows.append(
            {
                "actor": actor,
                "script": script,
                "score": round(score, 2),
                "scenario": scenario_text,
                "source_row": index + 2,
            }
        )

    return script_rows, failed_rows, errors


def _build_bulk_upload_rows_from_document(
    *,
    paragraphs: list[str],
    filename: Optional[str],
) -> tuple[list[dict[str, Any]], int, list[str], Optional[str]]:
    cleaned_paragraphs = [str(paragraph or "").strip() for paragraph in paragraphs if str(paragraph or "").strip()]
    if not cleaned_paragraphs:
        raise HTTPException(
            status_code=400,
            detail="The uploaded document is empty. Add a description paragraph followed by Actor, Script, Score, and Scenario lines.",
        )

    first_line_columns = _parse_document_row_columns(cleaned_paragraphs[0])
    if first_line_columns and _looks_like_bulk_upload_header(first_line_columns):
        description = _normalize_uploaded_document_title(filename)
        candidate_lines = cleaned_paragraphs
    else:
        description = cleaned_paragraphs[0]
        candidate_lines = cleaned_paragraphs[1:] if len(cleaned_paragraphs) > 1 else cleaned_paragraphs
    script_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    failed_rows = 0

    for index, line in enumerate(candidate_lines, start=2 if len(cleaned_paragraphs) > 1 else 1):
        columns = _parse_document_row_columns(line)
        if not columns:
            continue
        if _looks_like_bulk_upload_header(columns):
            continue

        actor, script, score_raw, scenario_text = columns[:4]
        actor = actor.strip()
        script = script.strip()
        scenario_text = scenario_text.strip()

        if not actor or not script or not scenario_text:
            failed_rows += 1
            errors.append(f"Line {index}: Actor, Script, and Scenario are required.")
            continue

        if _is_csr_actor(actor):
            try:
                score = float(score_raw)
            except Exception:
                failed_rows += 1
                errors.append(f"Line {index}: CSR rows need a numeric Score.")
                continue
            if score < 0:
                failed_rows += 1
                errors.append(f"Line {index}: Score must be zero or greater.")
                continue
        else:
            try:
                score = float(score_raw) if str(score_raw).strip() else 0.0
            except Exception:
                score = 0.0

        script_rows.append(
            {
                "actor": actor,
                "script": script,
                "score": round(max(score, 0.0), 2),
                "scenario": scenario_text,
                "source_row": index,
            }
        )

    if not script_rows:
        example_title = _normalize_uploaded_document_title(filename)
        raise HTTPException(
            status_code=400,
            detail=(
                "No structured scenario rows were found in the document. "
                "Use the first paragraph as the description, then add lines like "
                "`Actor | Script | Score | Scenario`. "
                f"Example title: {example_title}."
            ),
        )

    return script_rows, failed_rows, errors, description


def _normalize_script_rows_payload(
    rows: list[CallSimulationScenarioRowInput | dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row, CallSimulationScenarioRowInput):
            actor_name = row.actor_name
            script = row.script
            scenario_label = row.scenario
            score_value = row.score
            audio_url = row.audio_url
        else:
            actor_name = row.get("actor_name") or row.get("actor") or ""
            script = row.get("script") or ""
            scenario_label = row.get("scenario") or ""
            score_value = row.get("score")
            audio_url = row.get("audio_url")

        actor_text = str(actor_name or "").strip()
        script_text = str(script or "").strip()
        scenario_text = str(scenario_label or "").strip()
        audio_url_text = str(audio_url or "").strip()
        if not actor_text or not scenario_text:
            continue
        if not script_text and not audio_url_text:
            continue
        if _is_csr_actor(actor_text) and not script_text:
            continue

        normalized_rows.append(
            {
                "row_index": int(row.get("row_index") or row.get("source_row") or index) if isinstance(row, dict) else index,
                "actor": actor_text,
                "actor_name": actor_text,
                "script": script_text,
                "score": _coerce_nonnegative_float(score_value, fallback=0.0),
                "scenario": scenario_text,
                "audio_url": audio_url_text or None,
            }
        )

    return normalized_rows


def _build_call_simulation_row_assets(
    *,
    rows: list[CallSimulationScenarioRowInput | dict[str, Any]],
    expected_keywords: Optional[list[str]] = None,
) -> dict[str, Any]:
    normalized_rows = _normalize_script_rows_payload(rows)
    grouped_rows = _group_script_rows(normalized_rows)
    if not grouped_rows:
        raise HTTPException(
            status_code=400,
            detail="Add at least one Scenario group with a scored CSR Actor/Script row.",
        )

    aggregate_keywords = _normalize_keyword_list(expected_keywords)
    script_flow: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    variations: list[dict[str, Any]] = []

    for group_index, group in enumerate(grouped_rows):
        if len(group["member_rows"]) > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Scenario group {group['scenario_key']} contains multiple Member rows. "
                    "Each Scenario Group can contain only one Member row."
                ),
            )
        canonical_variant = max(
            group["csr_variants"],
            key=lambda item: (
                float(item.get("score") or 0.0),
                -int(item.get("row_index") or 0),
            ),
        )
        point_value = round(
            max(float(item.get("score") or 0.0) for item in group["csr_variants"]),
            2,
        )
        member_script = " ".join(
            str(item.get("script") or "").strip()
            for item in group["member_rows"]
            if str(item.get("script") or "").strip()
        ).strip()
        member_audio_url = next(
            (
                str(item.get("audio_url") or "").strip()
                for item in group["member_rows"]
                if str(item.get("audio_url") or "").strip()
            ),
            None,
        )
        member_actor_name = (
            str(group["member_rows"][0].get("actor_name") or "").strip()
            if group["member_rows"]
            else "Member"
        )
        is_last_group = group_index == len(grouped_rows) - 1
        has_member_reply = bool(member_script or member_audio_url)

        script_flow.append(
            {
                "step_id": f"scenario-{group['scenario_key']}",
                "suggested_csr_script": canonical_variant["script"],
                "member_response_text": member_script if has_member_reply else "",
                "point_value": point_value,
                "expected_keywords": aggregate_keywords,
                "actor_name": member_actor_name,
                "next_actor_name": member_actor_name if has_member_reply else None,
                "scenario": group["scenario_key"],
                "member_audio_url": member_audio_url if has_member_reply else None,
                "accepted_variants": [
                    {
                        "actor_name": row["actor_name"],
                        "script": row["script"],
                        "score": row["score"],
                        "scenario": row["scenario"],
                    }
                    for row in group["csr_variants"]
                ],
            }
        )

        steps.append(
            {
                "step_number": len(steps) + 1,
                "actor": "csr",
                "speaker_label": "CSR",
                "script": canonical_variant["script"],
                "expected_keywords": aggregate_keywords,
                "audio_url": None,
                "response_time_limit": None,
                "is_closing": is_last_group and not has_member_reply,
                "metadata": {
                    "point_value": point_value,
                    "script_flow_step_id": f"scenario-{group['scenario_key']}",
                    "actor_name": member_actor_name,
                    "scenario": member_script or group["scenario_key"],
                    "scenario_group": group["scenario_key"],
                    "accepted_variants": [
                        {
                            "actor_name": row["actor_name"],
                            "script": row["script"],
                            "score": row["score"],
                            "scenario": row["scenario"],
                        }
                        for row in group["csr_variants"]
                    ],
                    "member_script": member_script,
                    "member_audio_url": member_audio_url,
                },
            }
        )

        if has_member_reply:
            steps.append(
                {
                    "step_number": len(steps) + 1,
                    "actor": "member",
                    "speaker_label": member_actor_name,
                    "script": member_script,
                    "expected_keywords": [],
                    "audio_url": member_audio_url,
                    "response_time_limit": None,
                    "is_closing": is_last_group,
                    "metadata": {
                        "point_value": 0.0,
                        "actor_name": member_actor_name,
                        "scenario": member_script or group["scenario_key"],
                        "scenario_group": group["scenario_key"],
                        "member_script": member_script,
                        "member_audio_url": member_audio_url,
                    },
                }
            )

        variations.extend(
            [
                {
                    "actor_name": row["actor_name"],
                    "script": row["script"],
                    "score": row["score"],
                    "branching_logic": row["scenario"],
                }
                for row in group["csr_variants"]
            ]
        )

    first_group = grouped_rows[0]
    first_member_row = first_group["member_rows"][0] if first_group["member_rows"] else None
    first_csr_row = first_group["csr_variants"][0]

    return {
        "script_rows": normalized_rows,
        "scenario_groups": grouped_rows,
        "script_flow": script_flow,
        "steps": steps,
        "variations": variations,
        "first_member_row": first_member_row,
        "first_csr_row": first_csr_row,
    }


def _read_step_point_value(step: Optional[CallSimulationScenarioStepResponse | dict[str, Any]]) -> float:
    if not step:
        return 0.0
    metadata = None
    if isinstance(step, dict):
        metadata = _normalize_json_object(step.get("metadata"))
    else:
        metadata = _normalize_json_object(getattr(step, "metadata", None))

    candidate = metadata.get("point_value")
    try:
        resolved = float(candidate)
    except (TypeError, ValueError):
        resolved = 0.0
    return round(max(0.0, resolved), 2)


def _read_step_script_variants(
    step: Optional[CallSimulationScenarioStepResponse | dict[str, Any]],
) -> list[dict[str, Any]]:
    if not step:
        return []

    metadata = step.get("metadata") if isinstance(step, dict) else getattr(step, "metadata", None)
    metadata_object = _normalize_json_object(metadata)
    variants = metadata_object.get("accepted_variants")
    if isinstance(variants, list):
        normalized_variants: list[dict[str, Any]] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            script = str(variant.get("script") or "").strip()
            if not script:
                continue
            normalized_variants.append(
                {
                    "script": script,
                    "score": round(max(0.0, float(variant.get("score") or 0.0)), 2),
                    "actor_name": str(variant.get("actor_name") or "").strip() or "CSR",
                }
            )
        if normalized_variants:
            return normalized_variants

    fallback_script = step.get("script") if isinstance(step, dict) else getattr(step, "script", "")
    fallback_score = _read_step_point_value(step)
    if str(fallback_script or "").strip():
        return [
            {
                "script": str(fallback_script).strip(),
                "score": fallback_score,
                "actor_name": "CSR",
            }
        ]
    return []


def _select_best_script_variant(
    step: CallSimulationScenarioStepResponse,
    transcript: str,
) -> dict[str, Any]:
    variants = _read_step_script_variants(step)
    cleaned_transcript = normalize_text(transcript or "")
    if not variants:
        fallback_script = (step.script or "").strip()
        return {
            "script": fallback_script,
            "score": _read_step_point_value(step),
            "similarity": _phrase_similarity(cleaned_transcript, normalize_text(fallback_script)) if cleaned_transcript and fallback_script else 0.0,
        }

    best_variant = max(
        variants,
        key=lambda variant: (
            _phrase_similarity(cleaned_transcript, normalize_text(str(variant.get("script") or ""))),
            float(variant.get("score") or 0.0),
        ),
    )
    best_script = str(best_variant.get("script") or "").strip()
    return {
        "script": best_script,
        "score": round(max(0.0, float(best_variant.get("score") or 0.0)), 2),
        "similarity": _phrase_similarity(cleaned_transcript, normalize_text(best_script)) if cleaned_transcript and best_script else 0.0,
    }


def _read_step_scenario_label(step: Optional[CallSimulationScenarioStepResponse | dict[str, Any]]) -> Optional[str]:
    if not step:
        return None
    metadata = None
    if isinstance(step, dict):
        metadata = _normalize_json_object(step.get("metadata"))
    else:
        metadata = _normalize_json_object(getattr(step, "metadata", None))

    for key in ("scenario", "scenario_label", "scenario_text", "member_context"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_step_actor_label(step: Optional[CallSimulationScenarioStepResponse | dict[str, Any]]) -> Optional[str]:
    if not step:
        return None
    if isinstance(step, dict):
        speaker_label = step.get("speaker_label")
        if isinstance(speaker_label, str) and speaker_label.strip():
            return speaker_label.strip()
        metadata = _normalize_json_object(step.get("metadata"))
    else:
        speaker_label = getattr(step, "speaker_label", None)
        if isinstance(speaker_label, str) and speaker_label.strip():
            return speaker_label.strip()
        metadata = _normalize_json_object(getattr(step, "metadata", None))

    actor_label = metadata.get("actor_name")
    if isinstance(actor_label, str) and actor_label.strip():
        return actor_label.strip()
    return None


def _find_keyword_matches(text: str, keywords: Optional[list[str]]) -> list[str]:
    if not text or not keywords:
        return []
    normalized_text = text.lower()
    matches = {
        keyword.strip()
        for keyword in keywords
        if keyword and keyword.strip() and keyword.strip().lower() in normalized_text
    }
    return sorted(matches)


def _detect_forbidden_words(text: str, keywords: Optional[list[str]]) -> list[str]:
    return _find_keyword_matches(text, keywords)


def _score_aht(actual_seconds: int, target_seconds: int) -> float:
    if target_seconds <= 0:
        return 100.0
    if actual_seconds <= target_seconds:
        return 100.0
    overage_ratio = (actual_seconds - target_seconds) / target_seconds
    return round(max(0.0, 100.0 - (overage_ratio * 100.0)), 2)


def _score_target_delta(actual: float, target: float, penalty_multiplier: float = 100.0) -> float:
    if target <= 0:
        return 100.0
    delta_ratio = abs(actual - target) / target
    return round(max(0.0, 100.0 - (delta_ratio * penalty_multiplier)), 2)


def _score_dead_air(dead_air_seconds: float, target_dead_air_seconds: float) -> float:
    if target_dead_air_seconds <= 0:
        return 100.0 if dead_air_seconds <= 0 else 0.0
    if dead_air_seconds <= target_dead_air_seconds:
        return 100.0
    overage_ratio = (dead_air_seconds - target_dead_air_seconds) / target_dead_air_seconds
    return round(max(0.0, 100.0 - (overage_ratio * 55.0)), 2)


def _score_behavioral_hits(hit_count: int, keywords: Optional[list[str]]) -> float:
    normalized_keywords = _normalize_keyword_list(keywords)
    if not normalized_keywords:
        return 100.0
    target_hits = max(1, min(3, len(normalized_keywords)))
    return round(min(100.0, (hit_count / target_hits) * 100.0), 2)


def _estimate_dead_air_seconds(word_feedback: list[dict[str, Any]], response_duration: float) -> float:
    timed_words = []
    for item in word_feedback or []:
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            timed_words.append((float(start), float(end)))

    if not timed_words or response_duration <= 0:
        return 0.0

    timed_words.sort(key=lambda item: item[0])
    dead_air = 0.0
    previous_end = 0.0
    for start, end in timed_words:
        gap = max(0.0, start - previous_end)
        if gap > 0.75:
            dead_air += gap
        previous_end = max(previous_end, end)

    tail_gap = max(0.0, response_duration - previous_end)
    if tail_gap > 1.0:
        dead_air += tail_gap

    return round(dead_air, 2)


def _total_kpi_weight(config: BatchKPIConfigCreate | BatchKPIConfigUpdate | BatchKPIConfig) -> float:
    return round(
        float(getattr(config, "speech_to_text_weight", 0.0) or 0.0)
        + float(getattr(config, "aht_weight", 0.0) or 0.0)
        + float(getattr(config, "rate_of_speech_weight", 0.0) or 0.0)
        + float(getattr(config, "dead_air_weight", 0.0) or 0.0)
        + float(getattr(config, "empathy_statements_weight", 0.0) or 0.0)
        + float(getattr(config, "probing_questions_weight", 0.0) or 0.0)
        + float(getattr(config, "grammar_weight", 0.0) or 0.0)
        + float(getattr(config, "pronunciation_weight", 0.0) or 0.0)
        + float(getattr(config, "pacing_weight", 0.0) or 0.0),
        2,
    )


def _validate_kpi_weights(config: BatchKPIConfigCreate | BatchKPIConfigUpdate | BatchKPIConfig) -> None:
    total_weight = _total_kpi_weight(config)
    if total_weight < 95.0 or total_weight > 105.0:
        raise HTTPException(
            status_code=400,
            detail=f"KPI weights should total about 100%. Current total: {total_weight}%",
        )


def _average(values: list[float]) -> float:
    cleaned = [float(value) for value in values if value is not None]
    return round(sum(cleaned) / len(cleaned), 2) if cleaned else 0.0


def _format_asr_provider_label(provider: Optional[str]) -> str:
    labels = {
        "google_speech_to_text": "Google Speech-to-Text",
        "openai_whisper": "OpenAI Whisper",
        "heuristic_fallback": "Transcript Assist Fallback",
    }
    normalized = (provider or "").strip().lower()
    return labels.get(normalized, normalized.replace("_", " ").title() or "Unknown ASR")


def _estimate_script_duration_seconds(script: str) -> float:
    tokens = tokenize_text(script or "")
    if not tokens:
        return 1.5
    # Simulated member turns do not have reliable duration metadata, so estimate from word count.
    return round(max(1.5, len(tokens) / 2.6), 2)


def _phrase_similarity(left: str, right: str) -> float:
    normalized_left = normalize_text(left or "")
    normalized_right = normalize_text(right or "")
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _phrase_present(transcript: str, phrase: str) -> bool:
    normalized_transcript = normalize_text(transcript or "")
    normalized_phrase = normalize_text(phrase or "")
    if not normalized_transcript or not normalized_phrase:
        return False

    if normalized_phrase in normalized_transcript:
        return True

    phrase_tokens = [token for token in tokenize_text(normalized_phrase) if len(token) > 2]
    if not phrase_tokens:
        return False

    transcript_tokens = set(tokenize_text(normalized_transcript))
    matched_tokens = sum(1 for token in phrase_tokens if token in transcript_tokens)
    coverage = matched_tokens / len(phrase_tokens)
    return coverage >= 0.6 or _phrase_similarity(normalized_transcript, normalized_phrase) >= 0.55


def _extract_closing_spiel(steps: list[CallSimulationScenarioStepResponse]) -> str:
    closing_step = next(
        (
            step
            for step in reversed(steps)
            if step.actor == "csr" and (step.is_closing or step.script)
        ),
        None,
    )
    return (closing_step.script or "").strip() if closing_step else ""


def _determine_repeat_requirement(
    *,
    transcript: str,
    step: CallSimulationScenarioStepResponse,
    evaluation: dict[str, Any],
    assessment: dict[str, Any],
    matched_keywords: list[str],
) -> tuple[bool, Optional[str], float, dict[str, Any]]:
    cleaned_transcript = normalize_text(transcript or "")
    selected_variant = _select_best_script_variant(step, transcript)
    cleaned_script = normalize_text(str(selected_variant.get("script") or ""))
    similarity = float(selected_variant.get("similarity") or 0.0)

    if not cleaned_transcript:
        return True, "No recognizable speech was detected for this spiel.", similarity, selected_variant

    if assessment.get("status") != "completed":
        fallback_reason = str(assessment.get("error") or "").strip() or "Speech recognition could not validate the spiel."
        return True, fallback_reason, similarity, selected_variant

    speech_accuracy = float(evaluation.get("speech_to_text_accuracy") or 0.0)
    required_keywords = _normalize_keyword_list(step.expected_keywords)
    keyword_coverage = (
        len(matched_keywords) / len(required_keywords)
        if required_keywords
        else 1.0
    )

    accuracy_failed = speech_accuracy < MIN_STT_ACCURACY_FOR_PROGRESS
    similarity_failed = similarity < MIN_SCRIPT_SIMILARITY_FOR_PROGRESS
    keyword_failed = bool(required_keywords) and keyword_coverage < MIN_KEYWORD_COVERAGE_FOR_PROGRESS

    strong_script_match = similarity >= 0.82
    enough_keyword_coverage = not required_keywords or keyword_coverage >= MIN_KEYWORD_COVERAGE_FOR_PROGRESS
    should_repeat = (
        not strong_script_match
        and (
            similarity < 0.46
            or (similarity_failed and accuracy_failed)
            or (keyword_failed and similarity < 0.74)
        )
    )

    if not should_repeat and enough_keyword_coverage:
        return False, None, similarity, selected_variant

    reasons: list[str] = []
    if similarity_failed:
        reasons.append("the saved response did not follow the scripted spiel closely enough")
    if accuracy_failed:
        reasons.append("the recognized words were still too far from the target script")
    if keyword_failed:
        reasons.append("required keywords were missing")

    reason_text = "; ".join(reasons) if reasons else "the response needs to be repeated"
    return True, reason_text[0].upper() + reason_text[1:], similarity, selected_variant


def _build_keyword_compliance_summary(
    transcript: str,
    steps: list[CallSimulationScenarioStepResponse],
) -> dict[str, Any]:
    required_items = [
        {
            "id": "thank_you_for_calling",
            "label": "Thank you for calling",
            "required_phrase": "Thank you for calling",
        },
        {
            "id": "how_can_i_help",
            "label": "How can I help",
            "required_phrase": "How can I help",
        },
    ]

    closing_spiel = _extract_closing_spiel(steps)
    if closing_spiel:
        required_items.append(
            {
                "id": "closing_spiel",
                "label": "Closing spiel",
                "required_phrase": closing_spiel,
            }
        )

    matched_count = 0
    items: list[dict[str, Any]] = []
    for item in required_items:
        matched = _phrase_present(transcript, item["required_phrase"])
        matched_count += 1 if matched else 0
        items.append(
            {
                **item,
                "matched": matched,
            }
        )

    total_required = len(items)
    score = round((matched_count / total_required) * 100.0, 2) if total_required else 100.0
    missing = [item["label"] for item in items if not item["matched"]]

    return {
        "score": score,
        "matched_count": matched_count,
        "total_required": total_required,
        "missing": missing,
        "items": items,
    }


def _heuristic_sentiment_score(transcript: str) -> float:
    positive_tokens = {
        "thank",
        "glad",
        "happy",
        "help",
        "assist",
        "resolved",
        "appreciate",
        "sorry",
        "certainly",
        "absolutely",
    }
    negative_tokens = {
        "can't",
        "cannot",
        "won't",
        "problem",
        "issue",
        "delay",
        "angry",
        "upset",
        "frustrated",
        "no",
    }
    tokens = tokenize_text(transcript or "")
    if not tokens:
        return 0.0

    positive_hits = sum(1 for token in tokens if token in positive_tokens)
    negative_hits = sum(1 for token in tokens if token in negative_tokens)
    raw_score = (positive_hits - negative_hits) / max(len(tokens), 1)
    return round(max(-1.0, min(1.0, raw_score * 5.0)), 3)


def _analyze_sentiment_score(transcript: str) -> float:
    if not transcript or not transcript.strip():
        return 0.0

    api_key = (
        os.getenv("GOOGLE_NATURAL_LANGUAGE_API_KEY")
        or os.getenv("GOOGLE_CLOUD_NATURAL_LANGUAGE_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if api_key:
        try:
            response = requests.post(
                "https://language.googleapis.com/v1/documents:analyzeSentiment",
                params={"key": api_key},
                json={
                    "document": {
                        "type": "PLAIN_TEXT",
                        "language": "en",
                        "content": transcript,
                    },
                    "encodingType": "UTF8",
                },
                timeout=25,
            )
            response.raise_for_status()
            payload = response.json()
            document_sentiment = payload.get("documentSentiment") or {}
            score = document_sentiment.get("score")
            if isinstance(score, (int, float)):
                return round(max(-1.0, min(1.0, float(score))), 3)
        except Exception as exc:
            logger.warning("Google Natural Language sentiment fallback engaged: %s", exc)

    return _heuristic_sentiment_score(transcript)


def _sentiment_label(score: float) -> str:
    if score >= 0.3:
        return "positive"
    if score <= -0.3:
        return "at_risk"
    return "neutral"


def _build_default_kpi_config(batch_id: Optional[str] = None) -> BatchKPIConfig:
    return BatchKPIConfig(
        id="default-call-simulation-kpi",
        batch_id=batch_id or "default-batch",
        speech_to_text_weight=25.0,
        aht_weight=20.0,
        rate_of_speech_weight=15.0,
        dead_air_weight=15.0,
        empathy_statements_weight=10.0,
        probing_questions_weight=10.0,
        grammar_weight=2.5,
        pronunciation_weight=1.0,
        pacing_weight=1.0,
        forbidden_words_penalty=5.0,
        passing_score=DEFAULT_PASSING_SCORE,
        forbidden_words=[],
        empathy_keywords=[],
        probing_keywords=[],
        target_aht_seconds=120,
        target_ros_words_per_min=150.0,
        target_dead_air_seconds=3.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def _get_effective_kpi_config(db: Session, batch_id: Optional[str]) -> BatchKPIConfig:
    if not batch_id:
        return _build_default_kpi_config()

    config = db.query(BatchKPIConfig).filter(BatchKPIConfig.batch_id == batch_id).first()
    if config:
        return config
    return _build_default_kpi_config(batch_id=batch_id)


def _build_batch_target_kpis_payload(
    kpi_config: BatchKPIConfig,
    *,
    batch_id: Optional[str] = None,
) -> dict[str, Any]:
    resolved_batch_id = batch_id or getattr(kpi_config, "batch_id", None)
    target_aht_seconds = int(kpi_config.target_aht_seconds or 120)
    return {
        "speech_to_text_weight": float(kpi_config.speech_to_text_weight or 0.0),
        "aht_weight": float(kpi_config.aht_weight or 0.0),
        "rate_of_speech_weight": float(kpi_config.rate_of_speech_weight or 0.0),
        "dead_air_weight": float(kpi_config.dead_air_weight or 0.0),
        "empathy_statements_weight": float(kpi_config.empathy_statements_weight or 0.0),
        "probing_questions_weight": float(kpi_config.probing_questions_weight or 0.0),
        "grammar_weight": float(kpi_config.grammar_weight or 0.0),
        "pronunciation_weight": float(kpi_config.pronunciation_weight or 0.0),
        "pacing_weight": float(kpi_config.pacing_weight or 0.0),
        "forbidden_words_penalty": float(kpi_config.forbidden_words_penalty or 0.0),
        "passing_score": float(kpi_config.passing_score or DEFAULT_PASSING_SCORE),
        "target_aht_seconds": target_aht_seconds,
        "aht_seconds": target_aht_seconds,
        "target_ros_words_per_min": float(kpi_config.target_ros_words_per_min or 150.0),
        "target_dead_air_seconds": float(kpi_config.target_dead_air_seconds or 3.0),
        "forbidden_words": _normalize_keyword_list(kpi_config.forbidden_words),
        "empathy_keywords": _normalize_keyword_list(kpi_config.empathy_keywords),
        "probing_keywords": _normalize_keyword_list(kpi_config.probing_keywords),
        "batch_id": resolved_batch_id,
        "kpi_source": "batch_kpi_config",
    }


def _apply_batch_kpi_to_call_simulation_config(
    config: Optional[dict[str, Any]],
    *,
    kpi_config: BatchKPIConfig,
    batch_id: Optional[str] = None,
) -> dict[str, Any]:
    resolved_config = _normalize_json_object(config)
    existing_target_kpis = _normalize_json_object(resolved_config.get("target_kpis"))
    configured_passing_score = _read_call_scenario_passing_score_from_config(resolved_config)
    passing_score = configured_passing_score or float(kpi_config.passing_score or DEFAULT_PASSING_SCORE)
    return {
        **resolved_config,
        "target_kpis": {
            **existing_target_kpis,
            **_build_batch_target_kpis_payload(
                kpi_config,
                batch_id=batch_id,
            ),
            "passing_score": passing_score,
        },
        "passing_score": passing_score,
        "certification_threshold": passing_score,
    }


def _resolve_call_simulation_config_for_batch(
    db: Session,
    scenario: Optional[Scenario],
    *,
    batch_id: Optional[str] = None,
) -> dict[str, Any]:
    config = _resolve_call_simulation_config(scenario)
    if not batch_id:
        return config

    effective_kpi = _get_effective_kpi_config(db, batch_id)
    return _apply_batch_kpi_to_call_simulation_config(
        config,
        kpi_config=effective_kpi,
        batch_id=batch_id,
    )


def _calculate_weighted_score(
    *,
    speech_to_text_accuracy: float,
    aht_score: float,
    rate_of_speech_score: float,
    dead_air_score: float,
    empathy_score: float,
    probing_score: float,
    grammar_score: float,
    pronunciation_score: float,
    pacing_score: float,
    forbidden_word_penalty: float,
    kpi_config: BatchKPIConfig,
) -> float:
    weighted = (
        speech_to_text_accuracy * (float(kpi_config.speech_to_text_weight or 0.0) / 100.0)
        + aht_score * (float(kpi_config.aht_weight or 0.0) / 100.0)
        + rate_of_speech_score * (float(kpi_config.rate_of_speech_weight or 0.0) / 100.0)
        + dead_air_score * (float(kpi_config.dead_air_weight or 0.0) / 100.0)
        + empathy_score * (float(kpi_config.empathy_statements_weight or 0.0) / 100.0)
        + probing_score * (float(kpi_config.probing_questions_weight or 0.0) / 100.0)
        + grammar_score * (float(kpi_config.grammar_weight or 0.0) / 100.0)
        + pronunciation_score * (float(kpi_config.pronunciation_weight or 0.0) / 100.0)
        + pacing_score * (float(kpi_config.pacing_weight or 0.0) / 100.0)
    )
    return round(max(0.0, weighted - forbidden_word_penalty), 2)


def _build_scenario_assignment_summaries(
    db: Session,
    scenario_id: str,
    mappings: Optional[list[BatchScenarioMapping]] = None,
    *,
    include_empty_batches: bool = False,
) -> list[CallSimulationScenarioAssignmentSummary]:
    active_mappings = mappings
    if active_mappings is None:
        active_mappings = (
            db.query(BatchScenarioMapping)
            .filter(
                BatchScenarioMapping.scenario_id == scenario_id,
                BatchScenarioMapping.is_active == True,
            )
            .order_by(BatchScenarioMapping.assigned_at.desc())
            .all()
        )

    if not active_mappings:
        return []

    batch_ids = [mapping.batch_id for mapping in active_mappings]
    batches = {
        batch.id: batch
        for batch in db.query(Batch).filter(Batch.id.in_(batch_ids)).all()
    }
    assignments = (
        db.query(CallSimulationAssignment)
        .filter(
            CallSimulationAssignment.scenario_id == scenario_id,
            CallSimulationAssignment.batch_id.in_(batch_ids),
            CallSimulationAssignment.is_active == True,
        )
        .all()
    )
    assigned_trainees_by_batch: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        if assignment.batch_id and assignment.trainee_id:
            assigned_trainees_by_batch[assignment.batch_id].add(assignment.trainee_id)

    sessions = (
        db.query(SimSession)
        .filter(
            SimSession.scenario_id == scenario_id,
            SimSession.batch_id.in_(batch_ids),
            SimSession.status.in_(["completed", "failed"]),
        )
        .all()
    )
    sessions_by_batch: dict[str, list[SimSession]] = defaultdict(list)
    for session in sessions:
        if session.batch_id:
            sessions_by_batch[session.batch_id].append(session)

    summaries: list[CallSimulationScenarioAssignmentSummary] = []
    for mapping in active_mappings:
        batch = batches.get(mapping.batch_id)
        if not batch:
            continue

        trainee_count = len(assigned_trainees_by_batch.get(batch.id, set()))
        if trainee_count == 0 and not include_empty_batches:
            continue

        batch_sessions = sessions_by_batch.get(batch.id, [])
        completed_sessions = len(batch_sessions)
        passed_sessions = sum(1 for session in batch_sessions if session.pass_fail)
        latest_completed_at = max(
            (
                session.completed_at or session.created_at
                for session in batch_sessions
                if session.completed_at or session.created_at
            ),
            default=None,
        )
        summaries.append(
            CallSimulationScenarioAssignmentSummary(
                batch_id=batch.id,
                batch_name=batch.name,
                wave_number=batch.wave_number,
                assigned_at=mapping.assigned_at,
                trainee_count=trainee_count,
                completed_sessions=completed_sessions,
                passed_sessions=passed_sessions,
                average_score=round(
                    sum(float(session.weighted_score or 0.0) for session in batch_sessions) / completed_sessions,
                    1,
                )
                if completed_sessions
                else 0.0,
                pass_rate=round((passed_sessions / completed_sessions) * 100.0, 1)
                if completed_sessions
                else 0.0,
                latest_completed_at=latest_completed_at,
            )
        )

    return summaries


def _aggregate_assignment_summaries(
    summaries: list[CallSimulationScenarioAssignmentSummary],
) -> dict[str, Any]:
    completed_sessions = sum(summary.completed_sessions for summary in summaries)
    passed_sessions = sum(summary.passed_sessions for summary in summaries)
    latest_completed_at = max(
        (summary.latest_completed_at for summary in summaries if summary.latest_completed_at),
        default=None,
    )
    return {
        "member_count": sum(summary.trainee_count for summary in summaries),
        "completed_sessions": completed_sessions,
        "passed_sessions": passed_sessions,
        "average_score": round(
            sum(summary.average_score * summary.completed_sessions for summary in summaries)
            / completed_sessions,
            1,
        )
        if completed_sessions
        else 0.0,
        "pass_rate": round((passed_sessions / completed_sessions) * 100.0, 1)
        if completed_sessions
        else 0.0,
        "latest_completed_at": latest_completed_at,
    }


def _resolve_call_simulation_assignment_trainer_id(
    db: Session,
    *,
    mapping: Optional[BatchScenarioMapping] = None,
    scenario: Optional[Scenario] = None,
    batch: Optional[Batch] = None,
) -> Optional[str]:
    candidate_ids = [
        mapping.assigned_by if mapping else None,
        scenario.created_by if scenario else None,
        batch.created_by if batch else None,
    ]
    for candidate_id in candidate_ids:
        if candidate_id:
            return candidate_id

    fallback_trainer = (
        db.query(User)
        .filter(
            User.role.in_([UserRole.ADMIN, UserRole.TRAINER]),
            User.is_active == True,
        )
        .order_by(User.created_at.asc())
        .first()
    )
    return fallback_trainer.id if fallback_trainer else None


def _upsert_call_simulation_assignment(
    *,
    assignment: Optional[CallSimulationAssignment],
    scenario_id: str,
    trainee_id: str,
    trainer_id: str,
    batch_id: Optional[str],
    assigned_at: Optional[datetime],
    max_attempts: Optional[int] = None,
) -> tuple[CallSimulationAssignment, bool]:
    created = assignment is None
    resolved_max_attempts = _coerce_positive_int(
        max_attempts,
        fallback=_coerce_positive_int(DEFAULT_MAX_ATTEMPTS, fallback=3),
    )
    if assignment is None:
        assignment = CallSimulationAssignment(
            id=str(uuid.uuid4()),
            scenario_id=scenario_id,
            trainee_id=trainee_id,
            assigned_by=trainer_id,
            batch_id=batch_id,
            max_attempts=resolved_max_attempts,
            assigned_at=assigned_at or datetime.utcnow(),
            is_active=True,
        )
        return assignment, True

    changed = False
    next_assigned_at = assigned_at or assignment.assigned_at or datetime.utcnow()
    if assignment.assigned_by != trainer_id:
        assignment.assigned_by = trainer_id
        changed = True
    if assignment.batch_id != batch_id:
        assignment.batch_id = batch_id
        changed = True
    if int(assignment.max_attempts or 0) != resolved_max_attempts:
        assignment.max_attempts = resolved_max_attempts
        changed = True
    if assignment.assigned_at != next_assigned_at:
        assignment.assigned_at = next_assigned_at
        changed = True
    if not assignment.is_active:
        assignment.is_active = True
        changed = True

    return assignment, changed or created


def _get_active_call_simulation_assignment(
    db: Session,
    *,
    trainee_id: str,
    scenario_id: str,
) -> Optional[CallSimulationAssignment]:
    return (
        db.query(CallSimulationAssignment)
        .filter(
            CallSimulationAssignment.trainee_id == trainee_id,
            CallSimulationAssignment.scenario_id == scenario_id,
            CallSimulationAssignment.is_active == True,
        )
        .order_by(CallSimulationAssignment.assigned_at.desc(), CallSimulationAssignment.updated_at.desc())
        .first()
    )


def _sync_call_simulation_assignments_for_scenario(
    db: Session,
    scenario_id: str,
) -> bool:
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        return False

    existing_assignments = (
        db.query(CallSimulationAssignment)
        .filter(CallSimulationAssignment.scenario_id == scenario_id)
        .all()
    )
    assignment_lookup = {
        assignment.trainee_id: assignment for assignment in existing_assignments
    }

    mappings = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.scenario_id == scenario_id,
            BatchScenarioMapping.is_active == True,
        )
        .order_by(BatchScenarioMapping.assigned_at.desc())
        .all()
    )
    batches = {
        batch.id: batch
        for batch in db.query(Batch)
        .filter(Batch.id.in_([mapping.batch_id for mapping in mappings] or ["__none__"]))
        .all()
    }
    did_change = False

    active_mapping_ids = {mapping.batch_id for mapping in mappings if mapping.batch_id}
    valid_trainee_ids_by_batch: dict[str, set[str]] = defaultdict(set)
    for batch_id, batch in batches.items():
        valid_trainee_ids_by_batch[batch_id] = {
            trainee.id
            for trainee in batch.users
            if trainee.role == UserRole.TRAINEE and trainee.is_active
        }

    for assignment in assignment_lookup.values():
        is_valid = bool(scenario.is_published)
        if assignment.batch_id:
            is_valid = (
                is_valid
                and assignment.batch_id in active_mapping_ids
                and assignment.trainee_id in valid_trainee_ids_by_batch.get(assignment.batch_id, set())
            )
        else:
            trainee = db.query(User).filter(User.id == assignment.trainee_id).first()
            is_valid = is_valid and bool(trainee and trainee.role == UserRole.TRAINEE and trainee.is_active)

        if assignment.is_active != is_valid:
            assignment.is_active = is_valid
            did_change = True

    return did_change


def _deactivate_scenario_variations(
    db: Session,
    *,
    scenario_id: str,
) -> list[ScenarioVariation]:
    """Retire active variations without breaking historical sim-session links."""
    variations = (
        db.query(ScenarioVariation)
        .filter(ScenarioVariation.scenario_id == scenario_id)
        .all()
    )
    for variation in variations:
        variation.is_active = False
    return variations


def _sync_call_simulation_assignments_for_trainee(
    db: Session,
    trainee: User,
) -> bool:
    existing_assignments = (
        db.query(CallSimulationAssignment)
        .filter(CallSimulationAssignment.trainee_id == trainee.id)
        .all()
    )
    assignment_lookup = {
        (assignment.scenario_id, assignment.batch_id): assignment
        for assignment in existing_assignments
        if assignment.scenario_id and assignment.batch_id
    }

    batch_lookup = {batch.id: batch for batch in trainee.batches if batch.id}
    mappings = (
        db.query(BatchScenarioMapping)
        .join(Scenario, Scenario.id == BatchScenarioMapping.scenario_id)
        .filter(
            BatchScenarioMapping.batch_id.in_(list(batch_lookup.keys())),
            BatchScenarioMapping.is_active == True,
            Scenario.is_published == True,
        )
        .order_by(BatchScenarioMapping.assigned_at.desc())
        .all()
    )
    scenario_lookup = {
        scenario.id: scenario
        for scenario in db.query(Scenario)
        .filter(Scenario.id.in_([mapping.scenario_id for mapping in mappings] or ["__none__"]))
        .all()
    }

    latest_mapping_by_scenario: dict[str, BatchScenarioMapping] = {}
    for mapping in mappings:
        if mapping.scenario_id not in latest_mapping_by_scenario:
            latest_mapping_by_scenario[mapping.scenario_id] = mapping

    did_change = False
    valid_mapping_pairs = {
        (mapping.scenario_id, mapping.batch_id)
        for mapping in latest_mapping_by_scenario.values()
    }

    for mapping in latest_mapping_by_scenario.values():
        scenario = scenario_lookup.get(mapping.scenario_id)
        if not scenario or not scenario.is_published:
            continue

        batch = batch_lookup.get(mapping.batch_id)
        if not batch:
            continue

        trainer_id = _resolve_call_simulation_assignment_trainer_id(
            db,
            mapping=mapping,
            scenario=scenario,
            batch=batch,
        )
        resolved_max_attempts = _resolve_call_assignment_max_attempts(
            None,
            scenario=scenario,
            fallback=DEFAULT_MAX_ATTEMPTS,
        )
        assignment = assignment_lookup.get((scenario.id, mapping.batch_id))
        assignment, changed = _upsert_call_simulation_assignment(
            assignment=assignment,
            scenario_id=scenario.id,
            trainee_id=trainee.id,
            trainer_id=trainer_id or trainee.id,
            batch_id=mapping.batch_id,
            assigned_at=mapping.assigned_at or datetime.utcnow(),
            max_attempts=resolved_max_attempts,
        )
        if assignment not in existing_assignments:
            db.add(assignment)
            did_change = True
        elif changed:
            did_change = True

    for assignment in existing_assignments:
        scenario = scenario_lookup.get(assignment.scenario_id) or db.query(Scenario).filter(Scenario.id == assignment.scenario_id).first()
        is_valid = bool(
            trainee.is_active
            and scenario
            and scenario.is_published
            and assignment.batch_id
            and assignment.batch_id in batch_lookup
            and (assignment.scenario_id, assignment.batch_id) in valid_mapping_pairs
        )
        if assignment.is_active != is_valid:
            assignment.is_active = is_valid
            did_change = True

    return did_change


def _sync_sim_floor_assignments_for_scenario(db: Session, scenario_id: str) -> bool:
    """Backward-compatible alias for older Call Simulation references."""
    return _sync_call_simulation_assignments_for_scenario(db, scenario_id)


def _sync_sim_floor_assignments_for_trainee(db: Session, trainee: User) -> bool:
    """Backward-compatible alias for older Call Simulation references."""
    return _sync_call_simulation_assignments_for_trainee(db, trainee)


def _get_latest_sim_session_coaching_logs(
    db: Session,
    session_ids: list[str],
) -> dict[str, CoachingLog]:
    if not session_ids:
        return {}

    logs = (
        db.query(CoachingLog)
        .filter(CoachingLog.sim_session_id.in_(session_ids))
        .order_by(CoachingLog.created_at.desc(), CoachingLog.updated_at.desc())
        .all()
    )
    latest_logs: dict[str, CoachingLog] = {}
    for log in logs:
        if log.sim_session_id and log.sim_session_id not in latest_logs:
            latest_logs[log.sim_session_id] = log
    return latest_logs


def _upsert_sim_session_coaching_log(
    db: Session,
    *,
    session: SimSession,
    trainer_id: str,
    notes: Optional[str],
    verdict_status: str,
) -> Optional[CoachingLog]:
    normalized_verdict = (verdict_status or "pending").strip().lower()
    normalized_notes = (notes or "").strip()
    requested_status = "sent" if normalized_verdict in {"competent", "retake"} else "draft"
    existing_logs = (
        db.query(CoachingLog)
        .filter(CoachingLog.sim_session_id == session.id)
        .order_by(CoachingLog.updated_at.desc(), CoachingLog.created_at.desc())
        .all()
    )
    if normalized_verdict == "pending" and not normalized_notes:
        for existing_log in existing_logs:
            db.delete(existing_log)
        return None

    log = existing_logs[0] if existing_logs else None
    if not log:
        log = CoachingLog(
            coaching_id=generate_coaching_id(db),
            trainer_id=trainer_id,
            trainee_id=session.trainee_id,
            sim_session_id=session.id,
            source_type="call_simulation_session",
        )
        db.add(log)
    else:
        for duplicate_log in existing_logs[1:]:
            db.delete(duplicate_log)

    log.source_type = "call_simulation_session"
    log.trainer_id = trainer_id
    log.trainee_id = session.trainee_id
    log.sim_session_id = session.id
    log.practice_session_id = None
    log.batch_name = session.batch.name if session.batch else None
    log.lob = session.scenario.lob if session.scenario else None
    log.coaching_minutes = log.coaching_minutes or 15
    log.trainer_remarks = normalized_notes or None
    log.status = requested_status
    log.competency_status = normalize_competency_status(
        "not_competent" if normalized_verdict == "retake" else normalized_verdict
    )
    log.updated_at = datetime.utcnow()
    if requested_status != "acknowledged":
        log.acknowledged_at = None
    return log


def _summarize_sim_session_coaching(
    latest_logs: dict[str, CoachingLog],
) -> dict[str, int]:
    logs = list(latest_logs.values())
    return {
        "total_logs": len(logs),
        "pending_acknowledgement": sum(1 for log in logs if log.status == "sent"),
        "acknowledged": sum(1 for log in logs if log.status == "acknowledged"),
        "draft_logs": sum(1 for log in logs if log.status == "draft"),
        "competent": sum(
            1
            for log in logs
            if normalize_competency_status(log.competency_status) == "competent"
        ),
        "not_competent": sum(
            1
            for log in logs
            if normalize_competency_status(log.competency_status) == "not_competent"
        ),
    }


def _serialize_sim_session_response(
    db: Session,
    session: SimSession,
) -> dict[str, Any]:
    coaching_log = _get_latest_sim_session_coaching_logs(db, [session.id]).get(session.id)
    feedback_report = _get_supabase_call_simulation_feedback_reports([session.id]).get(session.id)
    payload = SimSessionResponse.model_validate(session).model_dump(mode="json")
    payload["feedback_report"] = feedback_report
    payload["coaching_id"] = coaching_log.coaching_id if coaching_log else None
    payload["coaching_status"] = coaching_log.status if coaching_log else None
    payload["coaching_acknowledged_at"] = (
        coaching_log.acknowledged_at.isoformat()
        if coaching_log and coaching_log.acknowledged_at
        else None
    )
    return payload


def _collect_session_audio_urls(
    session: SimSession,
    response_records: Optional[list[SessionResponseRecord]] = None,
) -> list[str]:
    urls: list[str] = []

    def add_url(candidate: Any) -> None:
        normalized = str(candidate or "").strip()
        if normalized and normalized not in urls:
            urls.append(normalized)

    add_url(session.audio_url)

    for entry in _session_turn_logs(session):
        if isinstance(entry, dict):
            add_url(entry.get("audio_url"))

    for entry in _session_transcript_log(session):
        if isinstance(entry, dict):
            add_url(entry.get("audio_url"))

    for record in response_records or []:
        add_url(record.audio_url)

    return urls


def _get_supabase_call_simulation_feedback_reports(
    session_ids: list[str],
) -> dict[str, dict[str, Any]]:
    normalized_session_ids = [
        session_id
        for session_id in {str(value).strip() for value in session_ids if str(value).strip()}
        if session_id
    ]
    if not normalized_session_ids:
        return {}

    supabase = get_supabase_client()
    if not supabase.is_available:
        return {}

    query_sources = (
        ("call_simulation_scores", "feedback_report"),
        ("call_simulation_ai_evaluations", "evaluation_payload"),
    )
    last_error: Optional[Exception] = None

    for relation_name, report_field in query_sources:
        try:
            result = (
                supabase.table(relation_name)
                .select(f"session_id, {report_field}")
                .in_("session_id", normalized_session_ids)
                .execute()
            )
        except Exception as exc:
            last_error = exc
            continue

        rows = result.data if isinstance(getattr(result, "data", None), list) else []
        payload: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            session_id = str(row.get("session_id") or "").strip()
            feedback_report = row.get(report_field)
            if session_id and isinstance(feedback_report, dict) and feedback_report:
                payload[session_id] = feedback_report
        if payload:
            return payload

    if last_error is not None:
        logger.warning("Unable to load Call Simulation feedback reports from Supabase: %s", last_error)
    return {}


def _normalize_json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_optional_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _resolve_call_simulation_step_audio_url(step: Any) -> Optional[str]:
    if step is None:
        return None
    metadata = _normalize_json_object(getattr(step, "step_metadata", None) or getattr(step, "metadata", None))
    return _normalize_optional_url(
        getattr(step, "prompt_audio", None)
        or getattr(step, "audio_url", None)
        or metadata.get("member_audio_url")
        or metadata.get("audio_url")
    )


def _normalize_uuid_candidate(value: Any) -> Optional[str]:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    return (
        normalized
        if re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            normalized,
            re.IGNORECASE,
        )
        else None
    )


def _get_owned_scenario_for_audio_management(
    db: Session,
    current_user: User,
    scenario_id: str,
) -> Scenario:
    scenario = _get_accessible_scenario(db, current_user, scenario_id)
    if current_user.role != UserRole.ADMIN and scenario.created_by != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You can only manage audio for Call Simulation scenarios that you created.",
        )
    return scenario


def _normalize_call_simulation_audio_content_type(content_type: Optional[str]) -> Optional[str]:
    normalized = str(content_type or "").split(";", 1)[0].strip().lower()
    if not normalized:
        return None
    return CALL_SIMULATION_AUDIO_MIME_ALIASES.get(normalized)


def _infer_call_simulation_audio_content_type_from_name(filename: Optional[str]) -> Optional[str]:
    extension = os.path.splitext(os.path.basename(filename or ""))[1].lower()
    return CALL_SIMULATION_ALLOWED_AUDIO_EXTENSIONS.get(extension)


def _sanitize_call_simulation_audio_stem(value: Optional[str], *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized[:80] or fallback


def _prepare_call_simulation_audio_upload(
    *,
    file: UploadFile,
    file_bytes: bytes,
    fallback_stem: str,
) -> dict[str, Any]:
    file_size = len(file_bytes or b"")
    if file_size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded audio asset is empty")
    if file_size > CALL_SIMULATION_AUDIO_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Call Simulation audio must be 50 MB or smaller.",
        )

    original_name = file.filename or fallback_stem
    original_extension = os.path.splitext(os.path.basename(original_name))[1].lower()
    canonical_content_type = _normalize_call_simulation_audio_content_type(file.content_type)
    extension_from_name = CALL_SIMULATION_ALLOWED_AUDIO_EXTENSIONS.get(original_extension)

    if extension_from_name and canonical_content_type and extension_from_name != canonical_content_type:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file type does not match its filename extension.",
        )

    resolved_content_type = canonical_content_type or extension_from_name
    if not resolved_content_type:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Upload MP3, WAV, OGG, or M4A audio.",
        )

    resolved_extension = original_extension or next(
        (
            extension
            for extension, content_type in CALL_SIMULATION_ALLOWED_AUDIO_EXTENSIONS.items()
            if content_type == resolved_content_type
        ),
        ".mp3",
    )
    sanitized_stem = _sanitize_call_simulation_audio_stem(
        os.path.splitext(os.path.basename(original_name))[0],
        fallback=fallback_stem,
    )

    return {
        "content_type": resolved_content_type,
        "extension": resolved_extension,
        "file_size": file_size,
        "sanitized_stem": sanitized_stem,
    }


def _resolve_call_simulation_storage_location(public_url: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    normalized_url = _normalize_optional_url(public_url)
    if not normalized_url or normalized_url.startswith("data:audio/"):
        return None, None

    parsed = urlparse(normalized_url)
    parsed_path = parsed.path or ""
    if normalized_url.startswith("/media/") or parsed_path.startswith("/media/"):
        media_path = normalized_url if normalized_url.startswith("/media/") else parsed_path
        return None, media_path.replace("\\", "/").lstrip("/")

    supabase = get_supabase_client()
    for bucket_name in (
        supabase.bucket_name,
        supabase.profile_bucket_name,
        supabase.call_simulation_bucket_name,
        supabase.call_simulation_asset_bucket_name,
        supabase.microlearning_bucket_name,
    ):
        marker = f"/storage/v1/object/public/{bucket_name}/"
        if marker not in parsed_path:
            continue
        relative_path = parsed_path.split(marker, 1)[1].lstrip("/")
        return bucket_name, unquote(relative_path)

    return None, None


def _is_local_media_public_url(public_url: Optional[str]) -> bool:
    normalized_url = _normalize_optional_url(public_url)
    if not normalized_url:
        return False
    if normalized_url.startswith("/media/"):
        return True
    return (urlparse(normalized_url).path or "").startswith("/media/")


def _is_supabase_storage_public_url(public_url: Optional[str]) -> bool:
    bucket_name, relative_path = _resolve_call_simulation_storage_location(public_url)
    return bool(bucket_name and relative_path)


def _require_supabase_storage_public_url(public_url: Optional[str], *, label: str) -> str:
    normalized_url = _normalize_optional_url(public_url)
    if not normalized_url or not _is_supabase_storage_public_url(normalized_url):
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be stored in Supabase Storage.",
        )
    return normalized_url


def _is_stored_call_simulation_audio_public_url(public_url: Optional[str]) -> bool:
    normalized_url = _normalize_optional_url(public_url)
    if not normalized_url:
        return False

    if _is_supabase_storage_public_url(normalized_url):
        return True
    return False


def _bool_from_metadata(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _is_call_simulation_speech_enabled_step(step: ScenarioFlow) -> bool:
    metadata = _normalize_json_object(getattr(step, "step_metadata", None))
    explicit = (
        _bool_from_metadata(metadata.get("speech_enabled"))
        if "speech_enabled" in metadata
        else _bool_from_metadata(metadata.get("generate_speech"))
    )
    if explicit is not None:
        return explicit

    actor = (getattr(step, "speaker_role", None) or "").strip().lower()
    # In the current Call Simulation product, CSR rows are trainee scoring
    # checkpoints. Member/customer/caller rows are the speech-enabled playback.
    return actor in {"member", "customer", "caller"}


def _resolve_call_simulation_step_speech_text(step: ScenarioFlow) -> str:
    actor = (getattr(step, "speaker_role", None) or "").strip().lower()
    if actor in {"csr", "agent", "trainee"}:
        return str(getattr(step, "expected_response", None) or "").strip()
    return str(getattr(step, "prompt_text", None) or "").strip()


def _resolve_call_simulation_step_voice(step: ScenarioFlow) -> tuple[str, str]:
    metadata = _normalize_json_object(getattr(step, "step_metadata", None))
    actor = (getattr(step, "speaker_role", None) or "").strip().lower()
    configured_voice = (
        str(
            metadata.get("voice_name")
            or metadata.get("tts_voice")
            or metadata.get("voice")
            or ""
        ).strip()
    )
    if configured_voice:
        return configured_voice, str(metadata.get("speaking_style") or "professional").strip() or "professional"
    if actor in {"csr", "agent", "trainee"}:
        return "Kore", "professionally"
    return "Puck", "casually"


def _stable_call_simulation_speech_filename(
    *,
    step: ScenarioFlow,
    text_value: str,
    content_extension: str,
) -> str:
    actor = (getattr(step, "speaker_role", None) or "speaker").strip().lower()
    actor_slug = re.sub(r"[^a-z0-9]+", "-", actor).strip("-") or "speaker"
    text_hash = hashlib.sha1(text_value.encode("utf-8")).hexdigest()[:12]
    step_number = int(getattr(step, "step_number", None) or 0)
    extension = re.sub(r"[^a-z0-9]+", "", str(content_extension or "wav").lower()) or "wav"
    return f"step-{step_number:02d}_{actor_slug}_{text_hash}.{extension}"


async def _synthesize_call_simulation_step_audio(
    *,
    text_value: str,
    voice_name: str,
    speaking_style: str,
    max_attempts: int = 3,
) -> dict[str, Any]:
    last_error = "Speech generation returned no audio."
    attempts = max(1, int(max_attempts or 1))
    for attempt in range(1, attempts + 1):
        try:
            result = await text_to_speech_for_persistence(
                text_value,
                voice_name=voice_name,
                speaking_style=speaking_style,
            )
            upload_bytes = _coerce_audio_bytes(result.get("audio_bytes"))
            if upload_bytes is None and isinstance(result.get("audio_base64"), str):
                upload_bytes = _coerce_audio_bytes(result.get("audio_base64"))
            if upload_bytes is None and isinstance(result.get("audio_url"), str):
                upload_bytes = _coerce_audio_bytes(result.get("audio_url"))
            if upload_bytes:
                return {
                    "audio_bytes": upload_bytes,
                    "content_type": str(result.get("audio_content_type") or "audio/wav").strip() or "audio/wav",
                    "extension": re.sub(r"[^a-z0-9]+", "", str(result.get("audio_extension") or "wav").lower()) or "wav",
                    "duration": float(result.get("duration") or 0.0),
                    "provider": str(result.get("provider") or "tts").strip() or "tts",
                    "attempts": attempt,
                }
            last_error = str(result.get("error") or last_error).strip() or last_error
        except Exception as exc:
            last_error = str(exc) or last_error

        if attempt < attempts:
            await asyncio.sleep(min(0.4 * attempt, 1.2))

    raise RuntimeError(last_error)


async def _generate_call_simulation_step_speech_asset(
    db: Session,
    *,
    scenario: Scenario,
    step: ScenarioFlow,
    trainer_id: str,
    replace_existing: bool,
    max_attempts: int = 3,
) -> dict[str, Any]:
    step_number = int(getattr(step, "step_number", None) or 0)
    actor = (getattr(step, "speaker_role", None) or "member").strip().lower() or "member"
    speaker_label = str(getattr(step, "speaker_label", None) or actor.title()).strip()
    text_value = _resolve_call_simulation_step_speech_text(step)
    existing_audio_url = _resolve_call_simulation_step_audio_url(step)

    base_item = {
        "step_number": step_number,
        "script_turn_id": getattr(step, "id", None),
        "actor": actor,
        "speaker_label": speaker_label,
        "text": text_value,
        "audio_url": existing_audio_url,
        "attempts": 0,
    }

    if not _is_call_simulation_speech_enabled_step(step):
        return {**base_item, "status": "skipped", "reason": "Step is not speech-enabled."}
    if not text_value:
        return {**base_item, "status": "failed", "error": f"{speaker_label} step {step_number} has no dialogue text."}

    if existing_audio_url and _is_stored_call_simulation_audio_public_url(existing_audio_url) and not replace_existing:
        audio_asset = _find_active_call_simulation_audio_asset(
            db,
            trainer_id=trainer_id,
            public_url=existing_audio_url,
            asset_kinds=["member-step", "step-prompts"],
            scenario_id=scenario.id,
            step_number=step_number,
        )
        if audio_asset is None or audio_asset.asset_kind != "member-step":
            audio_asset = _create_call_simulation_audio_asset_record(
                db,
                trainer_id=trainer_id,
                scenario_id=scenario.id,
                script_turn_id=step.id,
                step_number=step_number,
                asset_kind="member-step",
                source_type="existing_url",
                public_url=existing_audio_url,
                file_name=None,
                file_type=None,
                generated_text=text_value,
                asset_metadata={"relinked_by": "scenario_speech_generation"},
            )
            db.commit()
        return {
            **base_item,
            "status": "reused",
            "audio_url": existing_audio_url,
            "audio_asset": _serialize_call_simulation_audio_asset(audio_asset).model_dump() if audio_asset else None,
        }

    voice_name, speaking_style = _resolve_call_simulation_step_voice(step)
    synthesis = await _synthesize_call_simulation_step_audio(
        text_value=text_value,
        voice_name=voice_name,
        speaking_style=speaking_style,
        max_attempts=max_attempts,
    )
    upload_bytes = synthesis["audio_bytes"]
    get_tts_service().save_audio_locally(
        upload_bytes,
        scenario_id=scenario.id,
        step_number=step_number,
        asset_kind="member-step",
    )
    filename = _stable_call_simulation_speech_filename(
        step=step,
        text_value=text_value,
        content_extension=synthesis["extension"],
    )
    uploaded_audio_url = get_supabase_client().upload_call_simulation_asset(
        file_data=upload_bytes,
        trainer_id=trainer_id,
        scenario_id=scenario.id,
        asset_kind="member-step",
        filename=filename,
        content_type=synthesis["content_type"],
        upsert=True,
    )
    if not uploaded_audio_url or not _is_supabase_storage_public_url(uploaded_audio_url):
        raise RuntimeError("Supabase Storage did not return a stored audio URL for this dialogue.")

    stale_url = existing_audio_url if existing_audio_url and existing_audio_url != uploaded_audio_url else None
    if stale_url:
        stale_asset = _find_active_call_simulation_audio_asset(
            db,
            trainer_id=trainer_id,
            public_url=stale_url,
            asset_kinds=["member-step", "step-prompts"],
            scenario_id=scenario.id,
            step_number=step_number,
        )
        if stale_asset:
            _deactivate_call_simulation_audio_asset_record(db, asset=stale_asset, delete_storage=True)

    audio_asset = _find_active_call_simulation_audio_asset(
        db,
        trainer_id=trainer_id,
        public_url=uploaded_audio_url,
        asset_kinds=["member-step"],
        scenario_id=scenario.id,
        step_number=step_number,
    )
    if audio_asset is None:
        audio_asset = _create_call_simulation_audio_asset_record(
            db,
            trainer_id=trainer_id,
            scenario_id=scenario.id,
            script_turn_id=step.id,
            step_number=step_number,
            asset_kind="member-step",
            source_type="generated_tts",
            public_url=uploaded_audio_url,
            file_name=filename,
            file_type=synthesis["content_type"],
            file_size=len(upload_bytes),
            voice_used=voice_name,
            provider=synthesis["provider"],
            generated_text=text_value,
            asset_metadata={
                "storage_mode": "supabase",
                "voice_type": voice_name,
                "language": "en-US",
                "duration": round(max(float(synthesis["duration"]), 0.0), 2),
                "scenario_speech_generation": True,
            },
        )
    else:
        audio_asset.script_turn_id = step.id
        audio_asset.step_number = step_number
        audio_asset.file_name = filename
        audio_asset.file_type = synthesis["content_type"]
        audio_asset.file_size = len(upload_bytes)
        audio_asset.voice_used = voice_name
        audio_asset.provider = synthesis["provider"]
        audio_asset.generated_text = text_value
        audio_asset.asset_metadata = {
            **_normalize_json_object(audio_asset.asset_metadata),
            "storage_mode": "supabase",
            "voice_type": voice_name,
            "language": "en-US",
            "duration": round(max(float(synthesis["duration"]), 0.0), 2),
            "scenario_speech_generation": True,
        }
        db.add(audio_asset)

    step.prompt_audio = uploaded_audio_url
    metadata = _normalize_json_object(step.step_metadata)
    metadata["member_audio_url"] = uploaded_audio_url
    metadata["speech_generated_at"] = datetime.utcnow().isoformat()
    metadata["speech_status"] = "ready"
    metadata["speech_voice"] = voice_name
    step.step_metadata = metadata
    db.add(step)
    db.commit()

    return {
        **base_item,
        "status": "generated",
        "audio_url": uploaded_audio_url,
        "attempts": synthesis["attempts"],
        "provider": synthesis["provider"],
        "voice_used": voice_name,
        "audio_asset": _serialize_call_simulation_audio_asset(audio_asset).model_dump() if audio_asset else None,
    }


def _validate_optional_call_simulation_storage_url(public_url: Any, *, label: str) -> Optional[str]:
    normalized_url = _normalize_optional_url(public_url)
    if not normalized_url:
        return None
    if not _is_supabase_storage_public_url(normalized_url):
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be uploaded to Supabase Storage. Public, embedded, and local media URLs are not allowed.",
        )
    return normalized_url


def _validate_call_simulation_member_audio_assets(
    row_assets: dict[str, Any],
) -> None:
    missing_groups: list[str] = []
    invalid_groups: list[str] = []

    for group in list(row_assets.get("scenario_groups") or []):
        scenario_key = str(group.get("scenario_key") or "").strip() or "Unknown"
        member_rows = list(group.get("member_rows") or [])
        if not member_rows:
            continue

        member_audio_url = next(
            (
                _normalize_optional_url(row.get("audio_url"))
                for row in member_rows
                if _normalize_optional_url(row.get("audio_url"))
            ),
            None,
        )
        if not member_audio_url:
            missing_groups.append(scenario_key)
            continue
        if not _is_supabase_storage_public_url(member_audio_url):
            invalid_groups.append(scenario_key)

    if missing_groups:
        raise HTTPException(
            status_code=400,
            detail=(
                "Each Member AI row must have a stored audio asset before saving this Call Simulation. "
                f"Missing audio in Scenario Group{'s' if len(missing_groups) != 1 else ''}: {', '.join(missing_groups)}."
            ),
        )
    if invalid_groups:
        raise HTTPException(
            status_code=400,
            detail=(
                "Member AI audio must be stored in Supabase Storage before saving this Call Simulation. "
                f"Replace the audio in Scenario Group{'s' if len(invalid_groups) != 1 else ''}: {', '.join(invalid_groups)}."
            ),
        )


def _validate_call_simulation_launch_assets(
    scenario: Scenario,
    scenario_steps: list[CallSimulationScenarioStepResponse],
) -> None:
    _validate_optional_call_simulation_storage_url(
        scenario.ringer_audio_url,
        label="Trainer-uploaded ringer audio",
    )
    _validate_optional_call_simulation_storage_url(
        scenario.hold_audio_url,
        label="Trainer-uploaded notification audio",
    )

    missing_member_references: list[str] = []
    for step in scenario_steps:
        if str(getattr(step, "actor", "") or "").lower() != "member":
            continue
        if not str(getattr(step, "script", "") or "").strip():
            continue
        audio_url = _resolve_call_simulation_step_audio_url(step)
        if _is_stored_call_simulation_audio_public_url(audio_url):
            continue
        missing_member_references.append(f"step {int(step.step_number or 0)}")

    scenario_config = _normalize_json_object(getattr(scenario, "call_simulation_config", None))
    configured_script_flow = scenario_config.get("script_flow")
    if isinstance(configured_script_flow, list):
        for index, row in enumerate(configured_script_flow, start=1):
            normalized_row = _normalize_json_object(row)
            member_response_text = str(normalized_row.get("member_response_text") or "").strip()
            if not member_response_text:
                continue
            member_audio_url = _normalize_optional_url(normalized_row.get("member_audio_url"))
            if _is_stored_call_simulation_audio_public_url(member_audio_url):
                continue
            scenario_label = str(normalized_row.get("scenario") or "").strip() or str(index)
            reference = f"scenario {scenario_label}"
            if reference not in missing_member_references:
                missing_member_references.append(reference)

    if missing_member_references:
        raise HTTPException(
            status_code=400,
            detail=(
                "This Call Simulation is not speech-ready. Ask the trainer to generate Member AI speech "
                f"for {', '.join(missing_member_references)} before launching the trainee simulation."
            ),
        )


@router.post("/scenarios/{scenario_id}/speech")
async def generate_call_simulation_scenario_speech(
    scenario_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    scenario = _get_owned_scenario_for_audio_management(db, current_user, scenario_id)

    require_supabase = bool(payload.get("require_supabase", True))
    regenerate_all = bool(payload.get("regenerate_all", False))
    retry_failed_only = bool(payload.get("retry_failed_only", False))
    max_attempts = max(1, min(int(payload.get("max_attempts") or 3), 5))
    if require_supabase:
        _require_supabase_storage(
            "Scenario speech generation requires Supabase Storage so trainees can play the generated audio."
        )

    _require_server_side_tts(
        "Server-side TTS is unavailable. Configure GOOGLE_API_KEY or GEMINI_API_KEY, OPENAI_API_KEY, or AZURE_SPEECH_KEY/AZURE_SPEECH_REGION for deployed speech generation."
    )

    ordered_steps = sorted(
        list(getattr(scenario, "flow_steps", []) or []),
        key=lambda step: (int(step.step_number or 0), step.created_at or datetime.utcnow()),
    )
    if not ordered_steps:
        raise HTTPException(
            status_code=400,
            detail="This scenario has no persisted dialogue steps. Save the scenario before generating speech.",
        )

    items: list[dict[str, Any]] = []
    for step in ordered_steps:
        metadata = _normalize_json_object(getattr(step, "step_metadata", None))
        current_status = str(metadata.get("speech_status") or "").strip().lower()
        existing_audio_url = _resolve_call_simulation_step_audio_url(step)

        if retry_failed_only and current_status != "failed" and _is_stored_call_simulation_audio_public_url(existing_audio_url):
            items.append(
                {
                    "step_number": int(step.step_number or 0),
                    "script_turn_id": step.id,
                    "actor": (step.speaker_role or "member").strip().lower(),
                    "speaker_label": step.speaker_label,
                    "text": _resolve_call_simulation_step_speech_text(step),
                    "status": "reused",
                    "audio_url": existing_audio_url,
                    "attempts": 0,
                }
            )
            continue

        try:
            item = await _generate_call_simulation_step_speech_asset(
                db,
                scenario=scenario,
                step=step,
                trainer_id=str(current_user.id),
                replace_existing=regenerate_all or retry_failed_only,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            logger.warning(
                "Unable to generate Call Simulation speech for scenario=%s step=%s: %s",
                scenario.id,
                getattr(step, "step_number", None),
                exc,
            )
            metadata = _normalize_json_object(getattr(step, "step_metadata", None))
            metadata["speech_status"] = "failed"
            metadata["speech_error"] = str(exc)
            metadata["speech_failed_at"] = datetime.utcnow().isoformat()
            step.step_metadata = metadata
            db.add(step)
            db.commit()
            item = {
                "step_number": int(getattr(step, "step_number", None) or 0),
                "script_turn_id": getattr(step, "id", None),
                "actor": (getattr(step, "speaker_role", None) or "member").strip().lower(),
                "speaker_label": getattr(step, "speaker_label", None),
                "text": _resolve_call_simulation_step_speech_text(step),
                "status": "failed",
                "error": str(exc) or "Speech generation failed for this dialogue.",
                "attempts": max_attempts,
            }
        items.append(item)

    generated_count = sum(1 for item in items if item.get("status") == "generated")
    reused_count = sum(1 for item in items if item.get("status") == "reused")
    skipped_count = sum(1 for item in items if item.get("status") == "skipped")
    failed_count = sum(1 for item in items if item.get("status") == "failed")
    required_items = [item for item in items if item.get("status") != "skipped"]
    speech_ready = bool(required_items) and failed_count == 0 and all(
        _is_stored_call_simulation_audio_public_url(item.get("audio_url"))
        for item in required_items
    )

    scenario_config = _normalize_json_object(getattr(scenario, "call_simulation_config", None))
    scenario.call_simulation_config = {
        **scenario_config,
        "speech_ready": speech_ready,
        "speech_generated_at": datetime.utcnow().isoformat(),
        "speech_summary": {
            "total": len(required_items),
            "generated": generated_count,
            "reused": reused_count,
            "skipped": skipped_count,
            "failed": failed_count,
        },
    }
    scenario.updated_at = datetime.utcnow()
    db.add(scenario)

    _log_call_simulation_audio_action(
        db,
        user=current_user,
        action_type="generate_scenario_speech",
        scenario_id=scenario.id,
        status_value="failed" if failed_count else "success",
        error_detail=f"{failed_count} dialogue item(s) failed." if failed_count else None,
        metadata={
            "total": len(required_items),
            "generated": generated_count,
            "reused": reused_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "speech_ready": speech_ready,
        },
    )
    db.commit()

    return {
        "scenario_id": scenario.id,
        "speech_ready": speech_ready,
        "total": len(required_items),
        "generated": generated_count,
        "reused": reused_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "items": items,
    }


def _infer_call_simulation_audio_filename_from_url(
    public_url: Optional[str],
    *,
    fallback: str,
) -> str:
    normalized_url = _normalize_optional_url(public_url)
    if not normalized_url:
        return fallback
    if normalized_url.startswith("data:audio/"):
        return fallback

    parsed = urlparse(normalized_url)
    leaf = os.path.basename(parsed.path or normalized_url).strip()
    return leaf or fallback


def _serialize_call_simulation_audio_asset(
    asset: CallSimulationAudioAsset,
) -> CallSimulationAudioAssetResponse:
    return CallSimulationAudioAssetResponse.model_validate(asset)


def _create_call_simulation_audio_asset_record(
    db: Session,
    *,
    trainer_id: str,
    scenario_id: Optional[str],
    script_turn_id: Optional[str],
    step_number: Optional[int],
    asset_kind: str,
    source_type: str,
    public_url: Optional[str],
    file_name: Optional[str],
    file_type: Optional[str],
    file_size: Optional[int] = None,
    voice_used: Optional[str] = None,
    provider: Optional[str] = None,
    generated_text: Optional[str] = None,
    asset_metadata: Optional[dict[str, Any]] = None,
) -> Optional[CallSimulationAudioAsset]:
    normalized_url = _normalize_optional_url(public_url)
    if not normalized_url or normalized_url.startswith("data:audio/"):
        return None

    resolved_file_name = str(file_name or "").strip() or _infer_call_simulation_audio_filename_from_url(
        normalized_url,
        fallback=f"{asset_kind}.mp3",
    )
    resolved_file_type = str(file_type or "").strip() or _infer_call_simulation_audio_content_type_from_name(
        resolved_file_name,
    ) or "audio/mpeg"
    bucket_name, storage_path = _resolve_call_simulation_storage_location(normalized_url)

    asset = CallSimulationAudioAsset(
        id=str(uuid.uuid4()),
        trainer_id=str(trainer_id),
        scenario_id=_normalize_uuid_candidate(scenario_id),
        script_turn_id=_normalize_uuid_candidate(script_turn_id),
        step_number=int(step_number) if isinstance(step_number, int) or str(step_number or "").isdigit() else None,
        asset_kind=str(asset_kind or "").strip().lower() or "member-step",
        source_type=str(source_type or "").strip().lower() or "upload",
        file_name=resolved_file_name,
        file_type=resolved_file_type,
        file_size=int(file_size) if isinstance(file_size, int) or str(file_size or "").isdigit() else None,
        bucket_name=bucket_name,
        storage_path=storage_path,
        public_url=normalized_url,
        voice_used=str(voice_used or "").strip() or None,
        provider=str(provider or "").strip() or None,
        generated_text=str(generated_text or "").strip() or None,
        asset_metadata=_normalize_json_object(asset_metadata),
        is_active=True,
        deleted_at=None,
    )
    db.add(asset)
    db.flush()
    return asset


def _find_active_call_simulation_audio_asset(
    db: Session,
    *,
    trainer_id: str,
    public_url: Optional[str],
    asset_kinds: list[str],
    scenario_id: Optional[str] = None,
    step_number: Optional[int] = None,
) -> Optional[CallSimulationAudioAsset]:
    normalized_url = _normalize_optional_url(public_url)
    normalized_asset_kinds = [str(kind or "").strip().lower() for kind in asset_kinds if str(kind or "").strip()]
    if not normalized_url or not normalized_asset_kinds:
        return None

    candidates = (
        db.query(CallSimulationAudioAsset)
        .filter(
            CallSimulationAudioAsset.trainer_id == str(trainer_id),
            CallSimulationAudioAsset.public_url == normalized_url,
            CallSimulationAudioAsset.asset_kind.in_(normalized_asset_kinds),
            CallSimulationAudioAsset.is_active == True,
        )
        .order_by(CallSimulationAudioAsset.updated_at.desc(), CallSimulationAudioAsset.created_at.desc())
        .all()
    )
    if not candidates:
        return None

    for candidate in candidates:
        if scenario_id and candidate.scenario_id == scenario_id and (
            step_number is None or candidate.step_number == step_number
        ):
            return candidate
    for candidate in candidates:
        if candidate.scenario_id is None:
            return candidate
    return candidates[0]


def _deactivate_call_simulation_audio_asset_record(
    db: Session,
    *,
    asset: Optional[CallSimulationAudioAsset],
    delete_storage: bool,
) -> bool:
    if not asset or not asset.is_active:
        return False

    asset.is_active = False
    asset.deleted_at = datetime.utcnow()
    db.add(asset)

    if delete_storage:
        get_supabase_client().delete_by_public_url(asset.public_url)
    return True


def _sync_call_simulation_audio_assets_for_scenario(
    db: Session,
    *,
    scenario: Scenario,
    trainer_id: str,
    use_shared_ringer_audio: bool,
    use_shared_hold_audio: bool,
) -> list[CallSimulationAudioAsset]:
    linked_assets: list[CallSimulationAudioAsset] = []
    steps_by_number = {
        int(step.step_number or 0): step
        for step in list(getattr(scenario, "flow_steps", []) or [])
        if step and step.step_number is not None
    }

    target_bindings: list[dict[str, Any]] = []
    for step_number, step in steps_by_number.items():
        if (step.speaker_role or "").strip().lower() != "member":
            continue
        if not _normalize_optional_url(step.prompt_audio):
            continue
        target_bindings.append(
            {
                "allowed_asset_kinds": ["member-step"],
                "asset_kind": "member-step",
                "public_url": step.prompt_audio,
                "step_number": step_number,
                "script_turn_id": step.id,
            }
        )

    if not use_shared_ringer_audio and _normalize_optional_url(scenario.ringer_audio_url):
        target_bindings.append(
            {
                "allowed_asset_kinds": ["scenario-ringer", "ringer"],
                "asset_kind": "scenario-ringer",
                "public_url": scenario.ringer_audio_url,
                "step_number": None,
                "script_turn_id": None,
            }
        )
    if not use_shared_hold_audio and _normalize_optional_url(scenario.hold_audio_url):
        target_bindings.append(
            {
                "allowed_asset_kinds": ["scenario-hold", "hold"],
                "asset_kind": "scenario-hold",
                "public_url": scenario.hold_audio_url,
                "step_number": None,
                "script_turn_id": None,
            }
        )

    for binding in target_bindings:
        existing = _find_active_call_simulation_audio_asset(
            db,
            trainer_id=trainer_id,
            public_url=binding["public_url"],
            asset_kinds=binding["allowed_asset_kinds"],
            scenario_id=scenario.id,
            step_number=binding["step_number"],
        )
        if existing:
            existing.scenario_id = scenario.id
            existing.script_turn_id = _normalize_uuid_candidate(binding["script_turn_id"])
            existing.step_number = binding["step_number"]
            db.add(existing)
            linked_assets.append(existing)
            continue

        created = _create_call_simulation_audio_asset_record(
            db,
            trainer_id=trainer_id,
            scenario_id=scenario.id,
            script_turn_id=binding["script_turn_id"],
            step_number=binding["step_number"],
            asset_kind=binding["asset_kind"],
            source_type="manual_url",
            public_url=binding["public_url"],
            file_name=None,
            file_type=None,
            asset_metadata={"linked_from_scenario_save": True},
        )
        if created:
            linked_assets.append(created)

    return linked_assets


def _collect_scenario_audio_bindings(
    scenario: Scenario,
    *,
    use_shared_ringer_audio: bool,
    use_shared_hold_audio: bool,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    if not use_shared_ringer_audio and _normalize_optional_url(scenario.ringer_audio_url):
        bindings.append(
            {
                "asset_kind": "scenario-ringer",
                "step_number": None,
                "public_url": scenario.ringer_audio_url,
            }
        )
    if not use_shared_hold_audio and _normalize_optional_url(scenario.hold_audio_url):
        bindings.append(
            {
                "asset_kind": "scenario-hold",
                "step_number": None,
                "public_url": scenario.hold_audio_url,
            }
        )

    for step in list(getattr(scenario, "flow_steps", []) or []):
        if (step.speaker_role or "").strip().lower() != "member":
            continue
        if not _normalize_optional_url(step.prompt_audio):
            continue
        bindings.append(
            {
                "asset_kind": "member-step",
                "step_number": int(step.step_number or 0) or None,
                "public_url": step.prompt_audio,
            }
        )

    return bindings


def _cleanup_removed_scenario_audio_assets(
    db: Session,
    *,
    trainer_id: str,
    scenario_id: str,
    previous_bindings: list[dict[str, Any]],
    current_bindings: list[dict[str, Any]],
) -> None:
    current_keys = {
        (
            str(binding.get("asset_kind") or "").strip().lower(),
            binding.get("step_number"),
            _normalize_optional_url(binding.get("public_url")),
        )
        for binding in current_bindings
        if _normalize_optional_url(binding.get("public_url"))
    }

    for binding in previous_bindings:
        public_url = _normalize_optional_url(binding.get("public_url"))
        asset_kind = str(binding.get("asset_kind") or "").strip().lower()
        step_number = binding.get("step_number")
        binding_key = (asset_kind, step_number, public_url)
        if not public_url or binding_key in current_keys:
            continue

        stale_records_query = db.query(CallSimulationAudioAsset).filter(
            CallSimulationAudioAsset.trainer_id == str(trainer_id),
            CallSimulationAudioAsset.scenario_id == str(scenario_id),
            CallSimulationAudioAsset.asset_kind == asset_kind,
            CallSimulationAudioAsset.public_url == public_url,
            CallSimulationAudioAsset.is_active == True,
        )
        if step_number is None:
            stale_records_query = stale_records_query.filter(CallSimulationAudioAsset.step_number.is_(None))
        else:
            stale_records_query = stale_records_query.filter(CallSimulationAudioAsset.step_number == int(step_number))

        stale_records = stale_records_query.all()
        if not stale_records:
            continue

        stale_record_ids = {record.id for record in stale_records}
        is_still_referenced = (
            db.query(CallSimulationAudioAsset)
            .filter(
                CallSimulationAudioAsset.trainer_id == str(trainer_id),
                CallSimulationAudioAsset.public_url == public_url,
                CallSimulationAudioAsset.is_active == True,
                ~CallSimulationAudioAsset.id.in_(stale_record_ids),
            )
            .first()
            is not None
        )
        for stale_record in stale_records:
            _deactivate_call_simulation_audio_asset_record(
                db,
                asset=stale_record,
                delete_storage=not is_still_referenced,
            )


def _sanitize_supabase_script_flow(script_flow: Any) -> list[dict[str, Any]]:
    sanitized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(script_flow if isinstance(script_flow, list) else [], start=1):
        if not isinstance(step, dict):
            continue

        expected_keywords = [
            str(keyword).strip()
            for keyword in (step.get("expected_keywords") or [])
            if str(keyword).strip()
        ] if isinstance(step.get("expected_keywords"), list) else []

        sanitized_steps.append(
            {
                "step_id": str(step.get("step_id") or f"step-{index}").strip() or f"step-{index}",
                "suggested_csr_script": str(step.get("suggested_csr_script") or "").strip(),
                "member_response_text": str(step.get("member_response_text") or "").strip(),
                "point_value": _coerce_nonnegative_float(step.get("point_value"), fallback=0.0),
                "expected_keywords": expected_keywords,
                "member_audio_url": _normalize_optional_url(step.get("member_audio_url")),
            }
        )

    return sanitized_steps


def _read_supabase_scenario_group_from_step_id(step_id: Any) -> str:
    raw_step_id = str(step_id or "").strip()
    normalized = re.sub(r"^scenario[-_:]*", "", raw_step_id, flags=re.IGNORECASE).strip()
    return normalized or raw_step_id or "1"


def _build_supabase_scenario_group_summary(
    script_flow: list[dict[str, Any]],
) -> Optional[str]:
    ordered_groups: list[str] = []
    seen_groups: set[str] = set()
    for step in script_flow:
        scenario_group = _read_supabase_scenario_group_from_step_id(step.get("step_id"))
        if not scenario_group or scenario_group in seen_groups:
            continue
        seen_groups.add(scenario_group)
        ordered_groups.append(scenario_group)

    return ", ".join(ordered_groups) if ordered_groups else None


def _build_supabase_kpi_metric_rows_from_target_kpis(
    scenario_group_id: str,
    target_kpis: dict[str, Any],
) -> list[dict[str, Any]]:
    metric_candidates = [
        ("Script Accuracy", target_kpis.get("speech_to_text_weight")),
        ("AHT", target_kpis.get("aht_weight")),
        ("Rate of Speech", target_kpis.get("rate_of_speech_weight")),
        ("Dead Air", target_kpis.get("dead_air_weight")),
        ("Empathy", target_kpis.get("empathy_statements_weight")),
        ("Probing", target_kpis.get("probing_questions_weight")),
        ("Grammar", target_kpis.get("grammar_weight")),
        ("Pronunciation", target_kpis.get("pronunciation_weight")),
        ("Pacing", target_kpis.get("pacing_weight")),
    ]

    metric_rows: list[dict[str, Any]] = []
    for metric_name, raw_value in metric_candidates:
        try:
            weight_value = float(raw_value)
        except (TypeError, ValueError):
            continue

        metric_rows.append(
            {
                "scenario_group_id": scenario_group_id,
                "metric_name": metric_name,
                "weight_percentage": max(0, round(weight_value)),
            }
        )

    return metric_rows


def _build_supabase_authoring_script_rows(
    scenario_id: str,
    script_flow: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sequence_order = 1
    rows: list[dict[str, Any]] = []

    for step in script_flow:
        scenario_group = _read_supabase_scenario_group_from_step_id(step.get("step_id"))
        expected_keywords = [
            str(keyword).strip()
            for keyword in (step.get("expected_keywords") or [])
            if str(keyword).strip()
        ] if isinstance(step.get("expected_keywords"), list) else []
        csr_script = str(step.get("suggested_csr_script") or "").strip()

        if csr_script:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "actor_type": "CSR",
                    "content": csr_script,
                    "score_weight": _coerce_nonnegative_float(step.get("point_value"), fallback=0.0),
                    "sequence_order": sequence_order,
                    "scenario_group": scenario_group,
                    "audio_url": None,
                    "metadata": {
                        "step_id": step.get("step_id"),
                        "expected_keywords": expected_keywords,
                        "role": "csr",
                    },
                }
            )
            sequence_order += 1

        member_script = str(step.get("member_response_text") or "").strip()
        member_audio_url = _normalize_optional_url(step.get("member_audio_url"))
        if member_script or member_audio_url:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "actor_type": "Member",
                    "content": member_script,
                    "score_weight": 0,
                    "sequence_order": sequence_order,
                    "scenario_group": scenario_group,
                    "audio_url": member_audio_url,
                    "metadata": {
                        "step_id": step.get("step_id"),
                        "role": "member",
                    },
                }
            )
            sequence_order += 1

    return rows


def _sync_supabase_scenario_authoring_tables(
    db: Session,
    supabase: Any,
    *,
    source_scenario_id: str,
    trainer_id: str,
    title: Optional[str],
    description: Optional[str],
    topic: str,
    target_kpis: dict[str, Any],
    script_flow: list[dict[str, Any]],
    ringer_audio_url: Optional[str],
    hold_audio_url: Optional[str],
    difficulty: Optional[str],
    estimated_duration_seconds: Any,
    is_published: bool,
    metadata: dict[str, Any],
) -> None:
    scenario_id = _normalize_uuid_candidate(source_scenario_id)
    if not scenario_id:
        raise RuntimeError(
            "Scenario sync requires a UUID-shaped scenario id before it can mirror the trainer record to Supabase."
        )

    if not _legacy_call_simulation_mirror_relations_ready(
        db,
        ("scenarios", "scenario_groups", "scripts", "scenario_scripts", "kpi_metrics"),
        context=f"scenario authoring sync for scenario {source_scenario_id}",
    ):
        return

    trainer_uuid = _normalize_uuid_candidate(trainer_id)
    resolved_topic = str(topic or "").strip() or str(title or "").strip() or "Call Scenario"
    resolved_title = str(title or "").strip() or resolved_topic
    resolved_description = str(description or "").strip() or None
    resolved_metadata = _normalize_json_object(metadata)
    scenario_group = (
        str(resolved_metadata.get("scenario_group_label") or "").strip()
        or _build_supabase_scenario_group_summary(script_flow)
    )
    expected_keywords = list(
        dict.fromkeys(
            keyword
            for step in script_flow
            for keyword in (
                [
                    str(keyword_value).strip()
                    for keyword_value in (step.get("expected_keywords") or [])
                    if str(keyword_value).strip()
                ]
                if isinstance(step.get("expected_keywords"), list)
                else []
            )
        )
    )

    try:
        resolved_duration = float(estimated_duration_seconds)
    except (TypeError, ValueError):
        resolved_duration = 0.0

    scenario_payload = {
        "id": scenario_id,
        "title": resolved_title,
        "topic": resolved_topic,
        "description": resolved_description,
        "scenario_group": scenario_group,
        "opening_prompt": resolved_description or str(script_flow[0].get("suggested_csr_script") or "").strip() or resolved_topic,
        "expected_keywords": expected_keywords,
        "estimated_duration": int(round(resolved_duration)) if resolved_duration > 0 else None,
        "difficulty": str(difficulty or "").strip() or None,
        "purpose": "practice",
        "member_profile": {},
        "cxone_metadata": {},
        "call_simulation_config": {
            **resolved_metadata,
            "scenario_group_label": scenario_group,
            "target_kpis": target_kpis,
            "script_flow": script_flow,
        },
        "ringer_audio_url": _normalize_optional_url(ringer_audio_url),
        "hold_audio_url": _normalize_optional_url(hold_audio_url),
        "created_by": trainer_uuid,
        "is_published": bool(is_published),
        "is_draft": not bool(is_published),
        "updated_at": datetime.utcnow().isoformat(),
    }
    try:
        supabase.table("scenarios").upsert(scenario_payload, on_conflict="id").execute()
        supabase.table("scenario_groups").upsert(
            {
                "id": scenario_id,
                "title": resolved_title,
                "topic": resolved_topic,
                "description": resolved_description,
                "created_by": trainer_uuid,
            },
            on_conflict="id",
        ).execute()

        supabase.table("scripts").delete().eq("scenario_id", scenario_id).execute()
        script_rows = _build_supabase_authoring_script_rows(scenario_id, script_flow)
        if script_rows:
            supabase.table("scripts").insert(script_rows).execute()

        supabase.table("scenario_scripts").delete().eq("scenario_group_id", scenario_id).execute()
        scenario_script_rows = [
            {
                "scenario_group_id": scenario_id,
                "actor_type": row["actor_type"],
                "script_text": row["content"],
                "score_value": max(0, round(float(row.get("score_weight") or 0))),
                "order_index": int(row.get("sequence_order") or 0),
                "audio_url": _normalize_optional_url(row.get("audio_url")),
            }
            for row in script_rows
        ]
        if scenario_script_rows:
            supabase.table("scenario_scripts").insert(scenario_script_rows).execute()

        supabase.table("kpi_metrics").delete().eq("scenario_group_id", scenario_id).execute()
        kpi_metric_rows = _build_supabase_kpi_metric_rows_from_target_kpis(scenario_id, target_kpis)
        if kpi_metric_rows:
            supabase.table("kpi_metrics").insert(kpi_metric_rows).execute()
    except Exception as exc:
        if _is_missing_supabase_table_error(exc):
            _log_legacy_supabase_mirror_skip(
                f"scenario authoring sync for scenario {source_scenario_id}",
                exc,
            )
            return
        raise


def _sync_call_simulation_scenario_to_supabase_from_db(
    supabase: Any,
    *,
    db: Session,
    scenario: Scenario,
    trainer_id: str,
    batch_id: Optional[str] = None,
) -> dict[str, Any]:
    scenario_config = _resolve_call_simulation_config_for_batch(
        db,
        scenario,
        batch_id=str(batch_id) if batch_id else None,
    )
    sanitized_script_flow = _sanitize_supabase_script_flow(
        scenario_config.get("script_flow")
    )
    passing_score = _resolve_call_scenario_passing_score(
        scenario,
        float(
            _get_effective_kpi_config(db, str(batch_id) if batch_id else None).passing_score
            or DEFAULT_PASSING_SCORE
        ),
    )

    scenario_sync_data = {
        "source_scenario_id": str(scenario.id),
        "trainer_id": str(trainer_id),
        "title": scenario.title,
        "topic": str(scenario_config.get("topic") or scenario.lob or scenario.title),
        "description": scenario.description,
        "target_kpis": _normalize_json_object(scenario_config.get("target_kpis")),
        "script_flow": sanitized_script_flow,
        "ringer_audio_url": scenario.ringer_audio_url,
        "hold_audio_url": scenario.hold_audio_url,
        "difficulty": scenario.difficulty or "intermediate",
        "estimated_duration_seconds": scenario.estimated_duration or 300,
        "passing_score": passing_score,
        "is_published": bool(scenario.is_published),
        "is_active": True,
        "metadata": scenario_config,
    }

    if not _legacy_call_simulation_mirror_relations_ready(
        db,
        ("call_scenarios",),
        context=f"scenario sync for {scenario.id}",
    ):
        return scenario_sync_data

    try:
        supabase.table("call_scenarios").upsert(
            scenario_sync_data,
            ignore_duplicates=False,
        ).execute()
    except Exception as supabase_error:
        if _is_missing_supabase_table_error(supabase_error):
            _log_legacy_supabase_mirror_skip(
                f"scenario sync for {scenario.id}",
                supabase_error,
            )
            _sync_supabase_scenario_authoring_tables(
                db,
                supabase,
                source_scenario_id=scenario_sync_data["source_scenario_id"],
                trainer_id=scenario_sync_data["trainer_id"],
                title=scenario_sync_data.get("title"),
                description=scenario_sync_data.get("description"),
                topic=str(scenario_sync_data.get("topic") or ""),
                target_kpis=_normalize_json_object(scenario_sync_data.get("target_kpis")),
                script_flow=_sanitize_supabase_script_flow(scenario_sync_data.get("script_flow")),
                ringer_audio_url=_normalize_optional_url(scenario_sync_data.get("ringer_audio_url")),
                hold_audio_url=_normalize_optional_url(scenario_sync_data.get("hold_audio_url")),
                difficulty=_normalize_optional_url(scenario_sync_data.get("difficulty")),
                estimated_duration_seconds=scenario_sync_data.get("estimated_duration_seconds"),
                is_published=bool(scenario_sync_data.get("is_published", True)),
                metadata=_normalize_json_object(scenario_sync_data.get("metadata")),
            )
            return scenario_sync_data
        logger.warning(
            "Call Simulation Supabase upsert failed for scenario %s. Retrying with insert/update flow: %s",
            scenario.id,
            supabase_error,
        )
        try:
            existing = supabase.table("call_scenarios").select("id").eq(
                "source_scenario_id",
                scenario_sync_data["source_scenario_id"],
            ).execute()

            if existing.data and len(existing.data) > 0:
                supabase.table("call_scenarios").update(
                    scenario_sync_data
                ).eq(
                    "source_scenario_id",
                    scenario_sync_data["source_scenario_id"],
                ).execute()
            else:
                supabase.table("call_scenarios").insert(
                    scenario_sync_data
                ).execute()
        except Exception:
            try:
                legacy_existing = supabase.table("call_scenarios").select("id").eq(
                    "scenario_id",
                    scenario_sync_data["source_scenario_id"],
                ).execute()
            except Exception as legacy_error:
                if _is_missing_supabase_table_error(legacy_error):
                    _log_legacy_supabase_mirror_skip(
                        f"scenario sync fallback for {scenario.id}",
                        legacy_error,
                    )
                    _sync_supabase_scenario_authoring_tables(
                        db,
                        supabase,
                        source_scenario_id=scenario_sync_data["source_scenario_id"],
                        trainer_id=scenario_sync_data["trainer_id"],
                        title=scenario_sync_data.get("title"),
                        description=scenario_sync_data.get("description"),
                        topic=str(scenario_sync_data.get("topic") or ""),
                        target_kpis=_normalize_json_object(scenario_sync_data.get("target_kpis")),
                        script_flow=_sanitize_supabase_script_flow(scenario_sync_data.get("script_flow")),
                        ringer_audio_url=_normalize_optional_url(scenario_sync_data.get("ringer_audio_url")),
                        hold_audio_url=_normalize_optional_url(scenario_sync_data.get("hold_audio_url")),
                        difficulty=_normalize_optional_url(scenario_sync_data.get("difficulty")),
                        estimated_duration_seconds=scenario_sync_data.get("estimated_duration_seconds"),
                        is_published=bool(scenario_sync_data.get("is_published", True)),
                        metadata=_normalize_json_object(scenario_sync_data.get("metadata")),
                    )
                    return scenario_sync_data
                raise

            if legacy_existing.data and len(legacy_existing.data) > 0:
                supabase.table("call_scenarios").update(
                    scenario_sync_data
                ).eq(
                    "scenario_id",
                    scenario_sync_data["source_scenario_id"],
                ).execute()
            else:
                raise

    _sync_supabase_scenario_authoring_tables(
        db,
        supabase,
        source_scenario_id=scenario_sync_data["source_scenario_id"],
        trainer_id=scenario_sync_data["trainer_id"],
        title=scenario_sync_data.get("title"),
        description=scenario_sync_data.get("description"),
        topic=str(scenario_sync_data.get("topic") or ""),
        target_kpis=_normalize_json_object(scenario_sync_data.get("target_kpis")),
        script_flow=_sanitize_supabase_script_flow(scenario_sync_data.get("script_flow")),
        ringer_audio_url=_normalize_optional_url(scenario_sync_data.get("ringer_audio_url")),
        hold_audio_url=_normalize_optional_url(scenario_sync_data.get("hold_audio_url")),
        difficulty=_normalize_optional_url(scenario_sync_data.get("difficulty")),
        estimated_duration_seconds=scenario_sync_data.get("estimated_duration_seconds"),
        is_published=bool(scenario_sync_data.get("is_published", True)),
        metadata=_normalize_json_object(scenario_sync_data.get("metadata")),
    )

    return scenario_sync_data


def _get_latest_active_batch_mapping_for_scenario(
    db: Session,
    scenario_id: str,
) -> Optional[BatchScenarioMapping]:
    return (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.scenario_id == scenario_id,
            BatchScenarioMapping.is_active == True,
        )
        .order_by(BatchScenarioMapping.assigned_at.desc())
        .first()
    )


def _sync_call_simulation_scenario_mirror_or_raise(
    db: Session,
    *,
    scenario: Scenario,
    trainer_id: str,
    batch_id: Optional[str] = None,
    detail: str,
) -> None:
    try:
        supabase = _require_supabase_storage(detail)
        resolved_batch_id = batch_id
        if not resolved_batch_id:
            latest_mapping = _get_latest_active_batch_mapping_for_scenario(db, scenario.id)
            resolved_batch_id = latest_mapping.batch_id if latest_mapping else None

        _sync_call_simulation_scenario_to_supabase_from_db(
            supabase,
            db=db,
            scenario=scenario,
            trainer_id=trainer_id,
            batch_id=resolved_batch_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Call Simulation Supabase mirror sync failed for scenario %s: %s",
            getattr(scenario, "id", None),
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"{detail.rstrip('.')} Details: {exc}",
        ) from exc


def _sync_call_simulation_batch_mirror_or_raise(
    db: Session,
    *,
    batch_id: str,
    fallback_trainer_id: str,
    detail: str,
) -> None:
    try:
        supabase = _require_supabase_storage(detail)
        mappings = (
            db.query(BatchScenarioMapping)
            .filter(
                BatchScenarioMapping.batch_id == batch_id,
                BatchScenarioMapping.is_active == True,
            )
            .order_by(BatchScenarioMapping.assigned_at.desc())
            .all()
        )

        synced_scenario_ids: set[str] = set()
        for mapping in mappings:
            if not mapping.scenario_id or mapping.scenario_id in synced_scenario_ids:
                continue
            scenario = db.query(Scenario).filter(Scenario.id == mapping.scenario_id).first()
            if not scenario or not scenario.is_published:
                continue
            _sync_call_simulation_scenario_to_supabase_from_db(
                supabase,
                db=db,
                scenario=scenario,
                trainer_id=str(scenario.created_by or fallback_trainer_id),
                batch_id=batch_id,
            )
            synced_scenario_ids.add(mapping.scenario_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Call Simulation Supabase batch mirror sync failed for batch %s: %s",
            batch_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"{detail.rstrip('.')} Details: {exc}",
        ) from exc


def _delete_supabase_scenario_mirror(
    db: Session,
    supabase: Any,
    *,
    source_scenario_id: str,
) -> None:
    if _legacy_call_simulation_mirror_relations_ready(
        db,
        ("call_scenarios",),
        context=f"scenario delete for {source_scenario_id}",
    ):
        try:
            supabase.table("call_scenarios").delete().eq("source_scenario_id", source_scenario_id).execute()
        except Exception as exc:
            if _is_missing_supabase_table_error(exc):
                _log_legacy_supabase_mirror_skip(
                    f"scenario delete for {source_scenario_id}",
                    exc,
                )
            else:
                supabase.table("call_scenarios").delete().eq("scenario_id", source_scenario_id).execute()

    normalized_scenario_id = _normalize_uuid_candidate(source_scenario_id)
    if not normalized_scenario_id:
        return

    if not _legacy_call_simulation_mirror_relations_ready(
        db,
        ("scenario_scripts", "scripts", "kpi_metrics", "scenario_groups", "scenarios"),
        context=f"scenario authoring delete for {source_scenario_id}",
    ):
        return

    try:
        supabase.table("scenario_scripts").delete().eq("scenario_group_id", normalized_scenario_id).execute()
        supabase.table("scripts").delete().eq("scenario_id", normalized_scenario_id).execute()
        supabase.table("kpi_metrics").delete().eq("scenario_group_id", normalized_scenario_id).execute()
        supabase.table("scenario_groups").delete().eq("id", normalized_scenario_id).execute()
        supabase.table("scenarios").delete().eq("id", normalized_scenario_id).execute()
    except Exception as exc:
        if _is_missing_supabase_table_error(exc):
            _log_legacy_supabase_mirror_skip(
                f"scenario authoring delete for {source_scenario_id}",
                exc,
            )
            return
        raise


def _delete_call_simulation_scenario_mirror_or_raise(
    db: Session,
    *,
    source_scenario_id: str,
    detail: str,
) -> None:
    try:
        supabase = _require_supabase_storage(detail)
        _delete_supabase_scenario_mirror(
            db,
            supabase,
            source_scenario_id=source_scenario_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Call Simulation Supabase mirror cleanup failed for scenario %s: %s",
            source_scenario_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"{detail.rstrip('.')} Details: {exc}",
        ) from exc


def _get_trainer_call_audio_settings(current_user: User) -> dict[str, Any]:
    ui_preferences = _normalize_json_object(getattr(current_user, "ui_preferences", None))
    settings = _normalize_json_object(ui_preferences.get(CALL_SIMULATION_AUDIO_SETTINGS_KEY))
    return {
        "ringer_audio_url": _normalize_optional_url(settings.get("ringer_audio_url")),
        "hold_audio_url": _normalize_optional_url(settings.get("hold_audio_url")),
        "ringer_audio_source": _normalize_optional_url(settings.get("ringer_audio_source")),
        "hold_audio_source": _normalize_optional_url(settings.get("hold_audio_source")),
        "updated_at": _normalize_optional_url(settings.get("updated_at")),
    }


def _serialize_trainer_call_audio_settings(current_user: User) -> CallSimulationAudioSettingsResponse:
    return CallSimulationAudioSettingsResponse(**_get_trainer_call_audio_settings(current_user))


def _coerce_call_simulation_bool(value: Any, *, fallback: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return fallback


def _resolve_scenario_audio_urls(
    *,
    call_simulation_config: Optional[dict[str, Any]],
    trainer_audio_settings: dict[str, Any],
    current_ringer_audio_url: Any = None,
    current_hold_audio_url: Any = None,
    incoming_ringer_audio_url: Any = _UNSET,
    incoming_hold_audio_url: Any = _UNSET,
) -> tuple[Optional[str], Optional[str]]:
    resolved_config = _normalize_json_object(call_simulation_config)
    use_shared_ringer_audio = _coerce_call_simulation_bool(
        resolved_config.get("use_shared_ringer_audio"),
        fallback=True,
    )
    use_shared_hold_audio = _coerce_call_simulation_bool(
        resolved_config.get("use_shared_hold_audio"),
        fallback=True,
    )

    trainer_ringer_audio_url = _normalize_optional_url(trainer_audio_settings.get("ringer_audio_url"))
    trainer_hold_audio_url = _normalize_optional_url(trainer_audio_settings.get("hold_audio_url"))

    ringer_candidate = current_ringer_audio_url if incoming_ringer_audio_url is _UNSET else incoming_ringer_audio_url
    hold_candidate = current_hold_audio_url if incoming_hold_audio_url is _UNSET else incoming_hold_audio_url
    resolved_ringer_audio_url = trainer_ringer_audio_url if use_shared_ringer_audio else _normalize_optional_url(ringer_candidate)
    resolved_hold_audio_url = trainer_hold_audio_url if use_shared_hold_audio else _normalize_optional_url(hold_candidate)

    return (
        _validate_optional_call_simulation_storage_url(
            resolved_ringer_audio_url,
            label="Ringer audio",
        ),
        _validate_optional_call_simulation_storage_url(
            resolved_hold_audio_url,
            label="Hold audio",
        ),
    )


def _persist_trainer_call_audio_settings(
    db: Session,
    current_user: User,
    *,
    ringer_audio_url: Any = _UNSET,
    hold_audio_url: Any = _UNSET,
    ringer_audio_source: Any = _UNSET,
    hold_audio_source: Any = _UNSET,
) -> CallSimulationAudioSettingsResponse:
    ui_preferences = _normalize_json_object(getattr(current_user, "ui_preferences", None))
    existing_settings = _normalize_json_object(ui_preferences.get(CALL_SIMULATION_AUDIO_SETTINGS_KEY))

    settings = {
        "ringer_audio_url": _normalize_optional_url(existing_settings.get("ringer_audio_url")),
        "hold_audio_url": _normalize_optional_url(existing_settings.get("hold_audio_url")),
        "ringer_audio_source": _normalize_optional_url(existing_settings.get("ringer_audio_source")),
        "hold_audio_source": _normalize_optional_url(existing_settings.get("hold_audio_source")),
        "updated_at": _normalize_optional_url(existing_settings.get("updated_at")),
    }
    previous_settings = dict(settings)
    scenario_updates: dict[str, Optional[str]] = {}
    impacted_scenarios: list[Scenario] = []

    if ringer_audio_url is not _UNSET:
        next_url = _validate_optional_call_simulation_storage_url(
            ringer_audio_url,
            label="Shared ringer audio",
        )
        settings["ringer_audio_url"] = next_url
        settings["ringer_audio_source"] = None if next_url is None else _normalize_optional_url(ringer_audio_source) or "manual"
        scenario_updates["ringer_audio_url"] = next_url

    if hold_audio_url is not _UNSET:
        next_url = _validate_optional_call_simulation_storage_url(
            hold_audio_url,
            label="Shared hold audio",
        )
        settings["hold_audio_url"] = next_url
        settings["hold_audio_source"] = None if next_url is None else _normalize_optional_url(hold_audio_source) or "manual"
        scenario_updates["hold_audio_url"] = next_url

    settings["updated_at"] = datetime.utcnow().isoformat()
    ui_preferences[CALL_SIMULATION_AUDIO_SETTINGS_KEY] = settings
    current_user.ui_preferences = ui_preferences
    db.add(current_user)

    if scenario_updates:
        trainer_scenarios = (
            db.query(Scenario)
            .filter(Scenario.created_by == current_user.id)
            .all()
        )
        for scenario in trainer_scenarios:
            scenario_config = _resolve_call_simulation_config(scenario)
            scenario_changed = False
            if (
                "ringer_audio_url" in scenario_updates
                and _coerce_call_simulation_bool(
                    scenario_config.get("use_shared_ringer_audio"),
                    fallback=True,
                )
            ):
                scenario.ringer_audio_url = settings["ringer_audio_url"]
                scenario_changed = True
            if (
                "hold_audio_url" in scenario_updates
                and _coerce_call_simulation_bool(
                    scenario_config.get("use_shared_hold_audio"),
                    fallback=True,
                )
            ):
                scenario.hold_audio_url = settings["hold_audio_url"]
                scenario_changed = True
            if scenario_changed:
                impacted_scenarios.append(scenario)

    db.flush()
    for scenario in impacted_scenarios:
        _sync_call_simulation_scenario_mirror_or_raise(
            db,
            scenario=scenario,
            trainer_id=str(scenario.created_by or current_user.id),
            detail="Supabase sync is required before saving Call Simulation audio settings.",
        )

    for asset_kind in ("ringer", "hold"):
        current_url = settings.get(f"{asset_kind}_audio_url")
        current_source = settings.get(f"{asset_kind}_audio_source")
        previous_url = previous_settings.get(f"{asset_kind}_audio_url")

        if current_url:
            existing_record = _find_active_call_simulation_audio_asset(
                db,
                trainer_id=current_user.id,
                public_url=current_url,
                asset_kinds=[asset_kind],
            )
            if existing_record is None:
                source_type = "manual_url"
                if current_source == "upload":
                    source_type = "upload"
                elif current_source == "generated_tts":
                    source_type = "generated_tts"
                _create_call_simulation_audio_asset_record(
                    db,
                    trainer_id=current_user.id,
                    scenario_id=None,
                    script_turn_id=None,
                    step_number=None,
                    asset_kind=asset_kind,
                    source_type=source_type,
                    public_url=current_url,
                    file_name=None,
                    file_type=None,
                    asset_metadata={"scope": "shared"},
                )

        if previous_url and previous_url != current_url:
            stale_records = (
                db.query(CallSimulationAudioAsset)
                .filter(
                    CallSimulationAudioAsset.trainer_id == current_user.id,
                    CallSimulationAudioAsset.public_url == previous_url,
                    CallSimulationAudioAsset.asset_kind == asset_kind,
                    CallSimulationAudioAsset.is_active == True,
                    CallSimulationAudioAsset.scenario_id.is_(None),
                )
                .all()
            )
            for stale_record in stale_records:
                _deactivate_call_simulation_audio_asset_record(
                    db,
                    asset=stale_record,
                    delete_storage=False,
                )

    db.commit()
    db.refresh(current_user)

    supabase = get_supabase_client()
    if supabase.is_available:
        for asset_kind in ("ringer", "hold"):
            previous_url = previous_settings.get(f"{asset_kind}_audio_url")
            previous_source = previous_settings.get(f"{asset_kind}_audio_source")
            current_url = settings.get(f"{asset_kind}_audio_url")
            if previous_source in {"upload", "generated_tts"} and previous_url and previous_url != current_url:
                supabase.delete_by_public_url(previous_url)

    return _serialize_trainer_call_audio_settings(current_user)


def _resolve_call_simulation_config(scenario: Optional[Scenario]) -> dict[str, Any]:
    if not scenario:
        return {}
    return _normalize_json_object(getattr(scenario, "call_simulation_config", None))


def _read_call_scenario_passing_score_from_config(
    config: Optional[dict[str, Any]],
) -> Optional[float]:
    resolved_config = _normalize_json_object(config)
    legacy_target_kpis = _normalize_json_object(resolved_config.get("target_kpis"))
    for candidate in (
        resolved_config.get("passing_score"),
        resolved_config.get("certification_threshold"),
        legacy_target_kpis.get("passing_score"),
    ):
        try:
            if candidate is None:
                continue
            resolved = float(candidate)
        except (TypeError, ValueError):
            continue
        if 0.0 <= resolved <= 100.0:
            return round(resolved, 2)
    return None


def _resolve_call_scenario_max_attempts(
    scenario: Optional[Scenario],
    fallback: int = DEFAULT_MAX_ATTEMPTS,
) -> int:
    resolved_fallback = _coerce_positive_int(fallback, fallback=DEFAULT_MAX_ATTEMPTS)
    config = _resolve_call_simulation_config(scenario)
    for candidate in (
        config.get("max_attempts"),
        config.get("attempt_limit"),
        config.get("retake_limit"),
    ):
        try:
            resolved = int(candidate)
        except (TypeError, ValueError):
            continue
        if resolved >= 1:
            return resolved
    return resolved_fallback


def _resolve_call_assignment_max_attempts(
    assignment: Optional[CallSimulationAssignment],
    *,
    scenario: Optional[Scenario],
    fallback: int = DEFAULT_MAX_ATTEMPTS,
) -> int:
    scenario_fallback = _resolve_call_scenario_max_attempts(scenario, fallback=fallback)
    return _coerce_positive_int(
        assignment.max_attempts if assignment else None,
        fallback=scenario_fallback,
    )


def _resolve_call_scenario_passing_score(
    scenario: Optional[Scenario],
    fallback: float,
) -> float:
    configured_passing_score = _read_call_scenario_passing_score_from_config(
        _resolve_call_simulation_config(scenario)
    )
    for candidate in (configured_passing_score, fallback):
        try:
            if candidate is None:
                continue
            resolved = float(candidate)
        except (TypeError, ValueError):
            continue
        if 0.0 <= resolved <= 100.0:
            return round(resolved, 2)

    return DEFAULT_PASSING_SCORE


def _apply_call_scenario_passing_score(
    evaluation: dict[str, Any],
    *,
    scenario: Optional[Scenario],
    kpi_config: BatchKPIConfig,
) -> float:
    passing_score = _resolve_call_scenario_passing_score(
        scenario,
        float(kpi_config.passing_score or DEFAULT_PASSING_SCORE),
    )
    evaluation["passing_score"] = passing_score
    evaluation["pass_fail"] = float(evaluation.get("weighted_score") or 0.0) >= passing_score
    return passing_score


def _serialize_flow_step(step: ScenarioFlow) -> CallSimulationScenarioStepResponse:
    actor = (step.speaker_role or "").strip().lower()
    if not actor:
        actor = "csr" if step.step_type == "agent_response" else "member"

    script = (
        step.expected_response
        if actor == "csr"
        else step.prompt_text
    ) or ""

    return CallSimulationScenarioStepResponse(
        id=step.id,
        step_number=int(step.step_number or 0),
        actor=actor,
        speaker_label=step.speaker_label,
        script=script,
        expected_keywords=list(step.expected_keywords_for_step or []),
        audio_url=_resolve_call_simulation_step_audio_url(step),
        response_time_limit=step.response_time_limit,
        is_closing=bool(step.is_closing),
        metadata=_normalize_json_object(step.step_metadata),
    )


def _build_scenario_steps(
    scenario: Scenario,
    *,
    variations: Optional[list[ScenarioVariation]] = None,
) -> list[CallSimulationScenarioStepResponse]:
    ordered_flow_steps = sorted(
        list(getattr(scenario, "flow_steps", []) or []),
        key=lambda step: (step.step_number or 0, step.created_at or datetime.utcnow()),
    )
    if ordered_flow_steps:
        return [_serialize_flow_step(step) for step in ordered_flow_steps]

    active_variations = variations or [
        variation for variation in getattr(scenario, "variations", []) or []
        if variation.is_active
    ]
    fallback_steps: list[CallSimulationScenarioStepResponse] = []
    if scenario.opening_prompt:
        fallback_steps.append(
            CallSimulationScenarioStepResponse(
                step_number=1,
                actor="member",
                speaker_label="Member",
                script=scenario.opening_prompt,
                expected_keywords=[],
                audio_url=scenario.opening_prompt_audio,
                metadata={},
            )
        )
    if active_variations:
        fallback_steps.append(
            CallSimulationScenarioStepResponse(
                step_number=len(fallback_steps) + 1,
                actor="csr",
                speaker_label="CSR",
                script=active_variations[0].script,
                expected_keywords=list(scenario.expected_keywords or []),
                metadata={},
            )
        )
    elif scenario.opening_prompt:
        fallback_steps.append(
            CallSimulationScenarioStepResponse(
                step_number=len(fallback_steps) + 1,
                actor="csr",
                speaker_label="CSR",
                script=scenario.opening_prompt,
                expected_keywords=list(scenario.expected_keywords or []),
                metadata={},
            )
        )
    return fallback_steps


def _count_scenario_groups(
    steps: list[CallSimulationScenarioStepResponse],
) -> int:
    if not steps:
        return 0

    scenario_groups: set[str] = set()
    for step in steps:
        metadata = _normalize_json_object(getattr(step, "metadata", None))
        for key in ("scenario_group", "script_flow_step_id", "scenario", "scenario_label", "member_context"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                scenario_groups.add(value.strip())
                break

    if scenario_groups:
        return len(scenario_groups)

    csr_turn_count = len([step for step in steps if step.actor == "csr"])
    if csr_turn_count:
        return csr_turn_count

    return max(1, len(steps))


def _resolve_initial_csr_step_number(
    steps: list[CallSimulationScenarioStepResponse],
) -> int:
    """Start fresh trainee sessions on the first CSR turn, not a member prompt row."""
    for step in steps:
        if str(step.actor or "").strip().lower() == "csr":
            return int(step.step_number or 1)
    return int(steps[0].step_number or 1) if steps else 1


def _resolve_session_current_step_for_response(
    session: SimSession,
    steps: list[CallSimulationScenarioStepResponse],
) -> int:
    """Normalize the start step for fresh or legacy sessions before the trainee UI loads."""
    current_step = int(session.current_step or 1)
    if not steps:
        return current_step

    if current_step < 1:
        return _resolve_initial_csr_step_number(steps)

    step_lookup = {int(step.step_number or 0): step for step in steps}
    current_step_entry = step_lookup.get(current_step)
    if current_step_entry is None:
        return _resolve_initial_csr_step_number(steps)

    has_saved_turns = bool(_session_turn_logs(session))
    if not has_saved_turns and str(current_step_entry.actor or "").strip().lower() != "csr":
        return _resolve_initial_csr_step_number(steps)

    return current_step


def _resolve_next_pending_csr_step_number(
    session: SimSession,
    steps: list[CallSimulationScenarioStepResponse],
) -> Optional[int]:
    """Return the next CSR step that is still waiting for an accepted submission."""
    ordered_csr_steps = [
        step
        for step in sorted(steps, key=lambda item: int(item.step_number or 0))
        if _is_csr_actor(step.actor)
    ]
    if not ordered_csr_steps:
        return None

    latest_turn_by_step: dict[int, dict[str, Any]] = {}
    for item in _session_turn_logs(session):
        if not _is_csr_actor(str(item.get("actor") or "")):
            continue
        step_number = int(item.get("step_number") or 0)
        if step_number <= 0:
            continue
        existing_entry = latest_turn_by_step.get(step_number)
        if existing_entry is None or int(item.get("turn_attempt_number") or 0) >= int(
            existing_entry.get("turn_attempt_number") or 0
        ):
            latest_turn_by_step[step_number] = item

    for step in ordered_csr_steps:
        latest_turn = latest_turn_by_step.get(int(step.step_number or 0))
        if latest_turn is None:
            return int(step.step_number or 0)
        if not bool(latest_turn.get("accepted_for_progress")):
            return int(step.step_number or 0)

    return None


def _build_session_start_response(
    db: Session,
    *,
    session: SimSession,
    scenario: Scenario,
    batch_id: Optional[str],
    variation: Optional[ScenarioVariation] = None,
    kpi_config: Optional[BatchKPIConfig] = None,
) -> SimSessionStartResponse:
    effective_batch_id = batch_id or session.batch_id
    effective_kpi = kpi_config or _get_effective_kpi_config(db, effective_batch_id)
    effective_call_simulation_config = _resolve_call_simulation_config_for_batch(
        db,
        scenario,
        batch_id=effective_batch_id,
    )
    passing_score = _resolve_call_scenario_passing_score(
        scenario,
        float(effective_kpi.passing_score or DEFAULT_PASSING_SCORE),
    )
    steps = _build_scenario_steps(scenario)

    return SimSessionStartResponse(
        session_id=session.id,
        assignment_id=session.assignment_id,
        assigned_by_id=session.assigned_by_id,
        attempt_number=int(session.attempt_number or 1),
        max_attempts=int(session.max_attempts or 1),
        scenario_title=scenario.title,
        scenario_description=scenario.description,
        opening_prompt=scenario.opening_prompt,
        current_step=_resolve_session_current_step_for_response(session, steps),
        variation=ScenarioVariationResponse.model_validate(variation) if variation else None,
        kpi_config=BatchKPIConfigResponse.model_validate(effective_kpi),
        passing_score=passing_score,
        member_profile=_normalize_json_object(scenario.member_profile),
        cxone_metadata=_normalize_json_object(scenario.cxone_metadata),
        call_simulation_config=effective_call_simulation_config,
        ringer_audio_url=scenario.ringer_audio_url,
        hold_audio_url=scenario.hold_audio_url,
        steps=steps,
    )


def _serialize_scenario(
    db: Session,
    scenario: Scenario,
    *,
    batch: Optional[Batch] = None,
    mapping: Optional[BatchScenarioMapping] = None,
    variations: Optional[list[ScenarioVariation]] = None,
    highlighted_batch_id: Optional[str] = None,
) -> CallSimulationScenarioResponse:
    active_variations = variations
    if active_variations is None:
        active_variations = [variation for variation in scenario.variations if variation.is_active]
    steps = _build_scenario_steps(scenario, variations=active_variations)

    assignment_summaries = _build_scenario_assignment_summaries(
        db,
        scenario.id,
        include_empty_batches=bool(highlighted_batch_id or batch),
    )
    aggregate_summary = _aggregate_assignment_summaries(assignment_summaries)
    highlighted_summary = None
    if highlighted_batch_id:
        highlighted_summary = next(
            (
                summary
                for summary in assignment_summaries
                if summary.batch_id == highlighted_batch_id
            ),
            None,
        )
    if highlighted_summary is None and batch:
        highlighted_summary = next(
            (
                summary
                for summary in assignment_summaries
                if summary.batch_id == batch.id
            ),
            None,
        )

    summary_source = highlighted_summary
    if summary_source is None and assignment_summaries:
        summary_source = assignment_summaries[0]

    resolved_batch_id = None
    if summary_source is not None:
        resolved_batch_id = summary_source.batch_id
    elif highlighted_batch_id:
        resolved_batch_id = highlighted_batch_id
    elif batch is not None:
        resolved_batch_id = batch.id
    elif mapping is not None:
        resolved_batch_id = mapping.batch_id

    effective_call_simulation_config = _resolve_call_simulation_config_for_batch(
        db,
        scenario,
        batch_id=resolved_batch_id,
    )

    return CallSimulationScenarioResponse(
        id=scenario.id,
        title=scenario.title,
        description=scenario.description,
        opening_prompt=scenario.opening_prompt,
        difficulty=scenario.difficulty,
        purpose=scenario.purpose,
        expected_keywords=list(scenario.expected_keywords or []),
        estimated_duration=scenario.estimated_duration,
        member_profile=_normalize_json_object(scenario.member_profile),
        cxone_metadata=_normalize_json_object(scenario.cxone_metadata),
        call_simulation_config=effective_call_simulation_config,
        ringer_audio_url=scenario.ringer_audio_url,
        hold_audio_url=scenario.hold_audio_url,
        batch_id=(summary_source.batch_id if summary_source else batch.id if batch else None),
        batch_name=(summary_source.batch_name if summary_source else batch.name if batch else None),
        assigned_at=(summary_source.assigned_at if summary_source else mapping.assigned_at if mapping else None),
        is_published=bool(scenario.is_published),
        is_draft=bool(scenario.is_draft),
        variations_count=len(active_variations),
        variations=[
            ScenarioVariationResponse.model_validate(variation)
            for variation in active_variations
        ],
        steps_count=len(steps),
        steps=steps,
        assigned_batches=assignment_summaries,
        member_count=summary_source.trainee_count if summary_source else aggregate_summary["member_count"],
        completed_sessions=summary_source.completed_sessions if summary_source else aggregate_summary["completed_sessions"],
        passed_sessions=summary_source.passed_sessions if summary_source else aggregate_summary["passed_sessions"],
        average_score=summary_source.average_score if summary_source else aggregate_summary["average_score"],
        pass_rate=summary_source.pass_rate if summary_source else aggregate_summary["pass_rate"],
        latest_completed_at=(
            summary_source.latest_completed_at if summary_source else aggregate_summary["latest_completed_at"]
        ),
        created_at=scenario.created_at,
        updated_at=scenario.updated_at,
    )


def _get_trainer_batch_ids(db: Session, current_user: User) -> list[str]:
    if current_user.role == UserRole.ADMIN:
        return [batch_id for (batch_id,) in db.query(Batch.id).all()]
    return [
        batch_id
        for (batch_id,) in db.query(Batch.id)
        .filter(Batch.created_by == current_user.id)
        .all()
    ]


def _get_accessible_batch(db: Session, current_user: User, batch_id: str) -> Batch:
    query = db.query(Batch).filter(Batch.id == batch_id)
    if current_user.role != UserRole.ADMIN:
        query = query.filter(Batch.created_by == current_user.id)
    batch = query.first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


def _get_accessible_scenario(db: Session, current_user: User, scenario_id: str) -> Scenario:
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    if current_user.role == UserRole.ADMIN:
        return scenario

    trainer_batch_ids = _get_trainer_batch_ids(db, current_user)
    has_batch_mapping = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.scenario_id == scenario_id,
            BatchScenarioMapping.batch_id.in_(trainer_batch_ids or ["__none__"]),
        )
        .first()
        is not None
    )
    if scenario.created_by != current_user.id and not has_batch_mapping:
        raise HTTPException(status_code=403, detail="Access denied")

    return scenario


def _get_accessible_session(db: Session, current_user: User, session_id: str) -> SimSession:
    session = db.query(SimSession).filter(SimSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user.role == UserRole.TRAINEE:
        if session.trainee_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        assignment = _get_active_call_simulation_assignment(
            db,
            trainee_id=current_user.id,
            scenario_id=session.scenario_id,
        )
        if not assignment:
            raise HTTPException(
                status_code=403,
                detail="This Call Simulation is no longer assigned to your trainee workspace.",
            )

        trainee_batch_ids = {batch.id for batch in current_user.batches if batch.id}
        if assignment.batch_id and assignment.batch_id not in trainee_batch_ids:
            raise HTTPException(
                status_code=403,
                detail="The assigned batch for this Call Simulation is no longer active for your trainee profile.",
            )
        if assignment.batch_id and session.batch_id and assignment.batch_id != session.batch_id:
            raise HTTPException(
                status_code=403,
                detail="This session no longer matches the active batch assignment for the Call Simulation.",
            )

    if current_user.role in [UserRole.TRAINER, UserRole.ADMIN]:
        if current_user.role == UserRole.ADMIN:
            return session

        trainer_batch_ids = set(_get_trainer_batch_ids(db, current_user))
        if session.batch_id not in trainer_batch_ids:
            raise HTTPException(status_code=403, detail="Access denied")

    return session


def _evaluate_submission(
    *,
    transcript: str,
    audio_duration_seconds: int,
    kpi_config: BatchKPIConfig,
    assessment: Optional[dict[str, Any]] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    assessment = assessment or {}
    overrides = overrides or {}
    scores = assessment.get("scores") or {}

    transcript_confidence = float(
        overrides.get("transcript_confidence")
        if overrides.get("transcript_confidence") is not None
        else assessment.get("transcription_confidence") or 0.0
    )

    provider_confidence_score = float(
        overrides.get("speech_to_text_accuracy")
        if overrides.get("speech_to_text_accuracy") is not None
        else scores.get("transcription_confidence") or (transcript_confidence * 100.0)
    )
    alignment_accuracy = float(assessment.get("accuracy_percentage") or 0.0)
    if overrides.get("speech_to_text_accuracy") is not None:
        speech_to_text_accuracy = float(overrides["speech_to_text_accuracy"])
    elif provider_confidence_score or alignment_accuracy:
        speech_to_text_accuracy = round(
            (provider_confidence_score * 0.55) + (alignment_accuracy * 0.45),
            2,
        )
    else:
        speech_to_text_accuracy = 0.0

    rate_of_speech = overrides.get("rate_of_speech")
    if rate_of_speech is None:
        rate_of_speech = scores.get("speech_rate_wpm")
    if rate_of_speech is None and transcript and audio_duration_seconds > 0:
        rate_of_speech = round(
            (len(transcript.split()) / max(audio_duration_seconds, 1)) * 60.0,
            2,
        )
    rate_of_speech = float(rate_of_speech or 0.0)

    dead_air_seconds = overrides.get("dead_air_seconds")
    if dead_air_seconds is None:
        dead_air_seconds = _estimate_dead_air_seconds(
            assessment.get("word_feedback") or [],
            float(audio_duration_seconds),
        )
    dead_air_seconds = float(dead_air_seconds or 0.0)

    grammar_score = float(
        overrides.get("grammar_score")
        if overrides.get("grammar_score") is not None
        else scores.get("grammar_precision") or 0.0
    )
    pronunciation_score = float(
        overrides.get("pronunciation_score")
        if overrides.get("pronunciation_score") is not None
        else scores.get("phonetic_accuracy") or alignment_accuracy or 0.0
    )

    fluency_score = float(scores.get("fluency") or 0.0)
    ros_score = _score_target_delta(
        rate_of_speech,
        float(kpi_config.target_ros_words_per_min or 150.0),
        penalty_multiplier=95.0,
    ) if rate_of_speech > 0 else 0.0
    pacing_score = float(
        overrides.get("pacing_score")
        if overrides.get("pacing_score") is not None
        else round((fluency_score + ros_score) / 2.0, 2)
        if (fluency_score or ros_score)
        else 0.0
    )

    empathy_matches = _find_keyword_matches(transcript, kpi_config.empathy_keywords)
    probing_matches = _find_keyword_matches(transcript, kpi_config.probing_keywords)
    forbidden_matches = sorted(
        set(
            _detect_forbidden_words(transcript, kpi_config.forbidden_words)
            + list(overrides.get("detected_forbidden_words") or [])
        )
    )

    aht_actual = int(round(audio_duration_seconds))
    aht_target = int(kpi_config.target_aht_seconds or 120)
    aht_score = _score_aht(aht_actual, aht_target)
    dead_air_score = _score_dead_air(
        dead_air_seconds,
        float(kpi_config.target_dead_air_seconds or 3.0),
    )
    empathy_score = _score_behavioral_hits(
        len(empathy_matches),
        kpi_config.empathy_keywords,
    )
    probing_score = _score_behavioral_hits(
        len(probing_matches),
        kpi_config.probing_keywords,
    )
    forbidden_penalty = min(
        100.0,
        len(forbidden_matches) * float(kpi_config.forbidden_words_penalty or 0.0),
    )

    weighted_score = _calculate_weighted_score(
        speech_to_text_accuracy=speech_to_text_accuracy,
        aht_score=aht_score,
        rate_of_speech_score=ros_score,
        dead_air_score=dead_air_score,
        empathy_score=empathy_score,
        probing_score=probing_score,
        grammar_score=grammar_score,
        pronunciation_score=pronunciation_score,
        pacing_score=pacing_score,
        forbidden_word_penalty=forbidden_penalty,
        kpi_config=kpi_config,
    )

    pass_fail = weighted_score >= float(kpi_config.passing_score or DEFAULT_PASSING_SCORE)

    return {
        "transcript_confidence": transcript_confidence,
        "speech_to_text_accuracy": speech_to_text_accuracy,
        "aht_actual": aht_actual,
        "aht_score": aht_score,
        "aht_target": aht_target,
        "rate_of_speech": rate_of_speech,
        "rate_of_speech_score": ros_score,
        "dead_air_seconds": dead_air_seconds,
        "dead_air_score": dead_air_score,
        "empathy_matches": empathy_matches,
        "empathy_count": len(empathy_matches),
        "empathy_score": empathy_score,
        "probing_matches": probing_matches,
        "probing_count": len(probing_matches),
        "probing_score": probing_score,
        "grammar_score": grammar_score,
        "pronunciation_score": pronunciation_score,
        "pacing_score": pacing_score,
        "forbidden_matches": forbidden_matches,
        "forbidden_penalty": forbidden_penalty,
        "weighted_score": weighted_score,
        "pass_fail": pass_fail,
    }


def _build_ai_feedback(
    *,
    evaluation: dict[str, Any],
    kpi_config: BatchKPIConfig,
    assessment: Optional[dict[str, Any]] = None,
    fallback_message: Optional[str] = None,
) -> str:
    parts: list[str] = []
    passing_score = float(
        evaluation.get("passing_score")
        or kpi_config.passing_score
        or DEFAULT_PASSING_SCORE
    )
    weighted_score = float(evaluation.get("weighted_score") or 0.0)
    scenario_score = float(evaluation.get("scenario_score") or 0.0)
    kpi_score = float(evaluation.get("kpi_score") or 0.0)
    scenario_points_earned = float(evaluation.get("scenario_points_earned") or 0.0)
    scenario_points_total = float(evaluation.get("scenario_points_total") or 0.0)

    if fallback_message:
        parts.append(fallback_message)
    elif evaluation.get("pass_fail"):
        parts.append(
            f"Passed - Ready for coaching. Final score {weighted_score:.1f}% against a {passing_score:.0f}% target."
        )
    else:
        parts.append(
            f"Failed - Retake required. Final score {weighted_score:.1f}% and the target is {passing_score:.0f}%."
        )

    if scenario_points_total > 0:
        parts.append(
            f"Scenario score {scenario_score:.1f}% ({scenario_points_earned:.1f} of {scenario_points_total:.1f} points) and KPI score {kpi_score:.1f}% were combined for the final result."
        )

    if evaluation.get("forbidden_matches"):
        parts.append(
            "Forbidden words detected: "
            + ", ".join(evaluation["forbidden_matches"][:5])
            + "."
        )

    if not evaluation.get("empathy_count") and _normalize_keyword_list(kpi_config.empathy_keywords):
        parts.append("Add at least one empathy statement before moving into resolution.")

    if not evaluation.get("probing_count") and _normalize_keyword_list(kpi_config.probing_keywords):
        parts.append("Use more probing questions to confirm the customer's exact need.")

    keyword_compliance = _normalize_json_object(evaluation.get("keyword_compliance"))
    missing_items = list(keyword_compliance.get("missing") or [])
    if missing_items:
        parts.append(
            "Missing required call phrases: " + ", ".join(missing_items[:3]) + "."
        )

    sentiment_score = evaluation.get("sentiment_score")
    if isinstance(sentiment_score, (int, float)):
        parts.append(
            f"Sentiment score {float(sentiment_score):.2f} ({_sentiment_label(float(sentiment_score))})."
        )

    coaching_tips = list((assessment or {}).get("coaching_tips") or [])
    if coaching_tips:
        parts.append("Focus next on: " + " ".join(coaching_tips[:2]))

    if not parts:
        parts.append("Session completed.")

    return " ".join(part.strip() for part in parts if part and part.strip())


def _apply_evaluation_to_session(
    *,
    session: SimSession,
    transcript: str,
    audio_url: Optional[str],
    audio_duration_seconds: int,
    evaluation: dict[str, Any],
    ai_feedback: str,
    completed_at: Optional[datetime] = None,
) -> None:
    session.audio_url = audio_url
    session.audio_duration_seconds = audio_duration_seconds
    session.transcript = transcript
    session.transcript_confidence = evaluation.get("transcript_confidence")
    session.completed_at = completed_at or datetime.utcnow()
    session.status = "completed" if evaluation.get("pass_fail") else "failed"
    session.speech_to_text_accuracy = evaluation.get("speech_to_text_accuracy")
    session.aht_target = evaluation.get("aht_target")
    session.aht_actual = evaluation.get("aht_actual")
    session.rate_of_speech = evaluation.get("rate_of_speech")
    session.dead_air_seconds = evaluation.get("dead_air_seconds")
    session.empathy_statements_count = evaluation.get("empathy_count", 0)
    session.probing_questions_count = evaluation.get("probing_count", 0)
    session.grammar_score = evaluation.get("grammar_score")
    session.pronunciation_score = evaluation.get("pronunciation_score")
    session.pacing_score = evaluation.get("pacing_score")
    session.sentiment_score = evaluation.get("sentiment_score")
    session.keyword_compliance = _normalize_json_object(evaluation.get("keyword_compliance"))
    session.forbidden_words_count = len(evaluation.get("forbidden_matches") or [])
    session.forbidden_words_detected = evaluation.get("forbidden_matches") or []
    session.forbidden_word_penalty_applied = evaluation.get("forbidden_penalty", 0.0)
    session.weighted_score = evaluation.get("weighted_score")
    session.pass_fail = bool(evaluation.get("pass_fail"))
    session.ai_feedback = ai_feedback


def _session_outcome_label(session: SimSession) -> str:
    if bool(session.pass_fail):
        return "Passed / Completed"
    return "Failed / For Retake"


def _sync_call_simulation_certificate(
    db: Session,
    *,
    session: SimSession,
    scenario: Optional[Scenario],
    issuer_id: Optional[str] = None,
) -> None:
    passing_score = _resolve_call_scenario_passing_score(
        scenario,
        AUTO_CERTIFICATE_PASSING_SCORE,
    )
    weighted_score = float(session.weighted_score or 0.0)
    verdict_status = (session.trainer_verdict_status or "pending").strip().lower()
    should_issue = (
        verdict_status == "competent"
        or (
            weighted_score >= passing_score
            and verdict_status != "retake"
        )
    )

    existing_certificate = (
        db.query(CertificateRecord)
        .filter(
            CertificateRecord.source_type == "call_simulation_session",
            CertificateRecord.source_id == session.id,
        )
        .first()
    )

    if not should_issue:
        session.certificate_id = None
        if existing_certificate:
            db.delete(existing_certificate)
        return

    resolved_issuer_id = (
        issuer_id
        or session.trainer_evaluated_by
        or (scenario.created_by if scenario else None)
        or session.trainee_id
    )
    certificate, _ = award_certificate(
        db,
        trainee_id=session.trainee_id,
        issuer_id=resolved_issuer_id,
        source_type="call_simulation_session",
        source_id=session.id,
        achievement_title=scenario.title if scenario else "Call Simulation Scenario",
        achievement_type="completion",
        remarks=session.ai_feedback,
        score=weighted_score,
        issued_at=session.completed_at or datetime.utcnow(),
    )
    session.certificate_id = certificate.id


async def _replace_scenario_steps(
    db: Session,
    *,
    scenario: Scenario,
    steps: list[Any],
) -> list[ScenarioFlow]:
    existing_steps = (
        db.query(ScenarioFlow)
        .filter(ScenarioFlow.scenario_id == scenario.id)
        .all()
    )
    for step in existing_steps:
        db.delete(step)
    db.flush()

    # Generate multi-speaker conversation audio if available
    conversation_audio_url = None
    if gemini_tts_engine.is_available():
        try:
            # Build conversation script
            conversation_parts = []
            speaker_configs = []
            
            for step in sorted(steps, key=lambda item: item.step_number):
                actor = (step.actor or "").strip().lower()
                speaker_label = (step.speaker_label or ("CSR" if actor == "csr" else "Member")).strip()
                script = step.script.strip()
                if script:
                    conversation_parts.append(f"{speaker_label}: {script}")
                    
                    # Add speaker config if not already added
                    if not any(config['speaker'] == speaker_label for config in speaker_configs):
                        speaker_configs.append({
                            'speaker': speaker_label,
                            'voice_name': "Kore" if actor == "csr" else "Puck"
                        })
            
            if conversation_parts:
                conversation_text = "\n".join(conversation_parts)
                prompt = f"TTS the following conversation:\n{conversation_text}"
                
                audio_bytes = gemini_tts_engine.synthesize(
                    prompt,
                    multi_speaker_config=speaker_configs
                )
                if audio_bytes:
                    get_tts_service().save_audio_locally(
                        audio_bytes,
                        scenario_id=scenario.id,
                        step_number=0,
                        asset_kind="conversation-audio",
                    )
                    supabase_client = get_supabase_client()
                    conversation_audio_url = supabase_client.upload_call_simulation_asset(
                        file_data=audio_bytes,
                        trainer_id=scenario.created_by,
                        asset_kind="conversation-audio",
                        filename=f"{scenario.id}_conversation.wav",
                        scenario_id=scenario.id,
                        content_type="audio/wav"
                    )
        except Exception as exc:
            logger.warning("Failed to generate multi-speaker conversation audio: %s", exc)

    created_steps: list[ScenarioFlow] = []
    for index, step in enumerate(sorted(steps, key=lambda item: item.step_number), start=1):
        actor = (step.actor or "").strip().lower()
        speaker_label = (step.speaker_label or ("CSR" if actor == "csr" else "Member")).strip()
        keywords = _normalize_keyword_list(getattr(step, "expected_keywords", []) or [])
        metadata = _normalize_json_object(getattr(step, "metadata", None))
        
        # Use conversation audio for member steps, individual TTS as fallback
        step_audio_url = getattr(step, "audio_url", None)
        if not step_audio_url and step.script.strip():
            if actor != "csr" and conversation_audio_url:
                # For member steps, use the conversation audio
                step_audio_url = conversation_audio_url
            else:
                if get_tts_service().is_available():
                    try:
                        tts_result = await text_to_speech_for_persistence(
                            step.script.strip(),
                            voice_name="Kore" if actor == "csr" else "Puck",
                            speaking_style="professionally" if actor == "csr" else "casually",
                        )
                        upload_bytes = _coerce_audio_bytes(tts_result.get("audio_bytes"))
                        if upload_bytes is None and isinstance(tts_result.get("audio_base64"), str) and tts_result.get("audio_base64").strip():
                            try:
                                upload_bytes = b64decode(tts_result.get("audio_base64"))
                            except Exception:
                                upload_bytes = None

                        if upload_bytes:
                            get_tts_service().save_audio_locally(
                                upload_bytes,
                                scenario_id=scenario.id,
                                step_number=index,
                                asset_kind="step-prompt",
                            )
                            supabase_client = get_supabase_client()
                            step_audio_url = supabase_client.upload_call_simulation_asset(
                                file_data=upload_bytes,
                                trainer_id=scenario.created_by,
                                asset_kind="step-prompts",
                                filename=f"{scenario.id}_step_{index}.wav",
                                scenario_id=scenario.id,
                                content_type=str(tts_result.get("audio_content_type") or "audio/wav").strip() or "audio/wav",
                            )
                    except Exception as exc:
                        logger.warning("Failed to generate TTS for step %d: %s", index, exc)
                else:
                    logger.info(
                        "Skipping persisted TTS generation for step %d because no server-side TTS provider is available.",
                        index,
                    )
        
        created = ScenarioFlow(
            id=str(uuid.uuid4()),
            scenario_id=scenario.id,
            step_number=index,
            step_type="agent_response" if actor == "csr" else "customer_prompt",
            prompt_text=step.script.strip() if actor != "csr" else None,
            prompt_audio=step_audio_url,
            expected_response=step.script.strip() if actor == "csr" else None,
            expected_keywords_for_step=keywords,
            speaker_role=actor or ("csr" if step.step_number % 2 == 0 else "member"),
            speaker_label=speaker_label,
            is_closing=bool(getattr(step, "is_closing", False)),
            response_time_limit=getattr(step, "response_time_limit", None),
            step_metadata=metadata,
        )
        db.add(created)
        created_steps.append(created)
    db.flush()
    return created_steps


def _session_turn_logs(session: SimSession) -> list[dict[str, Any]]:
    return list(session.turn_logs or []) if isinstance(session.turn_logs, list) else []


def _session_transcript_log(session: SimSession) -> list[dict[str, Any]]:
    return list(session.transcript_log or []) if isinstance(session.transcript_log, list) else []


def _count_turn_attempts_for_step(turn_logs: list[dict[str, Any]], step_number: int) -> int:
    target_step = int(step_number or 0)
    if target_step <= 0:
        return 0

    count = 0
    for item in turn_logs:
        if int(item.get("step_number") or 0) != target_step:
            continue
        count += 1
    return count


def _session_member_speech_log(session: SimSession) -> list[dict[str, Any]]:
    """Extract member speech logs for Phase 2 audio reconstruction."""
    return list(session.member_speech_log or []) if isinstance(session.member_speech_log, list) else []


def _extract_member_speech_from_events(db: Session, session_id: str) -> list[dict[str, Any]]:
    """Extract member speech events from call simulation events for Phase 2 audio reconstruction."""
    try:
        events = (
            db.query(CallSimulationEvent)
            .filter(
                CallSimulationEvent.session_id == session_id,
                CallSimulationEvent.event_type == "member_speech_played",
            )
            .order_by(CallSimulationEvent.created_at)
            .all()
        )
        
        member_speeches = []
        for event in events:
            metadata = event.metadata_json or {}
            member_speeches.append({
                "step_number": event.step_number or metadata.get("step_number"),
                "audio_url": metadata.get("audio_url"),
                "script_preview": metadata.get("script_preview"),
                "event_id": event.id,
                "created_at": event.created_at.isoformat() if event.created_at else None,
            })
        return member_speeches
    except Exception as e:
        logging.warning(f"Failed to extract member speech events for session {session_id}: {e}")
        return []


def _select_latest_csr_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    latest_accepted: Optional[dict[str, Any]] = None
    for item in attempts:
        if bool(item.get("accepted_for_progress")):
            latest_accepted = item
    return latest_accepted or attempts[-1] if attempts else {}


def _selected_csr_turns_for_scoring(session: SimSession) -> list[dict[str, Any]]:
    grouped_turns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in _session_turn_logs(session):
        if str(item.get("actor", "")).lower() != "csr":
            continue
        step_number = int(item.get("step_number") or 0)
        if step_number <= 0:
            continue
        grouped_turns[step_number].append(item)

    selected_turns: list[dict[str, Any]] = []
    for step_number in sorted(grouped_turns):
        attempts = grouped_turns[step_number]
        selected_turns.append(_select_latest_csr_attempt(attempts))
    return selected_turns


def _calculate_turn_content_score(turn_log: dict[str, Any]) -> dict[str, float]:
    max_points = max(0.0, float(turn_log.get("point_value") or 0.0))
    if max_points <= 0:
        return {
            "point_value": 0.0,
            "earned_points": 0.0,
            "content_score_percent": 0.0,
        }

    similarity_percent = max(0.0, min(100.0, float(turn_log.get("script_similarity") or 0.0)))
    speech_accuracy = max(0.0, min(100.0, float(turn_log.get("speech_to_text_accuracy") or 0.0)))
    grammar_score = max(0.0, min(100.0, float(turn_log.get("grammar_score") or 0.0)))
    pronunciation_score = max(0.0, min(100.0, float(turn_log.get("pronunciation_score") or 0.0)))
    pacing_score = max(0.0, min(100.0, float(turn_log.get("pacing_score") or 0.0)))

    expected_keywords = _normalize_keyword_list(list(turn_log.get("expected_keywords") or []))
    matched_keywords = _normalize_keyword_list(list(turn_log.get("matched_keywords") or []))
    keyword_score = (
        min(100.0, (len(matched_keywords) / len(expected_keywords)) * 100.0)
        if expected_keywords
        else similarity_percent
    )

    content_score_percent = round(
        (similarity_percent * 0.5)
        + (keyword_score * 0.2)
        + (speech_accuracy * 0.15)
        + (grammar_score * 0.05)
        + (pronunciation_score * 0.05)
        + (pacing_score * 0.05),
        2,
    )
    content_score_percent = max(0.0, min(100.0, content_score_percent))
    earned_points = round(max_points * (content_score_percent / 100.0), 2)
    return {
        "point_value": round(max_points, 2),
        "earned_points": earned_points,
        "content_score_percent": content_score_percent,
    }


def _generate_coaching_insights(session: SimSession) -> dict[str, Any]:
    """
    Phase 5 Enhancement: Generate turn-level coaching insights for trainers.
    Analyzes turn logs to identify specific areas where trainee needs coaching.
    """
    insights = {
        "turns_requiring_focus": [],
        "common_issues": [],
        "strengths_to_build_on": [],
        "recommended_coaching_areas": [],
    }

    turn_logs = _session_turn_logs(session)
    if not turn_logs:
        return insights

    # Analyze turns for coaching opportunities
    low_performing_turns = []
    high_performing_turns = []

    for turn in turn_logs:
        if str(turn.get("actor", "")).lower() != "csr":
            continue

        content_score = float(turn.get("content_score_percent") or 0.0)
        step_number = turn.get("step_number")
        speaker_label = turn.get("speaker_label", "CSR")

        if content_score < 70:
            low_performing_turns.append({
                "step": step_number,
                "speaker": speaker_label,
                "score": round(content_score, 1),
                "similarity": float(turn.get("script_similarity") or 0.0),
                "reason": _identify_turn_weakness(turn),
            })
        elif content_score >= 85:
            high_performing_turns.append({
                "step": step_number,
                "speaker": speaker_label,
                "score": round(content_score, 1),
            })

    # Build coaching recommendations
    if low_performing_turns:
        insights["turns_requiring_focus"] = low_performing_turns
        insights["recommended_coaching_areas"] = list(set(
            t.get("reason") for t in low_performing_turns if t.get("reason")
        ))

    if high_performing_turns:
        insights["strengths_to_build_on"] = [
            f"Turn {t['step']}: Delivered well ({t['score']}%)"
            for t in high_performing_turns
        ]

    # Identify common issues across turns
    grammar_issues = sum(1 for t in turn_logs if float(t.get("grammar_score") or 0.0) < 75)
    pronunciation_issues = sum(1 for t in turn_logs if float(t.get("pronunciation_score") or 0.0) < 75)
    pacing_issues = sum(1 for t in turn_logs if float(t.get("pacing_score") or 0.0) < 75)

    if grammar_issues > 0:
        insights["common_issues"].append(f"Grammar/clarity issues in {grammar_issues} turn(s)")
    if pronunciation_issues > 0:
        insights["common_issues"].append(f"Pronunciation issues in {pronunciation_issues} turn(s)")
    if pacing_issues > 0:
        insights["common_issues"].append(f"Pacing/timing issues in {pacing_issues} turn(s)")

    return insights


def _identify_turn_weakness(turn: dict[str, Any]) -> str:
    """Identify the primary weakness in a turn for coaching purposes."""
    script_similarity = float(turn.get("script_similarity") or 0.0)
    grammar_score = float(turn.get("grammar_score") or 0.0)
    pronunciation_score = float(turn.get("pronunciation_score") or 0.0)
    pacing_score = float(turn.get("pacing_score") or 0.0)

    if script_similarity < 60:
        return "Script accuracy - did not follow the expected statement"
    elif grammar_score < 70:
        return "Grammar and clarity - focus on proper phrasing"
    elif pronunciation_score < 70:
        return "Pronunciation - practice clearer enunciation"
    elif pacing_score < 70:
        return "Pacing and timing - speed up or slow down delivery"
    else:
        return "Overall quality - needs refinement"


def _aggregate_turn_based_evaluation(
    *,
    session: SimSession,
    kpi_config: BatchKPIConfig,
    scenario_steps: Optional[list[CallSimulationScenarioStepResponse]] = None,
) -> tuple[str, int, dict[str, Any], str]:
    csr_turns = _selected_csr_turns_for_scoring(session)
    combined_transcript = " ".join(
        str(item.get("transcript") or item.get("text") or "").strip()
        for item in csr_turns
        if str(item.get("transcript") or item.get("text") or "").strip()
    ).strip()
    spoken_turn_duration = int(round(sum(float(item.get("duration_seconds") or 0.0) for item in csr_turns)))

    if not csr_turns:
        evaluation = _evaluate_submission(
            transcript=combined_transcript,
            audio_duration_seconds=0,
            kpi_config=kpi_config,
            overrides={
                "speech_to_text_accuracy": 0.0,
                "rate_of_speech": 0.0,
                "dead_air_seconds": 0.0,
                "grammar_score": 0.0,
                "pronunciation_score": 0.0,
                "pacing_score": 0.0,
            },
        )
        evaluation["weighted_score"] = 0.0
        evaluation["kpi_score"] = 0.0
        evaluation["scenario_score"] = 0.0
        evaluation["scenario_points_earned"] = 0.0
        evaluation["scenario_points_total"] = 0.0
        evaluation["pass_fail"] = False
        ai_feedback = _build_ai_feedback(
            evaluation=evaluation,
            kpi_config=kpi_config,
            fallback_message="No CSR turn was recorded for this attempt.",
        )
        return combined_transcript, spoken_turn_duration, evaluation, ai_feedback

    speech_accuracy = _average([float(item.get("speech_to_text_accuracy") or 0.0) for item in csr_turns])
    grammar_score = _average([float(item.get("grammar_score") or 0.0) for item in csr_turns])
    pronunciation_score = _average([float(item.get("pronunciation_score") or 0.0) for item in csr_turns])
    pacing_score = _average([float(item.get("pacing_score") or 0.0) for item in csr_turns])
    dead_air_seconds = round(sum(float(item.get("dead_air_seconds") or 0.0) for item in csr_turns), 2)
    total_words = len(combined_transcript.split())
    rate_of_speech = round((total_words / max(spoken_turn_duration, 1)) * 60.0, 2) if spoken_turn_duration else 0.0

    evaluation = _evaluate_submission(
        transcript=combined_transcript,
        audio_duration_seconds=spoken_turn_duration,
        kpi_config=kpi_config,
        overrides={
            "speech_to_text_accuracy": speech_accuracy,
            "rate_of_speech": rate_of_speech,
            "dead_air_seconds": dead_air_seconds,
            "grammar_score": grammar_score,
            "pronunciation_score": pronunciation_score,
            "pacing_score": pacing_score,
            "detected_forbidden_words": sorted(
                {
                    word
                    for item in csr_turns
                    for word in list(item.get("forbidden_matches") or [])
                }
            ),
        },
    )
    estimated_member_duration = 0.0
    if scenario_steps:
        max_recorded_csr_step = max(
            (int(item.get("step_number") or 0) for item in csr_turns),
            default=0,
        )
        estimated_member_duration = sum(
            _estimate_script_duration_seconds(step.script)
            for step in scenario_steps
            if str(step.actor or "").strip().lower() != "csr"
            and int(step.step_number or 0) <= max_recorded_csr_step
        )
    session_level_duration = int(
        round(
            (
                spoken_turn_duration
                + float(dead_air_seconds or 0.0)
                + estimated_member_duration
            )
            if scenario_steps
            else (
                float(session.audio_duration_seconds or 0.0)
                or sum(float(item.get("duration_seconds") or 0.0) for item in _session_turn_logs(session))
                or spoken_turn_duration
            )
        )
    )
    evaluation["aht_actual"] = session_level_duration
    evaluation["aht_score"] = _score_aht(
        session_level_duration,
        int(kpi_config.target_aht_seconds or evaluation.get("aht_target") or 120),
    )
    kpi_weighted_score = _calculate_weighted_score(
        speech_to_text_accuracy=float(evaluation.get("speech_to_text_accuracy") or 0.0),
        aht_score=float(evaluation.get("aht_score") or 0.0),
        rate_of_speech_score=float(evaluation.get("rate_of_speech_score") or 0.0),
        dead_air_score=float(evaluation.get("dead_air_score") or 0.0),
        empathy_score=float(evaluation.get("empathy_score") or 0.0),
        probing_score=float(evaluation.get("probing_score") or 0.0),
        grammar_score=float(evaluation.get("grammar_score") or 0.0),
        pronunciation_score=float(evaluation.get("pronunciation_score") or 0.0),
        pacing_score=float(evaluation.get("pacing_score") or 0.0),
        forbidden_word_penalty=float(evaluation.get("forbidden_penalty") or 0.0),
        kpi_config=kpi_config,
    )
    scenario_points_total = 0.0
    scenario_points_earned = 0.0
    for turn in csr_turns:
        content_score = _calculate_turn_content_score(turn)
        scenario_points_total += float(content_score["point_value"])
        scenario_points_earned += float(content_score["earned_points"])

    scenario_score = round(
        (scenario_points_earned / scenario_points_total) * 100.0,
        2,
    ) if scenario_points_total > 0 else round(kpi_weighted_score, 2)

    overall_weighted_score = round(
        (scenario_score * CONTENT_SCORE_WEIGHT) + (kpi_weighted_score * KPI_SCORE_WEIGHT),
        2,
    ) if scenario_points_total > 0 else round(kpi_weighted_score, 2)

    evaluation["kpi_score"] = round(kpi_weighted_score, 2)
    evaluation["scenario_score"] = round(scenario_score, 2)
    evaluation["scenario_points_earned"] = round(scenario_points_earned, 2)
    evaluation["scenario_points_total"] = round(scenario_points_total, 2)
    evaluation["weighted_score"] = overall_weighted_score
    evaluation["pass_fail"] = evaluation["weighted_score"] >= float(
        kpi_config.passing_score or DEFAULT_PASSING_SCORE
    )
    ai_feedback = _build_ai_feedback(
        evaluation=evaluation,
        kpi_config=kpi_config,
    )
    return combined_transcript, session_level_duration, evaluation, ai_feedback


def _failed_kpi_counts(
    sessions: list[SimSession],
    batch_kpi_config: Optional[BatchKPIConfig] = None,
) -> dict[str, int]:
    kpi_config = batch_kpi_config or _build_default_kpi_config()
    target_ros = float(kpi_config.target_ros_words_per_min or 150.0)
    target_dead_air = float(kpi_config.target_dead_air_seconds or 3.0)

    return {
        "speech_to_text_accuracy": sum(
            1
            for session in sessions
            if session.speech_to_text_accuracy is not None
            and session.speech_to_text_accuracy < 80
        ),
        "grammar": sum(
            1
            for session in sessions
            if session.grammar_score is not None and session.grammar_score < 80
        ),
        "pronunciation": sum(
            1
            for session in sessions
            if session.pronunciation_score is not None
            and session.pronunciation_score < 80
        ),
        "pacing": sum(
            1
            for session in sessions
            if session.pacing_score is not None and session.pacing_score < 80
        ),
        "aht": sum(
            1
            for session in sessions
            if session.aht_actual is not None
            and session.aht_target is not None
            and session.aht_actual > session.aht_target
        ),
        "rate_of_speech": sum(
            1
            for session in sessions
            if session.rate_of_speech is not None
            and _score_target_delta(session.rate_of_speech, target_ros, 95.0) < 80
        ),
        "dead_air": sum(
            1
            for session in sessions
            if session.dead_air_seconds is not None and session.dead_air_seconds > target_dead_air
        ),
        "empathy": sum(1 for session in sessions if (session.empathy_statements_count or 0) == 0),
        "probing": sum(1 for session in sessions if (session.probing_questions_count or 0) == 0),
        "forbidden_words": sum(1 for session in sessions if (session.forbidden_words_count or 0) > 0),
    }


def _resolve_report_period(
    *,
    month: Optional[int],
    year: Optional[int],
) -> tuple[Optional[datetime], Optional[datetime], str]:
    active_year = year or datetime.utcnow().year

    if month:
        from calendar import monthrange

        _, last_day = monthrange(active_year, month)
        return (
            datetime(active_year, month, 1),
            datetime(active_year, month, last_day, 23, 59, 59),
            datetime(active_year, month, 1).strftime("%B %Y"),
        )

    if year:
        return (
            datetime(active_year, 1, 1),
            datetime(active_year, 12, 31, 23, 59, 59),
            f"All Months {active_year}",
        )

    return None, None, "All Time"


# ==================== Trainer Scenario Management ====================


@router.post("/scenarios", response_model=CallSimulationScenarioResponse, status_code=201)
async def create_call_simulation_scenario(
    scenario_data: CallSimulationScenarioCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    batch = _get_accessible_batch(db, current_user, scenario_data.batch_id)
    trainer_audio_settings = _get_trainer_call_audio_settings(current_user)
    normalized_expected_keywords = _normalize_keyword_list(scenario_data.expected_keywords)
    row_assets = (
        _build_call_simulation_row_assets(
            rows=list(scenario_data.script_rows or []),
            expected_keywords=normalized_expected_keywords,
        )
        if scenario_data.script_rows
        else None
    )
    member_profile = _normalize_json_object(scenario_data.member_profile)
    cxone_metadata = _normalize_json_object(scenario_data.cxone_metadata)
    call_simulation_config = _normalize_json_object(scenario_data.call_simulation_config)

    if row_assets:
        _validate_call_simulation_member_audio_assets(row_assets)
        first_member_row = row_assets.get("first_member_row")
        first_csr_row = row_assets.get("first_csr_row")
        default_problem_statement = (
            str(first_member_row.get("script") or "").strip()
            if isinstance(first_member_row, dict)
            else str(first_csr_row.get("script") or "").strip()
        )
        default_member_name = (
            str(first_member_row.get("actor_name") or "").strip()
            if isinstance(first_member_row, dict)
            else "Scenario Member"
        ) or "Scenario Member"

        member_profile = {
            **member_profile,
            "name": str(member_profile.get("name") or default_member_name).strip() or default_member_name,
            "problem_statement": str(
                member_profile.get("problem_statement") or default_problem_statement
            ).strip() or default_problem_statement,
        }
        cxone_metadata = {
            **cxone_metadata,
            "member_name": str(cxone_metadata.get("member_name") or member_profile.get("name") or default_member_name).strip() or default_member_name,
            "problem_statement": str(
                cxone_metadata.get("problem_statement")
                or member_profile.get("problem_statement")
                or default_problem_statement
            ).strip() or default_problem_statement,
        }
        call_simulation_config = {
            **call_simulation_config,
            "mode": "dialer_call_scenario",
            "topic": str(call_simulation_config.get("topic") or scenario_data.title.strip()).strip(),
            "script_rows": row_assets["script_rows"],
            "scenario_groups": row_assets["scenario_groups"],
            "script_flow": row_assets["script_flow"],
            "show_actor_script_overlay": True,
            "require_hold_before_member_response": True,
        }

    call_simulation_config = _apply_batch_kpi_to_call_simulation_config(
        call_simulation_config,
        kpi_config=_get_effective_kpi_config(db, batch.id),
        batch_id=batch.id,
    )
    call_simulation_config = {
        **call_simulation_config,
        "use_shared_ringer_audio": _coerce_call_simulation_bool(
            call_simulation_config.get("use_shared_ringer_audio"),
            fallback=True,
        ),
        "use_shared_hold_audio": _coerce_call_simulation_bool(
            call_simulation_config.get("use_shared_hold_audio"),
            fallback=True,
        ),
    }
    resolved_ringer_audio_url, resolved_hold_audio_url = _resolve_scenario_audio_urls(
        call_simulation_config=call_simulation_config,
        trainer_audio_settings=trainer_audio_settings,
        incoming_ringer_audio_url=scenario_data.ringer_audio_url,
        incoming_hold_audio_url=scenario_data.hold_audio_url,
    )
    _validate_optional_call_simulation_storage_url(
        resolved_ringer_audio_url,
        label="Trainer-uploaded ringer audio",
    )

    new_scenario = Scenario(
        id=str(uuid.uuid4()),
        title=scenario_data.title.strip(),
        description=scenario_data.description,
        purpose=scenario_data.purpose,
        difficulty=scenario_data.difficulty,
        lob=batch.lob,
        opening_prompt=scenario_data.opening_prompt.strip(),
        expected_keywords=normalized_expected_keywords,
        estimated_duration=scenario_data.estimated_duration,
        member_profile=member_profile,
        cxone_metadata=cxone_metadata,
        call_simulation_config=call_simulation_config,
        ringer_audio_url=resolved_ringer_audio_url,
        hold_audio_url=resolved_hold_audio_url,
        created_by=current_user.id,
        is_published=True,
        is_draft=False,
    )
    db.add(new_scenario)
    db.flush()

    # Generate TTS audio for opening prompt if not provided and TTS is available
    if not new_scenario.opening_prompt_audio:
        if get_tts_service().is_available():
            try:
                tts_result = await text_to_speech_for_persistence(
                    new_scenario.opening_prompt,
                    voice_name="Kore",
                    speaking_style="professionally",
                )
                upload_bytes = _coerce_audio_bytes(tts_result.get("audio_bytes"))
                if upload_bytes is None and isinstance(tts_result.get("audio_base64"), str) and tts_result.get("audio_base64").strip():
                    try:
                        upload_bytes = b64decode(tts_result.get("audio_base64"))
                    except Exception:
                        upload_bytes = None

                if upload_bytes:
                    supabase_client = get_supabase_client()
                    audio_url = supabase_client.upload_call_simulation_asset(
                        file_data=upload_bytes,
                        trainer_id=current_user.id,
                        asset_kind="opening-prompts",
                        filename=f"{new_scenario.id}_opening.wav",
                        scenario_id=new_scenario.id,
                        content_type=str(tts_result.get("audio_content_type") or "audio/wav").strip() or "audio/wav",
                    )
                    if audio_url:
                        new_scenario.opening_prompt_audio = audio_url
            except Exception as exc:
                logger.warning("Failed to generate TTS for opening prompt: %s", exc)
        else:
            logger.info(
                "Skipping opening prompt TTS generation for new scenario %s because no server-side TTS provider is available.",
                new_scenario.id,
            )

    mapping = BatchScenarioMapping(
        id=str(uuid.uuid4()),
        batch_id=batch.id,
        scenario_id=new_scenario.id,
        assigned_by=current_user.id,
        is_active=True,
    )
    db.add(mapping)

    created_variations: list[ScenarioVariation] = []
    variation_payloads = row_assets["variations"] if row_assets else list(scenario_data.variations)
    for variation_data in variation_payloads:
        variation = ScenarioVariation(
            id=str(uuid.uuid4()),
            scenario_id=new_scenario.id,
            actor_name=str(getattr(variation_data, "actor_name", None) or variation_data["actor_name"]).strip(),
            script=str(getattr(variation_data, "script", None) or variation_data["script"]).strip(),
            score=_coerce_nonnegative_float(getattr(variation_data, "score", None) if not isinstance(variation_data, dict) else variation_data.get("score")),
            branching_logic=getattr(variation_data, "branching_logic", None) if not isinstance(variation_data, dict) else variation_data.get("branching_logic"),
            is_active=True,
        )
        db.add(variation)
        created_variations.append(variation)

    if row_assets:
        class _StepProxy:
            def __init__(self, payload: dict[str, Any]) -> None:
                for key, value in payload.items():
                    setattr(self, key, value)

        await _replace_scenario_steps(
            db,
            scenario=new_scenario,
            steps=[_StepProxy(step) for step in row_assets["steps"]],
        )
    elif scenario_data.steps:
        await _replace_scenario_steps(
            db,
            scenario=new_scenario,
            steps=scenario_data.steps,
        )

    _sync_call_simulation_audio_assets_for_scenario(
        db,
        scenario=new_scenario,
        trainer_id=str(current_user.id),
        use_shared_ringer_audio=_coerce_call_simulation_bool(
            call_simulation_config.get("use_shared_ringer_audio"),
            fallback=True,
        ),
        use_shared_hold_audio=_coerce_call_simulation_bool(
            call_simulation_config.get("use_shared_hold_audio"),
            fallback=True,
        ),
    )

    db.flush()
    _sync_call_simulation_assignments_for_scenario(db, new_scenario.id)
    _sync_call_simulation_scenario_mirror_or_raise(
        db,
        scenario=new_scenario,
        trainer_id=str(current_user.id),
        batch_id=batch.id,
        detail="Supabase sync is required before creating a Call Simulation scenario.",
    )
    db.commit()
    db.refresh(new_scenario)
    db.refresh(mapping)

    return _serialize_scenario(
        db,
        new_scenario,
        batch=batch,
        mapping=mapping,
        variations=created_variations,
        highlighted_batch_id=batch.id,
    )


@router.get("/scenarios", response_model=list[CallSimulationScenarioResponse])
async def list_trainer_scenarios(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    if current_user.role == UserRole.ADMIN:
        scenarios = (
            db.query(Scenario)
            .filter(Scenario.is_published == True)
            .order_by(Scenario.updated_at.desc())
            .all()
        )
    else:
        trainer_batch_ids = _get_trainer_batch_ids(db, current_user)
        mapped_scenario_ids = [
            scenario_id
            for (scenario_id,) in db.query(BatchScenarioMapping.scenario_id)
            .filter(
                BatchScenarioMapping.batch_id.in_(trainer_batch_ids or ["__none__"]),
                BatchScenarioMapping.is_active == True,
            )
            .distinct()
            .all()
        ]
        created_scenario_ids = [
            scenario_id
            for (scenario_id,) in db.query(Scenario.id)
            .filter(
                Scenario.created_by == current_user.id,
                Scenario.is_published == True,
            )
            .all()
        ]
        accessible_ids = sorted(set(mapped_scenario_ids + created_scenario_ids))
        if not accessible_ids:
            return []
        scenarios = (
            db.query(Scenario)
            .filter(Scenario.id.in_(accessible_ids))
            .order_by(Scenario.updated_at.desc())
            .all()
        )

    payload: list[CallSimulationScenarioResponse] = []
    for scenario in scenarios:
        variations = (
            db.query(ScenarioVariation)
            .filter(
                ScenarioVariation.scenario_id == scenario.id,
                ScenarioVariation.is_active == True,
            )
            .order_by(ScenarioVariation.created_at.asc())
            .all()
        )
        payload.append(_serialize_scenario(db, scenario, variations=variations))

    return payload


@router.get("/batch/{batch_id}/scenarios", response_model=list[CallSimulationScenarioResponse])
async def get_batch_scenarios(
    batch_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    batch = _get_accessible_batch(db, current_user, batch_id)

    mappings = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.batch_id == batch_id,
            BatchScenarioMapping.is_active == True,
        )
        .order_by(BatchScenarioMapping.assigned_at.desc())
        .all()
    )

    results: list[CallSimulationScenarioResponse] = []
    for mapping in mappings:
        scenario = db.query(Scenario).filter(Scenario.id == mapping.scenario_id).first()
        if not scenario:
            continue
        variations = (
            db.query(ScenarioVariation)
            .filter(
                ScenarioVariation.scenario_id == scenario.id,
                ScenarioVariation.is_active == True,
            )
            .all()
        )
        results.append(
            _serialize_scenario(
                db,
                scenario,
                batch=batch,
                mapping=mapping,
                variations=variations,
                highlighted_batch_id=batch.id,
            )
        )

    return results


@router.get("/scenarios/{scenario_id}", response_model=CallSimulationScenarioResponse)
async def get_call_simulation_scenario(
    scenario_id: str,
    batch_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    scenario = _get_accessible_scenario(db, current_user, scenario_id)

    batch = _get_accessible_batch(db, current_user, batch_id) if batch_id else None
    mapping = None
    if batch is not None:
        mapping = (
            db.query(BatchScenarioMapping)
            .filter(
                BatchScenarioMapping.scenario_id == scenario.id,
                BatchScenarioMapping.batch_id == batch.id,
                BatchScenarioMapping.is_active == True,
            )
            .order_by(BatchScenarioMapping.assigned_at.desc())
            .first()
        )

    if mapping is None:
        if batch is not None:
            batch = None
        mapping = (
            db.query(BatchScenarioMapping)
            .filter(
                BatchScenarioMapping.scenario_id == scenario.id,
                BatchScenarioMapping.is_active == True,
            )
            .order_by(BatchScenarioMapping.assigned_at.desc())
            .first()
        )
        if batch is None and mapping is not None:
            batch = db.query(Batch).filter(Batch.id == mapping.batch_id).first()
    variations = (
        db.query(ScenarioVariation)
        .filter(
            ScenarioVariation.scenario_id == scenario.id,
            ScenarioVariation.is_active == True,
        )
        .order_by(ScenarioVariation.created_at.asc())
        .all()
    )
    return _serialize_scenario(
        db,
        scenario,
        batch=batch,
        mapping=mapping,
        variations=variations,
        highlighted_batch_id=batch.id if batch else None,
    )


@router.put("/scenarios/{scenario_id}", response_model=CallSimulationScenarioResponse)
async def update_call_simulation_scenario(
    scenario_id: str,
    scenario_update: CallSimulationScenarioUpdate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    scenario = _get_owned_scenario_for_audio_management(db, current_user, scenario_id)
    previous_config = _resolve_call_simulation_config(scenario)
    previous_audio_bindings = _collect_scenario_audio_bindings(
        scenario,
        use_shared_ringer_audio=_coerce_call_simulation_bool(
            previous_config.get("use_shared_ringer_audio"),
            fallback=True,
        ),
        use_shared_hold_audio=_coerce_call_simulation_bool(
            previous_config.get("use_shared_hold_audio"),
            fallback=True,
        ),
    )
    trainer_audio_settings = _get_trainer_call_audio_settings(current_user)
    update_fields = set(scenario_update.model_fields_set or set())
    normalized_expected_keywords = (
        _normalize_keyword_list(scenario_update.expected_keywords)
        if scenario_update.expected_keywords is not None
        else list(scenario.expected_keywords or [])
    )
    row_assets = (
        _build_call_simulation_row_assets(
            rows=list(scenario_update.script_rows or []),
            expected_keywords=normalized_expected_keywords,
        )
        if scenario_update.script_rows is not None
        else None
    )

    if scenario_update.title is not None:
        scenario.title = scenario_update.title.strip()
    if scenario_update.description is not None:
        scenario.description = scenario_update.description
    if scenario_update.opening_prompt is not None:
        scenario.opening_prompt = scenario_update.opening_prompt.strip()
        # Regenerate TTS for opening prompt if changed and no audio provided
        if not scenario.opening_prompt_audio:
            if get_tts_service().is_available():
                try:
                    tts_result = await text_to_speech_for_persistence(
                        scenario.opening_prompt,
                        voice_name="Kore",
                        speaking_style="professionally",
                    )
                    upload_bytes = _coerce_audio_bytes(tts_result.get("audio_bytes"))
                    if upload_bytes is None and isinstance(tts_result.get("audio_base64"), str) and tts_result.get("audio_base64").strip():
                        try:
                            upload_bytes = b64decode(tts_result.get("audio_base64"))
                        except Exception:
                            upload_bytes = None

                    if upload_bytes:
                        supabase_client = get_supabase_client()
                        audio_url = supabase_client.upload_call_simulation_asset(
                            file_data=upload_bytes,
                            trainer_id=current_user.id,
                            asset_kind="opening-prompts",
                            filename=f"{scenario.id}_opening.wav",
                            scenario_id=scenario.id,
                            content_type=str(tts_result.get("audio_content_type") or "audio/wav").strip() or "audio/wav",
                        )
                        if audio_url:
                            scenario.opening_prompt_audio = audio_url
                except Exception as exc:
                    logger.warning("Failed to generate TTS for updated opening prompt: %s", exc)
            else:
                logger.info(
                    "Skipping opening prompt TTS generation for updated scenario %s because no server-side TTS provider is available.",
                    scenario.id,
                )
    if scenario_update.difficulty is not None:
        scenario.difficulty = scenario_update.difficulty
    if scenario_update.purpose is not None:
        scenario.purpose = scenario_update.purpose
    if scenario_update.expected_keywords is not None:
        scenario.expected_keywords = normalized_expected_keywords
    if scenario_update.estimated_duration is not None:
        scenario.estimated_duration = scenario_update.estimated_duration
    if scenario_update.member_profile is not None:
        scenario.member_profile = _normalize_json_object(scenario_update.member_profile)
    if scenario_update.cxone_metadata is not None:
        scenario.cxone_metadata = _normalize_json_object(scenario_update.cxone_metadata)
    if scenario_update.call_simulation_config is not None:
        scenario.call_simulation_config = _normalize_json_object(scenario_update.call_simulation_config)
    if scenario_update.is_published is not None:
        scenario.is_published = scenario_update.is_published
        scenario.is_draft = not scenario_update.is_published

    if row_assets:
        _validate_call_simulation_member_audio_assets(row_assets)
        first_member_row = row_assets.get("first_member_row")
        first_csr_row = row_assets.get("first_csr_row")
        default_problem_statement = (
            str(first_member_row.get("script") or "").strip()
            if isinstance(first_member_row, dict)
            else str(first_csr_row.get("script") or "").strip()
        )
        default_member_name = (
            str(first_member_row.get("actor_name") or "").strip()
            if isinstance(first_member_row, dict)
            else "Scenario Member"
        ) or "Scenario Member"

        scenario.member_profile = {
            **_normalize_json_object(scenario.member_profile),
            "name": str(_normalize_json_object(scenario.member_profile).get("name") or default_member_name).strip() or default_member_name,
            "problem_statement": str(
                _normalize_json_object(scenario.member_profile).get("problem_statement") or default_problem_statement
            ).strip() or default_problem_statement,
        }
        scenario.cxone_metadata = {
            **_normalize_json_object(scenario.cxone_metadata),
            "member_name": str(_normalize_json_object(scenario.cxone_metadata).get("member_name") or scenario.member_profile.get("name") or default_member_name).strip() or default_member_name,
            "problem_statement": str(
                _normalize_json_object(scenario.cxone_metadata).get("problem_statement")
                or scenario.member_profile.get("problem_statement")
                or default_problem_statement
            ).strip() or default_problem_statement,
        }
        scenario.call_simulation_config = {
            **_normalize_json_object(scenario.call_simulation_config),
            "mode": "dialer_call_scenario",
            "topic": str(
                _normalize_json_object(scenario.call_simulation_config).get("topic")
                or scenario.title
            ).strip() or scenario.title,
            "script_rows": row_assets["script_rows"],
            "scenario_groups": row_assets["scenario_groups"],
            "script_flow": row_assets["script_flow"],
            "show_actor_script_overlay": True,
            "require_hold_before_member_response": True,
        }

    scenario.call_simulation_config = {
        **_normalize_json_object(scenario.call_simulation_config),
        "use_shared_ringer_audio": _coerce_call_simulation_bool(
            _normalize_json_object(scenario.call_simulation_config).get("use_shared_ringer_audio"),
            fallback=True,
        ),
        "use_shared_hold_audio": _coerce_call_simulation_bool(
            _normalize_json_object(scenario.call_simulation_config).get("use_shared_hold_audio"),
            fallback=True,
        ),
    }
    scenario.ringer_audio_url, scenario.hold_audio_url = _resolve_scenario_audio_urls(
        call_simulation_config=scenario.call_simulation_config,
        trainer_audio_settings=trainer_audio_settings,
        current_ringer_audio_url=scenario.ringer_audio_url,
        current_hold_audio_url=scenario.hold_audio_url,
        incoming_ringer_audio_url=scenario_update.ringer_audio_url if "ringer_audio_url" in update_fields else _UNSET,
        incoming_hold_audio_url=scenario_update.hold_audio_url if "hold_audio_url" in update_fields else _UNSET,
    )
    _validate_optional_call_simulation_storage_url(
        scenario.ringer_audio_url,
        label="Trainer-uploaded ringer audio",
    )

    active_mapping = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.scenario_id == scenario.id,
            BatchScenarioMapping.is_active == True,
        )
        .order_by(BatchScenarioMapping.assigned_at.desc())
        .first()
    )
    batch = db.query(Batch).filter(Batch.id == active_mapping.batch_id).first() if active_mapping else None

    if scenario_update.batch_id:
        batch = _get_accessible_batch(db, current_user, scenario_update.batch_id)
        existing_mapping = (
            db.query(BatchScenarioMapping)
            .filter(
                BatchScenarioMapping.scenario_id == scenario.id,
                BatchScenarioMapping.batch_id == batch.id,
            )
            .first()
        )
        if existing_mapping:
            existing_mapping.is_active = True
            existing_mapping.assigned_by = current_user.id
            existing_mapping.assigned_at = datetime.utcnow()
            active_mapping = existing_mapping
        else:
            active_mapping = BatchScenarioMapping(
                id=str(uuid.uuid4()),
                batch_id=batch.id,
                scenario_id=scenario.id,
                assigned_by=current_user.id,
                is_active=True,
            )
            db.add(active_mapping)

        scenario.lob = batch.lob

    resolved_batch_id = batch.id if batch else active_mapping.batch_id if active_mapping else None
    if resolved_batch_id:
        scenario.call_simulation_config = _apply_batch_kpi_to_call_simulation_config(
            scenario.call_simulation_config,
            kpi_config=_get_effective_kpi_config(db, resolved_batch_id),
            batch_id=resolved_batch_id,
        )

    if scenario_update.variations is not None:
        _deactivate_scenario_variations(
            db,
            scenario_id=scenario.id,
        )

        for variation_data in scenario_update.variations:
            db.add(
                ScenarioVariation(
                    id=str(uuid.uuid4()),
                    scenario_id=scenario.id,
                    actor_name=variation_data.actor_name.strip(),
                    script=variation_data.script.strip(),
                    score=variation_data.score,
                    branching_logic=variation_data.branching_logic,
                    is_active=True,
                )
            )
    elif row_assets:
        _deactivate_scenario_variations(
            db,
            scenario_id=scenario.id,
        )
        for variation_data in row_assets["variations"]:
            db.add(
                ScenarioVariation(
                    id=str(uuid.uuid4()),
                    scenario_id=scenario.id,
                    actor_name=str(variation_data["actor_name"]).strip(),
                    script=str(variation_data["script"]).strip(),
                    score=_coerce_nonnegative_float(variation_data.get("score")),
                    branching_logic=variation_data.get("branching_logic"),
                    is_active=True,
                )
            )

    if row_assets:
        class _StepProxy:
            def __init__(self, payload: dict[str, Any]) -> None:
                for key, value in payload.items():
                    setattr(self, key, value)

        await _replace_scenario_steps(
            db,
            scenario=scenario,
            steps=[_StepProxy(step) for step in row_assets["steps"]],
        )
    elif scenario_update.steps is not None:
        await _replace_scenario_steps(
            db,
            scenario=scenario,
            steps=scenario_update.steps,
        )

    current_use_shared_ringer_audio = _coerce_call_simulation_bool(
        _normalize_json_object(scenario.call_simulation_config).get("use_shared_ringer_audio"),
        fallback=True,
    )
    current_use_shared_hold_audio = _coerce_call_simulation_bool(
        _normalize_json_object(scenario.call_simulation_config).get("use_shared_hold_audio"),
        fallback=True,
    )
    _sync_call_simulation_audio_assets_for_scenario(
        db,
        scenario=scenario,
        trainer_id=str(scenario.created_by or current_user.id),
        use_shared_ringer_audio=current_use_shared_ringer_audio,
        use_shared_hold_audio=current_use_shared_hold_audio,
    )
    current_audio_bindings = _collect_scenario_audio_bindings(
        scenario,
        use_shared_ringer_audio=current_use_shared_ringer_audio,
        use_shared_hold_audio=current_use_shared_hold_audio,
    )
    _cleanup_removed_scenario_audio_assets(
        db,
        trainer_id=str(scenario.created_by or current_user.id),
        scenario_id=scenario.id,
        previous_bindings=previous_audio_bindings,
        current_bindings=current_audio_bindings,
    )

    db.flush()
    _sync_call_simulation_assignments_for_scenario(db, scenario.id)
    _sync_call_simulation_scenario_mirror_or_raise(
        db,
        scenario=scenario,
        trainer_id=str(scenario.created_by or current_user.id),
        batch_id=resolved_batch_id,
        detail="Supabase sync is required before updating a Call Simulation scenario.",
    )
    db.commit()
    db.refresh(scenario)

    variations = (
        db.query(ScenarioVariation)
        .filter(
            ScenarioVariation.scenario_id == scenario.id,
            ScenarioVariation.is_active == True,
        )
        .order_by(ScenarioVariation.created_at.asc())
        .all()
    )
    if active_mapping and not batch:
        batch = db.query(Batch).filter(Batch.id == active_mapping.batch_id).first()

    return _serialize_scenario(
        db,
        scenario,
        batch=batch,
        mapping=active_mapping,
        variations=variations,
        highlighted_batch_id=batch.id if batch else None,
    )


@router.delete("/scenarios/{scenario_id}", response_model=SuccessResponse)
async def delete_call_simulation_scenario(
    scenario_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    scenario = _get_owned_scenario_for_audio_management(db, current_user, scenario_id)

    scenario.is_published = False
    scenario.is_draft = True

    mappings = (
        db.query(BatchScenarioMapping)
        .filter(BatchScenarioMapping.scenario_id == scenario.id)
        .all()
    )
    for mapping in mappings:
        mapping.is_active = False

    variations = (
        db.query(ScenarioVariation)
        .filter(ScenarioVariation.scenario_id == scenario.id)
        .all()
    )
    for variation in variations:
        variation.is_active = False

    _sync_call_simulation_assignments_for_scenario(db, scenario.id)
    _delete_call_simulation_scenario_mirror_or_raise(
        db,
        source_scenario_id=scenario.id,
        detail="Supabase sync is required before archiving a Call Simulation scenario.",
    )
    db.commit()
    return SuccessResponse(message="Scenario archived successfully")


@router.post("/scenarios/sync", response_model=SuccessResponse)
async def sync_scenario_to_supabase(
    sync_data: dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Sync scenario data to Supabase"""
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    try:
        supabase = _require_supabase_storage(
            "Supabase sync is required for call simulation scenario mirroring."
        )

        # Handle syncFromDatabase flag for bulk upload scenarios
        if sync_data.get("syncFromDatabase"):
            scenario_id = sync_data.get("scenarioId")
            if not scenario_id:
                raise HTTPException(status_code=400, detail="scenarioId required when syncFromDatabase is true")

            # Validate scenario exists and belongs to trainer
            scenario = _get_accessible_scenario(db, current_user, scenario_id)
            logger.debug("Syncing scenario %s from database to Supabase", scenario_id)
            sync_batch_id = sync_data.get("batchId")
            if sync_batch_id:
                _get_accessible_batch(db, current_user, str(sync_batch_id))
            elif sync_data.get("batch_id"):
                sync_batch_id = sync_data.get("batch_id")
                _get_accessible_batch(db, current_user, str(sync_batch_id))
            else:
                latest_mapping = (
                    db.query(BatchScenarioMapping)
                    .filter(
                        BatchScenarioMapping.scenario_id == scenario.id,
                        BatchScenarioMapping.is_active == True,
                    )
                    .order_by(BatchScenarioMapping.assigned_at.desc())
                    .first()
                )
                sync_batch_id = latest_mapping.batch_id if latest_mapping else None

            scenario_config = _resolve_call_simulation_config_for_batch(
                db,
                scenario,
                batch_id=str(sync_batch_id) if sync_batch_id else None,
            )
            sanitized_script_flow = _sanitize_supabase_script_flow(
                scenario_config.get("script_flow")
            )
            passing_score = _resolve_call_scenario_passing_score(
                scenario,
                float(
                    _get_effective_kpi_config(db, str(sync_batch_id) if sync_batch_id else None).passing_score
                    or DEFAULT_PASSING_SCORE
                ),
            )
            
            # Build scenario data from database scenario
            scenario_sync_data = {
                "source_scenario_id": str(scenario.id),
                "trainer_id": str(current_user.id),
                "title": scenario.title,
                "topic": str(scenario_config.get("topic") or scenario.lob or scenario.title),
                "description": scenario.description,
                "target_kpis": _normalize_json_object(scenario_config.get("target_kpis")),
                "script_flow": sanitized_script_flow,
                "ringer_audio_url": scenario.ringer_audio_url,
                "hold_audio_url": scenario.hold_audio_url,
                "difficulty": scenario.difficulty or "intermediate",
                "estimated_duration_seconds": scenario.estimated_duration or 300,
                "passing_score": passing_score,
                "is_published": bool(scenario.is_published),
                "is_active": True,
                "metadata": scenario_config,
            }
        else:
            # Handle full scenario data sync from frontend
            scenario_data = sync_data.get("scenario")
            if not scenario_data:
                raise HTTPException(status_code=400, detail="scenario data required")
            if not isinstance(scenario_data, dict):
                raise HTTPException(status_code=400, detail="scenario data must be an object")

            logger.debug("Syncing scenario data to Supabase: %s", scenario_data.get("scenarioId"))
            sanitized_script_flow = _sanitize_supabase_script_flow(
                scenario_data.get("scriptFlow")
            )

            scenario_sync_data = {
                "source_scenario_id": str(scenario_data.get("scenarioId")),
                "trainer_id": str(current_user.id),
                "title": str(scenario_data.get("title", "")),
                "topic": str(scenario_data.get("topic", "")),
                "description": scenario_data.get("description"),
                "target_kpis": _normalize_json_object(scenario_data.get("targetKpis")),
                "script_flow": sanitized_script_flow,
                "ringer_audio_url": scenario_data.get("ringerAudioUrl"),
                "hold_audio_url": scenario_data.get("holdAudioUrl"),
                "difficulty": scenario_data.get("difficulty") or "intermediate",
                "estimated_duration_seconds": scenario_data.get("estimatedDurationSeconds") or 300,
                "passing_score": float(scenario_data.get("passingScore", 80)),
                "is_published": bool(scenario_data.get("isPublished", True)),
                "is_active": bool(scenario_data.get("isActive", True)),
                "metadata": _normalize_json_object(scenario_data.get("metadata")),
            }

        if not _legacy_call_simulation_mirror_relations_ready(
            db,
            ("call_scenarios",),
            context=f"manual scenario sync for {scenario_sync_data.get('source_scenario_id')}",
        ):
            return SuccessResponse(
                message="Scenario data already persists through the primary Supabase Postgres connection. Legacy mirror tables are not enabled in this project."
            )

        try:
            # Try to upsert the scenario record - Supabase will use the unique index on source_scenario_id
            result = supabase.table("call_scenarios").upsert(
                scenario_sync_data,
                ignore_duplicates=False
            ).execute()
            
            logger.info("Synced scenario %s to Supabase", scenario_sync_data.get("source_scenario_id"))
        except Exception as supabase_error:
            if _is_missing_supabase_table_error(supabase_error):
                _log_legacy_supabase_mirror_skip(
                    f"manual scenario sync for {scenario_sync_data.get('source_scenario_id')}",
                    supabase_error,
                )
                return SuccessResponse(
                    message="Scenario saved through the primary Supabase Postgres connection. Legacy mirror tables are not enabled in this project."
                )
            # If upsert fails, try insert or update separately
            logger.warning("Upsert failed, trying insert/update approach: %s", supabase_error)
            try:
                # Check if record exists
                existing = supabase.table("call_scenarios").select("id").eq(
                    "source_scenario_id", scenario_sync_data["source_scenario_id"]
                ).execute()
                
                if existing.data and len(existing.data) > 0:
                    # Update existing
                    result = supabase.table("call_scenarios").update(
                        scenario_sync_data
                    ).eq("source_scenario_id", scenario_sync_data["source_scenario_id"]).execute()
                    logger.info("Updated scenario %s in Supabase", scenario_sync_data.get("source_scenario_id"))
                else:
                    # Insert new
                    result = supabase.table("call_scenarios").insert(
                        scenario_sync_data
                    ).execute()
                    logger.info("Inserted scenario %s into Supabase", scenario_sync_data.get("source_scenario_id"))
            except Exception as fallback_error:
                if _is_missing_supabase_table_error(fallback_error):
                    _log_legacy_supabase_mirror_skip(
                        f"manual scenario sync fallback for {scenario_sync_data.get('source_scenario_id')}",
                        fallback_error,
                    )
                    return SuccessResponse(
                        message="Scenario saved through the primary Supabase Postgres connection. Legacy mirror tables are not enabled in this project."
                    )
                logger.error("Fallback insert/update failed: %s", fallback_error)
                raise fallback_error

        _sync_supabase_scenario_authoring_tables(
            db,
            supabase,
            source_scenario_id=scenario_sync_data["source_scenario_id"],
            trainer_id=scenario_sync_data["trainer_id"],
            title=scenario_sync_data.get("title"),
            description=scenario_sync_data.get("description"),
            topic=str(scenario_sync_data.get("topic") or ""),
            target_kpis=_normalize_json_object(scenario_sync_data.get("target_kpis")),
            script_flow=_sanitize_supabase_script_flow(scenario_sync_data.get("script_flow")),
            ringer_audio_url=_normalize_optional_url(scenario_sync_data.get("ringer_audio_url")),
            hold_audio_url=_normalize_optional_url(scenario_sync_data.get("hold_audio_url")),
            difficulty=_normalize_optional_url(scenario_sync_data.get("difficulty")),
            estimated_duration_seconds=scenario_sync_data.get("estimated_duration_seconds"),
            is_published=bool(scenario_sync_data.get("is_published", True)),
            metadata=_normalize_json_object(scenario_sync_data.get("metadata")),
        )

        return SuccessResponse(message="Scenario sync completed successfully")
    except Exception as exc:
        logger.error("Error syncing scenario to Supabase: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync scenario to Supabase: {str(exc)}"
        ) from exc


@router.delete("/scenarios/sync", response_model=SuccessResponse)
async def delete_scenario_from_supabase(
    sync_data: dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    scenario_id = str(sync_data.get("scenarioId") or "").strip()
    if not scenario_id:
        raise HTTPException(status_code=400, detail="scenarioId required")

    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if scenario is not None:
        _get_accessible_scenario(db, current_user, scenario_id)

    try:
        supabase = _require_supabase_storage(
            "Supabase sync is required for call simulation scenario cleanup."
        )
        _delete_supabase_scenario_mirror(
            db,
            supabase,
            source_scenario_id=scenario_id,
        )
        return SuccessResponse(message="Scenario cleanup completed successfully")
    except Exception as exc:
        logger.error("Error deleting scenario from Supabase: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete scenario from Supabase: {str(exc)}"
        ) from exc


@router.post("/kpi/sync", response_model=SuccessResponse)
async def sync_kpi_to_supabase(
    sync_data: dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Sync KPI data to Supabase"""
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    try:
        supabase = _require_supabase_storage(
            "Supabase sync is required for call simulation KPI mirroring."
        )

        scenario_group_ids = sync_data.get("scenarioGroupIds", [])
        metrics = sync_data.get("metrics", [])

        if not isinstance(scenario_group_ids, list) or not scenario_group_ids:
            raise HTTPException(status_code=400, detail="scenarioGroupIds required")
        if not isinstance(metrics, list) or not metrics:
            raise HTTPException(status_code=400, detail="metrics required")

        normalized_scenario_group_ids = [
            scenario_group_id
            for scenario_group_id in (
                _normalize_uuid_candidate(value) for value in scenario_group_ids
            )
            if scenario_group_id
        ]
        if not normalized_scenario_group_ids:
            raise HTTPException(
                status_code=400,
                detail="scenarioGroupIds must contain at least one UUID-shaped scenario id",
            )

        normalized_metrics: list[dict[str, Any]] = []
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            metric_name = str(metric.get("metricName") or "").strip()
            if not metric_name:
                continue
            try:
                weight_value = float(metric.get("weightPercentage") or 0)
            except (TypeError, ValueError):
                weight_value = 0.0
            normalized_metrics.append(
                {
                    "metric_name": metric_name,
                    "weight_percentage": max(0, round(weight_value)),
                }
            )

        if not normalized_metrics:
            raise HTTPException(status_code=400, detail="metrics must contain at least one named KPI weight")

        if not _legacy_call_simulation_mirror_relations_ready(
            db,
            ("kpi_metrics",),
            context=f"kpi sync for scenario groups {', '.join(normalized_scenario_group_ids)}",
        ):
            return SuccessResponse(
                message="KPI settings are already saved in Supabase Postgres. Legacy mirror tables are not enabled in this project."
            )

        try:
            supabase.table("kpi_metrics").delete().in_(
                "scenario_group_id",
                normalized_scenario_group_ids,
            ).execute()

            metric_rows = [
                {
                    "scenario_group_id": scenario_group_id,
                    **metric,
                }
                for scenario_group_id in normalized_scenario_group_ids
                for metric in normalized_metrics
            ]
            supabase.table("kpi_metrics").insert(metric_rows).execute()
        except Exception as exc:
            if _is_missing_supabase_table_error(exc):
                _log_legacy_supabase_mirror_skip(
                    f"kpi sync for scenario groups {', '.join(normalized_scenario_group_ids)}",
                    exc,
                )
                return SuccessResponse(
                    message="KPI settings are already saved in Supabase Postgres. Legacy mirror tables are not enabled in this project."
                )
            raise

        return SuccessResponse(message="KPI sync completed successfully")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error syncing KPI to Supabase: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync KPI to Supabase: {str(exc)}"
        ) from exc


@router.post("/audio/sync", response_model=SuccessResponse)
async def sync_call_audio_to_supabase(
    sync_data: dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Mirror updated trainer call-tone settings to the Supabase scenario records."""
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    try:
        supabase = _require_supabase_storage(
            "Supabase sync is required for Call Simulation audio mirroring."
        )
        requested_ids = sync_data.get("scenario_ids") or sync_data.get("scenarioIds") or []
        normalized_requested_ids = [
            scenario_id
            for scenario_id in (
                _normalize_uuid_candidate(value) for value in requested_ids
            )
            if scenario_id
        ]

        scenario_query = db.query(Scenario).filter(
            Scenario.created_by == current_user.id,
            Scenario.is_published == True,
        )
        if normalized_requested_ids:
            scenario_query = scenario_query.filter(Scenario.id.in_(normalized_requested_ids))

        scenarios = scenario_query.order_by(Scenario.updated_at.desc()).all()
        synced_count = 0

        for scenario in scenarios:
            latest_mapping = (
                db.query(BatchScenarioMapping)
                .filter(
                    BatchScenarioMapping.scenario_id == scenario.id,
                    BatchScenarioMapping.is_active == True,
                )
                .order_by(BatchScenarioMapping.assigned_at.desc())
                .first()
            )
            _sync_call_simulation_scenario_to_supabase_from_db(
                supabase,
                db=db,
                scenario=scenario,
                trainer_id=current_user.id,
                batch_id=latest_mapping.batch_id if latest_mapping else None,
            )
            synced_count += 1

        logger.info(
            "Synchronized Call Simulation audio settings to Supabase for %s scenario(s) owned by trainer %s",
            synced_count,
            current_user.id,
        )
        return SuccessResponse(
            message=f"Call audio sync completed for {synced_count} Call Simulation scenario(s)"
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error syncing Call Simulation audio to Supabase: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync Call Simulation audio to Supabase: {str(exc)}",
        ) from exc


# ==================== Variation Management ====================


@router.post("/variations", response_model=ScenarioVariationResponse, status_code=201)
async def create_variation(
    variation_data: ScenarioVariationCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    _get_accessible_scenario(db, current_user, variation_data.scenario_id)

    new_variation = ScenarioVariation(
        id=str(uuid.uuid4()),
        scenario_id=variation_data.scenario_id,
        actor_name=variation_data.actor_name.strip(),
        script=variation_data.script.strip(),
        score=variation_data.score,
        branching_logic=variation_data.branching_logic,
        is_active=True,
    )
    db.add(new_variation)
    db.commit()
    db.refresh(new_variation)
    return ScenarioVariationResponse.model_validate(new_variation)


@router.get("/variations/{variation_id}", response_model=ScenarioVariationResponse)
async def get_variation(
    variation_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    variation = db.query(ScenarioVariation).filter(ScenarioVariation.id == variation_id).first()
    if not variation:
        raise HTTPException(status_code=404, detail="Variation not found")

    _get_accessible_scenario(db, current_user, variation.scenario_id)
    return ScenarioVariationResponse.model_validate(variation)


@router.put("/variations/{variation_id}", response_model=ScenarioVariationResponse)
async def update_variation(
    variation_id: str,
    variation_update: ScenarioVariationUpdate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    variation = db.query(ScenarioVariation).filter(ScenarioVariation.id == variation_id).first()
    if not variation:
        raise HTTPException(status_code=404, detail="Variation not found")

    _get_accessible_scenario(db, current_user, variation.scenario_id)

    if variation_update.actor_name is not None:
        variation.actor_name = variation_update.actor_name.strip()
    if variation_update.script is not None:
        variation.script = variation_update.script.strip()
    if variation_update.score is not None:
        variation.score = variation_update.score
    if variation_update.branching_logic is not None:
        variation.branching_logic = variation_update.branching_logic
    if variation_update.is_active is not None:
        variation.is_active = variation_update.is_active

    db.commit()
    db.refresh(variation)
    return ScenarioVariationResponse.model_validate(variation)


@router.delete("/variations/{variation_id}", response_model=SuccessResponse)
async def delete_variation(
    variation_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    variation = db.query(ScenarioVariation).filter(ScenarioVariation.id == variation_id).first()
    if not variation:
        raise HTTPException(status_code=404, detail="Variation not found")

    _get_accessible_scenario(db, current_user, variation.scenario_id)
    variation.is_active = False
    db.commit()
    return SuccessResponse(message="Variation archived successfully")


@router.get("/scenarios/{scenario_id}/variations", response_model=list[ScenarioVariationResponse])
async def list_variations(
    scenario_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    _get_accessible_scenario(db, current_user, scenario_id)

    variations = (
        db.query(ScenarioVariation)
        .filter(
            ScenarioVariation.scenario_id == scenario_id,
            ScenarioVariation.is_active == True,
        )
        .order_by(ScenarioVariation.created_at.asc())
        .all()
    )
    return [ScenarioVariationResponse.model_validate(variation) for variation in variations]


# ==================== Batch Mapping ====================


@router.post("/batch-mapping", response_model=BatchScenarioMappingCreate)
async def create_batch_mapping(
    mapping_data: BatchScenarioMappingCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    _get_accessible_batch(db, current_user, mapping_data.batch_id)
    _get_accessible_scenario(db, current_user, mapping_data.scenario_id)

    existing = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.batch_id == mapping_data.batch_id,
            BatchScenarioMapping.scenario_id == mapping_data.scenario_id,
        )
        .first()
    )
    if existing:
        existing.is_active = True
        existing.assigned_by = current_user.id
        existing.assigned_at = datetime.utcnow()
    else:
        db.add(
            BatchScenarioMapping(
                id=str(uuid.uuid4()),
                batch_id=mapping_data.batch_id,
                scenario_id=mapping_data.scenario_id,
                assigned_by=current_user.id,
                is_active=True,
            )
        )

    db.flush()
    _sync_call_simulation_assignments_for_scenario(db, mapping_data.scenario_id)
    scenario = db.query(Scenario).filter(Scenario.id == mapping_data.scenario_id).first()
    if scenario:
        _sync_call_simulation_scenario_mirror_or_raise(
            db,
            scenario=scenario,
            trainer_id=str(scenario.created_by or current_user.id),
            batch_id=mapping_data.batch_id,
            detail="Supabase sync is required before updating the Call Simulation batch mapping.",
        )
    db.commit()
    return BatchScenarioMappingCreate(
        batch_id=mapping_data.batch_id,
        scenario_id=mapping_data.scenario_id,
    )


@router.get("/assignment-targets", response_model=list[CallSimulationAssignmentTargetResponse])
async def get_assignment_targets(
    scenario_id: str = Query(...),
    batch_id: str = Query(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    _get_accessible_scenario(db, current_user, scenario_id)
    batch = _get_accessible_batch(db, current_user, batch_id)

    trainees = sorted(
        [user for user in batch.users if user.role == UserRole.TRAINEE and user.is_active],
        key=lambda trainee: (trainee.full_name or "").lower(),
    )
    trainee_ids = [trainee.id for trainee in trainees]
    assignments = (
        db.query(CallSimulationAssignment)
        .filter(
            CallSimulationAssignment.scenario_id == scenario_id,
            CallSimulationAssignment.batch_id == batch.id,
            CallSimulationAssignment.trainee_id.in_(trainee_ids or ["__none__"]),
        )
        .order_by(CallSimulationAssignment.updated_at.desc(), CallSimulationAssignment.assigned_at.desc())
        .all()
    )
    latest_assignments: dict[str, CallSimulationAssignment] = {}
    for assignment in assignments:
        if assignment.trainee_id not in latest_assignments:
            latest_assignments[assignment.trainee_id] = assignment

    return [
        CallSimulationAssignmentTargetResponse(
            trainee_id=trainee.id,
            trainee_name=trainee.full_name,
            trainee_email=trainee.email,
            language_dialect=trainee.language_dialect,
            batch_id=batch.id,
            batch_name=batch.name,
            wave_number=batch.wave_number,
            is_assigned=bool(latest_assignments.get(trainee.id) and latest_assignments[trainee.id].is_active),
            assignment_id=latest_assignments.get(trainee.id).id if latest_assignments.get(trainee.id) else None,
            assigned_at=latest_assignments.get(trainee.id).assigned_at if latest_assignments.get(trainee.id) else None,
            max_attempts=latest_assignments.get(trainee.id).max_attempts if latest_assignments.get(trainee.id) else None,
        )
        for trainee in trainees
    ]


@router.post("/assignments", response_model=CallSimulationAssignmentResponse)
async def assign_call_simulation_to_trainees(
    payload: CallSimulationAssignmentCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    scenario = _get_accessible_scenario(db, current_user, payload.scenario_id)
    batch = _get_accessible_batch(db, current_user, payload.batch_id)

    existing_mapping = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.scenario_id == scenario.id,
            BatchScenarioMapping.batch_id == batch.id,
        )
        .first()
    )
    if existing_mapping:
        existing_mapping.is_active = True
        existing_mapping.assigned_by = current_user.id
        existing_mapping.assigned_at = datetime.utcnow()
    else:
        existing_mapping = BatchScenarioMapping(
            id=str(uuid.uuid4()),
            batch_id=batch.id,
            scenario_id=scenario.id,
            assigned_by=current_user.id,
            is_active=True,
        )
        db.add(existing_mapping)

    batch_trainees = [
        trainee
        for trainee in batch.users
        if trainee.role == UserRole.TRAINEE and trainee.is_active
    ]
    batch_trainee_ids = {trainee.id for trainee in batch_trainees}
    requested_trainee_ids = {trainee_id for trainee_id in payload.trainee_ids if trainee_id}
    if requested_trainee_ids:
        invalid_ids = sorted(requested_trainee_ids.difference(batch_trainee_ids))
        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail="One or more selected trainees do not belong to the chosen batch.",
            )
        selected_trainee_ids = requested_trainee_ids
    else:
        selected_trainee_ids = set(batch_trainee_ids)

    existing_assignments = (
        db.query(CallSimulationAssignment)
        .filter(
            CallSimulationAssignment.scenario_id == scenario.id,
            CallSimulationAssignment.trainee_id.in_(list(batch_trainee_ids) or ["__none__"]),
        )
        .all()
    )
    assignment_lookup = {
        assignment.trainee_id: assignment for assignment in existing_assignments
    }

    assignments_created = 0
    assignments_updated = 0
    assignments_deactivated = 0
    assigned_at = datetime.utcnow()
    resolved_max_attempts = _resolve_call_assignment_max_attempts(
        None,
        scenario=scenario,
        fallback=payload.max_attempts or DEFAULT_MAX_ATTEMPTS,
    )

    for trainee_id in sorted(selected_trainee_ids):
        assignment, changed = _upsert_call_simulation_assignment(
            assignment=assignment_lookup.get(trainee_id),
            scenario_id=scenario.id,
            trainee_id=trainee_id,
            trainer_id=current_user.id,
            batch_id=batch.id,
            assigned_at=assigned_at,
            max_attempts=resolved_max_attempts,
        )
        if assignment_lookup.get(trainee_id) is None:
            db.add(assignment)
            assignment_lookup[trainee_id] = assignment
            assignments_created += 1
        elif changed:
            assignments_updated += 1

    for trainee_id in sorted(batch_trainee_ids.difference(selected_trainee_ids)):
        assignment = assignment_lookup.get(trainee_id)
        if assignment and assignment.is_active and assignment.batch_id == batch.id:
            assignment.is_active = False
            assignments_deactivated += 1

    db.flush()
    _sync_call_simulation_assignments_for_scenario(db, scenario.id)
    _sync_call_simulation_scenario_mirror_or_raise(
        db,
        scenario=scenario,
        trainer_id=str(scenario.created_by or current_user.id),
        batch_id=batch.id,
        detail="Supabase sync is required before saving Call Simulation assignments.",
    )
    db.commit()

    return CallSimulationAssignmentResponse(
        scenario_id=scenario.id,
        batch_id=batch.id,
        assigned_trainee_ids=sorted(selected_trainee_ids),
        max_attempts=resolved_max_attempts,
        assignments_created=assignments_created,
        assignments_updated=assignments_updated,
        assignments_deactivated=assignments_deactivated,
    )


# ==================== KPI Configuration ====================


@router.post("/kpi-config", response_model=BatchKPIConfigResponse, status_code=201)
async def create_kpi_config(
    config_data: BatchKPIConfigCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    _get_accessible_batch(db, current_user, config_data.batch_id)
    _validate_kpi_weights(config_data)

    existing = (
        db.query(BatchKPIConfig)
        .filter(BatchKPIConfig.batch_id == config_data.batch_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="KPI config already exists for this batch")

    new_config = BatchKPIConfig(
        id=str(uuid.uuid4()),
        batch_id=config_data.batch_id,
        speech_to_text_weight=config_data.speech_to_text_weight,
        aht_weight=config_data.aht_weight,
        rate_of_speech_weight=config_data.rate_of_speech_weight,
        dead_air_weight=config_data.dead_air_weight,
        empathy_statements_weight=config_data.empathy_statements_weight,
        probing_questions_weight=config_data.probing_questions_weight,
        grammar_weight=config_data.grammar_weight,
        pronunciation_weight=config_data.pronunciation_weight,
        pacing_weight=config_data.pacing_weight,
        forbidden_words_penalty=config_data.forbidden_words_penalty,
        passing_score=config_data.passing_score,
        forbidden_words=_normalize_keyword_list(config_data.forbidden_words),
        empathy_keywords=_normalize_keyword_list(config_data.empathy_keywords),
        probing_keywords=_normalize_keyword_list(config_data.probing_keywords),
        target_aht_seconds=config_data.target_aht_seconds,
        target_ros_words_per_min=config_data.target_ros_words_per_min,
        target_dead_air_seconds=config_data.target_dead_air_seconds,
    )
    db.add(new_config)
    db.flush()
    _sync_call_simulation_batch_mirror_or_raise(
        db,
        batch_id=config_data.batch_id,
        fallback_trainer_id=str(current_user.id),
        detail="Supabase sync is required before creating the Call Simulation KPI rubric.",
    )
    db.commit()
    db.refresh(new_config)
    return BatchKPIConfigResponse.model_validate(new_config)


@router.get("/kpi-config/{batch_id}", response_model=BatchKPIConfigResponse)
async def get_kpi_config(
    batch_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    if current_user.role in [UserRole.ADMIN, UserRole.TRAINER]:
        _get_accessible_batch(db, current_user, batch_id)
    elif current_user.role == UserRole.TRAINEE:
        if batch_id not in {batch.id for batch in current_user.batches}:
            raise HTTPException(status_code=403, detail="Access denied")

    config = db.query(BatchKPIConfig).filter(BatchKPIConfig.batch_id == batch_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="KPI config not found")
    return BatchKPIConfigResponse.model_validate(config)


@router.put("/kpi-config/{batch_id}", response_model=BatchKPIConfigResponse)
async def update_kpi_config(
    batch_id: str,
    config_update: BatchKPIConfigUpdate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    _get_accessible_batch(db, current_user, batch_id)

    config = db.query(BatchKPIConfig).filter(BatchKPIConfig.batch_id == batch_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="KPI config not found")

    update_data = config_update.model_dump(exclude_unset=True)
    list_fields = {"forbidden_words", "empathy_keywords", "probing_keywords"}
    for field, value in update_data.items():
        setattr(config, field, _normalize_keyword_list(value) if field in list_fields else value)

    _validate_kpi_weights(config)
    db.flush()
    _sync_call_simulation_batch_mirror_or_raise(
        db,
        batch_id=batch_id,
        fallback_trainer_id=str(current_user.id),
        detail="Supabase sync is required before updating the Call Simulation KPI rubric.",
    )
    db.commit()
    db.refresh(config)
    return BatchKPIConfigResponse.model_validate(config)


# ==================== Bulk Upload ====================


@router.post("/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload_scenarios(
    batch_id: str,
    scenario_title: Optional[str] = None,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    batch = _get_accessible_batch(db, current_user, batch_id)
    trainer_audio_settings = _get_trainer_call_audio_settings(current_user)

    filename = file.filename or ""
    normalized_filename = filename.lower()
    document_metadata = _parse_uploaded_document_metadata(filename)
    document_title = document_metadata["title"]
    document_topic = document_metadata["topic"]
    document_description = document_metadata["description"]
    description_override: Optional[str] = None

    try:
        contents = await file.read()
        pd = _ensure_pandas()
        if normalized_filename.endswith(".csv"):
            if pd is None:
                raise HTTPException(status_code=503, detail='Pandas is required to parse CSV uploads.')
            dataframe = pd.read_csv(io.BytesIO(contents))
            script_rows, failed_rows, errors = _build_bulk_upload_rows_from_dataframe(dataframe)
        elif normalized_filename.endswith(".xlsx") or normalized_filename.endswith(".xls"):
            if pd is None:
                raise HTTPException(status_code=503, detail='Pandas is required to parse Excel uploads.')
            dataframe = pd.read_excel(io.BytesIO(contents))
            script_rows, failed_rows, errors = _build_bulk_upload_rows_from_dataframe(dataframe)
        elif normalized_filename.endswith(".txt"):
            paragraphs = [
                line.strip()
                for line in contents.decode("utf-8-sig", errors="ignore").splitlines()
                if line.strip()
            ]
            script_rows, failed_rows, errors, description_override = _build_bulk_upload_rows_from_document(
                paragraphs=paragraphs,
                filename=filename,
            )
            pd = _ensure_pandas()
            if pd is None:
                # build dataframe-like list-of-dicts fallback when pandas not present
                dataframe = script_rows
            else:
                dataframe = pd.DataFrame(script_rows)
        elif normalized_filename.endswith(".docx"):
            paragraphs = _extract_docx_paragraphs(contents)
            script_rows, failed_rows, errors, description_override = _build_bulk_upload_rows_from_document(
                paragraphs=paragraphs,
                filename=filename,
            )
            dataframe = pd.DataFrame(script_rows)
        else:
            try:
                dataframe = pd.read_csv(io.BytesIO(contents))
                script_rows, failed_rows, errors = _build_bulk_upload_rows_from_dataframe(dataframe)
            except Exception:
                try:
                    dataframe = pd.read_excel(io.BytesIO(contents))
                    script_rows, failed_rows, errors = _build_bulk_upload_rows_from_dataframe(dataframe)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Error reading upload. Provide a valid CSV, Excel, TXT, or DOCX file "
                            f"that contains Actor, Script, Score, and Scenario rows: {exc}"
                        ),
                    ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Error reading upload. Provide a valid CSV, Excel, TXT, or DOCX file "
                f"that contains Actor, Script, Score, and Scenario rows: {exc}"
            ),
        ) from exc

    if not script_rows:
        raise HTTPException(
            status_code=400,
            detail="No valid scenario rows were found. Provide Actor, Script, Score, and Scenario values.",
        )

    row_assets = _build_call_simulation_row_assets(rows=script_rows)
    grouped_rows = row_assets["scenario_groups"]
    template_columns = (
        list(dataframe.columns)
        if {"Actor", "Script", "Score", "Scenario"}.issubset(set(dataframe.columns))
        else ["Actor", "Script", "Score", "Scenario"]
    )

    scenario_title_text = str(scenario_title or "").strip() or document_title
    first_group = grouped_rows[0]
    first_csr_row = row_assets["first_csr_row"]
    first_member_row = row_assets["first_member_row"]
    existing_mappings = (
        db.query(BatchScenarioMapping)
        .join(Scenario, Scenario.id == BatchScenarioMapping.scenario_id)
        .filter(
            BatchScenarioMapping.batch_id == batch.id,
            Scenario.title == scenario_title_text,
            Scenario.created_by == current_user.id,
        )
        .order_by(BatchScenarioMapping.assigned_at.desc())
        .all()
    )
    retired_scenario_ids = {
        str(mapping.scenario_id)
        for mapping in existing_mappings
        if mapping.scenario_id
    }
    for mapping in existing_mappings:
        mapping.is_active = False

    target_scenario = Scenario(
        id=str(uuid.uuid4()),
        title=scenario_title_text,
        created_by=current_user.id,
        opening_prompt="Answer the call, review the member context, and deliver the expected CSR spiel.",
    )
    db.add(target_scenario)
    db.flush()

    target_scenario.title = scenario_title_text
    target_scenario.description = (
        document_description
        or (description_override or "").strip()
        or f"Bulk uploaded scenario - {scenario_title_text}"
    )
    target_scenario.opening_prompt = (
        "Answer the call, review the member context, and deliver the expected CSR spiel."
    )
    target_scenario.difficulty = target_scenario.difficulty
    target_scenario.purpose = target_scenario.purpose
    target_scenario.lob = batch.lob
    target_scenario.expected_keywords = []
    target_scenario.member_profile = {
        "name": first_member_row["actor"] if first_member_row else "Member",
        "member_id": None,
        "plan_type": None,
        "verification_status": None,
        "problem_statement": first_member_row["script"] if first_member_row else first_csr_row["script"],
    }
    target_scenario.cxone_metadata = {
        "workspace_label": "MAX Mock Call",
        "crm_section": "Member Details",
        "source_template": file.filename,
        "document_title": document_title,
        "document_topic": document_topic,
        "document_description": document_description or description_override,
        "member_name": first_member_row["actor"] if first_member_row else "Member",
        "problem_statement": first_member_row["script"] if first_member_row else first_csr_row["script"],
    }
    target_scenario.call_simulation_config = _apply_batch_kpi_to_call_simulation_config(
        {
            "source": "bulk_upload",
            "source_file_name": filename,
            "source_document_title": document_title,
            "source_document_topic": document_topic,
            "source_document_description": document_description or description_override,
            "use_google_asr": True,
            "template_columns": template_columns,
            "topic": document_topic or scenario_title_text,
            "script_rows": row_assets["script_rows"],
            "scenario_groups": grouped_rows,
            "script_flow": row_assets["script_flow"],
            "interface": "nice-cxone",
            "show_actor_script_overlay": True,
            "require_hold_before_member_response": True,
        },
        kpi_config=_get_effective_kpi_config(db, batch.id),
        batch_id=batch.id,
    )
    target_scenario.ringer_audio_url = trainer_audio_settings.get("ringer_audio_url")
    target_scenario.hold_audio_url = trainer_audio_settings.get("hold_audio_url")
    target_scenario.is_published = True
    target_scenario.is_draft = False

    db.add(
        BatchScenarioMapping(
            id=str(uuid.uuid4()),
            batch_id=batch.id,
            scenario_id=target_scenario.id,
            assigned_by=current_user.id,
            is_active=True,
        )
    )

    uploaded_steps = row_assets["steps"]
    variations_created = 0
    for row in row_assets["variations"]:
        db.add(
            ScenarioVariation(
                id=str(uuid.uuid4()),
                scenario_id=target_scenario.id,
                actor_name=row["actor_name"],
                script=row["script"],
                score=row["score"],
                branching_logic=row["branching_logic"],
                is_active=True,
            )
        )
        variations_created += 1

    class _StepProxy:
        def __init__(self, payload: dict[str, Any]) -> None:
            for key, value in payload.items():
                setattr(self, key, value)

    await _replace_scenario_steps(
        db,
        scenario=target_scenario,
        steps=[_StepProxy(step) for step in uploaded_steps],
    )

    _sync_call_simulation_assignments_for_scenario(db, target_scenario.id)
    for retired_scenario_id in retired_scenario_ids:
        if retired_scenario_id != target_scenario.id:
            _sync_call_simulation_assignments_for_scenario(db, retired_scenario_id)
    db.flush()
    _sync_call_simulation_scenario_mirror_or_raise(
        db,
        scenario=target_scenario,
        trainer_id=str(current_user.id),
        batch_id=batch.id,
        detail="Supabase sync is required before finishing the Call Simulation bulk upload.",
    )
    db.commit()

    return BulkUploadResponse(
        scenario_id=target_scenario.id,
        variations_created=variations_created,
        failed_rows=failed_rows,
        errors=errors,
    )


@router.get("/bulk-upload-template")
async def get_bulk_upload_template(
    format: str = Query("csv", pattern="^(csv|xlsx|txt)$"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    rows = [
        ["Actor", "Script", "Score", "Scenario"],
        ["CSR", "Thank you for calling member support. May I have your full name?", 3, 1],
        ["CSR", "For security, may I verify your member ID?", 2, 1],
        ["Member", "I need help checking my account and a recent order.", "", 1],
        ["CSR", "I can help with that. May I verify your date of birth?", 3, 2],
        ["CSR", "Do you also have the order or reference number available?", 2, 2],
        ["Member", "I have the order number ready if you need it.", "", 2],
    ]

    if format == "txt":
        txt_lines = [
            "Call Verification and Account Support",
            "Member needs help verifying the account and checking a recent order.",
            "Actor | Script | Score | Scenario",
            "CSR | Thank you for calling member support. May I have your full name? | 3 | 1",
            "CSR | For security, may I verify your member ID? | 2 | 1",
            "Member | I need help checking my account and a recent order. | 0 | 1",
            "CSR | I can help with that. May I verify your date of birth? | 3 | 2",
            "Member | I have the order number ready if you need it. | 0 | 2",
        ]
        return Response(
            content="\n".join(txt_lines),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=call-simulation-template.txt"},
        )

    if format == "csv":
        csv_lines = [
            ",".join(
                '"' + str(value).replace('"', '""') + '"' if isinstance(value, str) else str(value)
                for value in row
            )
            for row in rows
        ]
        return Response(
            content="\n".join(csv_lines),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=call-simulation-template.csv"},
        )

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Scenarios"
    for row in rows:
        worksheet.append(row)

    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=call-simulation-template.xlsx"},
    )


# ==================== Trainer Asset Uploads ====================


@router.get("/audio-settings", response_model=CallSimulationAudioSettingsResponse)
async def get_call_simulation_audio_settings(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    return _serialize_trainer_call_audio_settings(current_user)


@router.put("/audio-settings", response_model=CallSimulationAudioSettingsResponse)
async def update_call_simulation_audio_settings(
    payload: CallSimulationAudioSettingsUpdate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    update_data = payload.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No Call Simulation audio changes were provided.")

    return _persist_trainer_call_audio_settings(
        db,
        current_user,
        ringer_audio_url=update_data.get("ringer_audio_url", _UNSET),
        hold_audio_url=update_data.get("hold_audio_url", _UNSET),
        ringer_audio_source="manual" if "ringer_audio_url" in update_data else _UNSET,
        hold_audio_source="manual" if "hold_audio_url" in update_data else _UNSET,
    )


@router.delete("/audio-settings", response_model=CallSimulationAudioSettingsResponse)
async def delete_call_simulation_audio_setting(
    asset_kind: str = Query(..., pattern="^(ringer|hold)$"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    if asset_kind == "ringer":
        return _persist_trainer_call_audio_settings(db, current_user, ringer_audio_url=None, ringer_audio_source=None)
    return _persist_trainer_call_audio_settings(db, current_user, hold_audio_url=None, hold_audio_source=None)


@router.get("/audio-assets", response_model=list[CallSimulationAudioAssetResponse])
async def list_call_simulation_audio_assets(
    scenario_id: Optional[str] = Query(None),
    scope: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    normalized_scope = str(scope or "").strip().lower()
    if scenario_id:
        _get_accessible_scenario(db, current_user, scenario_id)

    query = (
        db.query(CallSimulationAudioAsset)
        .filter(
            CallSimulationAudioAsset.trainer_id == current_user.id,
            CallSimulationAudioAsset.is_active == True,
        )
        .order_by(CallSimulationAudioAsset.updated_at.desc(), CallSimulationAudioAsset.created_at.desc())
    )

    if scenario_id:
        query = query.filter(CallSimulationAudioAsset.scenario_id == scenario_id)
    elif normalized_scope == "shared":
        query = query.filter(
            CallSimulationAudioAsset.scenario_id.is_(None),
            CallSimulationAudioAsset.asset_kind.in_(["ringer", "hold"]),
        )

    assets = query.all()
    return [_serialize_call_simulation_audio_asset(asset) for asset in assets]


@router.post("/assets/audio", response_model=CallSimulationAudioAssetUploadResponse)
async def upload_call_simulation_audio_asset(
    asset_kind: str = Form(...),
    scenario_id: Optional[str] = Form(None),
    step_number: Optional[int] = Form(None),
    replace_audio_url: Optional[str] = Form(None),
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    normalized_asset_kind = (asset_kind or "").strip().lower()
    allowed_asset_kinds = {"member-step", "ringer", "hold", "scenario-ringer", "scenario-hold"}
    if normalized_asset_kind not in allowed_asset_kinds:
        raise HTTPException(status_code=400, detail="Unsupported Call Simulation audio asset type")

    scenario_segment = "draft"
    if normalized_asset_kind in {"ringer", "hold"}:
        scenario_segment = "global"
    elif scenario_id:
        scenario = _get_owned_scenario_for_audio_management(db, current_user, scenario_id)
        scenario_segment = scenario.id

    file_bytes = await file.read()
    upload_meta = _prepare_call_simulation_audio_upload(
        file=file,
        file_bytes=file_bytes,
        fallback_stem="call-simulation-asset",
    )
    step_prefix = (
        f"step-{max(int(step_number or 0), 0):02d}_"
        if normalized_asset_kind == "member-step" and step_number
        else ""
    )
    storage_leaf = (
        f"{step_prefix}{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{upload_meta['sanitized_stem']}{upload_meta['extension']}"
    )

    supabase = get_supabase_client()
    audio_url = supabase.upload_call_simulation_asset(
        file_data=file_bytes,
        trainer_id=current_user.id,
        scenario_id=scenario_segment,
        asset_kind=normalized_asset_kind,
        filename=storage_leaf,
        content_type=str(upload_meta["content_type"]),
    )
    if not audio_url:
        raise HTTPException(
            status_code=503,
            detail="Supabase storage could not save the call simulation audio asset.",
        )

    settings_payload: Optional[CallSimulationAudioSettingsResponse] = None
    audio_asset: Optional[CallSimulationAudioAsset] = None
    if normalized_asset_kind in {"ringer", "hold"}:
        if normalized_asset_kind == "ringer":
            settings_payload = _persist_trainer_call_audio_settings(
                db,
                current_user,
                ringer_audio_url=audio_url,
                ringer_audio_source="upload",
            )
        else:
            settings_payload = _persist_trainer_call_audio_settings(
                db,
                current_user,
                hold_audio_url=audio_url,
                hold_audio_source="upload",
            )
        audio_asset = _find_active_call_simulation_audio_asset(
            db,
            trainer_id=current_user.id,
            public_url=audio_url,
            asset_kinds=[normalized_asset_kind],
        )
        if audio_asset is None:
            audio_asset = _create_call_simulation_audio_asset_record(
                db,
                trainer_id=current_user.id,
                scenario_id=None,
                script_turn_id=None,
                step_number=None,
                asset_kind=normalized_asset_kind,
                source_type="upload",
                public_url=audio_url,
                file_name=storage_leaf,
                file_type=str(upload_meta["content_type"]),
                file_size=int(upload_meta["file_size"]),
                asset_metadata={
                    "uploaded_filename": file.filename or storage_leaf,
                    "scope": "shared",
                },
            )
    else:
        audio_asset = _create_call_simulation_audio_asset_record(
            db,
            trainer_id=current_user.id,
            scenario_id=scenario_id,
            script_turn_id=None,
            step_number=step_number,
            asset_kind=normalized_asset_kind,
            source_type="upload",
            public_url=audio_url,
            file_name=storage_leaf,
            file_type=str(upload_meta["content_type"]),
            file_size=int(upload_meta["file_size"]),
            asset_metadata={
                "uploaded_filename": file.filename or storage_leaf,
            },
        )
        db.commit()

    if replace_audio_url and replace_audio_url != audio_url:
        if normalized_asset_kind in {"ringer", "hold"}:
            stale_shared_assets = (
                db.query(CallSimulationAudioAsset)
                .filter(
                    CallSimulationAudioAsset.trainer_id == current_user.id,
                    CallSimulationAudioAsset.public_url == replace_audio_url,
                    CallSimulationAudioAsset.asset_kind == normalized_asset_kind,
                    CallSimulationAudioAsset.is_active == True,
                    CallSimulationAudioAsset.scenario_id.is_(None),
                )
                .all()
            )
            for stale_asset in stale_shared_assets:
                _deactivate_call_simulation_audio_asset_record(
                    db,
                    asset=stale_asset,
                    delete_storage=True,
                )
            if stale_shared_assets:
                db.commit()
        else:
            stale_asset = _find_active_call_simulation_audio_asset(
                db,
                trainer_id=current_user.id,
                public_url=replace_audio_url,
                asset_kinds=[normalized_asset_kind],
                scenario_id=scenario_id,
                step_number=step_number,
            )
            if stale_asset:
                _deactivate_call_simulation_audio_asset_record(
                    db,
                    asset=stale_asset,
                    delete_storage=True,
                )
                db.commit()

    _log_call_simulation_audio_action(
        db,
        user=current_user,
        action_type="upload_audio" if not replace_audio_url else "replace_audio",
        scenario_id=None if scenario_segment in {"draft", "global"} else scenario_segment,
        step_number=step_number,
        file_name=storage_leaf,
        metadata={
            "asset_kind": normalized_asset_kind,
            "audio_url": audio_url,
            "bucket_name": getattr(audio_asset, "bucket_name", None),
            "storage_path": getattr(audio_asset, "storage_path", None),
            "file_size": int(upload_meta["file_size"]),
            "content_type": str(upload_meta["content_type"]),
        },
    )
    db.commit()

    return CallSimulationAudioAssetUploadResponse(
        audio_url=audio_url,
        asset_kind=normalized_asset_kind,
        filename=storage_leaf,
        scenario_id=None if scenario_segment in {"draft", "global"} else scenario_segment,
        settings=settings_payload,
        audio_asset=_serialize_call_simulation_audio_asset(audio_asset) if audio_asset else None,
    )


@router.delete("/audio-assets", response_model=SuccessResponse)
async def delete_call_simulation_audio_asset(
    payload: dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    normalized_asset_kind = str(payload.get("asset_kind") or "").strip().lower()
    allowed_asset_kinds = {"member-step", "ringer", "hold", "scenario-ringer", "scenario-hold"}
    if normalized_asset_kind not in allowed_asset_kinds:
        raise HTTPException(status_code=400, detail="Unsupported Call Simulation audio asset type")

    scenario_id = _normalize_uuid_candidate(payload.get("scenario_id"))
    if scenario_id and normalized_asset_kind not in {"ringer", "hold"}:
        _get_owned_scenario_for_audio_management(db, current_user, scenario_id)

    audio_url = _normalize_optional_url(payload.get("audio_url"))
    if not audio_url:
        raise HTTPException(status_code=400, detail="Audio URL is required to remove the asset.")

    step_number_value = payload.get("step_number")
    step_number = int(step_number_value) if isinstance(step_number_value, int) or str(step_number_value or "").isdigit() else None

    query = db.query(CallSimulationAudioAsset).filter(
        CallSimulationAudioAsset.trainer_id == current_user.id,
        CallSimulationAudioAsset.public_url == audio_url,
        CallSimulationAudioAsset.asset_kind == normalized_asset_kind,
        CallSimulationAudioAsset.is_active == True,
    )
    if scenario_id:
        query = query.filter(
            (CallSimulationAudioAsset.scenario_id == scenario_id)
            | (CallSimulationAudioAsset.scenario_id.is_(None))
        )
    if step_number is not None:
        query = query.filter(CallSimulationAudioAsset.step_number == step_number)

    matching_assets = query.all()
    if not matching_assets:
        get_supabase_client().delete_by_public_url(audio_url)
        return SuccessResponse(message="Audio asset removed.")

    matching_ids = {asset.id for asset in matching_assets}
    is_still_referenced = (
        db.query(CallSimulationAudioAsset)
        .filter(
            CallSimulationAudioAsset.trainer_id == current_user.id,
            CallSimulationAudioAsset.public_url == audio_url,
            CallSimulationAudioAsset.is_active == True,
            ~CallSimulationAudioAsset.id.in_(matching_ids),
        )
        .first()
        is not None
    )

    for asset in matching_assets:
        _deactivate_call_simulation_audio_asset_record(
            db,
            asset=asset,
            delete_storage=not is_still_referenced,
        )

    _log_call_simulation_audio_action(
        db,
        user=current_user,
        action_type="delete_audio",
        scenario_id=scenario_id,
        step_number=step_number,
        file_name=matching_assets[0].file_name if matching_assets else None,
        metadata={
            "asset_kind": normalized_asset_kind,
            "audio_url": audio_url,
            "deleted_storage": not is_still_referenced,
        },
    )
    db.commit()
    return SuccessResponse(message="Audio asset removed.")


# ==================== Trainee Simulation ====================


@router.get("/available")
async def get_available_scenarios(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)

    did_sync_assignments = _sync_call_simulation_assignments_for_trainee(db, current_user)
    if did_sync_assignments:
        db.commit()

    assignments = (
        db.query(CallSimulationAssignment)
        .filter(
            CallSimulationAssignment.trainee_id == current_user.id,
            CallSimulationAssignment.is_active == True,
        )
        .order_by(CallSimulationAssignment.assigned_at.desc(), CallSimulationAssignment.updated_at.desc())
        .all()
    )
    if not assignments:
        return {"scenarios": []}

    scenario_ids = [assignment.scenario_id for assignment in assignments]
    trainer_ids = [assignment.assigned_by for assignment in assignments if assignment.assigned_by]
    batch_ids = [assignment.batch_id for assignment in assignments if assignment.batch_id]
    scenarios = {
        scenario.id: scenario
        for scenario in db.query(Scenario)
        .filter(
            Scenario.id.in_(scenario_ids),
            Scenario.is_published == True,
        )
        .all()
    }
    trainers = {
        trainer.id: trainer
        for trainer in db.query(User)
        .filter(User.id.in_(trainer_ids or ["__none__"]))
        .all()
    }
    batches = {
        batch.id: batch
        for batch in db.query(Batch)
        .filter(Batch.id.in_(batch_ids or ["__none__"]))
        .all()
    }
    kpi_configs_by_batch = {
        config.batch_id: config
        for config in db.query(BatchKPIConfig)
        .filter(BatchKPIConfig.batch_id.in_(batch_ids or ["__none__"]))
        .all()
    }
    mappings = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.scenario_id.in_(scenario_ids),
            BatchScenarioMapping.is_active == True,
        )
        .all()
    )
    scenario_sessions = (
        db.query(SimSession)
        .filter(
            SimSession.trainee_id == current_user.id,
            SimSession.scenario_id.in_(scenario_ids),
        )
        .order_by(SimSession.created_at.desc())
        .all()
    )
    sessions_by_scenario: dict[str, list[SimSession]] = defaultdict(list)
    for session in scenario_sessions:
        sessions_by_scenario[session.scenario_id].append(session)
    latest_coaching_logs = _get_latest_sim_session_coaching_logs(
        db,
        [session.id for session in scenario_sessions],
    )

    scenarios_payload: list[dict[str, Any]] = []
    for assignment in assignments:
        scenario = scenarios.get(assignment.scenario_id)
        if not scenario:
            continue

        scenario_steps = _build_scenario_steps(scenario)
        assigned_batches = _build_scenario_assignment_summaries(
            db,
            scenario.id,
            mappings=[
                candidate
                for candidate in mappings
                if candidate.scenario_id == scenario.id
            ],
        )
        variation_count = (
            db.query(ScenarioVariation)
            .filter(
                ScenarioVariation.scenario_id == scenario.id,
                ScenarioVariation.is_active == True,
            )
            .count()
        )
        trainee_sessions = [
            session
            for session in sessions_by_scenario.get(scenario.id, [])
            if not assignment.batch_id or session.batch_id in {assignment.batch_id, None}
        ]
        configured_max_attempts = _resolve_call_assignment_max_attempts(
            assignment,
            scenario=scenario,
            fallback=DEFAULT_MAX_ATTEMPTS,
        )
        latest_session = trainee_sessions[0] if trainee_sessions else None
        latest_coaching_log = latest_coaching_logs.get(latest_session.id) if latest_session else None
        normalized_verdict = (latest_session.trainer_verdict_status or "pending").strip().lower() if latest_session else "pending"
        active_session = next((session for session in trainee_sessions if session.status == "in_progress"), None)
        completion_locked = bool(
            latest_session
            and (
                normalized_verdict == "competent"
                or (latest_session.pass_fail and normalized_verdict != "retake")
                or latest_session.certificate_id
            )
        )
        can_retake = False
        launch_blocked = False
        launch_block_reason: Optional[str] = None
        remaining_attempts: Optional[int] = None

        if latest_session:
            latest_attempt_number = int(latest_session.attempt_number or 1)
            latest_max_attempts = configured_max_attempts
            remaining_attempts = max(latest_max_attempts - latest_attempt_number, 0)

            if latest_session.status == "in_progress":
                launch_block_reason = "A call attempt is already in progress. Resume the softphone to continue."
            elif completion_locked:
                launch_blocked = True
                launch_block_reason = "This Call Simulation is already completed. Review your certificate instead of launching a new attempt."
            elif not latest_session.pass_fail:
                if latest_coaching_log and latest_coaching_log.status == "sent":
                    launch_blocked = True
                    launch_block_reason = "Acknowledge the trainer coaching notes before retaking this simulation."
                elif latest_attempt_number >= latest_max_attempts:
                    launch_blocked = True
                    launch_block_reason = "Maximum retake attempts reached for this Call Simulation."
                else:
                    can_retake = True

        if not launch_blocked:
            try:
                _validate_call_simulation_launch_assets(scenario, scenario_steps)
            except HTTPException as exc:
                launch_blocked = True
                launch_block_reason = str(exc.detail)

        scenario_batch = batches.get(assignment.batch_id) if assignment.batch_id else None
        effective_kpi_config = kpi_configs_by_batch.get(assignment.batch_id) if assignment.batch_id else None
        passing_score = _resolve_call_scenario_passing_score(
            scenario,
            float(effective_kpi_config.passing_score or DEFAULT_PASSING_SCORE)
            if effective_kpi_config
            else DEFAULT_PASSING_SCORE,
        )

        scenarios_payload.append(
            {
                "id": scenario.id,
                "assignment_id": assignment.id,
                "assigned_at": assignment.assigned_at.isoformat() if assignment.assigned_at else None,
                "assigned_by_id": assignment.assigned_by,
                "assigned_by_name": trainers.get(assignment.assigned_by).full_name
                if trainers.get(assignment.assigned_by)
                else None,
                "assignment_batch_id": assignment.batch_id,
                "assignment_batch_name": scenario_batch.name
                if scenario_batch
                else None,
                "assignment_wave_number": scenario_batch.wave_number
                if scenario_batch
                else None,
                "title": scenario.title,
                "topic": str(
                    _normalize_json_object(scenario.call_simulation_config).get("topic")
                    or scenario.title
                    or "Call scenario"
                ),
                "description": scenario.description,
                "difficulty": scenario.difficulty.value if scenario.difficulty else None,
                "estimated_duration": scenario.estimated_duration,
                "expected_duration_seconds": (
                    scenario.estimated_duration
                    or int(sum(_estimate_script_duration_seconds(step.script) for step in scenario_steps))
                )
                if scenario_steps
                else scenario.estimated_duration,
                "variation_count": variation_count,
                "scenario_groups_count": _count_scenario_groups(scenario_steps),
                "steps_count": len(scenario_steps),
                "passing_score": passing_score,
                "assigned_batches": [
                    {
                        "batch_id": summary.batch_id,
                        "batch_name": summary.batch_name,
                        "wave_number": summary.wave_number,
                        "assigned_at": summary.assigned_at.isoformat() if summary.assigned_at else None,
                    }
                    for summary in assigned_batches
                ],
                "attempt_count": len(trainee_sessions),
                "max_attempts": configured_max_attempts,
                "retake_required": bool(normalized_verdict == "retake" or can_retake),
                "competent": completion_locked,
                "latest_score": float(latest_session.weighted_score or 0.0) if latest_session else 0.0,
                "latest_session_id": latest_session.id if latest_session else None,
                "latest_status": latest_session.status if latest_session else None,
                "latest_completed_at": latest_session.completed_at.isoformat()
                if latest_session and latest_session.completed_at
                else None,
                "active_session_id": active_session.id if active_session else None,
                "latest_certificate_id": latest_session.certificate_id if latest_session else None,
                "can_retake": can_retake,
                "remaining_attempts": remaining_attempts,
                "launch_blocked": launch_blocked,
                "launch_block_reason": launch_block_reason,
            }
        )

    return {"scenarios": scenarios_payload}


@router.api_route("/tts", methods=["GET", "POST"])
async def synthesize_member_speech(
    text: str = Query(...),
    persist: bool = Query(False),
    require_supabase: bool = Query(False),
    scenario_id: Optional[str] = Query(None),
    step_number: Optional[int] = Query(None, ge=1),
    asset_kind: str = Query("member-step"),
    replace_audio_url: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    is_trainer_request = current_user.role in [UserRole.ADMIN, UserRole.TRAINER]
    if not is_trainer_request and current_user.role != UserRole.TRAINEE:
        raise HTTPException(status_code=403, detail="Call Simulation access required")

    normalized_text = text.strip()
    if not normalized_text:
        raise HTTPException(status_code=400, detail="Text is required")

    browser_fallback_message = "AI voice is using browser fallback mode."
    strict_persistence_message = (
        "Generated speech must be saved to supported storage, but that upload did not complete."
    )
    storage_warning: Optional[str] = None

    if persist and not get_tts_service().is_available():
        if require_supabase:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Server-side TTS is unavailable. Configure GOOGLE_API_KEY or GEMINI_API_KEY, "
                    "OPENAI_API_KEY, or AZURE_SPEECH_KEY/AZURE_SPEECH_REGION for deployed speech generation."
                ),
            )
        logger.warning(
            "Server-side TTS unavailable for optional persistence request. Falling back to browser audio inline."
        )
        storage_warning = (
            "Server-side TTS is unavailable. Generated speech is returned inline for draft playback."
        )
        persist = False

    try:
        synthesis_fn = text_to_speech_for_persistence if persist else text_to_speech
        audio_result = await synthesis_fn(
            normalized_text,
            voice_name="Puck",
            speaking_style="professional",
        )
    except Exception as exc:
        if persist:
            logger.warning("Call simulation TTS synthesis failed for persisted audio: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Generated speech could not be synthesized, so no Supabase audio was saved.",
            )
        logger.warning("Call simulation TTS synthesis failed. Browser fallback will be used: %s", exc)
        audio_result = {
            "audio_url": None,
            "audio_base64": None,
            "audio_bytes": None,
            "audio_content_type": "audio/wav",
            "audio_extension": "wav",
            "duration": 0,
            "provider": "browser_fallback",
            "error": str(exc),
            "fallback_mode": "browser",
        }

    audio_url = str(audio_result.get("audio_url") or "").strip() or None
    audio_base64 = str(audio_result.get("audio_base64") or "").strip() or None
    audio_bytes = _coerce_audio_bytes(audio_result.get("audio_bytes"))
    audio_content_type = str(audio_result.get("audio_content_type") or "audio/wav").strip() or "audio/wav"
    audio_extension = re.sub(r"[^a-z0-9]+", "", str(audio_result.get("audio_extension") or "wav").lower()) or "wav"
    duration = float(audio_result.get("duration") or 0.0)
    provider = str(audio_result.get("provider") or "").strip() or ("gemini" if gemini_tts_engine.is_available() else "browser_fallback")
    provider_error = str(audio_result.get("error") or "").strip() or None
    fallback_mode = str(audio_result.get("fallback_mode") or "").strip() or None

    storage_mode = "inline"
    audio_asset: Optional[CallSimulationAudioAsset] = None
    normalized_asset_kind = str(asset_kind or "").strip().lower() or "member-step"
    persisted_scenario: Optional[Scenario] = None
    persisted_flow_step: Optional[ScenarioFlow] = None
    persisted_script_flow_index: Optional[int] = None
    persisted_script_flow_rows: Optional[list[dict[str, Any]]] = None
    resolved_asset_owner_id = str(current_user.id)

    if persist:
        if normalized_asset_kind not in {"member-step", "ringer", "hold", "scenario-ringer", "scenario-hold"}:
            raise HTTPException(status_code=400, detail="Unsupported Call Simulation audio asset type")
        supabase = get_supabase_client()
        if not supabase.is_available:
            if require_supabase:
                raise HTTPException(status_code=503, detail=strict_persistence_message)
            supabase = None
            storage_warning = (
                "Supabase storage is unavailable. Generated speech will be returned as inline audio for draft playback."
            )

        normalized_replace_audio_url = _normalize_optional_url(replace_audio_url)
        scenario_segment = "draft"
        if is_trainer_request:
            if normalized_asset_kind in {"ringer", "hold"}:
                scenario_segment = "global"
            elif scenario_id:
                persisted_scenario = _get_owned_scenario_for_audio_management(db, current_user, scenario_id)
                scenario_segment = persisted_scenario.id
                if normalized_asset_kind == "member-step" and step_number is not None:
                    persisted_flow_step = next(
                        (
                            step
                            for step in list(getattr(persisted_scenario, "flow_steps", []) or [])
                            if int(step.step_number or 0) == int(step_number)
                        ),
                        None,
                    )
        else:
            if current_user.role != UserRole.TRAINEE:
                raise HTTPException(status_code=403, detail="Only trainees or trainers can persist Call Simulation member audio.")
            if normalized_asset_kind != "member-step":
                raise HTTPException(status_code=403, detail="Trainees can only persist Member AI step audio.")
            if not scenario_id or step_number is None:
                raise HTTPException(status_code=400, detail="Scenario and step details are required to persist trainee member audio.")

            persisted_scenario = (
                db.query(Scenario)
                .filter(
                    Scenario.id == scenario_id,
                    Scenario.is_published == True,
                )
                .first()
            )
            if not persisted_scenario:
                raise HTTPException(status_code=404, detail="Scenario not found")

            assignment = _get_active_call_simulation_assignment(
                db,
                trainee_id=current_user.id,
                scenario_id=persisted_scenario.id,
            )
            if not assignment:
                raise HTTPException(status_code=403, detail="Scenario is not assigned to your trainee workspace")

            persisted_step = next(
                (
                    step
                    for step in _build_scenario_steps(persisted_scenario)
                    if int(step.step_number or 0) == int(step_number)
                ),
                None,
            )
            if not persisted_step or persisted_step.actor != "member":
                raise HTTPException(status_code=400, detail="Only Member AI steps can generate persisted playback audio.")

            scenario_segment = persisted_scenario.id
            resolved_asset_owner_id = str(
                assignment.assigned_by
                or persisted_scenario.created_by
                or _resolve_call_simulation_assignment_trainer_id(
                    db,
                    scenario=persisted_scenario,
                )
                or ""
            ).strip()
            if not resolved_asset_owner_id:
                raise HTTPException(status_code=503, detail="A trainer owner could not be resolved for this Call Simulation audio asset.")

            persisted_flow_step = next(
                (
                    step
                    for step in list(getattr(persisted_scenario, "flow_steps", []) or [])
                    if int(step.step_number or 0) == int(step_number)
                ),
                None,
            )
            scenario_config = _normalize_json_object(getattr(persisted_scenario, "call_simulation_config", None))
            configured_script_flow = scenario_config.get("script_flow")
            if isinstance(configured_script_flow, list):
                persisted_script_flow_rows = [
                    _normalize_json_object(row)
                    for row in configured_script_flow
                    if isinstance(row, dict)
                ]
                for index, row in enumerate(persisted_script_flow_rows):
                    candidate_step_number = row.get("member_step_number")
                    try:
                        if int(candidate_step_number) == int(step_number):
                            persisted_script_flow_index = index
                            break
                    except (TypeError, ValueError):
                        continue

        upload_bytes: Optional[bytes] = audio_bytes
        if upload_bytes is None and audio_base64:
            try:
                upload_bytes = b64decode(audio_base64)
            except Exception as exc:
                logger.warning("Unable to decode generated TTS audio for persistence: %s", exc)
                upload_bytes = None

        if upload_bytes is None and audio_url:
            upload_bytes = _coerce_audio_bytes(audio_url)
            if upload_bytes is None:
                logger.warning(
                    "Generated TTS audio URL could not be decoded for persistence. "
                    "audio_url=%s",
                    audio_url[:128],
                )

        if upload_bytes is None and gemini_tts_engine.is_available():
            try:
                upload_bytes = gemini_tts_engine.synthesize(
                    normalized_text,
                    voice_name="Puck",
                    speaking_style="professional",
                )
                if upload_bytes:
                    audio_content_type = "audio/wav"
                    audio_extension = "wav"
                    provider = "gemini"
            except Exception as exc:
                logger.warning("Gemini TTS persistence synthesis failed: %s", exc)
                upload_bytes = None

        if upload_bytes is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Generated speech could not be synthesized by the backend, "
                    "so nothing was saved to Supabase."
                ),
            )

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        safe_leaf = re.sub(r"[^a-z0-9]+", "-", normalized_text.lower()).strip("-") or "member-script"
        prefix = f"step-{int(step_number):02d}_" if normalized_asset_kind == "member-step" and step_number else ""
        filename = f"{prefix}{timestamp}_{safe_leaf[:48]}.{audio_extension}"

        uploaded_audio_url: Optional[str] = None
        if supabase is not None:
            uploaded_audio_url = supabase.upload_call_simulation_asset(
                file_data=upload_bytes,
                trainer_id=resolved_asset_owner_id,
                scenario_id=scenario_segment,
                asset_kind=normalized_asset_kind,
                filename=filename,
                content_type=audio_content_type,
            )

        if uploaded_audio_url:
            if require_supabase and not _is_supabase_storage_public_url(uploaded_audio_url):
                if supabase is not None:
                    supabase.delete_by_public_url(uploaded_audio_url)
                raise HTTPException(status_code=503, detail=strict_persistence_message)
            audio_url = uploaded_audio_url
            storage_mode = "supabase"
            if normalized_asset_kind in {"ringer", "hold"}:
                audio_asset = _find_active_call_simulation_audio_asset(
                    db,
                    trainer_id=resolved_asset_owner_id,
                    public_url=uploaded_audio_url,
                    asset_kinds=[normalized_asset_kind],
                )
                if audio_asset is None:
                    audio_asset = _create_call_simulation_audio_asset_record(
                        db,
                        trainer_id=resolved_asset_owner_id,
                        scenario_id=None,
                        script_turn_id=None,
                        step_number=None,
                        asset_kind=normalized_asset_kind,
                        source_type="generated_tts",
                        public_url=uploaded_audio_url,
                        file_name=filename,
                        file_type=audio_content_type,
                        file_size=len(upload_bytes),
                        voice_used="Puck",
                        provider=provider,
                        generated_text=normalized_text,
                        asset_metadata={
                            "storage_mode": storage_mode,
                            "scope": "shared",
                            "voice_type": "Puck",
                            "language": "en-US",
                            "duration": round(max(duration, 0.0), 2),
                        },
                    )

                if normalized_asset_kind == "ringer":
                    _persist_trainer_call_audio_settings(
                        db,
                        current_user,
                        ringer_audio_url=uploaded_audio_url,
                        ringer_audio_source="generated_tts",
                    )
                else:
                    _persist_trainer_call_audio_settings(
                        db,
                        current_user,
                        hold_audio_url=uploaded_audio_url,
                        hold_audio_source="generated_tts",
                    )
            else:
                audio_asset = _create_call_simulation_audio_asset_record(
                    db,
                    trainer_id=resolved_asset_owner_id,
                    scenario_id=scenario_id,
                    script_turn_id=persisted_flow_step.id if persisted_flow_step else None,
                    step_number=step_number,
                    asset_kind=normalized_asset_kind,
                    source_type="generated_tts",
                    public_url=uploaded_audio_url,
                    file_name=filename,
                    file_type=audio_content_type,
                    file_size=len(upload_bytes),
                    voice_used="Puck",
                    provider=provider,
                    generated_text=normalized_text,
                    asset_metadata={
                        "storage_mode": storage_mode,
                        "voice_type": "Puck",
                        "language": "en-US",
                        "duration": round(max(duration, 0.0), 2),
                    },
                )
                if persisted_flow_step:
                    persisted_flow_step.prompt_audio = uploaded_audio_url
                    persisted_metadata = _normalize_json_object(persisted_flow_step.step_metadata)
                    persisted_metadata["member_audio_url"] = uploaded_audio_url
                    persisted_flow_step.step_metadata = persisted_metadata
                    db.add(persisted_flow_step)
                elif persisted_scenario and persisted_script_flow_rows is not None and persisted_script_flow_index is not None:
                    persisted_script_flow_rows[persisted_script_flow_index] = {
                        **persisted_script_flow_rows[persisted_script_flow_index],
                        "member_audio_url": uploaded_audio_url,
                    }
                    persisted_config = _normalize_json_object(getattr(persisted_scenario, "call_simulation_config", None))
                    persisted_scenario.call_simulation_config = {
                        **persisted_config,
                        "script_flow": persisted_script_flow_rows,
                    }
                    db.add(persisted_scenario)
                db.commit()

            _log_call_simulation_audio_action(
                db,
                user=current_user,
                action_type="regenerate_speech" if normalized_replace_audio_url else "generate_speech",
                scenario_id=scenario_id,
                step_number=step_number,
                file_name=filename,
                metadata={
                    "asset_kind": normalized_asset_kind,
                    "audio_url": uploaded_audio_url,
                    "provider": provider,
                    "voice_type": "Puck",
                    "language": "en-US",
                    "duration": round(max(duration, 0.0), 2),
                    "bucket_name": getattr(audio_asset, "bucket_name", None),
                    "storage_path": getattr(audio_asset, "storage_path", None),
                },
            )
            db.commit()

            if (
                normalized_replace_audio_url
                and normalized_replace_audio_url != uploaded_audio_url
            ):
                if normalized_asset_kind in {"ringer", "hold"}:
                    stale_shared_assets = (
                        db.query(CallSimulationAudioAsset)
                        .filter(
                            CallSimulationAudioAsset.trainer_id == resolved_asset_owner_id,
                            CallSimulationAudioAsset.public_url == normalized_replace_audio_url,
                            CallSimulationAudioAsset.asset_kind == normalized_asset_kind,
                            CallSimulationAudioAsset.is_active == True,
                            CallSimulationAudioAsset.scenario_id.is_(None),
                        )
                        .all()
                    )
                    for stale_asset in stale_shared_assets:
                        _deactivate_call_simulation_audio_asset_record(
                            db,
                            asset=stale_asset,
                            delete_storage=True,
                        )
                    if stale_shared_assets:
                        db.commit()
                else:
                    stale_asset = _find_active_call_simulation_audio_asset(
                        db,
                        trainer_id=resolved_asset_owner_id,
                        public_url=normalized_replace_audio_url,
                        asset_kinds=[normalized_asset_kind],
                        scenario_id=scenario_id,
                        step_number=step_number,
                    )
                    if stale_asset:
                        _deactivate_call_simulation_audio_asset_record(
                            db,
                            asset=stale_asset,
                            delete_storage=True,
                        )
                        db.commit()
        elif require_supabase:
            logger.warning(
                "Call Simulation TTS asset upload failed and persisted audio requires Supabase. "
                "trainer_id=%s scenario_segment=%s asset_kind=%s",
                resolved_asset_owner_id,
                scenario_segment,
                asset_kind,
            )
            raise HTTPException(status_code=503, detail=strict_persistence_message)
        audio_base64 = None

    if not audio_url:
        if audio_base64:
            try:
                audio_url = _build_audio_data_url(b64decode(audio_base64), audio_content_type)
            except Exception:
                audio_url = None
        elif audio_bytes is not None:
            try:
                audio_url = _build_audio_data_url(audio_bytes, audio_content_type)
            except Exception:
                audio_url = None

    if not audio_url:
        error_detail = provider_error or "Generated speech could not be synthesized by the backend."
        if fallback_mode == "browser" or provider == "browser_fallback":
            error_detail = (
                f"{error_detail} Server-side audio persistence requires a backend TTS provider. "
                "Configure server-side TTS credentials or enable local TTS on Windows."
            )
        logger.warning(
            "Call Simulation backend TTS returned no playable audio. error=%s",
            error_detail,
        )
        raise HTTPException(status_code=503, detail=error_detail)

    return {
        "audio_url": audio_url,
        "audio_base64": audio_base64,
        "duration": round(max(duration, 0.0), 2),
        "provider": provider,
        "storage_mode": storage_mode,
        "warning": storage_warning,
        "fallback_mode": fallback_mode,
        "audio_asset": _serialize_call_simulation_audio_asset(audio_asset).model_dump() if audio_asset else None,
    }


@router.post("/start", response_model=SimSessionStartResponse)
async def start_simulation(
    session_data: SimSessionCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)

    scenario = (
        db.query(Scenario)
        .filter(
            Scenario.id == session_data.scenario_id,
            Scenario.is_published == True,
        )
        .first()
    )
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    did_sync_assignments = _sync_sim_floor_assignments_for_trainee(db, current_user)
    if did_sync_assignments:
        db.flush()

    assignment = _get_active_call_simulation_assignment(
        db,
        trainee_id=current_user.id,
        scenario_id=scenario.id,
    )
    if not assignment:
        raise HTTPException(status_code=403, detail="Scenario is not assigned to your trainee workspace")

    trainee_batch_ids = {batch.id for batch in current_user.batches if batch.id}
    if assignment.batch_id and assignment.batch_id not in trainee_batch_ids:
        raise HTTPException(status_code=403, detail="Assigned scenario batch is no longer active for this trainee")

    latest_session_query = (
        db.query(SimSession)
        .filter(
            SimSession.trainee_id == current_user.id,
            SimSession.scenario_id == scenario.id,
        )
    )
    if assignment.batch_id:
        latest_session_query = latest_session_query.filter(
            SimSession.batch_id == assignment.batch_id
        )
    latest_session = latest_session_query.order_by(SimSession.created_at.desc()).first()
    batch_id = assignment.batch_id
    if not batch_id and session_data.batch_id and session_data.batch_id in trainee_batch_ids:
        batch_id = session_data.batch_id
    if not batch_id and latest_session and latest_session.batch_id in trainee_batch_ids:
        batch_id = latest_session.batch_id
    if not batch_id and len(trainee_batch_ids) == 1:
        batch_id = next(iter(trainee_batch_ids))
    if not batch_id and len(trainee_batch_ids) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "This Call Simulation assignment is missing its batch context. "
                "Ask the trainer to reassign it from a batch or relaunch it with a batch selection."
            ),
        )

    scenario_steps = _build_scenario_steps(scenario)
    _validate_call_simulation_launch_assets(scenario, scenario_steps)
    kpi_config = db.query(BatchKPIConfig).filter(BatchKPIConfig.batch_id == batch_id).first() if batch_id else None
    next_attempt_number = 1
    max_attempts = _resolve_call_assignment_max_attempts(
        assignment,
        scenario=scenario,
        fallback=DEFAULT_MAX_ATTEMPTS,
    )

    if latest_session:
        normalized_verdict = (latest_session.trainer_verdict_status or "pending").strip().lower()
        latest_coaching_log = _get_latest_sim_session_coaching_logs(db, [latest_session.id]).get(latest_session.id)
        latest_session_changed = False
        if int(latest_session.max_attempts or 0) != max_attempts:
            latest_session.max_attempts = max_attempts
            latest_session_changed = True
        if latest_session.assignment_id != assignment.id:
            latest_session.assignment_id = assignment.id
            latest_session_changed = True
        if latest_session.assigned_by_id != assignment.assigned_by:
            latest_session.assigned_by_id = assignment.assigned_by
            latest_session_changed = True

        if latest_session.status == "in_progress":
            if latest_session_changed:
                db.commit()
                db.refresh(latest_session)
            variation = (
                db.query(ScenarioVariation)
                .filter(ScenarioVariation.id == latest_session.scenario_variation_id)
                .first()
                if latest_session.scenario_variation_id
                else None
            )
            return _build_session_start_response(
                db,
                session=latest_session,
                scenario=scenario,
                batch_id=latest_session.batch_id or batch_id,
                variation=variation,
                kpi_config=kpi_config,
            )

        if (
            normalized_verdict == "competent"
            or latest_session.certificate_id
            or (latest_session.pass_fail and normalized_verdict != "retake")
        ):
            raise HTTPException(
                status_code=409,
                detail="This Call Simulation is already completed. Open your certificates or reports instead of launching a new attempt.",
            )

        if latest_coaching_log and latest_coaching_log.status == "sent":
            raise HTTPException(
                status_code=409,
                detail="Acknowledge the trainer coaching notes before retaking this Call Simulation.",
            )

        if int(latest_session.attempt_number or 1) >= max_attempts:
            raise HTTPException(status_code=409, detail="Maximum retake attempts reached for this Call Simulation.")

        next_attempt_number = int(latest_session.attempt_number or 1) + 1

    variation = (
        db.query(ScenarioVariation)
        .filter(
            ScenarioVariation.scenario_id == scenario.id,
            ScenarioVariation.is_active == True,
        )
        .order_by(func.random())
        .first()
    )
    initial_csr_step_number = _resolve_initial_csr_step_number(scenario_steps)

    new_session = SimSession(
        id=str(uuid.uuid4()),
        trainee_id=current_user.id,
        scenario_id=scenario.id,
        assignment_id=assignment.id,
        assigned_by_id=assignment.assigned_by,
        scenario_variation_id=variation.id if variation else None,
        batch_id=batch_id,
        status="in_progress",
        current_step=initial_csr_step_number,
        started_at=datetime.utcnow(),
        aht_target=(kpi_config.target_aht_seconds if kpi_config else 120),
        attempt_number=next_attempt_number,
        max_attempts=max_attempts,
        pass_fail=False,
        transcript_log=[],
        turn_logs=[],
        trainer_verdict_status="pending",
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return _build_session_start_response(
        db,
        session=new_session,
        scenario=scenario,
        batch_id=batch_id,
        variation=variation,
        kpi_config=kpi_config,
    )


@router.get("/session/{session_id}", response_model=SimSessionResponse)
async def get_session(
    session_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    session = _get_accessible_session(db, current_user, session_id)
    return _serialize_sim_session_response(db, session)


@router.post("/session/{session_id}/events", response_model=SuccessResponse)
async def log_session_event(
    session_id: str,
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
    authorization: Optional[str] = Header(None),
    user_agent: Optional[str] = Header(None),
    x_forwarded_for: Optional[str] = Header(None),
    x_real_ip: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    session = _get_accessible_session(db, current_user, session_id)

    event_type = str(payload.get("event_type") or payload.get("type") or "").strip()
    if not event_type:
        raise HTTPException(status_code=400, detail="event_type is required")
    if len(event_type) > 120:
        raise HTTPException(status_code=400, detail="event_type is too long")

    event_label = str(payload.get("event_label") or payload.get("label") or "").strip() or None
    step_number_value = payload.get("step_number")
    try:
        step_number = int(step_number_value) if step_number_value not in (None, "") else None
    except (TypeError, ValueError):
        step_number = None

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    client_ip = (x_forwarded_for or x_real_ip or "").split(",", 1)[0].strip() or None
    role_label = str(current_user.role.value if hasattr(current_user.role, "value") else current_user.role)
    db.add(
        CallSimulationEvent(
            session_id=session.id,
            scenario_id=session.scenario_id,
            user_id=current_user.id,
            role=role_label,
            event_type=event_type,
            event_label=event_label,
            step_number=step_number,
            metadata_json=metadata,
            ip_address=client_ip,
            browser=(user_agent or "").strip()[:255] or None,
            device=str(metadata.get("device") or metadata.get("platform") or "").strip()[:255] or None,
        )
    )
    create_audit_log(
        db,
        user=current_user,
        request=request,
        action_type=event_type,
        module_name="Call Simulation",
        entity_type="sim_session",
        entity_id=session.id,
        description=event_label or f"Call Simulation event: {event_type}",
        new_data={
            "event_type": event_type,
            "event_label": event_label,
            "step_number": step_number,
            "scenario_id": session.scenario_id,
        },
        batch_id=session.batch_id,
        trainee_id=session.trainee_id,
        trainer_id=session.assigned_by_id,
        session_id=session.id,
        endpoint=f"/api/call-simulation/session/{session.id}/events",
        http_method="POST",
        http_status=200,
        metadata={
            **metadata,
            "call_event": True,
            "role": role_label,
        },
    )
    db.commit()
    return SuccessResponse(message="Call Simulation event logged.")


@router.post("/session/{session_id}/turn", response_model=SimSessionTurnResponse)
async def submit_session_turn(
    session_id: str,
    step_number: int = Form(...),
    file: UploadFile = File(...),
    audio_duration_seconds: float = Form(...),
    live_transcript: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail="Session not in progress")

    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    steps = _build_scenario_steps(scenario)
    step = next((item for item in steps if item.step_number == step_number), None)
    if not step:
        raise HTTPException(status_code=404, detail="Scenario step not found")
    if step.actor != "csr":
        raise HTTPException(status_code=400, detail="Only CSR turns can be recorded")

    expected_step_number = _resolve_next_pending_csr_step_number(session, steps)
    if expected_step_number is None:
        raise HTTPException(
            status_code=409,
            detail="All CSR turns are already complete. End the call to generate the final evaluation.",
        )
    if step_number != expected_step_number:
        raise HTTPException(
            status_code=409,
            detail=f"Step {expected_step_number} is the next allowed CSR turn for this Call Simulation.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    original_name = file.filename or f"turn-{step_number}.webm"
    _, extension = os.path.splitext(original_name)
    safe_extension = extension if extension else ".webm"
    storage_leaf = f"step-{step_number}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}{safe_extension}"

    supabase = get_supabase_client()
    audio_url = supabase.upload_sim_floor_audio(
        file_data=file_bytes,
        trainee_id=current_user.id,
        scenario_id=scenario.id,
        session_id=session.id,
        filename=storage_leaf,
        content_type=file.content_type or "audio/webm",
    )
    if not audio_url:
        raise HTTPException(
            status_code=503,
            detail="Supabase storage could not save the submitted CSR turn recording.",
        )

    class _StepScenario:
        title = scenario.title
        expected_keywords = list(step.expected_keywords or [])
        flow_steps: list[Any] = []
        opening_prompt = step.script

    assessment = assess_audio_submission(
        audio_bytes=file_bytes,
        filename=original_name,
        mime_type=file.content_type or "audio/webm",
        scenario=_StepScenario(),
        reference_text=step.script,
        fallback_transcript=(live_transcript or None),
        response_duration=float(audio_duration_seconds),
        user_dialect=current_user.language_dialect,
    )
    asr_provider = str(assessment.get("provider") or "").strip() or None
    transcript_confidence = float(assessment.get("transcription_confidence") or 0.0)

    transcript = (
        assessment.get("transcription")
        or assessment.get("text")
        or live_transcript
        or ""
    ).strip()
    kpi_config = _get_effective_kpi_config(db, session.batch_id)
    duration_seconds = max(float(audio_duration_seconds or 0.0), 0.0)
    evaluation = _evaluate_submission(
        transcript=transcript,
        audio_duration_seconds=int(round(duration_seconds)),
        kpi_config=kpi_config,
        assessment=assessment if assessment.get("status") == "completed" else None,
        overrides=None if assessment.get("status") == "completed" else {
            "speech_to_text_accuracy": 0.0,
            "rate_of_speech": 0.0,
            "dead_air_seconds": duration_seconds,
            "grammar_score": 0.0,
            "pronunciation_score": 0.0,
            "pacing_score": 0.0,
        },
    )
    ai_feedback = _build_ai_feedback(
        evaluation=evaluation,
        kpi_config=kpi_config,
        assessment=assessment if assessment.get("status") == "completed" else None,
        fallback_message=assessment.get("error") if assessment.get("status") != "completed" else None,
    )
    matched_keywords = assessment.get("matched_keywords") or _find_keyword_matches(transcript, step.expected_keywords)
    requires_repeat, repeat_reason, script_similarity, selected_variant = _determine_repeat_requirement(
        transcript=transcript,
        step=step,
        evaluation=evaluation,
        assessment=assessment,
        matched_keywords=list(matched_keywords),
    )
    matched_script = str(selected_variant.get("script") or step.script or "").strip()
    point_value = round(max(0.0, float(selected_variant.get("score") or _read_step_point_value(step))), 2)
    scenario_label = _read_step_scenario_label(step)
    actor_label = _read_step_actor_label(step)

    turn_logs = _session_turn_logs(session)
    transcript_log = _session_transcript_log(session)
    step_attempt_number = _count_turn_attempts_for_step(turn_logs, step.step_number) + 1

    turn_log = {
        "turn_attempt_id": str(uuid.uuid4()),
        "turn_attempt_number": step_attempt_number,
        "step_number": step.step_number,
        "actor": "csr",
        "speaker_label": step.speaker_label or "CSR",
        "expected_script": matched_script,
        "expected_keywords": list(step.expected_keywords or []),
        "transcript": transcript,
        "audio_url": audio_url,
        "duration_seconds": round(duration_seconds, 2),
        "asr_provider": asr_provider,
        "asr_provider_label": _format_asr_provider_label(asr_provider),
        "transcript_confidence": transcript_confidence,
        "matched_keywords": list(matched_keywords),
        "speech_to_text_accuracy": float(evaluation.get("speech_to_text_accuracy") or 0.0),
        "grammar_score": float(evaluation.get("grammar_score") or 0.0),
        "pronunciation_score": float(evaluation.get("pronunciation_score") or 0.0),
        "pacing_score": float(evaluation.get("pacing_score") or 0.0),
        "rate_of_speech": float(evaluation.get("rate_of_speech") or 0.0),
        "dead_air_seconds": float(evaluation.get("dead_air_seconds") or 0.0),
        "forbidden_matches": list(evaluation.get("forbidden_matches") or []),
        "ai_feedback": ai_feedback,
        "accepted_for_progress": not requires_repeat,
        "requires_repeat": requires_repeat,
        "repeat_prompt": TURN_REPEAT_PROMPT if requires_repeat else None,
        "repeat_reason": repeat_reason,
        "script_similarity": round(script_similarity * 100.0, 2),
        "point_value": point_value,
        "scenario_label": scenario_label,
        "member_actor_label": actor_label,
        "created_at": datetime.utcnow().isoformat(),
    }
    turn_log.update(_calculate_turn_content_score(turn_log))
    turn_logs.append(turn_log)
    transcript_log.append(
        {
            "turn_attempt_id": turn_log["turn_attempt_id"],
            "turn_attempt_number": step_attempt_number,
            "step_number": step.step_number,
            "actor": "csr",
            "speaker_label": step.speaker_label or "CSR",
            "text": transcript,
            "transcript": transcript,
            "audio_url": audio_url,
            "expected_script": matched_script,
            "asr_provider": asr_provider,
            "transcript_confidence": transcript_confidence,
            "matched_keywords": list(matched_keywords),
            "duration_seconds": round(duration_seconds, 2),
            "accepted_for_progress": not requires_repeat,
            "requires_repeat": requires_repeat,
            "repeat_prompt": TURN_REPEAT_PROMPT if requires_repeat else None,
            "repeat_reason": repeat_reason,
            "script_similarity": round(script_similarity * 100.0, 2),
            "point_value": point_value,
            "scenario_label": scenario_label,
            "member_actor_label": actor_label,
            "earned_points": turn_log.get("earned_points", 0.0),
            "content_score_percent": turn_log.get("content_score_percent", 0.0),
            "created_at": datetime.utcnow().isoformat(),
        }
    )
    turn_logs.sort(
        key=lambda item: (
            int(item.get("step_number") or 0),
            int(item.get("turn_attempt_number") or 0),
        )
    )
    transcript_log.sort(
        key=lambda item: (
            int(item.get("step_number") or 0),
            int(item.get("turn_attempt_number") or 0),
        )
    )

    db.add(
        SessionResponseRecord(
            id=str(uuid.uuid4()),
            session_id=session.id,
            script_id=step.id,
            scenario_id=scenario.id,
            trainee_id=current_user.id,
            step_number=step.step_number,
            turn_attempt_number=step_attempt_number,
            actor_type="CSR",
            scenario_group=scenario_label,
            expected_script=matched_script,
            trainee_spoken_text=transcript,
            matched_score=float(turn_log.get("earned_points") or 0.0),
            grammar_score=float(turn_log.get("grammar_score") or 0.0),
            pronunciation_score=float(turn_log.get("pronunciation_score") or 0.0),
            pacing_score=float(turn_log.get("pacing_score") or 0.0),
            speech_to_text_accuracy=float(turn_log.get("speech_to_text_accuracy") or 0.0),
            transcript_confidence=transcript_confidence,
            audio_url=audio_url,
            ai_feedback=ai_feedback,
        )
    )

    progress_next_step = next((item.step_number for item in steps if item.step_number > step.step_number), None)
    next_step = step.step_number if requires_repeat else progress_next_step
    session.turn_logs = turn_logs
    session.transcript_log = transcript_log
    session.current_step = next_step or step.step_number
    session.audio_url = audio_url
    session.audio_duration_seconds = int(round(sum(float(item.get("duration_seconds") or 0.0) for item in turn_logs)))
    db.commit()
    db.refresh(session)

    return SimSessionTurnResponse(
        session_id=session.id,
        step_number=step.step_number,
        transcript=transcript,
        audio_url=audio_url,
        duration_seconds=round(duration_seconds, 2),
        asr_provider=asr_provider,
        asr_provider_label=_format_asr_provider_label(asr_provider),
        transcript_confidence=transcript_confidence,
        matched_keywords=list(turn_log["matched_keywords"]),
        speech_to_text_accuracy=float(turn_log["speech_to_text_accuracy"]),
        grammar_score=float(turn_log["grammar_score"]),
        pronunciation_score=float(turn_log["pronunciation_score"]),
        pacing_score=float(turn_log["pacing_score"]),
        rate_of_speech=float(turn_log["rate_of_speech"]),
        dead_air_seconds=float(turn_log["dead_air_seconds"]),
        ai_feedback=ai_feedback,
        requires_repeat=requires_repeat,
        repeat_prompt=TURN_REPEAT_PROMPT if requires_repeat else None,
        repeat_reason=repeat_reason,
        script_similarity=round(script_similarity * 100.0, 2),
        next_step=next_step,
        is_complete=(not requires_repeat and progress_next_step is None),
        transcript_log=transcript_log,
        turn_logs=turn_logs,
    )


@router.post("/session/{session_id}/member-speech")
async def request_session_member_speech(
    session_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail="Session is not in progress")

    script = str(payload.get("script") or "").strip()
    if not script:
        raise HTTPException(status_code=400, detail="Member script is required")

    try:
        step_number = int(payload.get("step_number"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Valid step_number is required")

    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Session scenario not found")

    steps = _build_scenario_steps(scenario)
    step = next((item for item in steps if item.step_number == step_number), None)
    if not step or _is_csr_actor(step.actor):
        raise HTTPException(status_code=400, detail="Requested step must be a Member AI step")

    saved_audio_url = _resolve_call_simulation_step_audio_url(step)
    if not _is_stored_call_simulation_audio_public_url(saved_audio_url):
        _log_call_simulation_audio_action(
            db,
            user=current_user,
            action_type="playback_member_speech",
            scenario_id=session.scenario_id,
            step_number=step_number,
            status_value="failed",
            error_detail="Saved Member AI audio is missing or is not stored in Supabase.",
            metadata={"session_id": session_id},
        )
        db.commit()
        raise HTTPException(
            status_code=404,
            detail=(
                "Saved Member AI speech is missing for this step. "
                "Ask the trainer to generate or upload the audio before launching the simulation."
            ),
        )

    audio_asset = _find_active_call_simulation_audio_asset(
        db,
        trainer_id=str(session.assigned_by_id or scenario.created_by or ""),
        public_url=saved_audio_url,
        asset_kinds=["member-step"],
        scenario_id=session.scenario_id,
        step_number=step_number,
    ) if (session.assigned_by_id or scenario.created_by) else None

    _log_call_simulation_audio_action(
        db,
        user=current_user,
        action_type="playback_member_speech",
        scenario_id=session.scenario_id,
        step_number=step_number,
        metadata={
            "session_id": session_id,
            "audio_url": saved_audio_url,
            "bucket_name": getattr(audio_asset, "bucket_name", None),
            "storage_path": getattr(audio_asset, "storage_path", None),
        },
    )
    db.commit()
    return {
        "audio_url": saved_audio_url,
        "duration": None,
        "provider": "stored_supabase",
        "storage_mode": "supabase",
        "audio_asset": _serialize_call_simulation_audio_asset(audio_asset).model_dump() if audio_asset else None,
    }


@router.post("/session/{session_id}/feedback")
async def generate_session_feedback(
    session_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    transcript = str(payload.get("transcript") or session.transcript or "").strip()
    script_flow = payload.get("scriptFlow")
    if not isinstance(script_flow, list):
        script_flow = []
    target_kpis = payload.get("targetKpis") or payload.get("target_kpis") or {}
    if not isinstance(target_kpis, dict):
        target_kpis = {}

    evaluation_payload = await generate_evaluation_feedback(
        {
            "transcript": transcript,
            "script_flow": script_flow,
            "target_kpis": target_kpis,
        }
    )

    report = evaluation_payload or {}
    script_accuracy = report.get("scriptAccuracy") if isinstance(report.get("scriptAccuracy"), dict) else {}
    grammar_and_pronunciation = report.get("grammarAndPronunciation") if isinstance(report.get("grammarAndPronunciation"), dict) else {}
    soft_skills = report.get("softSkills") if isinstance(report.get("softSkills"), dict) else {}
    pacing_and_aht = report.get("pacingAndAht") if isinstance(report.get("pacingAndAht"), dict) else {}

    provider = str(report.get("provider") or "fallback").strip() or "fallback"
    model = str(report.get("model") or "gemini-2.5-flash" if provider != "fallback" else "fallback").strip() or "fallback"
    overall_summary = str(report.get("overallSummary") or report.get("summary") or "Session feedback is ready for trainer review.").strip()
    coaching_tips = [str(item).strip() for item in (report.get("coachingTips") or []) if str(item).strip()]
    strengths = [str(item).strip() for item in (script_accuracy.get("strengths") or []) if str(item).strip()]
    areas_for_improvement = [str(item).strip() for item in (script_accuracy.get("misses") or []) if str(item).strip()]
    if not areas_for_improvement and coaching_tips:
        areas_for_improvement = coaching_tips[:3]

    total_score = float(payload.get("totalScore") or report.get("totalScore") or session.weighted_score or 0.0)
    passing_score = float(
        payload.get("passingScore")
        or target_kpis.get("passing_score")
        or target_kpis.get("passingScore")
        or report.get("passingScore")
        or 80.0
    )
    passed = bool(payload.get("passed") if payload.get("passed") is not None else report.get("passed") or total_score >= passing_score)

    kpi_breakdown: list[dict[str, Any]] = []
    if script_accuracy:
        kpi_breakdown.append(
            {
                "category": "Script Accuracy",
                "score": round(float(script_accuracy.get("score") or 0.0), 2),
                "feedback": str(script_accuracy.get("strengths") or ["Follow the suggested script more closely."])[0],
            }
        )
    if grammar_and_pronunciation:
        kpi_breakdown.append(
            {
                "category": "Grammar & Pronunciation",
                "score": round(float(grammar_and_pronunciation.get("score") or 0.0), 2),
                "feedback": str(grammar_and_pronunciation.get("notes") or ["Continue practicing clear, polished phrasing."])[0],
            }
        )
    if soft_skills:
        kpi_breakdown.append(
            {
                "category": "Soft Skills",
                "score": round(float(soft_skills.get("score") or 0.0), 2),
                "feedback": str(soft_skills.get("notes") or ["Maintain a confident, empathetic tone throughout the call."])[0],
            }
        )
    if pacing_and_aht:
        kpi_breakdown.append(
            {
                "category": "Pacing & AHT",
                "score": round(float((pacing_and_aht.get("score") if isinstance(pacing_and_aht.get("score"), (int, float)) else 0.0) or 0.0), 2),
                "feedback": str((pacing_and_aht.get("notes") or ["Keep pacing steady and concise."])[0]),
            }
        )

    if not kpi_breakdown:
        kpi_breakdown = [
            {
                "category": "Overall",
                "score": round(total_score, 2),
                "feedback": overall_summary,
            }
        ]

    report_payload = {
        "provider": provider,
        "model": model,
        "overallSummary": overall_summary,
        "summary": overall_summary,
        "totalScore": round(total_score, 2),
        "passingScore": round(passing_score, 2),
        "passed": passed,
        "strengths": strengths or ["Completed the call simulation and can be reviewed for coaching."],
        "areas_for_improvement": areas_for_improvement or ["Practice the core script and pacing to improve consistency."],
        "coaching_recommendation": " ".join(coaching_tips).strip() or overall_summary,
        "transcript_summary": transcript[:400] if transcript else "Transcript is ready for trainer review.",
        "kpi_breakdown": kpi_breakdown,
        "scriptAccuracy": {
            "score": round(float(script_accuracy.get("score") or 0.0), 2),
            "strengths": strengths,
            "misses": [str(item).strip() for item in (script_accuracy.get("misses") or []) if str(item).strip()],
        },
        "grammarAndPronunciation": {
            "score": round(float(grammar_and_pronunciation.get("score") or 0.0), 2),
            "notes": [str(item).strip() for item in (grammar_and_pronunciation.get("notes") or []) if str(item).strip()],
        },
        "softSkills": {
            "score": round(float(soft_skills.get("score") or 0.0), 2),
            "notes": [str(item).strip() for item in (soft_skills.get("notes") or []) if str(item).strip()],
        },
        "coachingTips": coaching_tips,
        "coachingInsights": _generate_coaching_insights(session),
    }
    return {"report": report_payload}


@router.post("/session/{session_id}/recording")
async def upload_session_recording(
    session_id: str,
    file: UploadFile = File(...),
    audio_duration_seconds: Optional[float] = Form(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded session recording is empty")

    original_name = file.filename or "session-recording.wav"
    _, extension = os.path.splitext(original_name)
    normalized_content_type = str(file.content_type or "").split(";", 1)[0].strip().lower()
    if normalized_content_type not in {"audio/mpeg", "audio/mp3"} and extension.lower() != ".mp3":
        raise HTTPException(
            status_code=400,
            detail="The final full-call recording must be uploaded as MP3.",
        )

    date_segment = datetime.utcnow().strftime("%Y%m%d")
    storage_leaf = f"{date_segment}/mockcall.mp3"
    resolved_content_type = "audio/mpeg"

    supabase = get_supabase_client()
    audio_url = supabase.upload_sim_floor_audio(
        file_data=file_bytes,
        trainee_id=current_user.id,
        scenario_id=scenario.id,
        session_id=session.id,
        filename=storage_leaf,
        content_type=resolved_content_type,
        save_local_backup=False,
    )
    if not audio_url:
        raise HTTPException(
            status_code=503,
            detail="Supabase storage could not save the session recording.",
        )

    session.audio_url = audio_url
    if audio_duration_seconds is not None:
        session.audio_duration_seconds = int(round(max(audio_duration_seconds, 0.0)))
    db.commit()

    return {
        "audio_url": audio_url,
        "audio_duration_seconds": session.audio_duration_seconds,
    }


@router.post("/session/{session_id}/finalize", response_model=SimSessionResponse)
async def finalize_session(
    session_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail="Session not in progress")

    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    kpi_config = _get_effective_kpi_config(db, session.batch_id)
    scenario_steps = _build_scenario_steps(scenario)
    csr_transcript, total_duration, evaluation, ai_feedback = _aggregate_turn_based_evaluation(
        session=session,
        kpi_config=kpi_config,
        scenario_steps=scenario_steps,
    )
    keyword_compliance = _build_keyword_compliance_summary(csr_transcript, scenario_steps)
    sentiment_score = _analyze_sentiment_score(csr_transcript)
    evaluation["keyword_compliance"] = keyword_compliance
    evaluation["sentiment_score"] = sentiment_score
    _apply_call_scenario_passing_score(
        evaluation,
        scenario=scenario,
        kpi_config=kpi_config,
    )
    ai_feedback = _build_ai_feedback(
        evaluation=evaluation,
        kpi_config=kpi_config,
    )

    full_transcript_log: list[dict[str, Any]] = []
    turn_lookup = {
        int(item.get("step_number") or 0): item
        for item in _selected_csr_turns_for_scoring(session)
    }
    existing_transcript_lookup = {
        int(item.get("step_number") or 0): item
        for item in _session_transcript_log(session)
    }
    timeline_cursor = 0.0
    for step in scenario_steps:
        existing_entry = existing_transcript_lookup.get(step.step_number, {})
        if step.actor == "csr":
            turn = turn_lookup.get(step.step_number, {})
            duration_seconds = round(
                float(turn.get("duration_seconds") or _estimate_script_duration_seconds(turn.get("transcript") or step.script)),
                2,
            )
            full_transcript_log.append(
                {
                    "step_number": step.step_number,
                    "actor": "csr",
                    "speaker_label": step.speaker_label or "CSR",
                    "text": turn.get("transcript") or "",
                    "transcript": turn.get("transcript") or "",
                    "audio_url": turn.get("audio_url"),
                    "expected_script": step.script,
                    "matched_keywords": list(turn.get("matched_keywords") or []),
                    "duration_seconds": duration_seconds,
                    "timeline_start_seconds": round(timeline_cursor, 2),
                    "timeline_end_seconds": round(timeline_cursor + duration_seconds, 2),
                    "coach_note": existing_entry.get("coach_note"),
                }
            )
        else:
            duration_seconds = round(_estimate_script_duration_seconds(step.script), 2)
            full_transcript_log.append(
                {
                    "step_number": step.step_number,
                    "actor": step.actor,
                    "speaker_label": step.speaker_label or "Member",
                    "text": step.script,
                    "transcript": step.script,
                    "audio_url": step.audio_url,
                    "duration_seconds": duration_seconds,
                    "timeline_start_seconds": round(timeline_cursor, 2),
                    "timeline_end_seconds": round(timeline_cursor + duration_seconds, 2),
                    "coach_note": existing_entry.get("coach_note"),
                }
            )
        timeline_cursor += duration_seconds
    session.transcript_log = full_transcript_log
    
    # Phase 2: Extract and populate member speech log for enhanced audio reconstruction
    member_speech_log = _extract_member_speech_from_events(db, session_id)
    if member_speech_log:
        session.member_speech_log = member_speech_log
    
    full_conversation_transcript = "\n".join(
        f"{str(item.get('speaker_label') or item.get('actor') or 'Speaker').strip()}: {str(item.get('transcript') or item.get('text') or '').strip()}"
        for item in full_transcript_log
        if str(item.get("transcript") or item.get("text") or "").strip()
    ).strip()

    _apply_evaluation_to_session(
        session=session,
        transcript=full_conversation_transcript or csr_transcript,
        audio_url=session.audio_url or next((item.get("audio_url") for item in _session_turn_logs(session) if item.get("audio_url")), None),
        audio_duration_seconds=total_duration,
        evaluation=evaluation,
        ai_feedback=ai_feedback,
    )
    _sync_call_simulation_certificate(
        db,
        session=session,
        scenario=scenario,
    )
    notify_call_simulation_completion(
        db,
        trainee=current_user,
        session=session,
        scenario=scenario,
    )
    db.commit()
    db.refresh(session)
    await _broadcast_call_simulation_completion(
        trainee=current_user,
        session=session,
        scenario=scenario,
    )
    return SimSessionResponse.model_validate(session)


@router.post("/session/{session_id}/submit", response_model=SimSessionResponse)
async def submit_session_audio(
    session_id: str,
    file: UploadFile = File(...),
    audio_duration_seconds: float = Form(...),
    live_transcript: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail="Session not in progress")

    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    variation = (
        db.query(ScenarioVariation)
        .filter(ScenarioVariation.id == session.scenario_variation_id)
        .first()
        if session.scenario_variation_id
        else None
    )
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    original_name = file.filename or "call-simulation-recording.webm"
    _, extension = os.path.splitext(original_name)
    safe_extension = extension if extension else ".webm"
    storage_leaf = f"session_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}{safe_extension}"

    supabase = get_supabase_client()
    audio_url = supabase.upload_sim_floor_audio(
        file_data=file_bytes,
        trainee_id=current_user.id,
        scenario_id=scenario.id,
        session_id=session.id,
        filename=storage_leaf,
        content_type=file.content_type or "audio/webm",
    )
    if not audio_url:
        raise HTTPException(
            status_code=503,
            detail="Supabase storage could not save the submitted call simulation recording.",
        )

    logger.info(
        "Audio saved for session %s via Supabase storage",
        session.id,
    )

    reference_text = variation.script if variation and variation.script else None
    assessment = assess_audio_submission(
        audio_bytes=file_bytes,
        filename=original_name,
        mime_type=file.content_type or "audio/webm",
        scenario=scenario,
        reference_text=reference_text,
        fallback_transcript=(live_transcript or None),
        response_duration=float(audio_duration_seconds),
        user_dialect=current_user.language_dialect,
    )

    transcript = (
        assessment.get("transcription")
        or assessment.get("text")
        or live_transcript
        or ""
    ).strip()
    kpi_config = _get_effective_kpi_config(db, session.batch_id)
    duration_seconds = int(round(audio_duration_seconds))
    fallback_message = None

    if assessment.get("status") != "completed":
        fallback_message = assessment.get("error") or "No recognizable speech was detected."
        evaluation = _evaluate_submission(
            transcript=transcript,
            audio_duration_seconds=duration_seconds,
            kpi_config=kpi_config,
            overrides={
                "speech_to_text_accuracy": 0.0,
                "rate_of_speech": 0.0,
                "dead_air_seconds": float(duration_seconds),
                "grammar_score": 0.0,
                "pronunciation_score": 0.0,
                "pacing_score": 0.0,
                "transcript_confidence": float(assessment.get("transcription_confidence") or 0.0),
            },
        )
        evaluation["weighted_score"] = 0.0
        evaluation["pass_fail"] = False
    else:
        evaluation = _evaluate_submission(
            transcript=transcript,
            audio_duration_seconds=duration_seconds,
            kpi_config=kpi_config,
            assessment=assessment,
        )

    evaluation["keyword_compliance"] = _build_keyword_compliance_summary(
        transcript,
        _build_scenario_steps(scenario),
    )
    evaluation["sentiment_score"] = _analyze_sentiment_score(transcript)
    _apply_call_scenario_passing_score(
        evaluation,
        scenario=scenario,
        kpi_config=kpi_config,
    )
    ai_feedback = _build_ai_feedback(
        evaluation=evaluation,
        kpi_config=kpi_config,
        assessment=assessment if assessment.get("status") == "completed" else None,
        fallback_message=fallback_message if assessment.get("status") != "completed" else None,
    )

    _apply_evaluation_to_session(
        session=session,
        transcript=transcript,
        audio_url=audio_url,
        audio_duration_seconds=duration_seconds,
        evaluation=evaluation,
        ai_feedback=ai_feedback,
    )
    _sync_call_simulation_certificate(
        db,
        session=session,
        scenario=scenario,
    )
    notify_call_simulation_completion(
        db,
        trainee=current_user,
        session=session,
        scenario=scenario,
    )

    db.commit()
    db.refresh(session)
    await _broadcast_call_simulation_completion(
        trainee=current_user,
        session=session,
        scenario=scenario,
    )
    return SimSessionResponse.model_validate(session)


@router.post("/session/{session_id}/complete", response_model=SimSessionResponse)
async def complete_session(
    session_id: str,
    payload: SimSessionCompleteRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail="Session not in progress")

    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    kpi_config = _get_effective_kpi_config(db, session.batch_id)
    evaluation = _evaluate_submission(
        transcript=payload.transcript,
        audio_duration_seconds=payload.audio_duration_seconds,
        kpi_config=kpi_config,
        overrides={
            "speech_to_text_accuracy": payload.speech_to_text_accuracy,
            "rate_of_speech": payload.rate_of_speech,
            "dead_air_seconds": payload.dead_air_seconds,
            "grammar_score": payload.grammar_score,
            "pronunciation_score": payload.pronunciation_score,
            "pacing_score": payload.pacing_score,
            "detected_forbidden_words": payload.detected_forbidden_words,
        },
    )
    if scenario:
        evaluation["keyword_compliance"] = _build_keyword_compliance_summary(
            payload.transcript,
            _build_scenario_steps(scenario),
        )
    evaluation["sentiment_score"] = _analyze_sentiment_score(payload.transcript)
    _apply_call_scenario_passing_score(
        evaluation,
        scenario=scenario,
        kpi_config=kpi_config,
    )
    ai_feedback = payload.ai_feedback or _build_ai_feedback(
        evaluation=evaluation,
        kpi_config=kpi_config,
    )

    _apply_evaluation_to_session(
        session=session,
        transcript=payload.transcript,
        audio_url=payload.audio_url,
        audio_duration_seconds=payload.audio_duration_seconds,
        evaluation=evaluation,
        ai_feedback=ai_feedback,
    )
    _sync_call_simulation_certificate(
        db,
        session=session,
        scenario=scenario,
    )
    notify_call_simulation_completion(
        db,
        trainee=current_user,
        session=session,
        scenario=scenario,
    )
    session.transcript_log = payload.transcript_log or session.transcript_log
    session.turn_logs = payload.turn_logs or session.turn_logs
    db.commit()
    db.refresh(session)
    await _broadcast_call_simulation_completion(
        trainee=current_user,
        session=session,
        scenario=scenario,
    )
    return SimSessionResponse.model_validate(session)


@router.post("/session/{session_id}/retake", response_model=SimSessionResponse)
async def retake_session(
    session_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)
    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    assignment = _get_active_call_simulation_assignment(
        db,
        trainee_id=current_user.id,
        scenario_id=session.scenario_id,
    )
    effective_max_attempts = _resolve_call_assignment_max_attempts(
        assignment,
        scenario=scenario,
        fallback=int(session.max_attempts or DEFAULT_MAX_ATTEMPTS),
    )

    if session.pass_fail:
        raise HTTPException(status_code=400, detail="Passed sessions do not require retakes")
    if session.attempt_number >= effective_max_attempts:
        raise HTTPException(status_code=400, detail="Maximum attempts reached")

    coaching_log = _get_latest_sim_session_coaching_logs(db, [session.id]).get(session.id)
    if coaching_log and coaching_log.status == "sent":
        raise HTTPException(
            status_code=400,
            detail="Acknowledge the trainer coaching notes before retaking this simulation.",
        )

    variation = (
        db.query(ScenarioVariation)
        .filter(
            ScenarioVariation.scenario_id == session.scenario_id,
            ScenarioVariation.is_active == True,
        )
        .order_by(func.random())
        .first()
    )
    scenario_steps = _build_scenario_steps(scenario) if scenario else []
    initial_csr_step_number = _resolve_initial_csr_step_number(scenario_steps)

    new_session = SimSession(
        id=str(uuid.uuid4()),
        trainee_id=current_user.id,
        scenario_id=session.scenario_id,
        assignment_id=assignment.id if assignment else session.assignment_id,
        assigned_by_id=assignment.assigned_by if assignment else session.assigned_by_id,
        scenario_variation_id=variation.id if variation else None,
        batch_id=session.batch_id,
        status="in_progress",
        current_step=initial_csr_step_number,
        started_at=datetime.utcnow(),
        aht_target=session.aht_target,
        attempt_number=session.attempt_number + 1,
        max_attempts=effective_max_attempts,
        pass_fail=False,
        transcript_log=[],
        turn_logs=[],
        trainer_verdict_status="pending",
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return SimSessionResponse.model_validate(new_session)


@router.post("/session/{session_id}/discard", response_model=SuccessResponse)
async def discard_session_attempt(
    session_id: str,
    authorization: Optional[str] = Header(None),
    request: Request = None,
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainee(current_user)
    session = _get_accessible_session(db, current_user, session_id)

    if session.status not in {"pending", "in_progress"}:
        raise HTTPException(
            status_code=400,
            detail="Only active Call Simulation attempts can be reset.",
        )

    # Log discard event for audit trail
    try:
        db.add(
            CallSimulationEvent(
                session_id=session.id,
                scenario_id=session.scenario_id,
                user_id=current_user.id,
                role="trainee",
                event_type="session_discarded",
                event_label="Trainee initiated retry",
                step_number=session.current_step,
                metadata_json={
                    "reason": "trainee_retry",
                    "status": session.status,
                    "attempt_number": session.attempt_number,
                },
            )
        )
    except Exception as exc:
        logging.warning(f"Failed to log session discard event: {exc}")

    response_records = (
        db.query(SessionResponseRecord)
        .filter(SessionResponseRecord.session_id == session.id)
        .all()
    )
    audio_urls = _collect_session_audio_urls(session, response_records)
    supabase = get_supabase_client()

    # Delete audio files from Supabase storage
    for audio_url in audio_urls:
        try:
            supabase.delete_by_public_url(audio_url)
        except Exception as exc:
            logger.warning(
                "Unable to delete discarded Call Simulation audio %s for session %s: %s",
                audio_url,
                session.id,
                exc,
            )

    # Delete response records
    for record in response_records:
        db.delete(record)

    # Delete session
    db.delete(session)
    db.commit()

    return SuccessResponse(
        message="The Call Simulation attempt was cleared. Start the call again to retry from step 1."
    )


@router.get("/session/{session_id}/can-retake")
async def check_retake_status(
    session_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    session = _get_accessible_session(db, current_user, session_id)
    assignment = _get_active_call_simulation_assignment(
        db,
        trainee_id=session.trainee_id,
        scenario_id=session.scenario_id,
    )
    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()
    effective_max_attempts = _resolve_call_assignment_max_attempts(
        assignment,
        scenario=scenario,
        fallback=int(session.max_attempts or DEFAULT_MAX_ATTEMPTS),
    )
    return {
        "can_retake": not session.pass_fail and session.attempt_number < effective_max_attempts,
        "attempt_number": session.attempt_number,
        "max_attempts": effective_max_attempts,
        "passed": session.pass_fail,
    }


# ==================== Coaching ====================


@router.get("/coaching/interactions")
async def get_interaction_history(
    batch_id: Optional[str] = Query(None),
    trainee_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    trainer_batch_ids = set(_get_trainer_batch_ids(db, current_user))
    query = db.query(SimSession).filter(SimSession.status.in_(["completed", "failed"]))

    if current_user.role != UserRole.ADMIN:
        if not trainer_batch_ids:
            return {"sessions": []}
        query = query.filter(SimSession.batch_id.in_(trainer_batch_ids))

    if batch_id:
        if current_user.role != UserRole.ADMIN and batch_id not in trainer_batch_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        query = query.filter(SimSession.batch_id == batch_id)

    if trainee_id:
        query = query.filter(SimSession.trainee_id == trainee_id)

    sessions = (
        query.order_by(SimSession.completed_at.desc(), SimSession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    scenario_map = {
        scenario.id: scenario
        for scenario in db.query(Scenario)
        .filter(Scenario.id.in_([session.scenario_id for session in sessions] or ["__none__"]))
        .all()
    }
    trainee_map = {
        user.id: user
        for user in db.query(User)
        .filter(User.id.in_([session.trainee_id for session in sessions] or ["__none__"]))
        .all()
    }
    batch_map = {
        batch.id: batch
        for batch in db.query(Batch)
        .filter(Batch.id.in_([session.batch_id for session in sessions if session.batch_id] or ["__none__"]))
        .all()
    }
    latest_coaching_logs = _get_latest_sim_session_coaching_logs(
        db,
        [session.id for session in sessions],
    )
    feedback_reports = _get_supabase_call_simulation_feedback_reports(
        [session.id for session in sessions]
    )

    payload = []
    for session in sessions:
        scenario = scenario_map.get(session.scenario_id)
        trainee = trainee_map.get(session.trainee_id)
        batch = batch_map.get(session.batch_id) if session.batch_id else None
        coaching_log = latest_coaching_logs.get(session.id)
        payload.append(
            {
                "id": session.id,
                "trainee_id": session.trainee_id,
                "trainee_name": trainee.full_name if trainee else "Unknown",
                "batch_id": session.batch_id,
                "batch_name": batch.name if batch else None,
                "batch_wave_number": batch.wave_number if batch else None,
                "scenario_title": scenario.title if scenario else "Unknown",
                "score": session.weighted_score or 0.0,
                "status": session.status or ("completed" if session.pass_fail else "failed"),
                "outcome_label": _session_outcome_label(session),
                "pass_fail": session.pass_fail,
                "attempt_number": session.attempt_number,
                "audio_url": session.audio_url,
                "audio_duration_seconds": session.audio_duration_seconds,
                "transcript": session.transcript,
                "transcript_log": session.transcript_log or [],
                "turn_logs": session.turn_logs or [],
                "ai_feedback": session.ai_feedback,
                "feedback_report": feedback_reports.get(session.id),
                "coaching_notes": session.coaching_notes,
                "grammar_score": session.grammar_score,
                "pronunciation_score": session.pronunciation_score,
                "pacing_score": session.pacing_score,
                "sentiment_score": session.sentiment_score,
                "keyword_compliance": session.keyword_compliance or {},
                "speech_to_text_accuracy": session.speech_to_text_accuracy,
                "rate_of_speech": session.rate_of_speech,
                "dead_air_seconds": session.dead_air_seconds,
                "forbidden_words_count": session.forbidden_words_count,
                "trainer_verdict_status": session.trainer_verdict_status or "pending",
                "trainer_verdict_notes": session.trainer_verdict_notes,
                "trainer_evaluated_at": session.trainer_evaluated_at.isoformat() if session.trainer_evaluated_at else None,
                "certificate_id": session.certificate_id,
                "coaching_id": coaching_log.coaching_id if coaching_log else None,
                "coaching_status": coaching_log.status if coaching_log else None,
                "coaching_acknowledged_at": coaching_log.acknowledged_at.isoformat() if coaching_log and coaching_log.acknowledged_at else None,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "created_at": session.created_at.isoformat() if session.created_at else None,
            }
        )

    return {"sessions": payload}


@router.put("/coaching/interactions/{session_id}/notes", response_model=SimSessionResponse)
async def update_coaching_notes(
    session_id: str,
    request: Request,
    note_update: SimSessionCoachingNoteUpdate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    session = _get_accessible_session(db, current_user, session_id)
    session.coaching_notes = note_update.notes.strip()
    _upsert_sim_session_coaching_log(
        db,
        session=session,
        trainer_id=current_user.id,
        notes=session.coaching_notes,
        verdict_status=session.trainer_verdict_status or "pending",
    )
    create_audit_log(
        db,
        user=current_user,
        request=request,
        action_type="trainer_started_coaching",
        module_name="Call Simulation",
        entity_type="sim_session",
        entity_id=session.id,
        description="Trainer saved Call Simulation coaching notes.",
        new_data={"notes_length": len(session.coaching_notes or "")},
        batch_id=session.batch_id,
        trainee_id=session.trainee_id,
        trainer_id=current_user.id,
        session_id=session.id,
        http_status=200,
    )
    db.commit()
    db.refresh(session)
    return _serialize_sim_session_response(db, session)


@router.put("/coaching/interactions/{session_id}/verdict", response_model=SimSessionResponse)
async def update_trainer_verdict(
    session_id: str,
    request: Request,
    verdict_update: SimSessionTrainerVerdictUpdate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    session = _get_accessible_session(db, current_user, session_id)
    scenario = db.query(Scenario).filter(Scenario.id == session.scenario_id).first()

    normalized_status = (verdict_update.verdict_status or "").strip().lower()
    if normalized_status in {"not_competent", "not-competent", "retake_required"}:
        normalized_status = "retake"
    if normalized_status not in {"pending", "competent", "retake"}:
        raise HTTPException(status_code=400, detail="Invalid verdict status")

    session.trainer_verdict_status = normalized_status
    session.trainer_verdict_notes = (verdict_update.notes or "").strip() or None
    if session.trainer_verdict_notes:
        session.coaching_notes = session.trainer_verdict_notes
    session.trainer_evaluated_by = current_user.id
    session.trainer_evaluated_at = datetime.utcnow()

    _sync_call_simulation_certificate(
        db,
        session=session,
        scenario=scenario,
        issuer_id=current_user.id,
    )

    _upsert_sim_session_coaching_log(
        db,
        session=session,
        trainer_id=current_user.id,
        notes=session.trainer_verdict_notes or session.coaching_notes,
        verdict_status=normalized_status,
    )
    create_audit_log(
        db,
        user=current_user,
        request=request,
        action_type="trainer_finished_coaching",
        module_name="Call Simulation",
        entity_type="sim_session",
        entity_id=session.id,
        description="Trainer submitted a Call Simulation coaching verdict.",
        new_data={
            "verdict_status": normalized_status,
            "has_notes": bool(session.trainer_verdict_notes or session.coaching_notes),
            "certificate_id": session.certificate_id,
        },
        batch_id=session.batch_id,
        trainee_id=session.trainee_id,
        trainer_id=current_user.id,
        session_id=session.id,
        http_status=200,
    )

    db.commit()
    db.refresh(session)
    return _serialize_sim_session_response(db, session)


# ==================== Analytics ====================


@router.get("/analytics/live")
async def get_live_analytics(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)

    trainer_batch_ids = _get_trainer_batch_ids(db, current_user)
    candidate_trainees: list[User] = []
    if current_user.role == UserRole.ADMIN:
        candidate_trainees = (
            db.query(User)
            .filter(User.role == UserRole.TRAINEE, User.is_active.is_(True))
            .all()
        )
    else:
        accessible_batches = (
            db.query(Batch)
            .filter(Batch.id.in_(trainer_batch_ids or ["__none__"]), Batch.is_active.is_(True))
            .all()
        )
        candidate_lookup: dict[str, User] = {}
        for batch in accessible_batches:
            for trainee in batch.users:
                if trainee.role != UserRole.TRAINEE or not trainee.is_active:
                    continue
                candidate_lookup[trainee.id] = trainee
        candidate_trainees = list(candidate_lookup.values())

    active_trainee_ids = {
        trainee.id
        for trainee in filter_to_supabase_active_users(db, candidate_trainees)
    }
    if not active_trainee_ids:
        return {
            "active_simulations": 0,
            "completed_today": 0,
            "pass_rate": 0.0,
            "total_passed": 0,
            "total_failed": 0,
            "pass_fail_by_batch": [],
            "top_failed_kpis": {},
            "coaching_summary": _summarize_sim_session_coaching({}),
        }

    active_query = db.query(SimSession).filter(SimSession.status == "in_progress")
    complete_query = db.query(SimSession).filter(SimSession.status.in_(["completed", "failed"]))
    active_query = active_query.filter(SimSession.trainee_id.in_(list(active_trainee_ids) or ["__none__"]))
    complete_query = complete_query.filter(SimSession.trainee_id.in_(list(active_trainee_ids) or ["__none__"]))
    if current_user.role != UserRole.ADMIN:
        if not trainer_batch_ids or not active_trainee_ids:
            return {
                "active_simulations": 0,
                "completed_today": 0,
                "pass_rate": 0.0,
                "total_passed": 0,
                "total_failed": 0,
                "pass_fail_by_batch": [],
                "top_failed_kpis": {},
                "coaching_summary": _summarize_sim_session_coaching({}),
            }
        active_query = active_query.filter(SimSession.batch_id.in_(trainer_batch_ids))
        complete_query = complete_query.filter(SimSession.batch_id.in_(trainer_batch_ids))

    active_count = active_query.count()

    today = datetime.utcnow().date()
    completed_today = (
        complete_query.filter(func.date(SimSession.created_at) == today).count()
    )

    completed_sessions = complete_query.all()
    latest_coaching_logs = _get_latest_sim_session_coaching_logs(
        db,
        [session.id for session in completed_sessions],
    )
    total_passed = sum(1 for session in completed_sessions if session.pass_fail)
    total_failed = sum(1 for session in completed_sessions if not session.pass_fail)
    total_sessions = total_passed + total_failed
    pass_rate = round((total_passed / total_sessions) * 100.0, 1) if total_sessions else 0.0

    batch_map = {
        batch.id: batch
        for batch in db.query(Batch)
        .filter(Batch.id.in_(list({session.batch_id for session in completed_sessions if session.batch_id}) or ["__none__"]))
        .all()
    }
    grouped_by_batch: dict[str, list[SimSession]] = defaultdict(list)
    for session in completed_sessions:
        if session.batch_id:
            grouped_by_batch[session.batch_id].append(session)

    pass_fail_by_batch = []
    for batch_id_value, sessions in grouped_by_batch.items():
        passed = sum(1 for session in sessions if session.pass_fail)
        failed = len(sessions) - passed
        batch = batch_map.get(batch_id_value)
        pass_fail_by_batch.append(
            {
                "batch_id": batch_id_value,
                "batch_name": batch.name if batch else "Unknown Batch",
                "passed": passed,
                "failed": failed,
                "pass_rate": round((passed / len(sessions)) * 100.0, 1) if sessions else 0.0,
            }
        )

    live_failed_kpis = _failed_kpi_counts(completed_sessions)

    return {
        "active_simulations": active_count,
        "completed_today": completed_today,
        "pass_rate": pass_rate,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "pass_fail_by_batch": sorted(
            pass_fail_by_batch,
            key=lambda item: item["batch_name"].lower(),
        ),
        "top_failed_kpis": live_failed_kpis,
        "coaching_summary": _summarize_sim_session_coaching(latest_coaching_logs),
    }


@router.get("/analytics/batch/{batch_id}")
async def get_batch_analytics(
    batch_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    batch = _get_accessible_batch(db, current_user, batch_id)

    sessions = (
        db.query(SimSession)
        .filter(
            SimSession.batch_id == batch_id,
            SimSession.status.in_(["completed", "failed"]),
        )
        .order_by(SimSession.created_at.asc())
        .all()
    )

    if not sessions:
        return {
            "batch_id": batch_id,
            "batch_name": batch.name,
            "total_sessions": 0,
            "avg_score": 0.0,
            "pass_rate": 0.0,
            "retakes": 0,
            "total_attempts": 0,
            "top_failed_kpis": {},
            "score_trends": [],
            "ai_feedback_trends": [],
            "attempts_by_trainee": [],
            "coaching_summary": _summarize_sim_session_coaching({}),
        }

    kpi_config = _get_effective_kpi_config(db, batch_id)
    latest_coaching_logs = _get_latest_sim_session_coaching_logs(
        db,
        [session.id for session in sessions],
    )
    total_sessions = len(sessions)
    total_attempts = sum(session.attempt_number or 1 for session in sessions)
    retakes = sum(max((session.attempt_number or 1) - 1, 0) for session in sessions)
    passed = sum(1 for session in sessions if session.pass_fail)
    avg_score = round(
        sum(float(session.weighted_score or 0.0) for session in sessions) / total_sessions,
        1,
    )
    pass_rate = round((passed / total_sessions) * 100.0, 1)

    score_trend_groups: dict[str, list[SimSession]] = defaultdict(list)
    for session in sessions:
        label = session.completed_at.strftime("%b %d") if session.completed_at else session.created_at.strftime("%b %d")
        score_trend_groups[label].append(session)

    score_trends = []
    ai_feedback_trends = []
    for label, grouped_sessions in score_trend_groups.items():
        score_trends.append(
            {
                "date": label,
                "avg_score": round(
                    sum(float(session.weighted_score or 0.0) for session in grouped_sessions)
                    / len(grouped_sessions),
                    1,
                ),
                "sessions": len(grouped_sessions),
            }
        )
        ai_feedback_trends.append(
            {
                "date": label,
                "grammar": round(
                    sum(float(session.grammar_score or 0.0) for session in grouped_sessions)
                    / len(grouped_sessions),
                    1,
                ),
                "pronunciation": round(
                    sum(float(session.pronunciation_score or 0.0) for session in grouped_sessions)
                    / len(grouped_sessions),
                    1,
                ),
                "pacing": round(
                    sum(float(session.pacing_score or 0.0) for session in grouped_sessions)
                    / len(grouped_sessions),
                    1,
                ),
            }
        )

    trainee_map = {
        user.id: user
        for user in db.query(User)
        .filter(User.id.in_([session.trainee_id for session in sessions] or ["__none__"]))
        .all()
    }
    attempts_by_trainee_groups: dict[str, list[SimSession]] = defaultdict(list)
    for session in sessions:
        attempts_by_trainee_groups[session.trainee_id].append(session)

    attempts_by_trainee = []
    for trainee_id, grouped_sessions in attempts_by_trainee_groups.items():
        trainee = trainee_map.get(trainee_id)
        latest_session = grouped_sessions[-1]
        attempts_by_trainee.append(
            {
                "trainee_id": trainee_id,
                "trainee_name": trainee.full_name if trainee else "Unknown",
                "total_sessions": len(grouped_sessions),
                "total_attempts": sum(session.attempt_number or 1 for session in grouped_sessions),
                "avg_score": round(
                    sum(float(session.weighted_score or 0.0) for session in grouped_sessions)
                    / len(grouped_sessions),
                    1,
                ),
                "latest_score": round(float(latest_session.weighted_score or 0.0), 1),
                "latest_pass_fail": bool(latest_session.pass_fail),
            }
        )

    return {
        "batch_id": batch_id,
        "batch_name": batch.name,
        "total_sessions": total_sessions,
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "retakes": retakes,
        "total_attempts": total_attempts,
        "top_failed_kpis": _failed_kpi_counts(sessions, batch_kpi_config=kpi_config),
        "score_trends": score_trends,
        "ai_feedback_trends": ai_feedback_trends,
        "coaching_summary": _summarize_sim_session_coaching(latest_coaching_logs),
        "attempts_by_trainee": sorted(
            attempts_by_trainee,
            key=lambda item: item["trainee_name"].lower(),
        ),
    }


# ==================== Reporting ====================


@router.get("/reports/batch/{batch_id}")
async def get_batch_report(
    batch_id: str,
    month: Optional[int] = Query(None, ge=1, le=12),
    year: Optional[int] = Query(None, ge=2020),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    _require_trainer(current_user)
    batch = _get_accessible_batch(db, current_user, batch_id)
    start_date, end_date, period_label = _resolve_report_period(month=month, year=year)

    trainees = [user for user in batch.users if user.role == UserRole.TRAINEE]
    trainee_ids = [user.id for user in trainees]
    did_prune_certificates = False
    for trainee_id in trainee_ids:
        sync_trainee_completion_certificates(db, trainee_id)
        did_prune_certificates = (
            prune_trainee_activity_certificates(db, trainee_id) > 0
            or did_prune_certificates
        )
    if did_prune_certificates:
        db.commit()

    session_query = db.query(SimSession).filter(
        SimSession.batch_id == batch_id,
        SimSession.status.in_(["completed", "failed"]),
    )
    if start_date and end_date:
        session_query = session_query.filter(
            SimSession.created_at >= start_date,
            SimSession.created_at <= end_date,
        )
    sessions = session_query.order_by(SimSession.created_at.desc()).all()

    mappings = (
        db.query(BatchScenarioMapping)
        .filter(
            BatchScenarioMapping.batch_id == batch_id,
            BatchScenarioMapping.is_active == True,
        )
        .order_by(BatchScenarioMapping.assigned_at.desc())
        .all()
    )
    scenario_ids = [mapping.scenario_id for mapping in mappings]
    active_assignments = (
        db.query(CallSimulationAssignment)
        .filter(
            CallSimulationAssignment.batch_id == batch_id,
            CallSimulationAssignment.scenario_id.in_(scenario_ids or ["__none__"]),
            CallSimulationAssignment.is_active == True,
        )
        .all()
    )
    assigned_counts_by_scenario: dict[str, int] = defaultdict(int)
    assigned_pairs: set[tuple[str, str]] = set()
    for assignment in active_assignments:
        pair = (assignment.scenario_id, assignment.trainee_id)
        if pair in assigned_pairs:
            continue
        assigned_pairs.add(pair)
        assigned_counts_by_scenario[assignment.scenario_id] += 1

    scenarios = {
        scenario.id: scenario
        for scenario in db.query(Scenario).filter(Scenario.id.in_(scenario_ids or ["__none__"])).all()
    }
    variations_by_scenario: dict[str, list[ScenarioVariation]] = defaultdict(list)
    for variation in (
        db.query(ScenarioVariation)
        .filter(
            ScenarioVariation.scenario_id.in_(scenario_ids or ["__none__"]),
            ScenarioVariation.is_active == True,
        )
        .order_by(ScenarioVariation.created_at.asc())
        .all()
    ):
        variations_by_scenario[variation.scenario_id].append(variation)

    sessions_by_scenario: dict[str, list[SimSession]] = defaultdict(list)
    sessions_by_trainee: dict[str, list[SimSession]] = defaultdict(list)
    for session in sessions:
        sessions_by_scenario[session.scenario_id].append(session)
        sessions_by_trainee[session.trainee_id].append(session)

    scenario_performance = []
    for mapping in mappings:
        scenario = scenarios.get(mapping.scenario_id)
        if not scenario:
            continue
        scenario_sessions = sessions_by_scenario.get(scenario.id, [])
        completed_sessions = len(scenario_sessions)
        passed_sessions = sum(1 for session in scenario_sessions if session.pass_fail)
        unique_takers = len({session.trainee_id for session in scenario_sessions})
        variations = variations_by_scenario.get(scenario.id, [])
        scenario_performance.append(
            {
                "scenario_id": scenario.id,
                "title": scenario.title,
                "members_assigned": assigned_counts_by_scenario.get(scenario.id, 0),
                "unique_takers": unique_takers,
                "csr_response_count": len(variations),
                "variation_scores": [
                    {
                        "actor_name": variation.actor_name,
                        "score": variation.score,
                    }
                    for variation in variations
                ],
                "completed_sessions": completed_sessions,
                "passed_sessions": passed_sessions,
                "average_score": round(
                    sum(float(session.weighted_score or 0.0) for session in scenario_sessions) / completed_sessions,
                    1,
                )
                if completed_sessions
                else 0.0,
                "pass_rate": round((passed_sessions / completed_sessions) * 100.0, 1)
                if completed_sessions
                else 0.0,
                "latest_attempt_at": max(
                    (
                        session.completed_at or session.created_at
                        for session in scenario_sessions
                        if session.completed_at or session.created_at
                    ),
                    default=None,
                ),
            }
        )

    trainee_performance = []
    for trainee in trainees:
        trainee_sessions = sessions_by_trainee.get(trainee.id, [])
        completed_sessions = len(trainee_sessions)
        passed_sessions = sum(1 for session in trainee_sessions if session.pass_fail)
        trainee_performance.append(
            {
                "trainee_id": trainee.id,
                "trainee_name": trainee.full_name,
                "total_sessions": completed_sessions,
                "retakes": sum(max((session.attempt_number or 1) - 1, 0) for session in trainee_sessions),
                "average_score": round(
                    sum(float(session.weighted_score or 0.0) for session in trainee_sessions) / completed_sessions,
                    1,
                )
                if completed_sessions
                else 0.0,
                "pass_rate": round((passed_sessions / completed_sessions) * 100.0, 1)
                if completed_sessions
                else 0.0,
                "latest_attempt_at": max(
                    (
                        session.completed_at or session.created_at
                        for session in trainee_sessions
                        if session.completed_at or session.created_at
                    ),
                    default=None,
                ),
            }
        )

    total_sessions = len(sessions)
    passed_sessions = sum(1 for session in sessions if session.pass_fail)
    report_kpi_config = _get_effective_kpi_config(db, batch_id)
    latest_coaching_logs = _get_latest_sim_session_coaching_logs(
        db,
        [session.id for session in sessions],
    )

    return {
        "batch_id": batch.id,
        "batch_name": batch.name,
        "wave_number": batch.wave_number,
        "period": period_label,
        "summary": {
            "total_trainees": len(trainees),
            "active_scenarios": len(scenario_performance),
            "total_sessions": total_sessions,
            "average_score": round(
                sum(float(session.weighted_score or 0.0) for session in sessions) / total_sessions,
                1,
            )
            if total_sessions
            else 0.0,
            "pass_rate": round((passed_sessions / total_sessions) * 100.0, 1)
            if total_sessions
            else 0.0,
            "retakes": sum(max((session.attempt_number or 1) - 1, 0) for session in sessions),
            "passing_score": float(report_kpi_config.passing_score or DEFAULT_PASSING_SCORE),
        },
        "kpi_scores": {
            "speech_to_text_accuracy": _average(
                [float(session.speech_to_text_accuracy) for session in sessions if session.speech_to_text_accuracy is not None]
            ),
            "grammar": _average(
                [float(session.grammar_score) for session in sessions if session.grammar_score is not None]
            ),
            "pronunciation": _average(
                [float(session.pronunciation_score) for session in sessions if session.pronunciation_score is not None]
            ),
            "pacing": _average(
                [float(session.pacing_score) for session in sessions if session.pacing_score is not None]
            ),
            "rate_of_speech": _average(
                [float(session.rate_of_speech) for session in sessions if session.rate_of_speech is not None]
            ),
            "dead_air": _average(
                [float(session.dead_air_seconds) for session in sessions if session.dead_air_seconds is not None]
            ),
        },
        "top_failed_kpis": _failed_kpi_counts(sessions, batch_kpi_config=report_kpi_config),
        "coaching_summary": _summarize_sim_session_coaching(latest_coaching_logs),
        "scenario_performance": sorted(
            scenario_performance,
            key=lambda item: (item["completed_sessions"], item["title"].lower()),
            reverse=True,
        ),
        "trainee_performance": sorted(
            trainee_performance,
            key=lambda item: (item["average_score"], item["trainee_name"].lower()),
            reverse=True,
        ),
    }


@router.get("/reports/trainee/{trainee_id}")
async def get_trainee_report(
    trainee_id: str,
    month: Optional[int] = Query(None, ge=1, le=12),
    year: Optional[int] = Query(None, ge=2020),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    current_user = await auth_utils.get_current_user(authorization, db)
    if current_user.role == UserRole.TRAINEE and current_user.id != trainee_id:
        raise HTTPException(status_code=403, detail="Access denied")

    trainee = db.query(User).filter(User.id == trainee_id, User.role == UserRole.TRAINEE).first()
    if not trainee:
        raise HTTPException(status_code=404, detail="Trainee not found")

    if current_user.role == UserRole.TRAINER:
        trainer_batch_ids = set(_get_trainer_batch_ids(db, current_user))
        trainee_batch_ids = {batch.id for batch in trainee.batches}
        if not trainer_batch_ids.intersection(trainee_batch_ids):
            raise HTTPException(status_code=403, detail="Access denied")

    did_sync_assignments = _sync_call_simulation_assignments_for_trainee(db, trainee)
    if did_sync_assignments:
        db.commit()

    sync_trainee_completion_certificates(db, trainee_id)
    if prune_trainee_activity_certificates(db, trainee_id):
        db.commit()

    active_scenario_ids = [
        scenario_id
        for (scenario_id,) in (
            db.query(CallSimulationAssignment.scenario_id)
            .filter(
                CallSimulationAssignment.trainee_id == trainee_id,
                CallSimulationAssignment.is_active == True,
            )
            .all()
        )
    ]

    start_date, end_date, period_label = _resolve_report_period(month=month, year=year)
    query = db.query(SimSession).filter(
        SimSession.trainee_id == trainee_id,
        SimSession.status.in_(["completed", "failed"]),
        SimSession.scenario_id.in_(active_scenario_ids or ["__none__"]),
    )
    if start_date and end_date:
        query = query.filter(
            SimSession.created_at >= start_date,
            SimSession.created_at <= end_date,
        )
    sessions = query.order_by(SimSession.created_at.desc()).all()

    scenario_ids = [session.scenario_id for session in sessions if session.scenario_id]
    scenario_titles = {
        scenario.id: scenario.title
        for scenario in db.query(Scenario).filter(Scenario.id.in_(scenario_ids or ["__none__"])).all()
    }

    sessions_by_scenario: dict[str, list[SimSession]] = defaultdict(list)
    for session in sessions:
        sessions_by_scenario[session.scenario_id].append(session)

    scenario_performance = []
    for scenario_id, scenario_sessions in sessions_by_scenario.items():
        completed_sessions = len(scenario_sessions)
        passed_sessions = sum(1 for session in scenario_sessions if session.pass_fail)
        scenario_performance.append(
            {
                "scenario_id": scenario_id,
                "title": scenario_titles.get(scenario_id, "Unknown Scenario"),
                "attempts": completed_sessions,
                "average_score": round(
                    sum(float(session.weighted_score or 0.0) for session in scenario_sessions) / completed_sessions,
                    1,
                )
                if completed_sessions
                else 0.0,
                "best_score": round(
                    max(float(session.weighted_score or 0.0) for session in scenario_sessions),
                    1,
                )
                if completed_sessions
                else 0.0,
                "pass_rate": round((passed_sessions / completed_sessions) * 100.0, 1)
                if completed_sessions
                else 0.0,
                "latest_attempt_at": max(
                    (
                        session.completed_at or session.created_at
                        for session in scenario_sessions
                        if session.completed_at or session.created_at
                    ),
                    default=None,
                ),
            }
        )

    total_sessions = len(sessions)
    passed_sessions = sum(1 for session in sessions if session.pass_fail)
    batch_id = sessions[0].batch_id if sessions else next((batch.id for batch in trainee.batches), None)
    report_kpi_config = _get_effective_kpi_config(db, batch_id)
    latest_coaching_logs = _get_latest_sim_session_coaching_logs(
        db,
        [session.id for session in sessions],
    )
    visible_session_ids = [session.id for session in sessions]
    call_simulation_certificates = (
        db.query(CertificateRecord)
        .filter(
            CertificateRecord.trainee_id == trainee_id,
            CertificateRecord.source_type == "call_simulation_session",
            CertificateRecord.source_id.in_(visible_session_ids or ["__none__"]),
        )
        .order_by(CertificateRecord.issued_at.desc())
        .all()
    )

    return {
        "trainee_id": trainee.id,
        "trainee_name": trainee.full_name,
        "period": period_label,
        "summary": {
            "total_sessions": total_sessions,
            "average_score": round(
                sum(float(session.weighted_score or 0.0) for session in sessions) / total_sessions,
                1,
            )
            if total_sessions
            else 0.0,
            "pass_rate": round((passed_sessions / total_sessions) * 100.0, 1)
            if total_sessions
            else 0.0,
            "retakes": sum(max((session.attempt_number or 1) - 1, 0) for session in sessions),
            "latest_score": round(float(sessions[0].weighted_score or 0.0), 1) if sessions else 0.0,
            "passing_score": float(report_kpi_config.passing_score or DEFAULT_PASSING_SCORE),
            "assigned_batches": [
                {
                    "batch_id": batch.id,
                    "batch_name": batch.name,
                    "wave_number": batch.wave_number,
                }
                for batch in trainee.batches
                if current_user.role != UserRole.TRAINER or batch.created_by == current_user.id
            ],
        },
        "kpi_scores": {
            "speech_to_text_accuracy": _average(
                [float(session.speech_to_text_accuracy) for session in sessions if session.speech_to_text_accuracy is not None]
            ),
            "grammar": _average(
                [float(session.grammar_score) for session in sessions if session.grammar_score is not None]
            ),
            "pronunciation": _average(
                [float(session.pronunciation_score) for session in sessions if session.pronunciation_score is not None]
            ),
            "pacing": _average(
                [float(session.pacing_score) for session in sessions if session.pacing_score is not None]
            ),
            "rate_of_speech": _average(
                [float(session.rate_of_speech) for session in sessions if session.rate_of_speech is not None]
            ),
            "dead_air": _average(
                [float(session.dead_air_seconds) for session in sessions if session.dead_air_seconds is not None]
            ),
        },
        "top_failed_kpis": _failed_kpi_counts(sessions, batch_kpi_config=report_kpi_config),
        "coaching_summary": _summarize_sim_session_coaching(latest_coaching_logs),
        "scenario_performance": sorted(
            scenario_performance,
            key=lambda item: (item["average_score"], item["title"].lower()),
            reverse=True,
        ),
        "recent_sessions": [
            {
                "session_id": session.id,
                "scenario_title": scenario_titles.get(session.scenario_id, "Unknown Scenario"),
                "score": float(session.weighted_score or 0.0),
                "status": "Passed" if session.pass_fail else "Retake Required",
                "attempt_number": session.attempt_number,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "audio_url": session.audio_url,
                "ai_feedback": session.ai_feedback,
                "trainer_verdict_status": session.trainer_verdict_status or "pending",
                "certificate_id": session.certificate_id,
                "coaching_id": latest_coaching_logs.get(session.id).coaching_id if latest_coaching_logs.get(session.id) else None,
                "coaching_status": latest_coaching_logs.get(session.id).status if latest_coaching_logs.get(session.id) else None,
                "coaching_acknowledged_at": latest_coaching_logs.get(session.id).acknowledged_at.isoformat() if latest_coaching_logs.get(session.id) and latest_coaching_logs.get(session.id).acknowledged_at else None,
            }
            for session in sessions[:10]
        ],
        "coaching_logs": [
            {
                "id": log.id,
                "coaching_id": log.coaching_id,
                "sim_session_id": log.sim_session_id,
                "status": log.status,
                "competency_status": normalize_competency_status(log.competency_status),
                "trainer_remarks": log.trainer_remarks,
                "acknowledged_at": log.acknowledged_at.isoformat() if log.acknowledged_at else None,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in latest_coaching_logs.values()
        ],
        "certificates": [
            {
                "certificate_id": certificate.id,
                "certificate_no": certificate.certificate_no,
                "scenario_session_id": certificate.source_id,
                "issued_at": certificate.issued_at.isoformat() if certificate.issued_at else None,
            }
            for certificate in call_simulation_certificates
        ],
    }
