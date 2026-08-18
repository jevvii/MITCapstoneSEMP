import asyncio
import json
import logging
import os
import sys
from time import perf_counter
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def _local_tts_enabled() -> bool:
    normalized = str(os.getenv("ENABLE_LOCAL_TTS", "")).strip().lower()
    if not normalized or normalized not in {"1", "true", "yes", "on"}:
        return False

    is_render = str(os.getenv("RENDER", "")).strip().lower() in {"1", "true", "yes", "on"}
    is_docker = Path("/.dockerenv").exists()
    if is_render or is_docker or os.name != "nt":
        logger.info(
            "Ignoring ENABLE_LOCAL_TTS because offline local TTS is only supported in local Windows development."
        )
        return False

    return True


def _load_pyttsx3():
    try:
        import pyttsx3
    except ImportError:
        return None

    return pyttsx3


def initialize_tts():
    if not _local_tts_enabled():
        logger.info("Local server-side TTS fallback is disabled.")
        return None
    pyttsx3 = _load_pyttsx3()
    if pyttsx3 is None:
        logger.warning("pyttsx3 is not installed. Local TTS generation is disabled.")
        return None

    try:
        engine = pyttsx3.init()
        return engine
    except Exception as e:
        logger.warning("Local TTS fallback is unavailable: %s", e)
        return None

# Create the engine instance to be used throughout your app
tts_engine = initialize_tts()

def speak_text(text):
    """Call this function whenever the BPO system needs to talk"""
    if tts_engine is None:
        logger.warning("Ignoring local TTS request because pyttsx3 is unavailable.")
        return
    tts_engine.say(text)
    tts_engine.runAndWait()

# --- Example Usage ---
if __name__ == "__main__":
    speak_text("Welcome to the Speech Enabled BPO Platform.")


# Defer importing heavy Azure Speech SDK until runtime to reduce startup cost.
speechsdk = None
PronunciationAssessmentConfig = None
PronunciationAssessmentGradingSystem = None
PronunciationAssessmentGranularity = None
AZURE_AVAILABLE = False

def ensure_azure_speech():
    """Attempt to import Azure Speech SDK at first use and set availability flags."""
    global speechsdk, PronunciationAssessmentConfig, PronunciationAssessmentGradingSystem, PronunciationAssessmentGranularity, AZURE_AVAILABLE
    if speechsdk is not None or AZURE_AVAILABLE:
        return
    try:
        import importlib

        mod = importlib.import_module('azure.cognitiveservices.speech')
        speechsdk = mod
        types_mod = importlib.import_module('azure.cognitiveservices.speech')
        PronunciationAssessmentConfig = getattr(types_mod, 'PronunciationAssessmentConfig', None)
        PronunciationAssessmentGradingSystem = getattr(types_mod, 'PronunciationAssessmentGradingSystem', None)
        PronunciationAssessmentGranularity = getattr(types_mod, 'PronunciationAssessmentGranularity', None)
        AZURE_AVAILABLE = True
    except Exception as e:
        speechsdk = None
        PronunciationAssessmentConfig = None
        PronunciationAssessmentGradingSystem = None
        PronunciationAssessmentGranularity = None
        AZURE_AVAILABLE = False
        logger.warning(f"Azure Speech SDK not available at runtime: {e}. Pronunciation features disabled.")

# Defer importing Google GenAI (Gemini) SDK until runtime.
genai = None
GEMINI_AVAILABLE = False

def ensure_genai():
    """Attempt to import google.genai at first use and set availability flag."""
    global genai, GEMINI_AVAILABLE
    if genai is not None or GEMINI_AVAILABLE:
        return
    try:
        import importlib

        genai = importlib.import_module('google.genai')
        GEMINI_AVAILABLE = True
    except Exception as e:
        genai = None
        GEMINI_AVAILABLE = False
        logger.warning(f"Google GenAI SDK not available at runtime: {e}. Gemini features disabled.")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

try:
    from .env_loader import load_backend_environment
except ImportError:
    from env_loader import load_backend_environment

try:
    from .config_validation import (
        extract_supabase_project_ref_from_key,
        extract_supabase_project_ref_from_url,
        is_usable_azure_speech_key,
        is_usable_supabase_publishable_key,
        is_usable_supabase_service_key,
        is_usable_supabase_url,
        normalize_env_value,
        resolve_gemini_api_key,
        resolve_supabase_publishable_key,
        resolve_supabase_service_key,
        resolve_supabase_url,
        supabase_key_matches_url,
    )
except ImportError:
    from config_validation import (
        extract_supabase_project_ref_from_key,
        extract_supabase_project_ref_from_url,
        is_usable_azure_speech_key,
        is_usable_supabase_publishable_key,
        is_usable_supabase_service_key,
        is_usable_supabase_url,
        normalize_env_value,
        resolve_gemini_api_key,
        resolve_supabase_publishable_key,
        resolve_supabase_service_key,
        resolve_supabase_url,
        supabase_key_matches_url,
    )

# Load environment variables using the shared backend resolution order.
load_backend_environment()
MEDIA_ROOT = Path(__file__).resolve().parent.parent / "media"
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)


def validate_environment():
    """Validate all required environment variables exist and are properly configured"""
    required = {
        'SECRET_KEY': 'JWT signing key (must be >= 32 characters, not default)',
        'BACKEND_URL': 'Backend public URL for CORS',
        'DATABASE_URL': 'PostgreSQL/Supabase database URL',
    }

    missing = []
    for var, description in required.items():
        if not normalize_env_value(os.getenv(var)):
            missing.append(f"{var}: {description}")

    supabase_url = resolve_supabase_url(os.getenv)
    if not is_usable_supabase_url(supabase_url):
        missing.append("SUPABASE_URL: Supabase project URL")
    if missing:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing)
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Validate SECRET_KEY strength
    secret_key = normalize_env_value(os.getenv('SECRET_KEY'))
    if secret_key == 'your-secret-key-change-in-production':
        raise RuntimeError("SECRET_KEY must be changed from default in production!")
    if len(secret_key) < 32:
        raise RuntimeError("SECRET_KEY must be at least 32 characters for security")

    # Validate URLs
    from urllib.parse import urlparse

    backend_url = normalize_env_value(os.getenv('BACKEND_URL'))
    try:
        urlparse(backend_url)
    except Exception as e:
        raise RuntimeError(f"Invalid BACKEND_URL format: {e}")

    try:
        urlparse(supabase_url)
    except Exception as e:
        raise RuntimeError(f"Invalid SUPABASE_URL format: {e}")

    publishable_key = resolve_supabase_publishable_key(os.getenv)
    if not is_usable_supabase_publishable_key(publishable_key):
        logger.warning(
            "SUPABASE_PUBLISHABLE_KEY is missing or invalid. The backend will still start, "
            "but Supabase Auth REST session issuance and realtime client tokens will be unavailable until a valid "
            "publishable/anon key is configured."
        )
    elif not supabase_key_matches_url(supabase_url, publishable_key):
        url_ref = extract_supabase_project_ref_from_url(supabase_url) or "unknown"
        key_ref = extract_supabase_project_ref_from_key(publishable_key) or "unknown"
        logger.warning(
            "SUPABASE_PUBLISHABLE_KEY does not match SUPABASE_URL. "
            "The backend will still start, but Supabase Auth REST session issuance will fail until both values target "
            "the same project. URL project=%s key project=%s",
            url_ref,
            key_ref,
        )

    service_key = resolve_supabase_service_key(os.getenv)
    if not service_key:
        logger.error(
            'Supabase service role key is not configured. Server-side Supabase admin features will stay disabled until SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY is set.'
        )
    elif not is_usable_supabase_service_key(service_key):
        logger.error(
            'Supabase service role key is configured but invalid. Server-side Supabase admin features will stay disabled until SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY contains a valid service-role JWT or sb_secret key.'
        )
    elif not supabase_key_matches_url(supabase_url, service_key):
        url_ref = extract_supabase_project_ref_from_url(supabase_url) or "unknown"
        key_ref = extract_supabase_project_ref_from_key(service_key) or "unknown"
        logger.error(
            "Supabase service role key does not match SUPABASE_URL. "
            "Server-side Supabase admin features will stay disabled until both values target the same project. "
            "URL project=%s key project=%s",
            url_ref,
            key_ref,
        )
    else:
        logger.info("Supabase admin storage configuration detected.")

    logger.info("Environment validation passed")


# NOTE: Validate environment during application startup (lifespan) instead of
# at import time. Running validation at import time may raise and cause the
# process to exit before the server binds to the port (Render port-scan
# failures). We'll perform validation inside the lifespan below and log any
# validation errors so the service can still bind and surface errors in logs.

# Configure Gemini-related availability messaging.
GEMINI_API_KEY = resolve_gemini_api_key(os.getenv)
if GEMINI_API_KEY:
    logger.info("Gemini API key detected for Gemini-enabled features")
elif GEMINI_AVAILABLE:
    logger.warning("Gemini API key not found; Gemini-enabled features disabled")
else:
    logger.warning("Google GenAI SDK not installed; Gemini-enabled features disabled")

# Support running from `backend/` as `uvicorn main:app`.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import route modules
from backend.routes import (
    auth_routes,
    user_routes,
    scenario_routes,
    assessment_routes,
    assessment_management_routes,
    assessment_redesign_routes,
    microlearning_routes,
    analytics_routes,
    admin_routes,
    trainer_routes,
    trainee_routes,
    settings_routes,
    workspace_routes,
    export_routes,
    certification_routes,
    notification_routes,
    call_simulation_routes,
    call_simulation_recordings,
    audit_routes,
)
try:
    from backend.routes import reading_assessment_routes
except Exception as exc:
    reading_assessment_routes = None
    logger.exception("Reading assessment routes are disabled because they failed to import: %s", exc)
from backend.database import Base, engine, SessionLocal
from backend.services.audit import should_audit_request, write_request_audit_log
from backend.services.sample_data_cleanup import cleanup_legacy_sample_dataset
from backend.services.speech_pipeline import (
    SpeechPipelineController,
    SpeechPipelineError,
)
from backend.supabase_client import get_supabase_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate environment early during startup so the process has already
    # bound its port when validation errors occur (avoids Render port-scan timeouts).
    try:
        validate_environment()
        app.state.validation_error = None
    except Exception as exc:
        msg = _summarize_startup_exception(exc)
        logger.error(
            "Environment validation failed during startup: %s. Continuing startup so the process can bind its port.",
            msg,
        )
        app.state.validation_error = msg
        # If strict mode is requested, re-raise to make startup fail fast.
        strict_mode = str(os.getenv("STRICT_ENV_VALIDATION", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if strict_mode:
            logger.error("STRICT_ENV_VALIDATION is enabled; aborting startup due to validation error.")
            raise

    if STARTUP_DATABASE_REACHABLE:
        ensure_admin_user()
        # Ensure the `public.profiles` table exists in case Supabase schema is not present
        # This allows the startup user-sync to insert rows without failing when the
        # hosting Postgres instance lacks Supabase-specific tables.
        try:
            ensure_profiles_table_exists()
        except Exception:
            logger.exception("Failed to ensure public.profiles table exists; continuing startup")

        sync_runtime_users_to_supabase_auth(fail_fast=False)
    else:
        logger.warning(
            "Skipping startup admin/bootstrap synchronization because the primary database is unreachable."
        )
    yield
    engine.dispose()
    logger.info("Database connection pool cleaned up")

app = FastAPI(
    title="Speech-Enabled BPO Platform",
    description="Comprehensive BPO training platform with speech assessment",
    version="2.0.0",
    lifespan=lifespan,
)

# Expose a simple health endpoint that reports validation and DB reachability.
app.state.validation_error = None


def _collect_validation_diagnostics() -> dict:
    """Gather lightweight environment diagnostics for the /health endpoint."""
    diagnostics = {}

    secret = normalize_env_value(os.getenv("SECRET_KEY"))
    diagnostics["SECRET_KEY"] = {
        "present": bool(secret),
        "length": len(secret) if secret else 0,
        "uses_default": secret == "your-secret-key-change-in-production",
        "ok": bool(secret) and len(secret) >= 32 and secret != "your-secret-key-change-in-production",
    }

    backend_url = normalize_env_value(os.getenv("BACKEND_URL"))
    try:
        parsed = urlparse(backend_url) if backend_url else None
        backend_url_valid = bool(parsed and parsed.scheme and parsed.netloc)
    except Exception:
        backend_url_valid = False
    diagnostics["BACKEND_URL"] = {"present": bool(backend_url), "valid": backend_url_valid}

    supabase_url = resolve_supabase_url(os.getenv)
    publishable = resolve_supabase_publishable_key(os.getenv)
    service = resolve_supabase_service_key(os.getenv)

    diagnostics["SUPABASE_URL"] = {"present": bool(supabase_url), "usable": is_usable_supabase_url(supabase_url)}
    diagnostics["SUPABASE_PUBLISHABLE_KEY"] = {
        "present": bool(publishable),
        "usable": is_usable_supabase_publishable_key(publishable),
        "matches_url": bool(supabase_url and publishable and supabase_key_matches_url(supabase_url, publishable)),
    }
    diagnostics["SUPABASE_SERVICE_KEY"] = {
        "present": bool(service),
        "usable": is_usable_supabase_service_key(service) if service else False,
        "matches_url": bool(supabase_url and service and supabase_key_matches_url(supabase_url, service)),
    }

    return diagnostics


@app.get("/health")
async def health():
    """Basic health endpoint used by platforms (non-fatal).

    Returns `status: ok` when environment validation passed and the
    runtime database probe succeeds, otherwise `degraded` with details.
    Includes diagnostics about keys and URLs to help debug validation failures.
    """
    validation_error = getattr(app.state, "validation_error", None)
    diagnostics = _collect_validation_diagnostics()
    strict_mode = str(os.getenv("STRICT_ENV_VALIDATION", "0")).strip() in {"1", "true", "yes", "on"}

    database_error = None
    database_reachable = False
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_reachable = True
    except Exception as exc:
        database_error = _summarize_startup_exception(exc)

    is_healthy = validation_error is None and database_reachable
    status_code = 200 if (is_healthy or not strict_mode) else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if is_healthy else "degraded",
            "strict_mode": strict_mode,
            "validation_error": validation_error,
            "database_reachable": database_reachable,
            "database_error": database_error,
            "diagnostics": diagnostics,
        },
    )

@app.exception_handler(OperationalError)
async def handle_database_operational_error(request: Request, exc: OperationalError):
    logger.warning(
        "Database connection error while serving %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": "The primary database is temporarily unavailable. Please try again in a moment.",
        },
    )


@app.middleware("http")
async def audit_request_middleware(request: Request, call_next):
    if not should_audit_request(request):
        return await call_next(request)

    content_type = (request.headers.get("content-type") or "").lower()
    content_length = 0
    try:
        content_length = int(request.headers.get("content-length") or 0)
    except ValueError:
        content_length = 0

    request_body: bytes | None = None
    if "multipart/form-data" not in content_type and content_length <= 10000:
        request_body = await request.body()

    started_at = perf_counter()
    http_status = 500
    try:
        response = await call_next(request)
        http_status = response.status_code
        return response
    finally:
        duration_ms = (perf_counter() - started_at) * 1000
        write_request_audit_log(
            request=request,
            http_status=http_status,
            duration_ms=duration_ms,
            request_body=request_body,
        )


def _summarize_startup_exception(exc: Exception) -> str:
    for line in str(exc).splitlines():
        normalized = line.strip()
        if normalized:
            return normalized
    return exc.__class__.__name__


def probe_startup_database_connectivity() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Primary database connection verified for startup maintenance.")
        return True
    except Exception as exc:
        logger.warning(
            "Primary database is unreachable during startup. "
            "Skipping database maintenance and bootstrap tasks until a later restart. Detail: %s",
            _summarize_startup_exception(exc),
        )
        return False


STARTUP_DATABASE_REACHABLE = probe_startup_database_connectivity()


def initialize_database_metadata() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema verified")
    except OperationalError:
        logger.exception(
            "Unable to verify the database schema during startup. Continuing in degraded mode until the database becomes reachable."
        )
    except Exception:
        logger.exception(
            "Unexpected error while verifying the database schema during startup. Continuing in degraded mode."
        )


if STARTUP_DATABASE_REACHABLE:
    initialize_database_metadata()


def ensure_user_settings_columns() -> None:
    """Backfill settings columns for existing databases created before UI settings were added."""
    try:
        inspector = inspect(engine)
        existing_columns = {column["name"] for column in inspector.get_columns("user")}
    except Exception:
        logger.exception("Unable to inspect user table for settings migration")
        return

    if not existing_columns:
        return

    column_definitions = {
        "lob": "VARCHAR(100)",
        "sidebar_state": "VARCHAR(20) DEFAULT 'default'",
        "big_font_scale": "FLOAT DEFAULT 1.0",
        "daltonism_mode": "VARCHAR(20) DEFAULT 'none'",
        "profile_image_url": "VARCHAR(500)",
        "ui_preferences": "JSONB DEFAULT '{}'::jsonb"
        if engine.dialect.name == "postgresql"
        else "JSON DEFAULT '{}'",
    }

    statements = [
        text(f'ALTER TABLE "user" ADD COLUMN {name} {definition}')
        for name, definition in column_definitions.items()
        if name not in existing_columns
    ]

    if not statements:
        return

    empty_json_literal = (
        "'{}'::jsonb" if engine.dialect.name == "postgresql" else "'{}'"
    )

    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(statement)

            connection.execute(
                text(
                    'UPDATE "user" SET '
                    "sidebar_state = COALESCE(sidebar_state, 'default'), "
                    "big_font_scale = COALESCE(big_font_scale, 1.0), "
                    "daltonism_mode = COALESCE(daltonism_mode, 'none'), "
                    f"ui_preferences = COALESCE(ui_preferences, {empty_json_literal})"
                )
            )
        logger.info("Applied user settings schema backfill for existing databases")
    except Exception:
        logger.exception("Failed to backfill user settings columns")


if STARTUP_DATABASE_REACHABLE:
    ensure_user_settings_columns()


def ensure_profiles_table_exists() -> None:
    """Create a minimal `public.profiles` table if it does not exist.

    Some deployments may not have the Supabase auth/profile schema. The
    startup sync attempts to insert into `public.profiles`; creating a
    minimal compatible table prevents startup failures while preserving
    Supabase-compatible columns.
    """
    if engine.dialect.name != "postgresql":
        return

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS public.profiles (
                        id UUID PRIMARY KEY,
                        full_name TEXT,
                        avatar_url TEXT,
                        role TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
                    )
                    """
                )
            )
        logger.info("Ensured minimal public.profiles table exists")
    except Exception:
        logger.exception("Failed to ensure public.profiles table exists")


def ensure_supabase_realtime_publication() -> None:
    """Register realtime-dependent tables in the Supabase publication when available."""
    if engine.dialect.name != "postgresql":
        return

    desired_tables = {
        "sim_session",
        "certificate_record",
        "coaching_log",
        "call_simulation_assignment",
        "notification_event",
    }

    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        publishable_tables = sorted(desired_tables.intersection(existing_tables))
        if not publishable_tables:
            return

        with engine.begin() as connection:
            publication_exists = connection.execute(
                text("select 1 from pg_publication where pubname = 'supabase_realtime'")
            ).first()
            if not publication_exists:
                logger.info("Supabase realtime publication not found; skipping publication sync")
                return

            existing_publication_tables = {
                row.tablename
                for row in connection.execute(
                    text(
                        "select tablename from pg_publication_tables "
                        "where pubname = 'supabase_realtime' and schemaname = 'public'"
                    )
                )
            }

            for table_name in publishable_tables:
                if table_name in existing_publication_tables:
                    continue
                connection.execute(
                    text(f"alter publication supabase_realtime add table public.{table_name}")
                )

        logger.info(
            "Ensured Supabase realtime publication includes: %s",
            ", ".join(publishable_tables),
        )
    except Exception:
        logger.exception("Unable to synchronize Supabase realtime publication")


if STARTUP_DATABASE_REACHABLE:
    ensure_supabase_realtime_publication()


def ensure_microlearning_assessment_schema() -> None:
    """Backfill microlearning module columns for existing databases."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect microlearning tables for schema backfill")
        return

    if "microlearning_module" not in existing_tables:
        return

    current_columns = {
        column["name"] for column in inspector.get_columns("microlearning_module")
    }
    json_definition = (
        "JSONB DEFAULT '{}'::jsonb"
        if engine.dialect.name == "postgresql"
        else "JSON DEFAULT '{}'"
    )
    empty_json_literal = (
        "'{}'::jsonb" if engine.dialect.name == "postgresql" else "'{}'"
    )

    statements = []
    if "type" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN type VARCHAR(50) DEFAULT 'video'"
            )
        )
    if "content_data" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                f"ADD COLUMN content_data {json_definition}"
            )
        )
    if "passing_score" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN passing_score INTEGER DEFAULT 75"
            )
        )
    if "audio_url" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN audio_url VARCHAR(500)"
            )
        )
    if "audio_transcript" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN audio_transcript TEXT"
            )
        )
    if "audio_tts_url" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN audio_tts_url VARCHAR(500)"
            )
        )
    if "audio_duration_seconds" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN audio_duration_seconds INTEGER"
            )
        )
    if "audio_language" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN audio_language VARCHAR(10) DEFAULT 'en-US'"
            )
        )
    if "assessment_method_id" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_module "
                "ADD COLUMN assessment_method_id VARCHAR(36)"
            )
        )

    if not statements:
        return

    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                text(
                    "UPDATE microlearning_module SET "
                    "type = COALESCE(type, 'video'), "
                    f"content_data = COALESCE(content_data, {empty_json_literal}), "
                    "passing_score = COALESCE(passing_score, 75), "
                    "audio_language = COALESCE(audio_language, 'en-US')"
                )
            )
        logger.info("Applied microlearning module schema backfill")
    except Exception:
        logger.exception("Failed to backfill microlearning module schema")


def ensure_microlearning_assignment_schema() -> None:
    """Backfill lifecycle columns for microlearning assignments."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect microlearning tables for assignment schema backfill")
        return

    if "microlearning_assignment" not in existing_tables:
        return

    try:
        current_columns = {
            column["name"] for column in inspector.get_columns("microlearning_assignment")
        }
    except Exception:
        logger.exception("Unable to inspect microlearning_assignment columns for schema backfill")
        return

    statements = []
    if "certificate_id" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_assignment "
                "ADD COLUMN certificate_id VARCHAR(36)"
            )
        )
    if "started_at" not in current_columns:
        statements.append(
            text(
                "ALTER TABLE microlearning_assignment "
                "ADD COLUMN started_at TIMESTAMP"
            )
        )

    if not statements:
        return

    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(statement)
        logger.info("Applied microlearning assignment lifecycle schema backfill")
    except Exception:
        logger.exception("Failed to backfill microlearning assignment lifecycle columns")


def ensure_microlearning_topic_category_schema() -> None:
    """Backfill topic category linkage for microlearning modules."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect microlearning tables for topic category backfill")
        return

    if "microlearning_module" not in existing_tables:
        return

    try:
        current_columns = {
            column["name"] for column in inspector.get_columns("microlearning_module")
        }
    except Exception:
        logger.exception("Unable to inspect microlearning_module columns for topic category backfill")
        return

    if "topic_category_id" in current_columns:
        return

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE microlearning_module "
                    "ADD COLUMN topic_category_id VARCHAR(36)"
                )
            )
        logger.info("Applied microlearning topic category schema backfill")
    except Exception:
        logger.exception("Failed to backfill microlearning topic category schema")


if STARTUP_DATABASE_REACHABLE:
    ensure_microlearning_assessment_schema()
    ensure_microlearning_assignment_schema()
    ensure_microlearning_topic_category_schema()


def ensure_user_dismissed_notifications_column() -> None:
    """Backfill dismissed_notifications column for existing databases."""
    try:
        inspector = inspect(engine)
        existing_columns = {column["name"] for column in inspector.get_columns("user")}
    except Exception:
        logger.exception(
            "Unable to inspect user table for dismissed_notifications migration"
        )
        return

    if "dismissed_notifications" in existing_columns:
        return

    json_definition = (
        "JSONB DEFAULT '[]'::jsonb"
        if engine.dialect.name == "postgresql"
        else "JSON DEFAULT '[]'"
    )

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    f'ALTER TABLE "user" ADD COLUMN dismissed_notifications {json_definition}'
                )
            )
        logger.info("Applied user dismissed_notifications schema backfill")
    except Exception:
        logger.exception("Failed to backfill dismissed_notifications column")


if STARTUP_DATABASE_REACHABLE:
    ensure_user_dismissed_notifications_column()


def ensure_batch_schema() -> None:
    """Backfill batch columns used by trainer assignment workflows."""
    try:
        inspector = inspect(engine)
        existing_columns = {column["name"] for column in inspector.get_columns("batch")}
    except Exception:
        logger.exception("Unable to inspect batch table for schema backfill")
        return

    statements = []
    if "is_active" not in existing_columns:
        statements.append(
            text('ALTER TABLE "batch" ADD COLUMN is_active BOOLEAN DEFAULT TRUE')
        )
    if "start_date" not in existing_columns:
        statements.append(
            text('ALTER TABLE "batch" ADD COLUMN start_date DATE')
        )
    if "end_date" not in existing_columns:
        statements.append(
            text('ALTER TABLE "batch" ADD COLUMN end_date DATE')
        )

    if not statements:
        return

    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                text('UPDATE "batch" SET is_active = COALESCE(is_active, TRUE)')
            )
        logger.info("Applied batch schema backfill")
    except Exception:
        logger.exception("Failed to backfill batch schema")


if STARTUP_DATABASE_REACHABLE:
    ensure_batch_schema()


def ensure_call_simulation_session_schema() -> None:
    """Backfill Call Simulation session columns for existing databases."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect Call Simulation tables for schema backfill")
        return

    if "sim_session" not in existing_tables:
        return

    try:
        existing_columns = {
            column["name"] for column in inspector.get_columns("sim_session")
        }
    except Exception:
        logger.exception("Unable to inspect sim_session columns for schema backfill")
        return

    json_definition = (
        "JSONB DEFAULT '[]'::jsonb"
        if engine.dialect.name == "postgresql"
        else "JSON DEFAULT '[]'"
    )
    json_object_definition = (
        "JSONB DEFAULT '{}'::jsonb"
        if engine.dialect.name == "postgresql"
        else "JSON DEFAULT '{}'"
    )

    statements = []
    if "coaching_notes" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN coaching_notes TEXT")
        )
    if "transcript_log" not in existing_columns:
        statements.append(
            text(f"ALTER TABLE sim_session ADD COLUMN transcript_log {json_definition}")
        )
    if "turn_logs" not in existing_columns:
        statements.append(
            text(f"ALTER TABLE sim_session ADD COLUMN turn_logs {json_definition}")
        )
    if "trainer_verdict_status" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN trainer_verdict_status VARCHAR(30) DEFAULT 'pending'")
        )
    if "trainer_verdict_notes" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN trainer_verdict_notes TEXT")
        )
    if "trainer_evaluated_by" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN trainer_evaluated_by VARCHAR(36)")
        )
    if "trainer_evaluated_at" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN trainer_evaluated_at TIMESTAMP")
        )
    if "certificate_id" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN certificate_id VARCHAR(36)")
        )
    if "assignment_id" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN assignment_id VARCHAR(36)")
        )
    if "assigned_by_id" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN assigned_by_id VARCHAR(36)")
        )
    if "sentiment_score" not in existing_columns:
        statements.append(
            text("ALTER TABLE sim_session ADD COLUMN sentiment_score FLOAT")
        )
    if "keyword_compliance" not in existing_columns:
        statements.append(
            text(f"ALTER TABLE sim_session ADD COLUMN keyword_compliance {json_object_definition}")
        )

    if not statements:
        return

    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(statement)
        logger.info("Applied Call Simulation session schema backfill")
    except Exception:
        logger.exception("Failed to backfill Call Simulation session schema")


if STARTUP_DATABASE_REACHABLE:
    ensure_call_simulation_session_schema()


def ensure_call_simulation_assignment_schema() -> None:
    """Backfill Call Simulation assignment columns for existing databases."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect Call Simulation assignment tables for schema backfill")
        return

    if "call_simulation_assignment" not in existing_tables:
        return

    try:
        existing_columns = {
            column["name"] for column in inspector.get_columns("call_simulation_assignment")
        }
    except Exception:
        logger.exception("Unable to inspect call_simulation_assignment columns for schema backfill")
        return

    statements = []
    if "max_attempts" not in existing_columns:
        statements.append(
            text("ALTER TABLE call_simulation_assignment ADD COLUMN max_attempts INTEGER DEFAULT 3")
        )

    if not statements:
        return

    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(statement)
        logger.info("Applied Call Simulation assignment schema backfill")
    except Exception:
        logger.exception("Failed to backfill Call Simulation assignment schema")


if STARTUP_DATABASE_REACHABLE:
    ensure_call_simulation_assignment_schema()


def ensure_call_simulation_scenario_schema() -> None:
    """Backfill Scenario and ScenarioFlow fields for multi-turn Call Simulation sessions."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect Call Simulation scenario tables for schema backfill")
        return

    json_object_definition = (
        "JSONB DEFAULT '{}'::jsonb"
        if engine.dialect.name == "postgresql"
        else "JSON DEFAULT '{}'"
    )

    try:
        with engine.begin() as connection:
            if "scenario" in existing_tables:
                scenario_columns = {
                    column["name"] for column in inspector.get_columns("scenario")
                }
                scenario_statements = []
                if "member_profile" not in scenario_columns:
                    scenario_statements.append(
                        text(f"ALTER TABLE scenario ADD COLUMN member_profile {json_object_definition}")
                    )
                if "cxone_metadata" not in scenario_columns:
                    scenario_statements.append(
                        text(f"ALTER TABLE scenario ADD COLUMN cxone_metadata {json_object_definition}")
                    )
                if "call_simulation_config" not in scenario_columns:
                    scenario_statements.append(
                        text(f"ALTER TABLE scenario ADD COLUMN call_simulation_config {json_object_definition}")
                    )
                if "ringer_audio_url" not in scenario_columns:
                    scenario_statements.append(
                        text("ALTER TABLE scenario ADD COLUMN ringer_audio_url VARCHAR(500)")
                    )
                if "hold_audio_url" not in scenario_columns:
                    scenario_statements.append(
                        text("ALTER TABLE scenario ADD COLUMN hold_audio_url VARCHAR(500)")
                    )
                for statement in scenario_statements:
                    connection.execute(statement)

            if "scenario_flow" in existing_tables:
                flow_columns = {
                    column["name"] for column in inspector.get_columns("scenario_flow")
                }
                flow_statements = []
                if "speaker_role" not in flow_columns:
                    flow_statements.append(
                        text("ALTER TABLE scenario_flow ADD COLUMN speaker_role VARCHAR(20) DEFAULT 'member'")
                    )
                if "speaker_label" not in flow_columns:
                    flow_statements.append(
                        text("ALTER TABLE scenario_flow ADD COLUMN speaker_label VARCHAR(100)")
                    )
                if "step_metadata" not in flow_columns:
                    flow_statements.append(
                        text(f"ALTER TABLE scenario_flow ADD COLUMN step_metadata {json_object_definition}")
                    )
                for statement in flow_statements:
                    connection.execute(statement)
        logger.info("Applied Call Simulation scenario schema backfill")
    except Exception:
        logger.exception("Failed to backfill Call Simulation scenario schema")


if STARTUP_DATABASE_REACHABLE:
    ensure_call_simulation_scenario_schema()


def ensure_call_simulation_reporting_views() -> None:
    """Expose stable Supabase-facing reporting views for the Call Simulation module."""
    if engine.dialect.name != "postgresql":
        return

    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect Call Simulation tables for reporting views")
        return

    view_statements: list[str] = []

    if "scenario" in existing_tables:
        view_statements.append(
            """
            CREATE OR REPLACE VIEW public.call_simulation_scenarios AS
            SELECT
              id,
              title,
              description,
              COALESCE(NULLIF(call_simulation_config ->> 'topic', ''), title) AS topic,
              opening_prompt,
              expected_keywords,
              estimated_duration,
              difficulty,
              purpose,
              member_profile,
              cxone_metadata,
              call_simulation_config,
              call_simulation_config -> 'script_flow' AS script_flow,
              call_simulation_config -> 'target_kpis' AS target_kpis,
              ringer_audio_url,
              hold_audio_url,
              created_by AS trainer_id,
              is_published,
              is_draft,
              created_at,
              updated_at
            FROM public.scenario
            """
        )

    if "scenario_flow" in existing_tables:
        view_statements.append(
            """
            CREATE OR REPLACE VIEW public.call_simulation_script_turns AS
            SELECT
              id,
              scenario_id,
              step_number,
              COALESCE(NULLIF(speaker_role, ''), 'member') AS actor,
              speaker_label,
              COALESCE(NULLIF(prompt_text, ''), NULLIF(expected_response, ''), '') AS script,
              expected_keywords_for_step AS expected_keywords,
              prompt_audio AS audio_url,
              step_metadata,
              COALESCE(
                NULLIF(step_metadata ->> 'member_audio_url', ''),
                NULLIF(step_metadata ->> 'audio_url', ''),
                prompt_audio
              ) AS member_audio_url,
              created_at,
              updated_at
            FROM public.scenario_flow
            """
        )

    if "batch_kpi_config" in existing_tables:
        view_statements.append(
            """
            CREATE OR REPLACE VIEW public.call_simulation_kpis AS
            SELECT
              id,
              batch_id,
              speech_to_text_weight,
              aht_weight,
              rate_of_speech_weight,
              dead_air_weight,
              empathy_statements_weight,
              probing_questions_weight,
              grammar_weight,
              pronunciation_weight,
              pacing_weight,
              forbidden_words_penalty,
              passing_score,
              forbidden_words,
              empathy_keywords,
              probing_keywords,
              target_aht_seconds,
              target_ros_words_per_min,
              target_dead_air_seconds,
              jsonb_build_object(
                'speech_to_text_weight', speech_to_text_weight,
                'aht_weight', aht_weight,
                'rate_of_speech_weight', rate_of_speech_weight,
                'dead_air_weight', dead_air_weight,
                'empathy_statements_weight', empathy_statements_weight,
                'probing_questions_weight', probing_questions_weight,
                'grammar_weight', grammar_weight,
                'pronunciation_weight', pronunciation_weight,
                'pacing_weight', pacing_weight,
                'forbidden_words_penalty', forbidden_words_penalty,
                'passing_score', passing_score,
                'forbidden_words', forbidden_words,
                'empathy_keywords', empathy_keywords,
                'probing_keywords', probing_keywords,
                'target_aht_seconds', target_aht_seconds,
                'target_ros_words_per_min', target_ros_words_per_min,
                'target_dead_air_seconds', target_dead_air_seconds
              ) AS criteria_payload,
              created_at,
              updated_at
            FROM public.batch_kpi_config
            """
        )

    if "call_simulation_assignment" in existing_tables:
        if "sim_session" in existing_tables:
            view_statements.append(
                """
                CREATE OR REPLACE VIEW public.call_simulation_assignments AS
                SELECT
                  assignment.id,
                  assignment.scenario_id,
                  assignment.trainee_id,
                  assignment.assigned_by AS trainer_id,
                  assignment.batch_id,
                  assignment.max_attempts,
                  assignment.trainer_notes,
                  assignment.is_active,
                  assignment.assigned_at,
                  assignment.updated_at,
                  COALESCE(session_stats.latest_attempt_number, 0) AS latest_attempt_number,
                  GREATEST(COALESCE(session_stats.latest_attempt_number, 0) - 1, 0) AS retake_count,
                  latest_session.id AS latest_session_id,
                  latest_session.status AS latest_session_status,
                  latest_session.pass_fail AS latest_pass_fail,
                  latest_session.weighted_score AS latest_score,
                  latest_session.completed_at AS latest_completed_at,
                  latest_session.audio_url AS latest_recording_url
                FROM public.call_simulation_assignment AS assignment
                LEFT JOIN LATERAL (
                  SELECT
                    session.id,
                    session.status,
                    session.pass_fail,
                    session.weighted_score,
                    session.completed_at,
                    session.audio_url
                  FROM public.sim_session AS session
                  WHERE CAST(session.assignment_id AS text) = CAST(assignment.id AS text)
                  ORDER BY COALESCE(session.completed_at, session.created_at) DESC, session.attempt_number DESC
                  LIMIT 1
                ) AS latest_session ON TRUE
                LEFT JOIN LATERAL (
                  SELECT COALESCE(MAX(session.attempt_number), 0) AS latest_attempt_number
                  FROM public.sim_session AS session
                  WHERE CAST(session.assignment_id AS text) = CAST(assignment.id AS text)
                ) AS session_stats ON TRUE
                """
            )
        else:
            view_statements.append(
                """
                CREATE OR REPLACE VIEW public.call_simulation_assignments AS
                SELECT
                  id,
                  scenario_id,
                  trainee_id,
                  assigned_by AS trainer_id,
                  batch_id,
                  max_attempts,
                  trainer_notes,
                  is_active,
                  assigned_at,
                  updated_at,
                  CAST(0 AS integer) AS latest_attempt_number,
                  CAST(0 AS integer) AS retake_count,
                  CAST(NULL AS text) AS latest_session_id,
                  CAST(NULL AS text) AS latest_session_status,
                  CAST(NULL AS boolean) AS latest_pass_fail,
                  CAST(NULL AS numeric) AS latest_score,
                  CAST(NULL AS timestamp) AS latest_completed_at,
                  CAST(NULL AS text) AS latest_recording_url
                FROM public.call_simulation_assignment
                """
            )

    if "sim_session" in existing_tables:
        if "call_simulation_scores" in existing_tables:
            view_statements.append(
                """
                CREATE OR REPLACE VIEW public.call_simulation_attempts AS
                SELECT
                  session.id,
                  session.trainee_id,
                  session.scenario_id,
                  session.assignment_id,
                  session.assigned_by_id AS trainer_id,
                  session.batch_id,
                  session.status,
                  session.attempt_number,
                  GREATEST(COALESCE(session.attempt_number, 1) - 1, 0) AS retake_count,
                  session.max_attempts,
                  session.transcript,
                  session.transcript_log,
                  session.turn_logs,
                  session.audio_url,
                  session.audio_url AS recording_url,
                  session.audio_duration_seconds AS call_duration_seconds,
                  session.speech_to_text_accuracy,
                  session.grammar_score,
                  session.pronunciation_score,
                  session.pacing_score,
                  session.rate_of_speech,
                  session.dead_air_seconds,
                  session.sentiment_score,
                  session.keyword_compliance,
                  session.weighted_score AS final_score,
                  session.pass_fail,
                  session.ai_feedback,
                  score.id AS supabase_score_record_id,
                  score.passing_score,
                  COALESCE(score.full_transcript, session.transcript) AS full_transcript,
                  score.feedback_report AS evaluation_payload,
                  session.coaching_notes,
                  session.trainer_verdict_status,
                  session.trainer_verdict_notes,
                  session.trainer_evaluated_by,
                  session.trainer_evaluated_at,
                  session.certificate_id,
                  session.started_at,
                  session.completed_at,
                  session.created_at,
                  session.updated_at
                FROM public.sim_session AS session
                LEFT JOIN public.call_simulation_scores AS score
                  ON CAST(score.session_id AS text) = CAST(session.id AS text)
                """
            )
        else:
            view_statements.append(
                """
                CREATE OR REPLACE VIEW public.call_simulation_attempts AS
                SELECT
                  session.id,
                  session.trainee_id,
                  session.scenario_id,
                  session.assignment_id,
                  session.assigned_by_id AS trainer_id,
                  session.batch_id,
                  session.status,
                  session.attempt_number,
                  GREATEST(COALESCE(session.attempt_number, 1) - 1, 0) AS retake_count,
                  session.max_attempts,
                  session.transcript,
                  session.transcript_log,
                  session.turn_logs,
                  session.audio_url,
                  session.audio_url AS recording_url,
                  session.audio_duration_seconds AS call_duration_seconds,
                  session.speech_to_text_accuracy,
                  session.grammar_score,
                  session.pronunciation_score,
                  session.pacing_score,
                  session.rate_of_speech,
                  session.dead_air_seconds,
                  session.sentiment_score,
                  session.keyword_compliance,
                  session.weighted_score AS final_score,
                  session.pass_fail,
                  session.ai_feedback,
                  CAST(NULL AS text) AS supabase_score_record_id,
                  CAST(NULL AS numeric) AS passing_score,
                  session.transcript AS full_transcript,
                  CAST(NULL AS jsonb) AS evaluation_payload,
                  session.coaching_notes,
                  session.trainer_verdict_status,
                  session.trainer_verdict_notes,
                  session.trainer_evaluated_by,
                  session.trainer_evaluated_at,
                  session.certificate_id,
                  session.started_at,
                  session.completed_at,
                  session.created_at,
                  session.updated_at
                FROM public.sim_session AS session
                """
            )

        view_statements.append(
            """
            CREATE OR REPLACE VIEW public.call_simulation_recordings AS
            SELECT
              session.id,
              session.trainee_id,
              session.scenario_id,
              session.assignment_id,
              session.batch_id,
              session.audio_url AS recording_url,
              session.audio_url AS recording_path,
              session.audio_duration_seconds AS call_duration_seconds,
              session.transcript,
              session.transcript_log,
              session.turn_logs,
              session.weighted_score AS final_score,
              session.pass_fail,
              session.attempt_number,
              GREATEST(COALESCE(session.attempt_number, 1) - 1, 0) AS retake_count,
              session.started_at,
              session.completed_at,
              session.created_at,
              session.updated_at
            FROM public.sim_session AS session
            WHERE session.audio_url IS NOT NULL
            """
        )

    if "call_simulation_scores" in existing_tables:
        if "sim_session" in existing_tables:
            view_statements.append(
                """
                CREATE OR REPLACE VIEW public.call_simulation_ai_evaluations AS
                SELECT
                  score.id,
                  score.session_id,
                  score.scenario_id,
                  score.call_scenario_id,
                  score.trainee_id,
                  score.trainee_name,
                  score.scenario_topic,
                  score.total_score,
                  score.passing_score,
                  score.is_passed,
                  COALESCE(score.full_transcript, session.transcript) AS full_transcript,
                  session.transcript_log,
                  session.turn_logs,
                  session.audio_url AS recording_url,
                  session.audio_duration_seconds AS call_duration_seconds,
                  session.attempt_number,
                  GREATEST(COALESCE(session.attempt_number, 1) - 1, 0) AS retake_count,
                  session.max_attempts,
                  session.batch_id,
                  session.assignment_id,
                  session.coaching_notes,
                  session.trainer_verdict_status,
                  session.trainer_verdict_notes,
                  session.completed_at,
                  score.feedback_report AS evaluation_payload,
                  score.certificate_id,
                  score.supabase_certificate_id,
                  score.created_at,
                  score.updated_at
                FROM public.call_simulation_scores AS score
                LEFT JOIN public.sim_session AS session
                  ON CAST(session.id AS text) = CAST(score.session_id AS text)
                """
            )
        else:
            view_statements.append(
                """
                CREATE OR REPLACE VIEW public.call_simulation_ai_evaluations AS
                SELECT
                  id,
                  session_id,
                  scenario_id,
                  call_scenario_id,
                  trainee_id,
                  trainee_name,
                  scenario_topic,
                  total_score,
                  passing_score,
                  is_passed,
                  full_transcript,
                  CAST(NULL AS jsonb) AS transcript_log,
                  CAST(NULL AS jsonb) AS turn_logs,
                  CAST(NULL AS text) AS recording_url,
                  CAST(NULL AS integer) AS call_duration_seconds,
                  CAST(NULL AS integer) AS attempt_number,
                  CAST(0 AS integer) AS retake_count,
                  CAST(NULL AS integer) AS max_attempts,
                  CAST(NULL AS text) AS batch_id,
                  CAST(NULL AS text) AS assignment_id,
                  CAST(NULL AS text) AS coaching_notes,
                  CAST(NULL AS text) AS trainer_verdict_status,
                  CAST(NULL AS text) AS trainer_verdict_notes,
                  CAST(NULL AS timestamp) AS completed_at,
                  feedback_report AS evaluation_payload,
                  certificate_id,
                  supabase_certificate_id,
                  created_at,
                  updated_at
                FROM public.call_simulation_scores
                """
            )

    if "coaching_log" in existing_tables:
        view_statements.append(
            """
            CREATE OR REPLACE VIEW public.call_simulation_coaching_notes AS
            SELECT
              coaching.id,
              coaching.coaching_id,
              coaching.sim_session_id AS session_id,
              coaching.trainer_id,
              coaching.trainee_id,
              coaching.batch_name,
              coaching.lob,
              coaching.coaching_minutes,
              coaching.strengths,
              coaching.opportunities,
              coaching.action_plan,
              coaching.target_date,
              coaching.status,
              coaching.competency_status,
              coaching.trainer_remarks,
              coaching.acknowledged_at,
              session.scenario_id,
              session.batch_id,
              session.audio_url AS recording_url,
              session.transcript,
              session.attempt_number,
              session.pass_fail,
              session.trainer_verdict_status,
              session.completed_at,
              coaching.created_at,
              coaching.updated_at
            FROM public.coaching_log AS coaching
            LEFT JOIN public.sim_session AS session
              ON CAST(session.id AS text) = CAST(coaching.sim_session_id AS text)
            """
        )
    elif "coaching_logs" in existing_tables:
        view_statements.append(
            """
            CREATE OR REPLACE VIEW public.call_simulation_coaching_notes AS
            SELECT
              coaching.id,
              coaching.coaching_id,
              coaching.session_id,
              coaching.trainer_id,
              coaching.trainee_id,
              CAST(NULL AS text) AS batch_name,
              CAST(NULL AS text) AS lob,
              CAST(NULL AS integer) AS coaching_minutes,
              coaching.strengths,
              coaching.opportunities,
              coaching.action_plan,
              CAST(coaching.target_date AS timestamp) AS target_date,
              coaching.status,
              CAST(NULL AS text) AS competency_status,
              CAST(NULL AS text) AS trainer_remarks,
              coaching.acknowledged_at,
              session.scenario_id,
              session.batch_id,
              session.audio_url AS recording_url,
              session.transcript,
              session.attempt_number,
              session.pass_fail,
              session.trainer_verdict_status,
              session.completed_at,
              coaching.created_at,
              coaching.updated_at
            FROM public.coaching_logs AS coaching
            LEFT JOIN public.sim_session AS session
              ON CAST(session.id AS text) = CAST(coaching.session_id AS text)
            """
        )

    if not view_statements:
        return

    reporting_view_names = (
        "public.call_simulation_assignments",
        "public.call_simulation_attempts",
        "public.call_simulation_recordings",
        "public.call_simulation_ai_evaluations",
        "public.call_simulation_coaching_notes",
        "public.call_simulation_kpis",
    )

    try:
        with engine.begin() as connection:
            for statement in view_statements:
                view_name = next(
                    (candidate for candidate in reporting_view_names if candidate in statement),
                    None,
                )
                # Postgres rejects CREATE OR REPLACE VIEW when column names or order
                # change. These reporting views are derived, so rebuild them cleanly.
                if engine.dialect.name == "postgresql" and view_name and "CREATE OR REPLACE VIEW" in statement:
                    try:
                        connection.execute(text(f"DROP VIEW IF EXISTS {view_name} CASCADE"))
                    except Exception:
                        logger.debug("Drop of %s view failed or unnecessary", view_name)
                    create_stmt = statement.replace("CREATE OR REPLACE VIEW", "CREATE VIEW")
                    connection.execute(text(create_stmt))
                else:
                    connection.execute(text(statement))
        logger.info("Ensured Call Simulation reporting views are available in Supabase Postgres")
    except Exception:
        logger.exception("Failed to create Call Simulation reporting views")


if STARTUP_DATABASE_REACHABLE:
    ensure_call_simulation_reporting_views()


def ensure_certification_schema() -> None:
    """Backfill certificate settings and certificate record columns for older databases."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect certification tables for schema backfill")
        return

    json_definition = (
        "JSONB DEFAULT '{}'::jsonb"
        if engine.dialect.name == "postgresql"
        else "JSON DEFAULT '{}'"
    )
    empty_json_literal = (
        "'{}'::jsonb" if engine.dialect.name == "postgresql" else "'{}'"
    )

    certification_columns = {
        "logo_url": "TEXT",
        "manager_signature_url": "TEXT",
        "dry_seal_url": "TEXT",
        "signatory_title": "VARCHAR(255) DEFAULT 'Authorized Signatory'",
        "certificate_prefix": "VARCHAR(50) DEFAULT 'SPV'",
        "certificate_title": "VARCHAR(255) DEFAULT 'Certificate of Completion'",
        "certificate_subtitle": "VARCHAR(255) DEFAULT 'Issued for completed trainee tasks and assessments'",
        "certificate_intro": "TEXT DEFAULT 'This certificate is proudly presented to'",
        "certificate_outro": (
            "TEXT DEFAULT 'for successfully completing the training requirement shown below "
            "through St. Peter Velle Technical Training Center, Inc.'"
        ),
        "certificate_footer": (
            "TEXT DEFAULT 'This certificate is stored in the platform database and may be "
            "verified through the official certificate record.'"
        ),
    }

    certificate_record_columns = {
        "source_type": "VARCHAR(50) DEFAULT 'competency_verdict'",
        "source_id": "VARCHAR(36)",
        "achievement_type": "VARCHAR(50) DEFAULT 'completion'",
        "template_snapshot": json_definition,
    }
    coaching_log_columns = {
        "source_type": "VARCHAR(30) DEFAULT 'practice_session'",
        "sim_session_id": "VARCHAR(36)",
        "competency_status": "VARCHAR(20) DEFAULT 'pending'",
    }

    try:
        with engine.begin() as connection:
            if "certification_settings" in existing_tables:
                current_columns = {
                    column["name"]
                    for column in inspector.get_columns("certification_settings")
                }
                for name, definition in certification_columns.items():
                    if name not in current_columns:
                        connection.execute(
                            text(
                                f"ALTER TABLE certification_settings ADD COLUMN {name} {definition}"
                            )
                        )
                    elif (
                        engine.dialect.name == "postgresql"
                        and name in {"logo_url", "manager_signature_url", "dry_seal_url"}
                    ):
                        column_info = next(
                            column
                            for column in inspector.get_columns("certification_settings")
                            if column["name"] == name
                        )
                        if "VARCHAR" in str(column_info["type"]).upper():
                            connection.execute(
                                text(
                                    f"ALTER TABLE certification_settings ALTER COLUMN {name} TYPE TEXT"
                                )
                            )

                connection.execute(
                    text(
                        "UPDATE certification_settings SET "
                        "signatory_title = COALESCE(signatory_title, 'Authorized Signatory'), "
                        "certificate_prefix = COALESCE(certificate_prefix, 'SPV'), "
                        "certificate_title = COALESCE(certificate_title, 'Certificate of Completion'), "
                        "certificate_subtitle = COALESCE(certificate_subtitle, 'Issued for completed trainee tasks and assessments'), "
                        "certificate_intro = COALESCE(certificate_intro, 'This certificate is proudly presented to'), "
                        "certificate_outro = COALESCE(certificate_outro, 'for successfully completing the training requirement shown below through St. Peter Velle Technical Training Center, Inc.'), "
                        "certificate_footer = COALESCE(certificate_footer, 'This certificate is stored in the platform database and may be verified through the official certificate record.')"
                    )
                )

            if "certificate_record" in existing_tables:
                current_columns = {
                    column["name"]
                    for column in inspector.get_columns("certificate_record")
                }
                for name, definition in certificate_record_columns.items():
                    if name not in current_columns:
                        connection.execute(
                            text(
                                f"ALTER TABLE certificate_record ADD COLUMN {name} {definition}"
                            )
                        )

                connection.execute(
                    text(
                        "UPDATE certificate_record SET "
                        "source_type = COALESCE(source_type, 'competency_verdict'), "
                        "source_id = COALESCE(source_id, verdict_id), "
                        "achievement_type = COALESCE(achievement_type, 'competency'), "
                        f"template_snapshot = COALESCE(template_snapshot, {empty_json_literal})"
                    )
                )

            if "coaching_log" in existing_tables:
                current_columns = {
                    column["name"] for column in inspector.get_columns("coaching_log")
                }
                for name, definition in coaching_log_columns.items():
                    if name not in current_columns:
                        connection.execute(
                            text(
                                f"ALTER TABLE coaching_log ADD COLUMN {name} {definition}"
                            )
                        )

                connection.execute(
                    text(
                        "UPDATE coaching_log SET "
                        "source_type = COALESCE(source_type, CASE WHEN sim_session_id IS NOT NULL THEN 'sim_floor_session' ELSE 'practice_session' END), "
                        "competency_status = COALESCE(competency_status, 'pending')"
                    )
                )
        logger.info("Applied certification schema backfill for existing databases")
    except Exception:
        logger.exception("Failed to backfill certification schema")


if STARTUP_DATABASE_REACHABLE:
    ensure_certification_schema()


def ensure_mcq_assessment_schema() -> None:
    """Backfill MCQ navigation columns for persisted assessment data."""
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        logger.exception("Unable to inspect MCQ assessment schema")
        return

    if not {"mcq_category", "mcq_assessment", "mcq_submission"}.intersection(existing_tables):
        return

    try:
        empty_array_literal = "'[]'::jsonb" if engine.dialect.name == "postgresql" else "'[]'"
        with engine.begin() as connection:
            if "certification_settings" in existing_tables:
                settings_columns = {
                    column["name"] for column in inspector.get_columns("certification_settings")
                }
                if "mcq_passing_threshold" in settings_columns:
                    connection.execute(
                        text(
                            "UPDATE certification_settings SET "
                            "mcq_passing_threshold = CASE "
                            "WHEN mcq_passing_threshold IS NULL OR mcq_passing_threshold < 90 THEN 90 "
                            "ELSE mcq_passing_threshold "
                            "END"
                        )
                    )

            if "mcq_category" in existing_tables:
                category_columns = {
                    column["name"] for column in inspector.get_columns("mcq_category")
                }
                if "selected_question_ids" not in category_columns:
                    connection.execute(
                        text(
                            "ALTER TABLE mcq_category "
                            "ADD COLUMN selected_question_ids JSONB DEFAULT '[]'::jsonb"
                            if engine.dialect.name == "postgresql"
                            else "ALTER TABLE mcq_category "
                            "ADD COLUMN selected_question_ids JSON DEFAULT '[]'"
                        )
                    )
                connection.execute(
                    text(
                        "UPDATE mcq_category SET "
                        f"selected_question_ids = COALESCE(selected_question_ids, {empty_array_literal}), "
                        "passing_threshold = CASE "
                        "WHEN passing_threshold IS NULL OR passing_threshold < 90 THEN 90 "
                        "ELSE passing_threshold "
                        "END"
                    )
                )

            if "mcq_assessment" in existing_tables:
                assessment_columns = {
                    column["name"] for column in inspector.get_columns("mcq_assessment")
                }
                if "time_limit_minutes" not in assessment_columns:
                    connection.execute(
                        text(
                            "ALTER TABLE mcq_assessment "
                            "ADD COLUMN time_limit_minutes INTEGER DEFAULT 30"
                        )
                    )
                if "question_snapshot" not in assessment_columns:
                    connection.execute(
                        text(
                            "ALTER TABLE mcq_assessment "
                            "ADD COLUMN question_snapshot JSONB DEFAULT '[]'::jsonb"
                            if engine.dialect.name == "postgresql"
                            else "ALTER TABLE mcq_assessment "
                            "ADD COLUMN question_snapshot JSON DEFAULT '[]'"
                        )
                    )
                connection.execute(
                    text(
                        "UPDATE mcq_assessment SET "
                        "time_limit_minutes = COALESCE(time_limit_minutes, 30), "
                        f"question_snapshot = COALESCE(question_snapshot, {empty_array_literal})"
                    )
                )

            if "mcq_submission" in existing_tables:
                submission_columns = {
                    column["name"] for column in inspector.get_columns("mcq_submission")
                }
                if "review" not in submission_columns:
                    connection.execute(
                        text(
                            "ALTER TABLE mcq_submission "
                            "ADD COLUMN review JSONB DEFAULT '[]'::jsonb"
                            if engine.dialect.name == "postgresql"
                            else "ALTER TABLE mcq_submission "
                            "ADD COLUMN review JSON DEFAULT '[]'"
                        )
                    )
                if "attempt_count" not in submission_columns:
                    connection.execute(
                        text(
                            "ALTER TABLE mcq_submission "
                            "ADD COLUMN attempt_count INTEGER DEFAULT 1"
                        )
                    )
                connection.execute(
                    text(
                        "UPDATE mcq_submission SET "
                        f"review = COALESCE(review, {empty_array_literal}), "
                        "attempt_count = CASE "
                        "WHEN attempt_count IS NULL OR attempt_count < 1 THEN 1 "
                        "ELSE attempt_count "
                        "END"
                    )
                )
            if {"mcq_submission", "mcq_assessment", "mcq_category"}.issubset(existing_tables):
                connection.execute(
                    text(
                        "UPDATE mcq_submission SET "
                        "is_passed = CASE "
                        "WHEN COALESCE(score_percentage, 0) >= COALESCE(("
                        "  SELECT CASE "
                        "    WHEN c.passing_threshold IS NULL OR c.passing_threshold < 90 THEN 90 "
                        "    ELSE c.passing_threshold "
                        "  END "
                        "  FROM mcq_assessment AS a "
                        "  JOIN mcq_category AS c ON c.id = a.category_id "
                        "  WHERE a.id = mcq_submission.assessment_id"
                        "), 90) "
                        "THEN TRUE ELSE FALSE END"
                    )
                )
        logger.info("Applied MCQ assessment navigation schema backfill")
    except Exception:
        logger.exception("Failed to backfill MCQ assessment navigation schema")


if STARTUP_DATABASE_REACHABLE:
    ensure_mcq_assessment_schema()


def cleanup_legacy_seed_data() -> None:
    """Keep retired sample/demo content out of the live product views."""
    db = SessionLocal()
    try:
        cleanup_summary = cleanup_legacy_sample_dataset(db)
        db.commit()
        if cleanup_summary.get("changed"):
            logger.info("Legacy sample cleanup applied: %s", cleanup_summary)
    except Exception:
        db.rollback()
        logger.exception("Failed to clean up legacy sample data")
    finally:
        db.close()


def _normalize_origin(value: str | None) -> str:
    candidate = normalize_env_value(value or "")
    if not candidate:
        return ""

    try:
        parsed = urlparse(candidate)
    except Exception:
        return ""

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def build_allowed_cors_origins() -> list[str]:
    configured_values = [
        os.getenv("FRONTEND_URL"),
        os.getenv("BACKEND_URL"),
        os.getenv("RENDER_EXTERNAL_URL"),
    ]
    configured_values.extend(
        segment.strip()
        for segment in (os.getenv("APP_ALLOWED_ORIGINS") or "").split(",")
        if segment.strip()
    )

    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    normalized_origins: list[str] = []
    seen: set[str] = set()
    for raw_value in [*configured_values, *defaults]:
        normalized = _normalize_origin(raw_value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_origins.append(normalized)

    return normalized_origins


# Add CORS middleware to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=build_allowed_cors_origins(),
    allow_origin_regex=r"^https://([a-z0-9-]+\.)*(onrender\.com|vercel\.app)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")

if STARTUP_DATABASE_REACHABLE:
    cleanup_legacy_seed_data()

# Ensure admin user exists
def ensure_admin_user():
    from backend.database import SessionLocal
    from backend.models import User, UserRole
    from backend import auth_utils
    from backend.default_credentials import ADMIN_EMAIL, ADMIN_PASSWORD

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        if not existing:
            admin = User(
                email=ADMIN_EMAIL,
                full_name="Admin User",
                password_hash=auth_utils.hash_password(ADMIN_PASSWORD),
                role=UserRole.ADMIN,
                is_active=True,
                lob="Administration",
                department="Management"
            )
            db.add(admin)
            db.commit()
            logger.info("Admin user created: %s", ADMIN_EMAIL)
        else:
            logger.info("Admin user already exists")
    except Exception as e:
        logger.exception("Failed to ensure admin user: %s", e)
    finally:
        db.close()


def sync_runtime_users_to_supabase_auth(*, fail_fast: bool) -> None:
    """Keep Supabase auth.users aligned with the platform users on every start."""
    from backend.database import SessionLocal
    from backend.models import User
    from backend.services.supabase_auth_service import sync_user_to_supabase_auth

    db = SessionLocal()
    created = 0
    updated = 0
    skipped = 0
    mismatched_ids = 0

    try:
        users = db.query(User).order_by(User.created_at.asc(), User.email.asc()).all()
        for user in users:
            result = sync_user_to_supabase_auth(db, user, update_password=False)
            status = result.get("status")
            if status == "created":
                created += 1
            elif status == "skipped":
                skipped += 1
            else:
                updated += 1

            if not result.get("matched_local_id", True):
                mismatched_ids += 1

        db.commit()
        logger.info(
            "Synchronized %s users to Supabase Auth (created=%s, updated=%s, skipped=%s, mismatched_ids=%s)",
            len(users),
            created,
            updated,
            skipped,
            mismatched_ids,
        )
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to synchronize users to Supabase Auth: %s", exc)
        if fail_fast:
            raise RuntimeError(
                "Supabase Auth synchronization failed during backend startup."
            ) from exc
    finally:
        db.close()


# Include route blueprints
app.include_router(auth_routes.router)
app.include_router(user_routes.router)
app.include_router(scenario_routes.router)
app.include_router(assessment_routes.router)
app.include_router(assessment_management_routes.router)
app.include_router(microlearning_routes.router)
app.include_router(analytics_routes.router)
app.include_router(admin_routes.router)
app.include_router(trainer_routes.router)
app.include_router(trainee_routes.router)
app.include_router(settings_routes.router)
app.include_router(workspace_routes.router)
app.include_router(export_routes.router)
app.include_router(certification_routes.router)
app.include_router(notification_routes.router)
app.include_router(call_simulation_routes.router)
app.include_router(call_simulation_recordings.router)
app.include_router(assessment_redesign_routes.router)
app.include_router(audit_routes.router)
if reading_assessment_routes is not None:
    app.include_router(reading_assessment_routes.router)

# Azure Speech Configuration
SPEECH_KEY = normalize_env_value(os.getenv("AZURE_SPEECH_KEY"))
SPEECH_REGION = normalize_env_value(os.getenv("AZURE_SPEECH_REGION")) or "eastus"
# Ensure Azure Speech SDK is imported if available at runtime
ensure_azure_speech()
if AZURE_AVAILABLE and is_usable_azure_speech_key(SPEECH_KEY):
    SPEECH_CONFIG = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY, region=SPEECH_REGION
    )
    SPEECH_CONFIG.speech_recognition_language = "en-US"
else:
    SPEECH_CONFIG = None
    if AZURE_AVAILABLE:
        logger.info("Azure Speech credentials are not configured. Pronunciation assessment is disabled.")

VOICE_PIPELINE_CONTROLLER = SpeechPipelineController()


def assess_pronunciation(
    audio_bytes: bytes, reference_text: Optional[str] = None
) -> dict:
    """
    Assess pronunciation using Azure Speech Service with Pronunciation Assessment.

    Args:
        audio_bytes: PCM 16-bit 16kHz audio data
        reference_text: The text the user should read (for pronunciation assessment)

    Returns:
        Dictionary with transcription, pronunciation scores, and word-level feedback
    """
    ensure_azure_speech()
    if not AZURE_AVAILABLE:
        return {"status": "error", "error": "Azure Speech SDK not available"}
    if not SPEECH_CONFIG:
        return {
            "status": "error",
            "error": "Azure Speech credentials are not configured",
        }

    try:
        # Create a push audio input stream
        push_stream = speechsdk.audio.PushAudioInputStream()
        push_stream.write(audio_bytes)
        push_stream.close()

        # Create audio configuration
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        # Initialize pronunciation assessment config
        if reference_text:
            # Full pronunciation assessment with reference text
            assessment_config = PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=PronunciationAssessmentGradingSystem.HundredMark,
                granularity=PronunciationAssessmentGranularity.Word,
            )
            assessment_config.enable_miscue(True)  # Detect mispronounced words
        else:
            # Speech recognition only (no assessment)
            assessment_config = None

        # Create speech recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=SPEECH_CONFIG, audio_config=audio_config
        )

        # Apply pronunciation assessment if configured
        if assessment_config:
            assessment_config.apply_on(recognizer)

        # Recognize speech
        logger.info("Starting speech recognition...")
        result = recognizer.recognize_once()

        # Process the result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            logger.info(f"Recognized text: {result.text}")

            response = {
                "status": "success",
                "text": result.text,
                "reference_text": reference_text,
            }

            # Parse pronunciation assessment if available
            if assessment_config and result.properties:
                try:
                    json_result = result.properties.get(
                        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                    )
                    if json_result:
                        pronunciation_result = json.loads(json_result)
                        logger.info(f"Pronunciation Assessment: {pronunciation_result}")

                        # Extract overall scores
                        nbest = pronunciation_result.get("NBest", [{}])[0]
                        response["overall_scores"] = {
                            "accuracy": nbest.get("PronunciationAssessment", {}).get(
                                "AccuracyScore", 0
                            ),
                            "fluency": nbest.get("PronunciationAssessment", {}).get(
                                "FluencyScore", 0
                            ),
                            "completeness": nbest.get(
                                "PronunciationAssessment", {}
                            ).get("CompletenessScore", 0),
                            "prosody": nbest.get("PronunciationAssessment", {}).get(
                                "ProsodyScore", 0
                            ),
                        }

                        # Extract word-level feedback
                        words = []
                        for word_info in nbest.get("Words", []):
                            word_assessment = word_info.get(
                                "PronunciationAssessment", {}
                            )
                            words.append(
                                {
                                    "word": word_info.get("Word", ""),
                                    "accuracy": word_assessment.get("AccuracyScore", 0),
                                    "error_type": word_assessment.get(
                                        "ErrorType", "None"
                                    ),
                                }
                            )

                        response["words"] = words
                except Exception as e:
                    logger.warning(f"Error parsing pronunciation assessment: {e}")
            else:
                # Basic confidence score from recognition
                response["overall_scores"] = {
                    "accuracy": 0,
                    "fluency": 0,
                    "completeness": 0,
                    "prosody": 0,
                }
                response["words"] = []

            return response

        elif result.reason == speechsdk.ResultReason.NoMatch:
            logger.warning("No speech detected")
            return {
                "status": "no_match",
                "text": None,
                "error": "No speech detected. Please speak clearly.",
                "overall_scores": {
                    "accuracy": 0,
                    "fluency": 0,
                    "completeness": 0,
                    "prosody": 0,
                },
                "words": [],
            }

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_message = f"Error: {cancellation.reason}"
            if cancellation.error_details:
                error_message += f" - {cancellation.error_details}"
            logger.error(error_message)
            return {
                "status": "error",
                "text": None,
                "error": error_message,
                "overall_scores": {
                    "accuracy": 0,
                    "fluency": 0,
                    "completeness": 0,
                    "prosody": 0,
                },
                "words": [],
            }

    except Exception as e:
        error_message = f"Speech assessment error: {str(e)}"
        logger.error(error_message)
        return {
            "status": "error",
            "text": None,
            "error": error_message,
            "overall_scores": {
                "accuracy": 0,
                "fluency": 0,
                "completeness": 0,
                "prosody": 0,
            },
            "words": [],
        }


@app.websocket("/ws/speech")
async def speech_endpoint(websocket: WebSocket):
    """WebSocket endpoint for the voice controller pipeline."""
    await websocket.accept()
    logger.info("Client connected to speech endpoint")

    try:
        await websocket.send_json(
            {
                "status": "ready",
                "pipeline": {
                    "stages": ["audio_in", "asr", "processing", "tts", "audio_out"],
                    "supports_audio_output": VOICE_PIPELINE_CONTROLLER.tts_engine.is_available(),
                    "processor_uses_gemini": bool(resolve_gemini_api_key(os.getenv)),
                },
            }
        )

        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send_json(
                    {"status": "error", "error": "Invalid JSON payload."}
                )
                continue

            message_type = str(message.get("type") or "").strip().lower()
            history = message.get("history")
            history_payload = history if isinstance(history, list) else None

            raw_context_hint = (
                message.get("context_hint")
                or message.get("context")
                or message.get("prompt")
            )
            context_hint = (
                raw_context_hint.strip()
                if isinstance(raw_context_hint, str) and raw_context_hint.strip()
                else None
            )
            voice_name = (
                message.get("voice_name")
                if isinstance(message.get("voice_name"), str)
                else None
            )
            user_dialect = (
                message.get("user_dialect")
                if isinstance(message.get("user_dialect"), str)
                else None
            )
            fallback_transcript = (
                message.get("fallback_transcript")
                if isinstance(message.get("fallback_transcript"), str)
                else None
            )
            synthesize = message.get("synthesize", True) is not False

            try:
                if message_type == "audio":
                    audio_data = message.get("audio")
                    if not isinstance(audio_data, str) or not audio_data.strip():
                        await websocket.send_json(
                            {
                                "status": "error",
                                "error": "Audio messages must include a base64 audio payload in 'audio'.",
                            }
                        )
                        continue

                    await websocket.send_json(
                        {"status": "processing", "stage": "asr"}
                    )
                    result = await asyncio.to_thread(
                        VOICE_PIPELINE_CONTROLLER.process_audio_turn,
                        encoded_audio=audio_data,
                        mime_type=(
                            message.get("mime_type")
                            if isinstance(message.get("mime_type"), str)
                            else "audio/webm"
                        ),
                        context_hint=context_hint,
                        history=history_payload,
                        synthesize=synthesize,
                        voice_name=voice_name,
                        fallback_transcript=fallback_transcript,
                        user_dialect=user_dialect,
                    )

                    await websocket.send_json(
                        {
                            "status": "response",
                            "transcript": result["transcript"],
                            "text": result["reply_text"],
                            "audio": result["audio_base64"],
                            "audio_mime_type": result["audio_mime_type"],
                            "pipeline": result["pipeline"],
                        }
                    )
                elif message_type == "text":
                    text = message.get("text")
                    if not isinstance(text, str) or not text.strip():
                        await websocket.send_json(
                            {
                                "status": "error",
                                "error": "Text messages must include a non-empty 'text' value.",
                            }
                        )
                        continue

                    await websocket.send_json(
                        {"status": "processing", "stage": "processing"}
                    )
                    result = await asyncio.to_thread(
                        VOICE_PIPELINE_CONTROLLER.process_text_turn,
                        text=text,
                        context_hint=context_hint,
                        history=history_payload,
                        synthesize=synthesize,
                        voice_name=voice_name,
                    )

                    await websocket.send_json(
                        {
                            "status": "response",
                            "transcript": result["transcript"],
                            "text": result["reply_text"],
                            "audio": result["audio_base64"],
                            "audio_mime_type": result["audio_mime_type"],
                            "pipeline": result["pipeline"],
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "status": "error",
                            "error": "Unsupported message type. Use 'audio' or 'text'.",
                        }
                    )
            except SpeechPipelineError as exc:
                logger.warning("Speech pipeline error: %s", exc)
                await websocket.send_json({"status": "error", "error": str(exc)})

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass


from fastapi.responses import RedirectResponse


@app.get("/health", include_in_schema=False)
async def health():
    database_status = {
        "status": "connected",
        "detail": "Primary database connection is healthy.",
    }
    supabase_client = get_supabase_client()
    supabase_status = {
        "status": "connected" if supabase_client.is_available else "error",
        "detail": (
            "Supabase storage and admin APIs are configured."
            if supabase_client.is_available
            else getattr(
                supabase_client,
                "status_detail",
                "Supabase storage is not configured.",
            )
        ),
    }
    http_status = 200

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        database_status = {
            "status": "error",
            "detail": str(exc),
        }
        http_status = 503
    finally:
        db.close()

    if not supabase_client.is_available:
        http_status = 503

    return JSONResponse(
        status_code=http_status,
        content={
            "status": "ok" if http_status == 200 else "degraded",
            "services": {
                "database": database_status,
                "supabase": supabase_status,
            },
            "urls": {
                "backend_url": normalize_env_value(os.getenv("BACKEND_URL")),
                "frontend_url": normalize_env_value(os.getenv("FRONTEND_URL")),
            },
            "render": {
                "enabled": normalize_env_value(os.getenv("RENDER")).lower() in {"1", "true", "yes", "on"},
                "external_url": normalize_env_value(os.getenv("RENDER_EXTERNAL_URL")),
                "service_name": normalize_env_value(os.getenv("RENDER_SERVICE_NAME")),
            },
        },
    )


@app.get("/")
async def root():
    # when the backend is browsed directly, redirect to the frontend
    # development server if available; otherwise return basic JSON.
    frontend_url = _normalize_origin(os.getenv("FRONTEND_URL"))
    if frontend_url:
        return RedirectResponse(url=frontend_url)
    return {
        "message": "Speech-Enabled BPO Platform Backend",
        "version": "2.0.0",
        "features": [
            "speech-recognition",
            "pronunciation-assessment",
            "word-level-scoring",
        ],
        "endpoints": {
            "webSocket": "/ws/speech",
            "docs": "/docs",
        },
    }
