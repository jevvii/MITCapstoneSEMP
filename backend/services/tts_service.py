"""
TTS Service for Call Simulation
Provides text-to-speech functionality for AI member responses.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config_validation import is_usable_azure_speech_key, normalize_env_value
from .gemini_tts import GeminiTextToSpeechEngine

# Lazy-load Azure Speech and OpenAI SDKs at runtime to reduce startup cost
_azure_speech = None
_openai = None

def _ensure_azure_speech():
    global _azure_speech
    if _azure_speech is not None:
        return _azure_speech
    try:
        import importlib

        _azure_speech = importlib.import_module('azure.cognitiveservices.speech')
        return _azure_speech
    except Exception as exc:
        logger.info('Azure Speech SDK not available: %s', exc)
        _azure_speech = None
        return None

def _ensure_openai():
    global _openai
    if _openai is not None:
        return _openai
    try:
        import importlib

        mod = importlib.import_module('openai')
        _openai = getattr(mod, 'OpenAI', mod)
        return _openai
    except Exception as exc:
        logger.info('OpenAI SDK not available: %s', exc)
        _openai = None
        return None

logger = logging.getLogger(__name__)
OPENAI_TTS_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}


def _env_flag(name: str, default: bool = False) -> bool:
    normalized = normalize_env_value(os.getenv(name)).lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _default_local_tts_enabled() -> bool:
    normalized = normalize_env_value(os.getenv("ENABLE_LOCAL_TTS")).lower()
    if normalized in {"0", "false", "no", "off"}:
        return False

    # Offline local fallback is meant for local desktop development, not Render/Linux containers.
    is_render = normalize_env_value(os.getenv("RENDER")).lower() in {"1", "true", "yes", "on"}
    is_docker = os.path.exists("/.dockerenv")
    if is_render or is_docker or os.name != "nt":
        if normalized in {"1", "true", "yes", "on"}:
            logger.info(
                "Ignoring ENABLE_LOCAL_TTS because offline server-side fallback is only supported in local Windows development."
            )
        return False

    # Default to enabled for local Windows development so trainer-generated member audio
    # still works when Gemini/Azure/OpenAI speech is unavailable.
    return True


class TTSService:
    """Text-to-Speech service for call simulation."""

    def __init__(self) -> None:
        self.gemini_tts = GeminiTextToSpeechEngine()
        self.azure_speech_key = normalize_env_value(os.getenv("AZURE_SPEECH_KEY"))
        self.azure_speech_region = normalize_env_value(os.getenv("AZURE_SPEECH_REGION")) or "eastus"
        self.azure_voice_name = normalize_env_value(os.getenv("AZURE_TTS_VOICE")) or "en-US-JennyNeural"
        self.openai_api_key = normalize_env_value(os.getenv("OPENAI_API_KEY"))
        self.openai_tts_model = normalize_env_value(os.getenv("OPENAI_TTS_MODEL")) or "gpt-4o-mini-tts"
        self.openai_voice_name = normalize_env_value(os.getenv("OPENAI_TTS_VOICE")) or "marin"
        self.openai_client = None
        OpenAI = _ensure_openai()
        if OpenAI and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI TTS client: %s", exc)
                self.openai_client = None
        self.enable_local_tts = _default_local_tts_enabled()
        if not self.enable_local_tts:
            # If no cloud TTS providers are available (e.g., Gemini auth failure)
            # enable local Windows fallback automatically for local development so
            # trainers can still generate and persist audio for call simulation.
            if (
                os.name == "nt"
                and not self.gemini_tts.is_available()
                and not self._azure_tts_available()
                and not self._openai_tts_available()
            ):
                logger.info(
                    "No cloud TTS providers available; enabling local Windows TTS fallback for persistence."
                )
                self.enable_local_tts = True
            else:
                logger.info(
                    "Local server-side TTS fallback is disabled. Browser fallback will be used when Gemini, Azure, and OpenAI audio are unavailable."
                )

    def _gtts_available(self) -> bool:
        try:
            from gtts import gTTS  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return (
            self.gemini_tts.is_available()
            or self._azure_tts_available()
            or self._openai_tts_available()
            or self._gtts_available()
            or self._local_tts_available()
        )

    def _synthesize_gtts(
        self,
        text: str,
        language_code: str = "en-US",
    ) -> Optional[dict[str, Any]]:
        """Synthesize speech using gTTS (Google Translate TTS web API)"""
        try:
            from gtts import gTTS

            lang = language_code.split("-")[0].lower() if language_code else "en"
            tts = gTTS(text=text.strip(), lang=lang, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            audio_bytes = buffer.getvalue()

            if not audio_bytes or len(audio_bytes) < 64:
                raise RuntimeError("gTTS generated empty or invalid audio data.")

            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            words = len((text or "").split())
            duration = max(1.0, (words / 150.0) * 60.0) if words > 0 else 0.0

            logger.info("Successfully generated gTTS call simulation audio: %d bytes, %.2fs", len(audio_bytes), duration)
            return {
                "audio_url": None,
                "audio_base64": audio_base64,
                "audio_bytes": audio_bytes,
                "audio_content_type": "audio/mpeg",
                "audio_extension": "mp3",
                "duration": duration,
                "provider": "gtts",
                "error": None,
            }
        except Exception as e:
            logger.warning("gTTS speech synthesis failed: %s", e)
            return None

    def _azure_tts_available(self) -> bool:
        # Determine availability by attempting to load the Azure Speech module at runtime
        try:
            azure = _ensure_azure_speech()
        except Exception:
            azure = None
        return bool(
            azure
            and is_usable_azure_speech_key(self.azure_speech_key)
            and self.azure_speech_region
        )

    def _openai_tts_available(self) -> bool:
        # OpenAI availability is determined by the local _openai loader and initialized client
        try:
            from . import tts_service  # ensure module context
        except Exception:
            pass
        return bool(_openai and self.openai_client and self.openai_api_key)

    def _resolve_openai_voice_name(self, voice_name: Optional[str]) -> str:
        requested_voice = normalize_env_value(voice_name)
        if requested_voice in OPENAI_TTS_VOICES:
            return requested_voice

        configured_voice = normalize_env_value(self.openai_voice_name)
        if configured_voice in OPENAI_TTS_VOICES:
            return configured_voice

        return "marin"

    def _windows_sapi_available(self) -> bool:
        return self.enable_local_tts and os.name == "nt"

    def _local_tts_available(self) -> bool:
        if not self.enable_local_tts:
            return False
        if self._windows_sapi_available():
            return True
        try:
            import pyttsx3  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _result_has_audio_payload(result: Optional[dict[str, Any]]) -> bool:
        if not result:
            return False

        audio_bytes = result.get("audio_bytes")
        if isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) > 0:
            return True

        audio_base64 = result.get("audio_base64")
        return isinstance(audio_base64, str) and bool(audio_base64.strip())

    async def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        speaking_style: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Convert text to speech.

        Args:
            text: The text to convert to speech
            voice_name: Optional voice name
            speaking_style: Optional speaking style (e.g., 'professional', 'friendly')

        Returns:
            Dictionary with audio_url, audio_base64, and duration
        """
        if not text:
            return {
                "audio_url": None,
                "audio_base64": None,
                "audio_bytes": None,
                "audio_content_type": "audio/wav",
                "audio_extension": "wav",
                "duration": 0,
                "provider": None,
                "error": "Text is required.",
            }

        # Try Gemini TTS first
        gemini_error: Optional[str] = None
        if self.gemini_tts.is_available():
            try:
                audio_bytes = self.gemini_tts.synthesize(
                    text=text,
                    voice_name=voice_name or "Puck",
                    speaking_style=speaking_style or "professional",
                )

                if audio_bytes:
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    return {
                        "audio_url": None,
                        "audio_base64": audio_base64,
                        "audio_bytes": audio_bytes,
                        "audio_content_type": "audio/wav",
                        "audio_extension": "wav",
                        "duration": len(audio_bytes) / 16000,  # Approximate duration
                        "provider": "gemini",
                        "error": None,
                    }
            except Exception as e:
                logger.warning("Gemini TTS failed. Browser fallback may be used: %s", e)
                gemini_error = str(e)

            if not gemini_error:
                gemini_error = self.gemini_tts.last_error or "Gemini TTS returned no audio."

        # Try Azure TTS next when it is configured.
        azure_error: Optional[str] = None
        if self._azure_tts_available():
            try:
                azure_result = self._synthesize_azure(
                    text,
                    voice_name=voice_name or self.azure_voice_name,
                )
                if azure_result:
                    return azure_result
            except Exception as e:
                logger.warning("Azure TTS failed. Browser fallback may be used: %s", e)
                azure_error = str(e)

            if not azure_error:
                azure_error = "Azure TTS returned no audio."

        # Try OpenAI TTS next when it is configured.
        openai_error: Optional[str] = None
        if self._openai_tts_available():
            try:
                openai_result = self._synthesize_openai(
                    text,
                    voice_name=voice_name,
                    speaking_style=speaking_style,
                )
                if openai_result:
                    return openai_result
            except Exception as e:
                logger.warning("OpenAI TTS failed. Browser fallback may be used: %s", e)
                openai_error = str(e)

            if not openai_error:
                openai_error = "OpenAI TTS returned no audio."

        # Try gTTS next (works in all environments including Linux/Render without API keys)
        gtts_error: Optional[str] = None
        if self._gtts_available():
            try:
                gtts_result = self._synthesize_gtts(text)
                if gtts_result and gtts_result.get("audio_bytes"):
                    return gtts_result
            except Exception as e:
                logger.warning("gTTS synthesis failed: %s", e)
                gtts_error = str(e)

        if not self.enable_local_tts:
            provider_errors = [error for error in [gemini_error, azure_error, openai_error, gtts_error] if error]
            fallback_instructions = (
                "Server-side TTS is unavailable. Configure GOOGLE_API_KEY or GEMINI_API_KEY, OPENAI_API_KEY, "
                "or AZURE_SPEECH_KEY/AZURE_SPEECH_REGION for deployed speech generation."
            )
            if os.name == "nt":
                fallback_instructions += " On Windows, set ENABLE_LOCAL_TTS=1 to enable local Windows TTS fallback."
            return {
                "audio_url": None,
                "audio_base64": None,
                "audio_bytes": None,
                "audio_content_type": "audio/wav",
                "audio_extension": "wav",
                "duration": 0,
                "provider": "browser_fallback",
                "error": ". ".join(provider_errors)
                if provider_errors
                else fallback_instructions,
                "fallback_mode": "browser",
            }

        # Fallback to native Windows SAPI or pyttsx3 only for local development.
        fallback_result = await self._fallback_tts(text)
        if fallback_result.get("audio_bytes") or fallback_result.get("audio_base64"):
            if not fallback_result.get("error"):
                fallback_result["error"] = " | ".join(
                    error
                    for error in [gemini_error, azure_error, openai_error]
                    if error
                ) or None
            return fallback_result

        fallback_error = str(fallback_result.get("error") or "").strip()
        provider_errors = [error for error in [gemini_error, azure_error, openai_error] if error]
        if provider_errors and fallback_error:
            fallback_result["error"] = f"{'. '.join(provider_errors)}. Fallback TTS failed: {fallback_error}"
        elif provider_errors:
            fallback_result["error"] = ". ".join(provider_errors)
        return fallback_result

    async def synthesize_for_persistence(
        self,
        text: str,
        voice_name: Optional[str] = None,
        speaking_style: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate audio that is intended to be saved server-side.

        This retries the local/offline fallback one more time before giving up so
        trainer flows can still persist generated speech when the network TTS
        providers are temporarily unavailable.
        """
        try:
            result = await self.synthesize(text, voice_name, speaking_style)
        except Exception as exc:
            logger.warning("Primary persistable TTS synthesis failed: %s", exc)
            result = {
                "audio_url": None,
                "audio_base64": None,
                "audio_bytes": None,
                "audio_content_type": "audio/wav",
                "audio_extension": "wav",
                "duration": 0,
                "provider": None,
                "error": str(exc),
                "fallback_mode": "browser",
            }

        if self._result_has_audio_payload(result):
            return result

        if not self._local_tts_available():
            return result

        try:
            fallback_result = await self._fallback_tts(text)
        except Exception as exc:
            logger.warning("Persistable local TTS fallback failed: %s", exc)
            return result

        if not self._result_has_audio_payload(fallback_result):
            primary_error = str(result.get("error") or "").strip()
            fallback_error = str(fallback_result.get("error") or "").strip()
            if primary_error and fallback_error:
                fallback_result["error"] = f"{primary_error}. Fallback TTS failed: {fallback_error}"
            elif primary_error:
                fallback_result["error"] = primary_error
            return fallback_result

        primary_error = str(result.get("error") or "").strip()
        fallback_error = str(fallback_result.get("error") or "").strip()
        combined_error = " | ".join(part for part in [primary_error, fallback_error] if part) or None
        fallback_result["error"] = combined_error
        return fallback_result

    def _synthesize_azure(
        self,
        text: str,
        *,
        voice_name: str,
    ) -> Optional[dict[str, Any]]:
        if not self._azure_tts_available() or not text.strip():
            return None

        azure = _ensure_azure_speech()
        if not azure:
            logger.info("Azure Speech SDK not available at runtime for TTS.")
            return None

        speech_config = azure.SpeechConfig(
            subscription=self.azure_speech_key,
            region=self.azure_speech_region,
        )
        speech_config.speech_synthesis_voice_name = voice_name
        speech_config.set_speech_synthesis_output_format(
            azure.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )
        synthesizer = azure.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        result = synthesizer.speak_text_async(text).get()

        if result.reason != azure.ResultReason.SynthesizingAudioCompleted:
            details = getattr(result, "cancellation_details", None)
            detail_text = getattr(details, "error_details", None) or getattr(details, "reason", None)
            raise RuntimeError(
                f"Azure TTS failed to synthesize audio. {detail_text or ''}".strip()
            )

        audio_bytes = bytes(result.audio_data or b"")
        if not audio_bytes:
            raise RuntimeError("Azure TTS completed without returning audio data.")

        return {
            "audio_url": None,
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "audio_bytes": audio_bytes,
            "audio_content_type": "audio/wav",
            "audio_extension": "wav",
            "duration": len(audio_bytes) / 32000,
            "provider": "azure_speech",
            "error": None,
        }

    def _synthesize_openai(
        self,
        text: str,
        *,
        voice_name: Optional[str],
        speaking_style: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if not self._openai_tts_available() or not text.strip():
            return None

        resolved_voice = self._resolve_openai_voice_name(voice_name)
        instructions = (
            speaking_style.strip()
            if isinstance(speaking_style, str) and speaking_style.strip()
            else "Speak in a professional and natural customer-service tone."
        )
        response = self.openai_client.audio.speech.create(
            model=self.openai_tts_model,
            voice=resolved_voice,
            input=text,
            instructions=instructions,
            response_format="wav",
        )
        audio_bytes = response.read()
        if not audio_bytes:
            raise RuntimeError("OpenAI TTS completed without returning audio data.")

        return {
            "audio_url": None,
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "audio_bytes": audio_bytes,
            "audio_content_type": "audio/wav",
            "audio_extension": "wav",
            "duration": len(audio_bytes) / 32000,
            "provider": "openai_tts",
            "error": None,
        }

    def _synthesize_windows_sapi(self, text: str) -> Optional[dict[str, Any]]:
        if not self._windows_sapi_available() or not text.strip():
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        encoded_path = base64.b64encode(temp_path.encode("utf-8")).decode("ascii")
        script = f"""
$ErrorActionPreference = 'Stop'
$text = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_text}'))
$outputPath = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_path}'))
$voice = New-Object -ComObject SAPI.SpVoice
$stream = New-Object -ComObject SAPI.SpFileStream
$stream.Open($outputPath, 3, $false)
try {{
    $voice.AudioOutputStream = $stream
    [void]$voice.Speak($text)
}} finally {{
    try {{ $stream.Close() }} catch {{}}
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($stream) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($voice) | Out-Null
}}
"""

        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(stderr or "Windows SAPI speech synthesis failed.")

            with open(temp_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            if len(audio_bytes) <= 64:
                raise RuntimeError("Windows SAPI created an empty or invalid audio file.")

            return {
                "audio_url": None,
                "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                "audio_bytes": audio_bytes,
                "audio_content_type": "audio/wav",
                "audio_extension": "wav",
                "duration": len(audio_bytes) / 32000,
                "provider": "windows_sapi",
                "error": None,
            }
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                logger.warning("Unable to remove temporary Windows SAPI audio file: %s", temp_path)

    async def _fallback_tts(self, text: str) -> dict[str, Any]:
        """Fallback TTS using native Windows SAPI, gTTS, then pyttsx3."""
        if self._windows_sapi_available():
            try:
                windows_result = self._synthesize_windows_sapi(text)
                if windows_result and windows_result.get("audio_bytes"):
                    return windows_result
            except Exception as exc:
                logger.warning("Windows SAPI fallback TTS failed: %s", exc)

        if self._gtts_available():
            try:
                gtts_result = self._synthesize_gtts(text)
                if gtts_result and gtts_result.get("audio_bytes"):
                    return gtts_result
            except Exception as exc:
                logger.warning("gTTS fallback TTS failed: %s", exc)

        if not self.enable_local_tts:
            return {
                "audio_url": None,
                "audio_base64": None,
                "audio_bytes": None,
                "audio_content_type": "audio/wav",
                "audio_extension": "wav",
                "duration": 0,
                "provider": None,
                "error": "Local server-side TTS fallback is disabled.",
            }

        windows_error: Optional[str] = None
        try:
            windows_result = self._synthesize_windows_sapi(text)
            if windows_result:
                return windows_result
        except Exception as exc:
            windows_error = str(exc)
            logger.warning("Windows SAPI fallback TTS failed: %s", exc)

        try:
            import pyttsx3

            engine = pyttsx3.init()

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name

            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            # Read the file
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()

            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Clean up
            os.remove(temp_path)

            return {
                "audio_url": None,
                "audio_base64": audio_base64,
                "audio_bytes": audio_bytes,
                "audio_content_type": "audio/wav",
                "audio_extension": "wav",
                "duration": len(audio_bytes) / 16000,
                "provider": "pyttsx3",
                "error": windows_error,
            }

        except Exception as e:
            logger.warning("Local pyttsx3 fallback TTS failed: %s", e)
            error_parts = [part for part in [windows_error, str(e)] if part]
            return {
                "audio_url": None,
                "audio_base64": None,
                "audio_bytes": None,
                "audio_content_type": "audio/wav",
                "audio_extension": "wav",
                "duration": 0,
                "provider": None,
                "error": " | ".join(error_parts) if error_parts else str(e),
            }

    def save_audio_locally(
        self,
        audio_bytes: bytes,
        scenario_id: Optional[str] = None,
        step_number: Optional[int] = None,
        asset_kind: str = "member-step",
    ) -> Optional[str]:
        """
        Save TTS audio to local folder for backup/archive purposes.

        Args:
            audio_bytes: The audio data to save
            scenario_id: Optional scenario ID for organization
            step_number: Optional step number for organization
            asset_kind: Type of asset (member-step, ringer, hold, etc.)

        Returns:
            Local file path if successful, None otherwise
        """
        try:
            project_root = Path(__file__).resolve().parents[2]
            base_dir = project_root / "media" / "tts-audio"
            base_dir.mkdir(parents=True, exist_ok=True)

            if scenario_id:
                safe_scenario_id = "".join(
                    character if character.isalnum() or character in {"-", "_"} else "-"
                    for character in str(scenario_id)
                ).strip("-_") or "draft"
                scenario_dir = base_dir / "scenarios" / safe_scenario_id
                scenario_dir.mkdir(parents=True, exist_ok=True)
            else:
                scenario_dir = base_dir / "global"
                scenario_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            step_prefix = f"step-{int(step_number):02d}_" if step_number else ""
            safe_asset_kind = "".join(
                character if character.isalnum() or character in {"-", "_"} else "-"
                for character in str(asset_kind or "member-step")
            ).strip("-_") or "member-step"
            filename = f"{step_prefix}{timestamp}_{safe_asset_kind}.wav"

            file_path = scenario_dir / filename

            # Write audio to file
            file_path.write_bytes(audio_bytes)
            logger.info(
                "TTS audio saved locally: %s (size=%d bytes)",
                file_path,
                len(audio_bytes)
            )

            return str(file_path)

        except Exception as e:
            logger.warning("Failed to save TTS audio locally: %s", e)
            return None


# Singleton instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get the singleton TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


async def text_to_speech(
    text: str,
    voice_name: Optional[str] = None,
    speaking_style: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert text to speech.

    Args:
        text: The text to convert
        voice_name: Optional voice name
        speaking_style: Optional speaking style

    Returns:
        Dictionary with audio data
    """
    service = get_tts_service()
    return await service.synthesize(text, voice_name, speaking_style)


async def text_to_speech_for_persistence(
    text: str,
    voice_name: Optional[str] = None,
    speaking_style: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert text to speech with an extra local/offline retry suitable for assets
    that must be persisted to storage.
    """
    service = get_tts_service()
    return await service.synthesize_for_persistence(text, voice_name, speaking_style)
