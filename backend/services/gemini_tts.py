"""
Gemini Text-to-Speech service for generating audio from text.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import wave
from typing import Any, Optional

# Lazy-load google.genai to avoid importing heavy SDK at process startup
_genai = None
_genai_types = None
GEMINI_TTS_AVAILABLE = False

def _ensure_genai():
    global _genai, _genai_types, GEMINI_TTS_AVAILABLE
    if _genai is not None or _genai_types is not None:
        return _genai, _genai_types
    try:
        import importlib

        _genai = importlib.import_module('google.genai')
        _genai_types = importlib.import_module('google.genai').types
        GEMINI_TTS_AVAILABLE = True
        return _genai, _genai_types
    except Exception as exc:
        # Ensure a logger is available even if module-level logger wasn't defined earlier
        try:
            _logger = logging.getLogger(__name__)
            _logger.info('Gemini genai not available: %s', exc)
        except Exception:
            pass
        _genai = None
        _genai_types = None
        GEMINI_TTS_AVAILABLE = False
        return None, None

from ..config_validation import normalize_env_value, resolve_gemini_api_key

logger = logging.getLogger(__name__)
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
FALLBACK_GEMINI_TTS_MODELS = (
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro-preview-tts",
)


class GeminiTextToSpeechEngine:
    """Gemini-powered TTS engine for generating audio responses."""

    def __init__(self) -> None:
        self.api_key = resolve_gemini_api_key(os.getenv)
        self.model_name = normalize_env_value(os.getenv("GEMINI_TTS_MODEL")) or DEFAULT_GEMINI_TTS_MODEL
        self.client = None
        self.last_error: Optional[str] = None
        self.disabled_reason: Optional[str] = None
        genai, _types = _ensure_genai()
        if genai and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                logger.warning("Failed to initialize Gemini TTS client: %s", exc)
                self.last_error = str(exc)
                self.client = None

    def is_available(self) -> bool:
        return bool(GEMINI_TTS_AVAILABLE and self.client and self.api_key and not self.disabled_reason)

    def synthesize(
        self,
        text: str,
        *,
        voice_name: Optional[str] = None,
        speaking_style: Optional[str] = None,
        multi_speaker_config: Optional[list] = None
    ) -> Optional[bytes]:
        """
        Generate speech audio from text using Gemini TTS.

        Args:
            text: The text to convert to speech
            voice_name: Voice to use (e.g., 'Kore', 'Puck', etc.) - for single speaker
            speaking_style: Speaking style instruction (e.g., 'cheerfully')
            multi_speaker_config: List of speaker configs for multi-speaker conversations
                Each config should be a dict with 'speaker' and 'voice_config' keys

        Returns:
            WAV audio bytes if successful, None otherwise
        """
        if not self.is_available() or not text.strip():
            return None

        try:
            self.last_error = None
            # Format the text with speaking style if provided
            content_text = text.strip()
            if speaking_style:
                content_text = f"Say {speaking_style}: {content_text}"

            config_kwargs = {}

            if multi_speaker_config:
                # Multi-speaker configuration
                genai, types = _ensure_genai()
                if not types:
                    logger.warning('Gemini types unavailable; cannot configure multi-speaker.')
                    return None
                speaker_configs = []
                for speaker_config in multi_speaker_config:
                    speaker_configs.append(
                        types.SpeakerVoiceConfig(
                            speaker=speaker_config['speaker'],
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=speaker_config['voice_name']
                                )
                            )
                        )
                    )

                config_kwargs['speech_config'] = types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=speaker_configs
                    )
                )
            else:
                # Single speaker configuration
                genai, types = _ensure_genai()
                if not types:
                    logger.warning('Gemini types unavailable; cannot configure voice.')
                    return None
                config_kwargs['speech_config'] = types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name or "Kore",
                        )
                    ),
                )

            for model_name in self._candidate_models():
                try:
                    genai, types = _ensure_genai()
                    if not genai or not types:
                        self.last_error = 'Gemini genai SDK not available at runtime.'
                        return None

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=content_text,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            **config_kwargs
                        )
                    )
                except Exception as exc:
                    self.last_error = str(exc)
                    if self._is_permanent_auth_error(exc):
                        self.disabled_reason = self.last_error
                        logger.error(
                            "Disabling Gemini TTS for the current process after a permanent authentication error: %s",
                            exc,
                        )
                        return None
                    if self._is_model_selection_error(exc):
                        logger.warning(
                            "Gemini TTS model %s was rejected. Trying the next configured model: %s",
                            model_name,
                            exc,
                        )
                        continue
                    logger.warning("Gemini TTS synthesis failed with model %s: %s", model_name, exc)
                    return None

                audio_payload = self._extract_audio_payload(response)
                if audio_payload:
                    audio_bytes, mime_type = audio_payload
                    normalized_mime_type = (mime_type or "").lower()
                    if audio_bytes[:4] == b"RIFF" or "wav" in normalized_mime_type or "wave" in normalized_mime_type:
                        return audio_bytes
                    if "mpeg" in normalized_mime_type or "mp3" in normalized_mime_type or "ogg" in normalized_mime_type:
                        return audio_bytes
                    return self._create_wav_file(audio_bytes)

                self.last_error = f"Gemini TTS model {model_name} returned no audio."
                logger.warning("Gemini TTS response from model %s did not include any inline audio payload.", model_name)

        except Exception as exc:
            self.last_error = str(exc)
            if self._is_permanent_auth_error(exc):
                self.disabled_reason = self.last_error
                logger.error(
                    "Disabling Gemini TTS for the current process after a permanent authentication error: %s",
                    exc,
                )
            logger.warning("Gemini TTS synthesis failed: %s", exc)
            return None

        return None

    def _candidate_models(self) -> list[str]:
        ordered_models = [self.model_name, DEFAULT_GEMINI_TTS_MODEL, *FALLBACK_GEMINI_TTS_MODELS]
        unique_models: list[str] = []
        for model_name in ordered_models:
            normalized_name = normalize_env_value(model_name)
            if normalized_name and normalized_name not in unique_models:
                unique_models.append(normalized_name)
        return unique_models

    def _is_permanent_auth_error(self, exc: Exception) -> bool:
        message = str(exc or "").lower()
        return (
            "permission_denied" in message
            or "reported as leaked" in message
            or ("api key" in message and "leak" in message)
            or ("api key" in message and "invalid" in message)
        )

    def _is_model_selection_error(self, exc: Exception) -> bool:
        message = str(exc or "").lower()
        return (
            "model" in message
            and (
                "not found" in message
                or "unsupported" in message
                or "not supported" in message
                or "unknown" in message
                or "does not exist" in message
            )
        )

    def _extract_audio_payload(self, response: Any) -> Optional[tuple[bytes, Optional[str]]]:
        """Scan Gemini candidates/parts until an inline audio payload is found."""
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if not inline_data:
                    continue
                audio_bytes = self._coerce_audio_bytes(getattr(inline_data, "data", None))
                if audio_bytes:
                    return audio_bytes, getattr(inline_data, "mime_type", None)
        return None

    def _coerce_audio_bytes(self, payload: Any) -> Optional[bytes]:
        """Normalize Gemini inline audio into raw bytes."""
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
        if isinstance(payload, str):
            try:
                return base64.b64decode(payload, validate=True)
            except Exception:
                logger.warning("Gemini TTS returned string audio data that could not be base64-decoded.")
                return None
        return None

    def _create_wav_file(self, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> bytes:
        """Create a WAV file from PCM audio data."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)
        return buffer.getvalue()


# Global instance for easy access
gemini_tts_engine = GeminiTextToSpeechEngine()
