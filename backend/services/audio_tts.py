"""
Text-to-Speech service for accessibility features.
Converts audio transcripts to speech for trainees who prefer listening.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

from ..config_validation import (
    is_usable_azure_speech_key,
    normalize_env_value,
    resolve_gemini_api_key,
)

logger = logging.getLogger(__name__)
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
FALLBACK_GEMINI_TTS_MODELS = (
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro-preview-tts",
)


def _load_gtts():
    try:
        from gtts import gTTS
        return gTTS
    except Exception:
        return None


def _estimate_spoken_duration_seconds(text: str, words_per_minute: int = 150) -> float:
    words = len((text or "").split())
    if words <= 0:
        return 0.0
    return max(1.0, (words / max(1, words_per_minute)) * 60.0)

def _default_local_tts_enabled() -> bool:
    normalized = normalize_env_value(os.getenv("ENABLE_LOCAL_TTS")).lower()
    if not normalized or normalized not in {"1", "true", "yes", "on"}:
        return False

    # Local offline TTS is only supported for local desktop development.
    is_render = normalize_env_value(os.getenv("RENDER")).lower() in {"1", "true", "yes", "on"}
    is_docker = os.path.exists("/.dockerenv")
    if is_render or is_docker or os.name != "nt":
        logger.info(
            "Ignoring ENABLE_LOCAL_TTS because offline pyttsx3 fallback is only supported in local Windows development."
        )
        return False

    return True


def _load_pyttsx3():
    try:
        import pyttsx3
    except ImportError:
        return None

    return pyttsx3


# Lazy-load google.genai, Azure Speech, and OpenAI SDKs to avoid importing heavy SDKs at startup
_genai = None
_genai_types = None
_azure_speech = None
_openai = None

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


@dataclass
class TTSResult:
    """Result of text-to-speech conversion"""

    audio_bytes: bytes
    format: str  # "wav", "mp3"
    duration_seconds: float
    provider: str  # "gemini", "pyttsx3"
    error: Optional[str] = None  # Error message if synthesis failed


class TextToSpeechService:
    """Multi-provider text-to-speech service for accessibility"""

    def __init__(self):
        # Gemini TTS configuration
        self.gemini_api_key = resolve_gemini_api_key(os.getenv)
        self.gemini_tts_model = normalize_env_value(os.getenv("GEMINI_TTS_MODEL")) or DEFAULT_GEMINI_TTS_MODEL

        # Azure TTS configuration
        self.azure_speech_key = normalize_env_value(os.getenv("AZURE_SPEECH_KEY"))
        self.azure_speech_region = normalize_env_value(os.getenv("AZURE_SPEECH_REGION")) or "eastus"
        self.azure_voice_name = normalize_env_value(os.getenv("AZURE_TTS_VOICE")) or "en-US-JennyNeural"

        # OpenAI TTS configuration
        self.openai_api_key = normalize_env_value(os.getenv("OPENAI_API_KEY"))
        self.openai_tts_model = normalize_env_value(os.getenv("OPENAI_TTS_MODEL")) or "gpt-4o-mini-tts"
        self.openai_voice_name = normalize_env_value(os.getenv("OPENAI_TTS_VOICE")) or "marin"
        self.openai_client = None

        self.enable_local_tts = _default_local_tts_enabled()
        self.gemini_client = None
        self.gemini_types = None
        genai, _types = _ensure_genai()
        if genai and self.gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                self.gemini_types = _types
                logger.info("Gemini TTS client initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini TTS client: {e}")
                self.gemini_client = None
                self.gemini_types = None

        OpenAI = _ensure_openai()
        if OpenAI and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI TTS client initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI TTS client: {e}")

        # pyttsx3 offline TTS
        self.pyttsx3_engine = None
        if self.enable_local_tts:
            pyttsx3 = _load_pyttsx3()
            if pyttsx3 is None:
                logger.warning("pyttsx3 is not installed. Local TTS generation is disabled.")
            else:
                try:
                    self.pyttsx3_engine = pyttsx3.init()
                    # Configure voice properties
                    self.pyttsx3_engine.setProperty("rate", 150)  # Words per minute
                    self.pyttsx3_engine.setProperty("volume", 1.0)  # 0.0 to 1.0
                    logger.info("pyttsx3 TTS engine initialized.")
                except Exception as e:
                    logger.warning(f"Failed to initialize pyttsx3 engine: {e}")
        else:
            logger.info(
                "Local microlearning TTS fallback is disabled. Browser fallback should be used when server audio is unavailable."
            )

        # Automatically enable Windows local fallback if no cloud providers are configured.
        if (
            os.name == "nt"
            and not self.gemini_client
            and not self._azure_tts_available()
            and not self._openai_tts_available()
            and not self.enable_local_tts
        ):
            logger.info(
                "No Gemini/Azure/OpenAI TTS providers configured; enabling local Windows TTS fallback."
            )
            self.enable_local_tts = True

    def _gtts_available(self) -> bool:
        return _load_gtts() is not None

    def is_available(self) -> bool:
        """Check if any TTS provider is available"""
        return bool(
            self.gemini_client
            or self._azure_tts_available()
            or self._openai_tts_available()
            or self._gtts_available()
            or self.pyttsx3_engine
            or self._windows_sapi_available()
        )

    def get_available_providers(self) -> list[str]:
        """Get list of available TTS providers"""
        providers = []
        if self.gemini_client:
            providers.append("gemini")
        if self._azure_tts_available():
            providers.append("azure_speech")
        if self._openai_tts_available():
            providers.append("openai")
        if self._gtts_available():
            providers.append("gtts")
        if self.pyttsx3_engine:
            providers.append("pyttsx3")
        if self._windows_sapi_available():
            providers.append("windows_sapi")
        return providers

    def _azure_tts_available(self) -> bool:
        azure = _ensure_azure_speech()
        return bool(
            azure
            and is_usable_azure_speech_key(self.azure_speech_key)
            and self.azure_speech_region
        )

    def _openai_tts_available(self) -> bool:
        return bool(_openai and self.openai_client and self.openai_api_key)

    def _select_provider(self) -> str:
        if self._azure_tts_available():
            return "azure"
        if self._openai_tts_available():
            return "openai"
        if self.gemini_client:
            return "gemini"
        if self._gtts_available():
            return "gtts"
        if self._windows_sapi_available():
            return "windows_sapi"
        return "pyttsx3"

    def _synthesize_gtts(
        self,
        text: str,
        language_code: str = "en-US",
    ) -> Optional[TTSResult]:
        """Synthesize speech using gTTS (Google Translate TTS web API)"""
        gTTS = _load_gtts()
        if not gTTS or not text or not text.strip():
            return None

        try:
            lang = language_code.split("-")[0].lower() if language_code else "en"
            tts = gTTS(text=text.strip(), lang=lang, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            audio_bytes = buffer.getvalue()

            if not audio_bytes or len(audio_bytes) < 64:
                raise RuntimeError("gTTS generated empty or invalid audio data.")

            duration = _estimate_spoken_duration_seconds(text)
            logger.info(f"Successfully generated gTTS audio: {len(audio_bytes)} bytes, {duration:.2f}s")
            return TTSResult(
                audio_bytes=audio_bytes,
                format="mp3",
                duration_seconds=duration,
                provider="gtts",
                error=None,
            )
        except Exception as e:
            error_msg = f"gTTS synthesis failed: {e}"
            logger.warning(error_msg)
            return TTSResult(
                audio_bytes=b"",
                format="mp3",
                duration_seconds=0,
                provider="gtts",
                error=error_msg,
            )

    def _synthesize_gemini(
        self,
        text: str,
        language_code: str,
        voice_name: Optional[str],
    ) -> Optional[TTSResult]:
        """Synthesize speech using Gemini TTS"""
        if not self.gemini_client or not self.gemini_types:
            logger.warning("Gemini client or Gemini types are not available")
            return None

        types = self.gemini_types
        try:
            # Map language code to voice
            # Gemini supports: Kore (en-US), Puck (en-US), etc.
            if not voice_name:
                voice_name = "Kore"  # Default professional voice

            # Determine speaking style based on content
            speaking_style = self._detect_speaking_style(text)

            config_kwargs = {
                "speech_config": types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    ),
                )
            }

            logger.info(f"Generating Gemini TTS with voice: {voice_name}, style: {speaking_style}")
            
            response = self.gemini_client.models.generate_content(
                model=self.gemini_tts_model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    **config_kwargs,
                ),
            )

            # Extract audio data - handle different response structures
            audio_data = None
            
            # Try to extract from candidates
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    
                    # Check for parts in content
                    if hasattr(content, 'parts') and content.parts and len(content.parts) > 0:
                        part = content.parts[0]
                        
                        # Check for inline_data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if hasattr(part.inline_data, 'data'):
                                audio_data = part.inline_data.data
                            elif isinstance(part.inline_data, dict) and 'data' in part.inline_data:
                                audio_data = part.inline_data['data']
            
            if not audio_data:
                error_msg = "No audio data found in Gemini response. Response structure: " + str(response)
                logger.warning(error_msg)
                return TTSResult(
                    audio_bytes=b"",
                    format="wav",
                    duration_seconds=0,
                    provider="gemini",
                    error=error_msg,
                )

            # Convert to WAV format if needed
            import wave

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)

            audio_bytes = buffer.getvalue()
            
            if not audio_bytes or len(audio_bytes) < 44:  # WAV header is at least 44 bytes
                error_msg = f"Generated audio is invalid or empty (size: {len(audio_bytes)} bytes)"
                logger.error(error_msg)
                return TTSResult(
                    audio_bytes=b"",
                    format="wav",
                    duration_seconds=0,
                    provider="gemini",
                    error=error_msg,
                )

            # Estimate duration (rough)
            duration = len(audio_data) / (24000 * 2) if audio_data else 0

            logger.info(f"Successfully generated Gemini TTS audio: {len(audio_bytes)} bytes, {duration:.2f}s")
            
            return TTSResult(
                audio_bytes=audio_bytes,
                format="wav",
                duration_seconds=duration,
                provider="gemini",
                error=None,
            )

        except Exception as e:
            error_msg = f"Gemini TTS synthesis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                duration_seconds=0,
                provider="gemini",
                error=error_msg,
            )

    def _synthesize_pyttsx3(
        self,
        text: str,
        language_code: str,
        voice_name: Optional[str],
    ) -> Optional[TTSResult]:
        """Synthesize speech using pyttsx3 (offline)"""
        if not self.pyttsx3_engine:
            error_msg = "pyttsx3 engine not available"
            logger.warning(error_msg)
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                duration_seconds=0,
                provider="pyttsx3",
                error=error_msg,
            )

        try:
            # Get available voices
            voices = self.pyttsx3_engine.getProperty("voices")

            # Try to find a matching voice
            selected_voice = None
            lang_prefix = language_code.split("-")[0].lower()

            for voice in voices:
                if lang_prefix in voice.languages:
                    selected_voice = voice.id
                    break

            if selected_voice:
                self.pyttsx3_engine.setProperty("voice", selected_voice)

            if voice_name and voice_name.isdigit():
                # Voice ID provided
                self.pyttsx3_engine.setProperty("voice", voice_name)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            self.pyttsx3_engine.save_to_file(text, temp_filename)
            self.pyttsx3_engine.runAndWait()

            # Read the temp file
            with open(temp_filename, "rb") as f:
                audio_bytes = f.read()

            try:
                os.remove(temp_filename)
            except OSError:
                pass

            if not audio_bytes or len(audio_bytes) < 44:
                error_msg = f"pyttsx3 generated invalid audio (size: {len(audio_bytes)} bytes)"
                logger.error(error_msg)
                return TTSResult(
                    audio_bytes=b"",
                    format="wav",
                    duration_seconds=0,
                    provider="pyttsx3",
                    error=error_msg,
                )

            # Estimate duration
            duration = len(audio_bytes) / (22050 * 2)  # Default pyttsx3 rate

            logger.info(f"Successfully generated pyttsx3 TTS audio: {len(audio_bytes)} bytes, {duration:.2f}s")
            
            return TTSResult(
                audio_bytes=audio_bytes,
                format="wav",
                duration_seconds=duration,
                provider="pyttsx3",
                error=None,
            )

        except Exception as e:
            error_msg = f"pyttsx3 TTS synthesis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                duration_seconds=0,
                provider="pyttsx3",
                error=error_msg,
            )

    def _synthesize_azure(
        self,
        text: str,
        language_code: str,
        voice_name: Optional[str],
    ) -> Optional[TTSResult]:
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
        speech_config.speech_synthesis_voice_name = voice_name or self.azure_voice_name
        speech_config.set_speech_synthesis_output_format(
            azure.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        synthesizer = azure.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        result = synthesizer.speak_text_async(text).get()

        if result.reason != azure.ResultReason.SynthesizingAudioCompleted:
            details = getattr(result, "cancellation_details", None)
            detail_text = getattr(details, "error_details", None) or getattr(details, "reason", None)
            error_msg = f"Azure TTS failed to synthesize audio. {detail_text or ''}".strip()
            logger.error(error_msg)
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                duration_seconds=0,
                provider="azure_speech",
                error=error_msg,
            )

        audio_bytes = bytes(result.audio_data or b"")
        if not audio_bytes:
            error_msg = "Azure TTS completed without returning audio data."
            logger.error(error_msg)
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                duration_seconds=0,
                provider="azure_speech",
                error=error_msg,
            )

        return TTSResult(
            audio_bytes=audio_bytes,
            format="mp3",
            duration_seconds=_estimate_spoken_duration_seconds(text),
            provider="azure_speech",
            error=None,
        )

    def _synthesize_openai(
        self,
        text: str,
        language_code: str,
        voice_name: Optional[str],
    ) -> Optional[TTSResult]:
        if not self._openai_tts_available() or not text.strip():
            return None

        resolved_voice = voice_name or self.openai_voice_name
        instructions = (
            "Speak in a professional and natural customer-service tone."
            if not isinstance(language_code, str)
            else f"Speak in a professional and natural customer-service tone."
        )

        try:
            response = self.openai_client.audio.speech.create(
                model=self.openai_tts_model,
                voice=resolved_voice,
                input=text,
                instructions=instructions,
                response_format="mp3",
            )
            audio_bytes = response.read()
            if not audio_bytes:
                raise RuntimeError("OpenAI TTS completed without returning audio data.")

            return TTSResult(
                audio_bytes=audio_bytes,
                format="mp3",
                duration_seconds=_estimate_spoken_duration_seconds(text),
                provider="openai_tts",
                error=None,
            )
        except Exception as e:
            error_msg = f"OpenAI TTS synthesis failed: {e}"
            logger.error(error_msg, exc_info=True)
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                duration_seconds=0,
                provider="openai_tts",
                error=error_msg,
            )

    def _windows_sapi_available(self) -> bool:
        return self.enable_local_tts and os.name == "nt"

    def _synthesize_windows_sapi(self, text: str) -> Optional[TTSResult]:
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
    try {{ $stream.Close() }} catch {{ }}
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

            return TTSResult(
                audio_bytes=audio_bytes,
                format="wav",
                duration_seconds=len(audio_bytes) / 32000,
                provider="windows_sapi",
                error=None,
            )
        except Exception as e:
            logger.warning("Windows SAPI fallback TTS failed: %s", e)
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                duration_seconds=0,
                provider="windows_sapi",
                error=str(e),
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _fallback_tts(self, text: str) -> TTSResult:
        if self._windows_sapi_available():
            result = self._synthesize_windows_sapi(text)
            if result and result.audio_bytes:
                return result

        if self.pyttsx3_engine:
            result = self._synthesize_pyttsx3(text, "en-US", None)
            if result and result.audio_bytes:
                return result

        error_msg = "No local fallback TTS provider available."
        logger.warning(error_msg)
        return TTSResult(
            audio_bytes=b"",
            format="wav",
            duration_seconds=0,
            provider="local_fallback",
            error=error_msg,
        )

    def synthesize(
        self,
        text: str,
        language_code: str = "en-US",
        provider: Optional[str] = None,
        voice_name: Optional[str] = None,
    ) -> Optional[TTSResult]:
        """
        Convert text to speech audio.

        Args:
            text: Text to convert to speech
            language_code: Language code (affects voice selection)
            provider: Force specific provider ("gemini", "azure", "openai", "pyttsx3") or None for auto
            voice_name: Voice name (for Gemini: "Kore", "Puck"; for pyttsx3: voice ID)

        Returns:
            TTSResult with audio bytes and metadata, or None if no providers available
        """

        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None

        attempted_providers = []
        if not provider:
            provider = self._select_provider()

        provider_sequence = [provider]
        for candidate in ["azure", "openai", "gemini", "gtts", "pyttsx3"]:
            if candidate not in provider_sequence:
                provider_sequence.append(candidate)

        for provider_choice in provider_sequence:
            if provider_choice in attempted_providers:
                continue
            attempted_providers.append(provider_choice)
            if provider_choice == "gemini" and self.gemini_client:
                result = self._synthesize_gemini(text, language_code, voice_name)
            elif provider_choice == "azure" and self._azure_tts_available():
                result = self._synthesize_azure(text, language_code, voice_name)
            elif provider_choice == "openai" and self._openai_tts_available():
                result = self._synthesize_openai(text, language_code, voice_name)
            elif provider_choice == "gtts" and self._gtts_available():
                result = self._synthesize_gtts(text, language_code)
            elif provider_choice == "pyttsx3" and self.pyttsx3_engine:
                result = self._synthesize_pyttsx3(text, language_code, voice_name)
            else:
                result = None

            if result and result.audio_bytes:
                return result

        if self._windows_sapi_available():
            fallback_result = self._synthesize_windows_sapi(text)
            if fallback_result and fallback_result.audio_bytes:
                return fallback_result

        logger.error("Unable to synthesize speech with any available provider.")
        return TTSResult(
            audio_bytes=b"",
            format="wav",
            duration_seconds=0,
            provider=provider or "unknown",
            error=(
                "No TTS provider succeeded. Configure your Gemini/OpenAI/Azure credentials, "
                "or install pyttsx3 and enable local TTS on Windows."
            ),
        )

    def _detect_speaking_style(self, text: str) -> str:
        """Detect appropriate speaking style based on text content"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["question", "quiz", "answer", "?"]):
            return "professionally"
        if any(word in text_lower for word in ["!", "excited", "great", "amazing"]):
            return "cheerfully"
        if any(word in text_lower for word in ["sorry", "apologize", "regret"]):
            return "sadly"
        return "calmly"


# Global instance for easy access
text_to_speech_service = TextToSpeechService()
