"""
Speech-to-Text service for transcribing microlearning audio content.
Supports multiple providers: Google Speech-to-Text, OpenAI Whisper, and local Vosk.
"""

from __future__ import annotations

import base64
import asyncio
import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

from ..config_validation import normalize_env_value

logger = logging.getLogger(__name__)
_MP3_MIME_MARKERS = (
    "mp3",
    "mpeg",
    "mpga",
    "mpg",
)


def _looks_like_vosk_model_dir(model_path: str) -> bool:
    required_directories = ("am", "conf", "graph")
    return all(os.path.isdir(os.path.join(model_path, segment)) for segment in required_directories)

# Try to import providers
try:
    from google.cloud import speech_v1 as speech
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    speech = None
    GOOGLE_SPEECH_AVAILABLE = False

# Lazy-load OpenAI SDK when needed
_openai = None

def _ensure_openai():
    global _openai
    if _openai is not None:
        return _openai
    try:
        import importlib

        mod = importlib.import_module('openai')
        # Prefer exported OpenAI class if present
        _openai = getattr(mod, 'OpenAI', mod)
        return _openai
    except Exception as exc:
        logger.info('OpenAI SDK not available: %s', exc)
        _openai = None
        return None

# Try Vosk for local transcription
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    Model = None
    KaldiRecognizer = None
    VOSK_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Result of audio transcription"""
    text: str
    confidence: float  # 0.0 to 1.0
    provider: str  # "google", "whisper", "vosk"
    language_code: str
    words: Optional[list] = None  # Word-level timestamps if available
    duration_seconds: Optional[float] = None


class SpeechToTextService:
    """Multi-provider speech-to-text transcription service"""

    def __init__(self):
        # Google Cloud Speech configuration
        self.google_api_key = normalize_env_value(
            os.getenv("GOOGLE_CLOUD_SPEECH_API_KEY")
            or os.getenv("GOOGLE_SPEECH_API_KEY")
        )
        self.google_client = None
        if GOOGLE_SPEECH_AVAILABLE and self.google_api_key:
            try:
                self.google_client = speech.Client(api_key=self.google_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Google Speech client: {e}")

        # OpenAI Whisper configuration (lazy-loaded)
        self.openai_api_key = normalize_env_value(os.getenv("OPENAI_API_KEY"))
        self.openai_client = None
        OpenAI = _ensure_openai()
        if OpenAI and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

        # Vosk local model
        self.vosk_model = None
        self.vosk_recognizer = None
        if VOSK_AVAILABLE:
            model_path = os.getenv("VOSK_MODEL_PATH", "model")
            if os.path.exists(model_path) and _looks_like_vosk_model_dir(model_path):
                try:
                    self.vosk_model = Model(model_path)
                    self.vosk_recognizer = KaldiRecognizer(self.vosk_model, 16000)
                    logger.info("✓ Vosk model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load Vosk model: {e}")
            elif os.path.exists(model_path):
                logger.info(
                    "Skipping Vosk model initialization because %s does not contain a complete Vosk model.",
                    model_path,
                )

    def is_available(self) -> bool:
        """Check if any transcription provider is available"""
        return bool(
            self.google_client 
            or self.openai_client 
            or (self.vosk_model and self.vosk_recognizer)
        )

    def get_available_providers(self) -> list[str]:
        """Get list of available transcription providers"""
        providers = []
        if self.google_client:
            providers.append("google")
        if self.openai_client:
            providers.append("whisper")
        if self.vosk_model and self.vosk_recognizer:
            providers.append("vosk")
        return providers

    async def transcribe_from_url(
        self,
        audio_url: str,
        language_code: str = "en-US",
        provider: Optional[str] = None,
        mime_type: str = "audio/webm",
    ) -> str:
        """Fetch an audio asset by URL and return its transcript text."""
        audio_bytes = await asyncio.to_thread(self._read_audio_url, audio_url)
        result = await asyncio.to_thread(
            self.transcribe,
            audio_bytes=audio_bytes,
            language_code=language_code,
            provider=provider,
            mime_type=mime_type,
        )

        if not result or not result.text:
            raise RuntimeError("Transcription failed")

        return result.text

    def _read_audio_url(self, audio_url: str) -> bytes:
        normalized_url = normalize_env_value(audio_url)
        if not normalized_url:
            raise RuntimeError("Audio URL is empty")

        if normalized_url.startswith(("http://", "https://")):
            import requests

            response = requests.get(normalized_url, timeout=30)
            response.raise_for_status()
            return response.content

        raise RuntimeError("Unsupported audio URL")

    def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "en-US",
        provider: Optional[str] = None,
        mime_type: str = "audio/mp3",
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_bytes: Raw audio file bytes
            language_code: Language code (e.g., "en-US", "en-GB")
            provider: Force specific provider ("google", "whisper", "vosk") or None for auto
            mime_type: MIME type of audio (for provider detection)
        
        Returns:
            TranscriptionResult with text, confidence, and metadata
        """
        # Auto-select provider if not specified
        if not provider:
            provider = self._select_provider(mime_type)
        
        if provider == "google":
            return self._transcribe_google(audio_bytes, language_code, mime_type)
        elif provider == "whisper":
            return self._transcribe_whisper(audio_bytes, language_code)
        elif provider == "vosk":
            return self._transcribe_vosk(audio_bytes, language_code)
        else:
            logger.error(f"Unknown or unavailable provider: {provider}")
            return None

    def _select_provider(self, mime_type: str) -> str:
        """Select best available provider"""
        if self.google_client:
            return "google"
        if self.openai_client:
            return "whisper"
        if self.vosk_model:
            return "vosk"
        return "vosk"  # Fallback (will fail gracefully)

    def _transcribe_google(
        self, 
        audio_bytes: bytes, 
        language_code: str,
        mime_type: str = "audio/mp3",
    ) -> Optional[TranscriptionResult]:
        """Transcribe using Google Cloud Speech-to-Text"""
        if not self.google_client:
            logger.warning("Google Speech client not available")
            return None

        try:
            # Encode audio to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            normalized_mime_type = (mime_type or "").lower()
            uses_mp3_encoding = any(marker in normalized_mime_type for marker in _MP3_MIME_MARKERS)

            # Configure recognition
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MP3
                if uses_mp3_encoding
                else speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="default",
                use_enhanced=True,  # Use video/phone model for better quality
            )

            audio = speech.RecognitionAudio(content=audio_bytes)

            # Perform transcription
            operation = self.google_client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=300)  # 5 minute timeout

            # Extract results
            transcripts = []
            confidence = 0.0
            word_list = []

            for result in response.results:
                alternative = result.alternatives[0]
                transcripts.append(alternative.transcript)
                confidence = max(confidence, alternative.confidence)
                
                # Extract word-level data if available
                if hasattr(alternative, 'words') and alternative.words:
                    for word in alternative.words:
                        word_list.append({
                            "word": word.word,
                            "start": word.start_time.seconds + word.start_time.nanos / 1e9,
                            "end": word.end_time.seconds + word.end_time.nanos / 1e9,
                        })

            full_text = " ".join(transcripts)
            
            # Estimate duration from audio size (rough estimate)
            duration = len(audio_bytes) / (16000 * 2)  # 16kHz, 16-bit mono

            return TranscriptionResult(
                text=full_text,
                confidence=confidence,
                provider="google",
                language_code=language_code,
                words=word_list if word_list else None,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Google transcription failed: {e}")
            return None

    def _transcribe_whisper(
        self, 
        audio_bytes: bytes, 
        language_code: str
    ) -> Optional[TranscriptionResult]:
        """Transcribe using OpenAI Whisper"""
        if not self.openai_client:
            logger.warning("OpenAI client not available")
            return None

        try:
            # Create audio file object from bytes
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.mp3"

            # Determine model based on language
            model = "whisper-1"

            # Perform transcription
            response = self.openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language_code.split("-")[0],  # Whisper uses just "en" not "en-US"
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

            # Extract results
            text = response.text
            
            # Calculate confidence (Whisper doesn't provide direct confidence)
            # Use average of word-level confidence if available
            confidence = 0.85  # Default assumption for Whisper
            word_list = []

            if hasattr(response, 'words') and response.words:
                confidences = []
                for word in response.words:
                    word_list.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    })
                    if hasattr(word, 'confidence'):
                        confidences.append(word.confidence)
                
                if confidences:
                    confidence = sum(confidences) / len(confidences)

            # Get duration if available
            duration = None
            if hasattr(response, 'duration'):
                duration = response.duration

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                provider="whisper",
                language_code=language_code,
                words=word_list if word_list else None,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return None

    def _transcribe_vosk(
        self, 
        audio_bytes: bytes, 
        language_code: str
    ) -> Optional[TranscriptionResult]:
        """Transcribe using local Vosk model"""
        if not self.vosk_model or not self.vosk_recognizer:
            logger.warning("Vosk model not available")
            return None

        try:
            import wave
            import json

            # Vosk requires specific audio format (16kHz, 16-bit, mono)
            # Convert if needed - for now assume input is compatible
            # In production, you'd use pydub to convert
            
            # Reset recognizer for new audio
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, 16000)

            # Process audio in chunks
            # For simplicity, we'll process the whole thing
            if self.vosk_recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get("text", "")
            else:
                # Get partial result
                partial = json.loads(self.vosk_recognizer.PartialResult())
                text = partial.get("partial", "")

            # Vosk doesn't provide confidence scores
            # Estimate based on text quality
            confidence = 0.75 if text else 0.0

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                provider="vosk",
                language_code=language_code,
                words=None,
                duration_seconds=len(audio_bytes) / (16000 * 2),
            )

        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            return None


# Global instance for easy access
speech_to_text_service = SpeechToTextService()


def get_transcription_service() -> SpeechToTextService:
    """Return the shared speech-to-text service instance."""
    return speech_to_text_service


__all__ = [
    "SpeechToTextService",
    "TranscriptionResult",
    "speech_to_text_service",
    "get_transcription_service",
]
