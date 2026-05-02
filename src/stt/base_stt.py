"""
Speech-to-Text (STT) Service Implementation

Primary: OpenAI Whisper (local, free, no API key required)
Fallback: SpeechRecognition with Google Web Speech API (free)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
import asyncio
import tempfile
import os
import io

logger = logging.getLogger(__name__)

# Detect available STT backends
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.info("openai-whisper not installed. Install with: pip install openai-whisper")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.info("SpeechRecognition not installed. Install with: pip install SpeechRecognition")


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text implementations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass

    def is_ready(self) -> bool:
        return self.is_initialized


class STTService(BaseSTT):
    """
    Production STT service.
    Primary: OpenAI Whisper (local inference, free, no API key).
    Fallback: SpeechRecognition + Google Web Speech API (free).
    """

    MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB
    MAX_RETRIES = 2
    RETRY_DELAY = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.model_name = None
        self.backend = None  # "whisper" or "speech_recognition"

    async def initialize(self) -> None:
        """Load Whisper model or fall back to SpeechRecognition."""
        self.model_name = self.config.get("model", "base")

        # Check for ffmpeg (required by whisper)
        ffmpeg_available = False
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ffmpeg not found. Whisper will not work. Falling back to SpeechRecognition.")

        if WHISPER_AVAILABLE and ffmpeg_available:
            try:
                logger.info(f"Loading Whisper model '{self.model_name}'...")
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, whisper.load_model, self.model_name
                )
                self.backend = "whisper"
                logger.info(f"Whisper '{self.model_name}' loaded successfully")
            except Exception as e:
                logger.warning(f"Whisper load failed: {e}")
                if SR_AVAILABLE:
                    self.backend = "speech_recognition"
                    logger.info("Falling back to SpeechRecognition")
                else:
                    raise RuntimeError(f"No STT backend available: {e}")
        elif SR_AVAILABLE:
            self.backend = "speech_recognition"
            logger.info("Using SpeechRecognition backend (ffmpeg missing or Whisper not installed)")
        else:
            raise RuntimeError(
                "No STT backend available. "
                "Install ffmpeg and openai-whisper or SpeechRecognition."
            )

        self.is_initialized = True
        logger.info(f"STT initialized with backend: {self.backend}")

    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """Transcribe audio bytes to text with retry logic."""
        if not self.is_ready():
            raise RuntimeError("STT service not initialized")

        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Audio data is empty")

        if len(audio_bytes) > self.MAX_AUDIO_SIZE:
            raise ValueError(
                f"Audio too large ({len(audio_bytes)} bytes). "
                f"Max: {self.MAX_AUDIO_SIZE} bytes"
            )

        # Check for OpenAI API key to use Whisper API (Cloud)
        openai_key = os.getenv("OPENAI_API_KEY")
        
        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if openai_key:
                    return await self._transcribe_openai_whisper(audio_bytes, openai_key, **kwargs)
                elif self.backend == "whisper":
                    return await self._transcribe_whisper(audio_bytes, **kwargs)
                else:
                    return await self._transcribe_sr(audio_bytes, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    logger.warning(
                        f"STT attempt {attempt + 1} failed: {e}. Retrying..."
                    )
                    await asyncio.sleep(self.RETRY_DELAY)

        raise RuntimeError(
            f"STT failed after {self.MAX_RETRIES + 1} attempts: {last_error}"
        )

    async def _transcribe_openai_whisper(self, audio_bytes: bytes, api_key: str, **kwargs) -> str:
        """Transcribe using OpenAI Whisper API (Cloud)."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        # Whisper API requires a file-like object with a name
        buffer = io.BytesIO(audio_bytes)
        buffer.name = "audio.wav"
        
        transcript = await client.audio.transcriptions.create(
            model="whisper-1", 
            file=buffer,
            language=kwargs.get("language")
        )
        return transcript.text.strip()

    async def _transcribe_whisper(self, audio_bytes: bytes, **kwargs) -> str:
        """Transcribe using local Whisper model."""
        suffix = ".wav"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    temp_path,
                    language=kwargs.get("language"),
                    fp16=False,
                ),
            )
            text = result.get("text", "").strip()
            if not text:
                logger.warning("Whisper returned empty transcription")
                return ""
            logger.info(f"Whisper transcription ({len(text)} chars)")
            return text
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def _transcribe_sr(self, audio_bytes: bytes, **kwargs) -> str:
        """Transcribe using SpeechRecognition (Google Web Speech)."""
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(io.BytesIO(audio_bytes))

        with audio_file as source:
            audio = recognizer.record(source)

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None, recognizer.recognize_google, audio
            )
            return text.strip()
        except sr.UnknownValueError:
            logger.warning("SpeechRecognition could not understand audio")
            return ""
        except sr.RequestError as e:
            raise RuntimeError(f"SpeechRecognition API error: {e}")

    async def cleanup(self) -> None:
        """Release model resources."""
        self.model = None
        self.backend = None
        self.is_initialized = False
        logger.info("STT service cleaned up")