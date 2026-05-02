"""
Text-to-Speech (TTS) Service Implementation

Primary: Edge TTS (free Microsoft neural voices, no API key required)
Fallback: gTTS (Google Translate TTS, free, no API key required)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
import asyncio
import io
import os

logger = logging.getLogger(__name__)

# Detect available TTS backends
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.info("edge-tts not installed. Install with: pip install edge-tts")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.info("gTTS not installed. Install with: pip install gTTS")


class BaseTTS(ABC):
    """
    Abstract base class for Text-to-Speech implementations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass

    def is_ready(self) -> bool:
        return self.is_initialized


class TTSService(BaseTTS):
    """
    Production TTS service.
    Primary: Edge TTS (free Microsoft neural voices, async-native).
    Fallback: gTTS (Google Translate TTS, free).
    """

    MAX_CHUNK_SIZE = 4000  # Characters per chunk for long text
    MAX_RETRIES = 2
    RETRY_DELAY = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self.voice_id = None
        self.model = None
        # Indian Voices
        self.en_in_voice = "en-IN-NeerjaNeural"
        self.hi_in_voice = "hi-IN-MadhurNeural"
        self.voice = self.en_in_voice
        self.backend = None  # "edge_tts" or "gtts"

    async def initialize(self) -> None:
        """Initialize TTS with best available backend."""
        self.voice = self.config.get("voice", "en-US-AriaNeural")

        if EDGE_TTS_AVAILABLE:
            self.backend = "edge_tts"
            logger.info(f"TTS initialized with Edge TTS (voice: {self.voice})")
        elif GTTS_AVAILABLE:
            self.backend = "gtts"
            logger.info("TTS initialized with gTTS fallback")
        else:
            raise RuntimeError(
                "No TTS backend available. Install edge-tts or gTTS."
            )

        self.is_initialized = True

    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Convert text to speech audio bytes (MP3)."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        clean_text = text.strip()

        # Split long text into manageable chunks
        chunks = self._split_text(clean_text)
        logger.info(
            f"Synthesizing {len(clean_text)} chars in {len(chunks)} chunk(s)"
        )

        # Check for ElevenLabs API key
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Select voice based on language
        language = kwargs.get("language", "en").lower()
        target_voice = self.hi_in_voice if language == "hi" else self.en_in_voice
        kwargs["voice"] = target_voice  # Pass to backend implementations

        all_audio = b""
        for i, chunk in enumerate(chunks):
            last_error = None
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    if elevenlabs_key:
                        audio = await self._synthesize_elevenlabs(chunk, elevenlabs_key, **kwargs)
                    elif self.backend == "edge_tts":
                        audio = await self._synthesize_edge(chunk, **kwargs)
                    else:
                        audio = await self._synthesize_gtts(chunk, **kwargs)
                    all_audio += audio
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self.MAX_RETRIES:
                        logger.warning(
                            f"TTS chunk {i+1} attempt {attempt+1} "
                            f"failed: {e}. Retrying..."
                        )
                        await asyncio.sleep(self.RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"TTS failed on chunk {i+1} after "
                    f"{self.MAX_RETRIES + 1} attempts: {last_error}"
                )

        if not all_audio:
            raise RuntimeError("TTS produced empty audio output")

        return all_audio

    async def _synthesize_elevenlabs(self, text: str, api_key: str, **kwargs) -> bytes:
        """Synthesize with ElevenLabs API."""
        import httpx
        voice_id = kwargs.get("voice_id", "pNInz6obpgDQGcFmaJgB") # Adam voice default
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"ElevenLabs API error: {resp.text}")
            return resp.content

    async def _synthesize_edge(self, text: str, **kwargs) -> bytes:
        """Synthesize with Edge TTS (async-native)."""
        voice = kwargs.get("voice", self.voice)
        communicate = edge_tts.Communicate(text, voice)

        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        if not audio_bytes:
            raise RuntimeError("Edge TTS returned empty audio")

        return audio_bytes

    async def _synthesize_gtts(self, text: str, **kwargs) -> bytes:
        """Synthesize with gTTS (sync, run in executor)."""
        loop = asyncio.get_event_loop()

        def _generate():
            tts = gTTS(text=text, lang=kwargs.get("lang", "en"))
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()

        return await loop.run_in_executor(None, _generate)

    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """Stream synthesis — wraps synthesize() into a BytesIO buffer."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")

        audio_data = await self.synthesize(text, **kwargs)
        buf = io.BytesIO(audio_data)
        return buf

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks on sentence boundaries."""
        if len(text) <= self.MAX_CHUNK_SIZE:
            return [text]

        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= self.MAX_CHUNK_SIZE:
                chunks.append(remaining)
                break

            # Find best split point: sentence end, then space
            split_pos = remaining.rfind(". ", 0, self.MAX_CHUNK_SIZE)
            if split_pos != -1:
                split_pos += 1  # Include the period
            else:
                split_pos = remaining.rfind(" ", 0, self.MAX_CHUNK_SIZE)
            if split_pos == -1:
                split_pos = self.MAX_CHUNK_SIZE

            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()

        return [c for c in chunks if c]

    async def cleanup(self) -> None:
        """Release TTS resources."""
        self.client = None
        self.voice = None
        self.backend = None
        self.is_initialized = False
        logger.info("TTS service cleaned up")

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """List available Edge TTS voices."""
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")

        if self.backend == "edge_tts" and EDGE_TTS_AVAILABLE:
            voices = await edge_tts.list_voices()
            return [
                {
                    "voice_id": v["ShortName"],
                    "name": v["FriendlyName"],
                    "locale": v["Locale"],
                    "gender": v["Gender"],
                }
                for v in voices
            ]
        return []