"""
Audio Customer Support Agent Pipeline

This module orchestrates the complete STT -> LLM -> TTS pipeline.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService

logger = logging.getLogger(__name__)


@dataclass
class TranscriptData:
    """Transcript information for a conversation turn."""
    user_input: str
    agent_response: str


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


class AudioSupportPipeline:
    """
    Main pipeline class that orchestrates STT -> LLM -> TTS flow.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False
        
        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            self.logger.info("Initializing Audio Support Pipeline...")
            
            # 1. Initialize STT service
            self.logger.info("Initializing STT service...")
            self.stt = STTService(self.config.stt_config)
            await self.stt.initialize()
            
            # 2. Initialize LLM agent
            self.logger.info("Initializing LLM agent...")
            self.llm_agent = CustomerSupportAgent(self.config.llm_config)
            await self.llm_agent.initialize()
            
            # 3. Initialize TTS service
            self.logger.info("Initializing TTS service...")
            self.tts = TTSService(self.config.tts_config)
            await self.tts.initialize()
            
            # 4. Verify components
            if not all([self.stt.is_ready(), self.llm_agent.is_initialized, self.tts.is_ready()]):
                raise RuntimeError("Some pipeline components failed to initialize")
            
            self.is_initialized = True
            self.logger.info("Pipeline initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            await self.cleanup()
            raise
    
    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        """
        Process audio input through the complete pipeline.
        
        STT -> LLM -> TTS
        """
        audio, _, _ = await self.process_audio_with_transcript(audio_bytes, **kwargs)
        return audio

    async def process_audio_with_transcript(self, audio_bytes: bytes, **kwargs) -> Tuple[bytes, TranscriptData, int]:
        """
        Process audio input and return audio response, transcript, and timing.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        try:
            # Step 1 - Speech to Text
            self.logger.info("Converting speech to text...")
            try:
                text_input = await asyncio.wait_for(
                    self.stt.transcribe(audio_bytes, **kwargs),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                raise RuntimeError("STT transcription timed out")
                
            self.logger.info(f"Transcribed text: {text_input}")
            
            if not text_input:
                self.logger.warning("Empty transcription result")
                text_input = "Hello?" 
            
            # Step 2 - Process with LLM Agent
            self.logger.info("Processing query with LLM agent...")
            try:
                agent_response = await asyncio.wait_for(
                    self.llm_agent.process_query(text_input, **kwargs),
                    timeout=45.0
                )
            except asyncio.TimeoutError:
                raise RuntimeError("LLM processing timed out")
                
            self.logger.info(f"Agent response: {agent_response}")
            
            # Step 3 - Text to Speech
            self.logger.info("Converting response to speech...")
            try:
                response_audio = await asyncio.wait_for(
                    self.tts.synthesize(agent_response, **kwargs),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                raise RuntimeError("TTS synthesis timed out")
                
            processing_time_ms = int((time.time() - start_time) * 1000)
            transcript = TranscriptData(user_input=text_input, agent_response=agent_response)
            
            self.logger.info(f"Audio response generated in {processing_time_ms}ms")
            
            return response_audio, transcript, processing_time_ms
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        """
        Process text input. LLM -> TTS.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            # Process with LLM Agent
            self.logger.info(f"Processing text query: {text_input}")
            try:
                agent_response = await asyncio.wait_for(
                    self.llm_agent.process_query(text_input, **kwargs),
                    timeout=45.0
                )
            except asyncio.TimeoutError:
                raise RuntimeError("LLM processing timed out")
            
            # Convert to speech
            self.logger.info("Converting response to speech...")
            try:
                response_audio = await asyncio.wait_for(
                    self.tts.synthesize(agent_response, **kwargs),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                raise RuntimeError("TTS synthesis timed out")
            
            return agent_response, response_audio
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """Check the health status of all pipeline components."""
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }
    
    async def cleanup(self) -> None:
        """Cleanup all pipeline resources."""
        self.logger.info("Cleaning up pipeline resources...")
        
        try:
            if self.stt:
                await self.stt.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.tts:
                await self.tts.cleanup()
                
            self.stt = None
            self.llm_agent = None
            self.tts = None
            self.is_initialized = False
            
            self.logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True
) -> AudioSupportPipeline:
    """Factory function to create and initialize a pipeline."""
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging
    )
    
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    
    return pipeline