"""
FastAPI Server for Audio Customer Support Agent

REST API endpoints for testing the audio support pipeline.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
import os
import time
import base64

from dotenv import load_dotenv

from src.pipeline import AudioSupportPipeline, create_pipeline, PipelineConfig, TranscriptData

# Load environment variables
load_dotenv()

class TextRequest(BaseModel):
    """Request model for text-based queries."""
    text: str
    parameters: Optional[Dict[str, Any]] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    stt: bool
    llm: bool
    tts: bool

class TranscriptModel(BaseModel):
    """Transcript details."""
    user_input: str
    agent_response: str

class AudioResponse(BaseModel):
    """Response model for audio chat."""
    success: bool
    audio_response: str  # Base64 encoded audio
    transcript: TranscriptModel
    processing_time_ms: int

class TextResponse(BaseModel):
    """Response model for text queries."""
    response_text: str
    audio_available: bool
    processing_time_ms: int


app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for testing the STT -> LLM -> TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioSupportPipeline] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on server startup."""
    global pipeline
    
    try:
        logger.info("Starting Audio Support Agent API server...")
        
        # Configure services. The actual classes handle logic of reading env if config values are missing.
        stt_config = {
            "model": "base"
        }
        
        llm_config = {
            "temperature": 0.7
        }
        
        tts_config = {
            "voice": "en-US-AriaNeural"
        }
        
        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        logger.info("Pipeline configuration loaded and initialized successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        # We don't raise here so the server still starts and we can see /health failures for debugging


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup pipeline resources on server shutdown."""
    global pipeline
    
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint. Returns status of all components."""
    global pipeline
    
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            stt=False,
            llm=False,
            tts=False
        )
    
    try:
        components = await pipeline.health_check()
        
        # Check actual values returned from pipeline.py
        stt_ready = components.get("stt_ready", False)
        llm_ready = components.get("llm_ready", False)
        tts_ready = components.get("tts_ready", False)
        
        all_healthy = stt_ready and llm_ready and tts_ready
        
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            stt=stt_ready,
            llm=llm_ready,
            tts=tts_ready
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            stt=False,
            llm=False,
            tts=False
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """Process text query through the LLM agent."""
    global pipeline
    
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Pipeline not initialized"
        )
        
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty"
        )
    
    try:
        start_time = time.time()
        
        response_text, response_audio = await pipeline.process_text(
            request.text, 
            **request.parameters
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return TextResponse(
            response_text=response_text,
            audio_available=True if response_audio else False,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )


@app.post("/chat/audio", response_model=AudioResponse)
async def chat_audio(audio: UploadFile = File(...), language: str = "en"):
    """Process audio query through the complete pipeline."""
    global pipeline
    
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Pipeline not initialized"
        )
    
    try:
        audio_bytes = await audio.read()
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Empty audio file"
            )
            
        # Max 25 MB
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio file too large. Max size is 25MB."
            )
        
        # Use the new pipeline method
        response_audio, transcript, processing_time = await pipeline.process_audio_with_transcript(audio_bytes, language=language)
        
        # Base64 encode the audio response
        audio_b64 = base64.b64encode(response_audio).decode('utf-8')
        
        return AudioResponse(
            success=True,
            audio_response=audio_b64,
            transcript=TranscriptModel(
                user_input=transcript.user_input,
                agent_response=transcript.agent_response
            ),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str, language: str = "en"):
    """Convert text to audio using TTS (for testing TTS alone)."""
    global pipeline
    
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Pipeline not initialized"
        )
        
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty"
        )
    
    try:
        if not pipeline.tts:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="TTS not available"
            )
        
        audio_bytes = await pipeline.tts.synthesize(text, language=language)
        
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """Debug endpoint for testing STT component independently."""
    global pipeline
    
    if not pipeline or not pipeline.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Pipeline not initialized"
        )
    
    try:
        audio_bytes = await audio.read()
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Empty audio file"
            )
        
        if not pipeline.stt:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="STT not available"
            )
        
        transcription = await pipeline.stt.transcribe(audio_bytes)
        
        return {"transcription": transcription}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )