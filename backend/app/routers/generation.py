from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging

from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["generation"])
ollama_service = OllamaService()

class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None

@router.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text using the default model (non-streaming)"""
    try:
        response = await ollama_service.generate(
            prompt=request.prompt
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/generate/stream")
async def generate_stream(request: GenerationRequest):
    """Stream text generation from the default model"""
    try:
        return StreamingResponse(
            ollama_service.generate_stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Streaming generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming generation failed: {str(e)}")