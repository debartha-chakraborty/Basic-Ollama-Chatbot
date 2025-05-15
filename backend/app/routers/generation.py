from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
from services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["generation"])
rag_service = RAGService()

class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None

class IngestionRequest(BaseModel):
    limit: Optional[int] = 10

@router.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text using RAG pipeline (non-streaming)"""
    try:
        response = await rag_service.generate(
            prompt=request.prompt
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/generate/stream")
async def generate_stream(request: GenerationRequest):
    """Stream text generation from the RAG pipeline"""
    try:
        return StreamingResponse(
            rag_service.generate_stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Streaming generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming generation failed: {str(e)}")

@router.post("/ingest/federal-register")
async def ingest_federal_register(request: IngestionRequest):
    """Ingest documents from the Federal Register API into the RAG system"""
    try:
        documents = await rag_service.ingest_federal_register_documents(limit=request.limit)
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(documents)} documents from Federal Register",
            "document_count": len(documents)
        }
    except Exception as e:
        logger.error(f"Federal Register ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
