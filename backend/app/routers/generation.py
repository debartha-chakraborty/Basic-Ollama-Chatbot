import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging
from ..services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["generation"])
rag_service = RAGService()

class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    
class ToolSelection(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class AgenticGenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    suggested_tools: Optional[List[ToolSelection]] = None

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

@router.post("/agentic/generate")
async def agentic_generate_text(request: AgenticGenerationRequest):
    """Generate text using Agentic RAG pipeline with intelligent tool selection"""
    try:
        # The agentic RAG system will automatically select the best tools
        response = await rag_service.generate(
            prompt=request.prompt
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Agentic generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic generation failed: {str(e)}")

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

@router.post("/agentic/generate/stream")
async def agentic_generate_stream(request: AgenticGenerationRequest):
    """Stream text generation from the Agentic RAG pipeline with intelligent tool selection"""
    try:
        return StreamingResponse(
            rag_service.generate_stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Agentic streaming generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic streaming generation failed: {str(e)}")

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

@router.get("/tools")
async def list_available_tools():
    """List all available tools in the Agentic RAG system"""
    try:
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": getattr(tool, "args_schema", {})
            }
            for tool in rag_service.tools
        ]
        return {"tools": tools}
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")

@router.post("/search/date-filtered")
async def date_filtered_search(
    query: str = Query(..., description="The search query"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format")
):
    """Directly use the date-filtered search tool"""
    try:
        results = await asyncio.to_thread(
            rag_service.date_filtered_search,
            query=query,
            start_date=start_date,
            end_date=end_date
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Date-filtered search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Date-filtered search failed: {str(e)}")

@router.post("/search/agency-filtered")
async def agency_filtered_search(
    query: str = Query(..., description="The search query"),
    agency: str = Query(..., description="The agency to filter by")
):
    """Directly use the agency-filtered search tool"""
    try:
        results = await asyncio.to_thread(
            rag_service.agency_filtered_search,
            query=query,
            agency=agency
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Agency-filtered search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agency-filtered search failed: {str(e)}")

@router.post("/search/keyword")
async def keyword_search(
    query: str = Query(..., description="The keyword to search for")
):
    """Directly use the keyword search tool"""
    try:
        results = await asyncio.to_thread(
            rag_service.keyword_search,
            query=query
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Keyword search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")

@router.post("/search/federal-register-api")
async def search_federal_register_api(
    query: str = Query(..., description="The query to search in the Federal Register API")
):
    """Directly search the Federal Register API"""
    try:
        results = await rag_service.search_federal_register_api(query=query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Federal Register API search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Federal Register API search failed: {str(e)}")
