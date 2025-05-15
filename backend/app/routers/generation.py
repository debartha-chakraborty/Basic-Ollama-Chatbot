import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging
from services.rag_service import RAGService

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

@router.post("/search/advanced")
async def advanced_search(
    query: str = Query(..., description="The search query"),
    min_confidence: float = Query(0.7, description="Minimum confidence score for results"),
    max_results: int = Query(10, description="Maximum number of results to return"),
    include_metadata: bool = Query(True, description="Whether to include metadata in results")
):
    """Perform advanced search with confidence scoring and metadata filtering"""
    try:
        results = await rag_service.advanced_search(
            query=query,
            min_confidence=min_confidence,
            max_results=max_results,
            include_metadata=include_metadata
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Advanced search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")

@router.post("/search/topic")
async def topic_search(
    query: str = Query(..., description="The search query"),
    topic: str = Query(..., description="The specific topic or category to search within"),
    include_related: bool = Query(True, description="Whether to include related topics")
):
    """Search within specific topics or categories"""
    try:
        results = await rag_service.topic_search(
            query=query,
            topic=topic,
            include_related=include_related
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Topic search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Topic search failed: {str(e)}")

@router.post("/search/citation")
async def citation_search(
    query: str = Query(..., description="The search query"),
    citation_type: str = Query(..., description="Type of citation (e.g., 'law', 'regulation', 'policy')"),
    jurisdiction: Optional[str] = Query(None, description="Specific jurisdiction to search within")
):
    """Search for specific types of citations and references"""
    try:
        results = await rag_service.citation_search(
            query=query,
            citation_type=citation_type,
            jurisdiction=jurisdiction
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Citation search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Citation search failed: {str(e)}")

@router.post("/search/cross-reference")
async def cross_reference_search(
    document_id: str = Query(..., description="The document ID to find references for"),
    reference_type: str = Query(..., description="Type of reference (e.g., 'cited_by', 'cites', 'related')"),
    depth: int = Query(1, description="Depth of reference chain to follow")
):
    """Find documents that reference or are referenced by a specific document"""
    try:
        results = await rag_service.cross_reference_search(
            document_id=document_id,
            reference_type=reference_type,
            depth=depth
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Cross-reference search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cross-reference search failed: {str(e)}")

@router.post("/document/summarize")
async def summarize_document(
    query: str = Query(..., description="The query to find the document to summarize")
):
    """Generate a concise summary of a document"""
    try:
        results = await rag_service.summarize_document(query=query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Document summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document summarization failed: {str(e)}")

@router.post("/document/compare")
async def compare_documents(
    query: str = Query(..., description="The query to find documents to compare")
):
    """Compare multiple documents for similarities and differences"""
    try:
        results = await rag_service.compare_documents(query=query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Document comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document comparison failed: {str(e)}")

@router.post("/search/document-type")
async def document_type_search(
    query: str = Query(..., description="The search query"),
    doc_type: str = Query(..., description="Type of document (e.g., 'Rule', 'Proposed Rule', 'Notice')")
):
    """Search for specific types of Federal Register documents"""
    try:
        results = await rag_service.document_type_search(
            query=query,
            doc_type=doc_type
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Document type search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document type search failed: {str(e)}")

@router.post("/search/public-comment")
async def public_comment_search(
    query: str = Query(..., description="The search query"),
    comment_status: str = Query(..., description="Status of public comments (e.g., 'open', 'closed')"),
    days_remaining: Optional[int] = Query(None, description="Days remaining for comments")
):
    """Search for documents with specific public comment status"""
    try:
        results = await rag_service.public_comment_search(
            query=query,
            comment_status=comment_status,
            days_remaining=days_remaining
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Public comment search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Public comment search failed: {str(e)}")

@router.post("/search/regulatory-impact")
async def regulatory_impact_search(
    query: str = Query(..., description="The search query"),
    impact_type: str = Query(..., description="Type of regulatory impact (e.g., 'economic', 'environmental', 'health')"),
    significance_level: Optional[str] = Query(None, description="Significance level of the impact")
):
    """Search for documents based on their regulatory impact"""
    try:
        results = await rag_service.regulatory_impact_search(
            query=query,
            impact_type=impact_type,
            significance_level=significance_level
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Regulatory impact search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Regulatory impact search failed: {str(e)}")

@router.post("/search/document-relationship")
async def document_relationship_search(
    query: str = Query(..., description="The search query"),
    relationship_type: str = Query(..., description="Type of relationship (e.g., 'amends', 'rescinds', 'supersedes')"),
    target_document: Optional[str] = Query(None, description="Target document number")
):
    """Search for documents based on their relationships to other documents"""
    try:
        results = await rag_service.document_relationship_search(
            query=query,
            relationship_type=relationship_type,
            target_document=target_document
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Document relationship search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document relationship search failed: {str(e)}")

@router.post("/search/agency-hierarchy")
async def agency_hierarchy_search(
    query: str = Query(..., description="The search query"),
    parent_agency: Optional[str] = Query(None, description="Parent agency name"),
    include_subagencies: bool = Query(True, description="Whether to include subagencies")
):
    """Search for documents based on agency hierarchy"""
    try:
        results = await rag_service.agency_hierarchy_search(
            query=query,
            parent_agency=parent_agency,
            include_subagencies=include_subagencies
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Agency hierarchy search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agency hierarchy search failed: {str(e)}")

@router.post("/search/document-timeline")
async def document_timeline_search(
    query: str = Query(..., description="The search query"),
    timeline_type: str = Query(..., description="Type of timeline (e.g., 'rulemaking', 'comment', 'effective')"),
    include_related: bool = Query(True, description="Whether to include related documents")
):
    """Search for documents based on their timeline in the rulemaking process"""
    try:
        results = await rag_service.document_timeline_search(
            query=query,
            timeline_type=timeline_type,
            include_related=include_related
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Document timeline search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document timeline search failed: {str(e)}")

@router.post("/document/analyze")
async def analyze_document(
    query: str = Query(..., description="The query to find the document to analyze")
):
    """Perform detailed analysis of a document's content, structure, and implications"""
    try:
        results = await rag_service.analyze_document(query=query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")

@router.post("/document/compliance")
async def check_regulatory_compliance(
    query: str = Query(..., description="The query to find documents for compliance analysis")
):
    """Check compliance requirements and obligations in regulations"""
    try:
        results = await rag_service.check_regulatory_compliance(query=query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Regulatory compliance check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Regulatory compliance check failed: {str(e)}")
