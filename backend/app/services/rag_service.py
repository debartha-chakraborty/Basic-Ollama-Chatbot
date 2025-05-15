import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import logging
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any, Union, Callable
from datetime import datetime
from config import get_settings
import PyPDF2
import io
from bs4 import BeautifulSoup

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Document
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import google.generativeai as genai

# LangChain imports for agentic capabilities
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import Tool, StructuredTool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.render import format_tool_to_openai_function
from datetime import datetime, timedelta
from urllib.parse import quote
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalregister.gov/api/v1/documents"

class SearchQuery(BaseModel):
    query: str = Field(description="The query to search for")
    
class DateRangeQuery(BaseModel):
    query: str = Field(description="The query to search for")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    
class AgencyQuery(BaseModel):
    query: str = Field(description="The query to search for")
    agency: str = Field(description="The specific agency to search within")

class DocumentTypeQuery(BaseModel):
    query: str = Field(description="The query to search for")
    doc_type: str = Field(description="Type of document (e.g., 'Rule', 'Proposed Rule', 'Notice')")

class PublicCommentQuery(BaseModel):
    query: str = Field(description="The query to search for")
    comment_status: str = Field(description="Status of public comments (e.g., 'open', 'closed')")
    days_remaining: Optional[int] = Field(default=None, description="Days remaining for comments")

class RegulatoryImpactQuery(BaseModel):
    query: str = Field(description="The query to search for")
    impact_type: str = Field(description="Type of regulatory impact (e.g., 'economic', 'environmental', 'health')")
    significance_level: Optional[str] = Field(default=None, description="Significance level of the impact")

class DocumentRelationshipQuery(BaseModel):
    query: str = Field(description="The query to search for")
    relationship_type: str = Field(description="Type of relationship (e.g., 'amends', 'rescinds', 'supersedes')")
    target_document: Optional[str] = Field(default=None, description="Target document number")

class AgencyHierarchyQuery(BaseModel):
    query: str = Field(description="The query to search for")
    parent_agency: Optional[str] = Field(default=None, description="Parent agency name")
    include_subagencies: bool = Field(default=True, description="Whether to include subagencies")

class DocumentTimelineQuery(BaseModel):
    query: str = Field(description="The query to search for")
    timeline_type: str = Field(description="Type of timeline (e.g., 'rulemaking', 'comment', 'effective')")
    include_related: bool = Field(default=True, description="Whether to include related documents")

class AdvancedSearchQuery(BaseModel):
    query: str = Field(description="The query to search for")
    min_confidence: float = Field(default=0.7, description="Minimum confidence score for results")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    include_metadata: bool = Field(default=True, description="Whether to include metadata in results")

class TopicSearchQuery(BaseModel):
    query: str = Field(description="The query to search for")
    topic: str = Field(description="The specific topic or category to search within")
    include_related: bool = Field(default=True, description="Whether to include related topics")

class CitationSearchQuery(BaseModel):
    query: str = Field(description="The query to search for")
    citation_type: str = Field(description="Type of citation (e.g., 'law', 'regulation', 'policy')")
    jurisdiction: Optional[str] = Field(default=None, description="Specific jurisdiction to search within")

class CrossReferenceQuery(BaseModel):
    document_id: str = Field(description="The document ID to find references for")
    reference_type: str = Field(description="Type of reference (e.g., 'cited_by', 'cites', 'related')")
    depth: int = Field(default=1, description="Depth of reference chain to follow")

class RAGService:
    def __init__(self):
        self.settings = get_settings()
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chromadb_path)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_client.get_or_create_collection("rag_docs"))
        
        # Use a local embedding model instead of the default OpenAI one
        self.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        
        # Pass the embed_model explicitly
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
        
        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Initialize LangChain Gemini model for the agent
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.settings.gemini_api_key,
            temperature=0.1
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Set up the agent with tools
        self._setup_agent()
        
        logger.info("Initialized Agentic RAGService with Gemini, ChromaDB, and LangChain tools.")

    def _setup_agent(self):
        """Set up the agent with appropriate tools for intelligent retrieval"""
        # Create a retriever from the index
        retriever = self.index.as_retriever(similarity_top_k=5)
        
        # Define our tools
        self.tools = [
            # Basic semantic search tool
            create_retriever_tool(
                retriever,
                name="semantic_search",
                description="Searches documents using semantic similarity. Use this for general queries when you don't need filtering by date or agency."
            ),
            
            # Tool for date-filtered searches
            StructuredTool.from_function(
                func=self.date_filtered_search,
                name="date_filtered_search",
                description="Search for documents within a specific date range. Useful when the user mentions time periods or dates.",
                args_schema=DateRangeQuery
            ),
            
            # Tool for agency-specific searches
            StructuredTool.from_function(
                func=self.agency_filtered_search,
                name="agency_filtered_search",
                description="Search for documents from a specific government agency. Use this when the user mentions a specific agency or department.",
                args_schema=AgencyQuery
            ),
            
            # Tool for keyword-based searches with metadata filtering
            StructuredTool.from_function(
                func=self.keyword_search,
                name="keyword_search",
                description="Search for documents using exact keyword matches. Use this when semantic search might not capture the specific terminology.",
                args_schema=SearchQuery
            ),
            
            # Tool for federal register API searches
            StructuredTool.from_function(
                func=self.search_federal_register_api,
                name="search_federal_register_api",
                description="Search the Federal Register API directly. Use this when the information might be very recent or not in the local database.",
                args_schema=SearchQuery
            )
        ]
        
        # Create the agent prompt with required variables and memory
        agent_prompt = PromptTemplate.from_template("""
            You are an expert research assistant specializing in finding precise information from government documents and official sources.
            
            Your goal is to retrieve the most accurate, relevant, and authoritative information to answer questions about government policies, regulations, programs, and services.
            
            You have access to the following tools:
            
            {tools}
            
            Previous conversation history:
            {chat_history}
            
            When researching, follow these principles:
            - Prioritize primary sources (official government websites, legislation, policy documents)
            - Consider the recency and applicability of information
            - Look for specific sections, clauses, or paragraphs that directly address the query
            - Gather sufficient context to ensure accurate interpretation
            - Cross-reference information when possible for verification
            
            Use the following format for your responses:
            
            Question: the input question you must answer
            Thought: carefully analyze what information is needed and where to find it
            Action: the action to take, should be one of [{tool_names}]
            Action Input: precise, targeted input to the action that will yield relevant results
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer based on authoritative government information
            Final Answer: the comprehensive answer to the original question, citing specific government sources
            
            Begin!
            
            Question: {input}
            Thought:{agent_scratchpad}
            """)
        
        # Create the agent with memory
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=agent_prompt
        )
        
        # Create the agent executor with memory
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True,
            memory=self.memory
        )
        
        logger.info("Agent setup completed with tools and memory configured.")

    async def fetch_url_content(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch content from a URL (HTML or PDF)."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/pdf' in content_type:
                        # Handle PDF content
                        pdf_content = await response.read()
                        pdf_file = io.BytesIO(pdf_content)
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                    
                    elif 'text/html' in content_type:
                        # Handle HTML content
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        # Get text content
                        text = soup.get_text(separator='\n', strip=True)
                        return text
                    
                    else:
                        logger.warning(f"Unsupported content type: {content_type}")
                        return ""
                        
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return ""

    async def agentic_retrieve(self, prompt: str) -> str:
        """Use the agent to intelligently retrieve the most relevant information for the prompt"""
        try:
            # Create an enhanced input prompt that encourages better information retrieval
            enhanced_prompt = f"""
            I need to find specific, relevant information from government documents to answer:
            
            {prompt}
            
            When searching, prioritize:
            - Official government sources and documentation
            - Information that directly addresses the core question
            - Recent and up-to-date information when available
            - Complete context to ensure accurate understanding
            """
            
            # Run the agent using the enhanced prompt
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"input": enhanced_prompt}
            )
            
            # Check if we have intermediate steps for debugging
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    logger.debug(f"Agent step: {step}")
            
            # Return the agent's output
            return result.get("output", "No relevant information found.")
        except Exception as e:
            logger.error(f"Agent retrieval error: {e}")
            # Fallback to traditional retrieval if agent fails
            try:
                retriever = self.index.as_retriever(similarity_top_k=3)
                nodes = await asyncio.to_thread(retriever.retrieve, prompt)
                return "\n".join([n.get_content() for n in nodes])
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {fallback_error}")
                return "Unable to retrieve relevant information at this time."

    async def generate(self, prompt: str, **kwargs) -> str:
        # Use the agent to retrieve context intelligently
        context = await self.agentic_retrieve(prompt)
        
        # Create a structured prompt for generation that leverages the retrieved context
        full_prompt = f"""
        Context from government documents:
        {context}
        
        Based on the above official information, please provide a thorough and accurate response to:
        
        User: {prompt}
        
        Your response should:
        - Rely strictly on the provided government information
        - Cite specific regulations, policies, or documents where applicable
        - Be clear and accessible to someone unfamiliar with government terminology
        - Indicate if certain aspects of the question cannot be answered with the available information
        
        Assistant:
        """
        
        # Call Gemini API
        response = await asyncio.to_thread(self.model.generate_content, full_prompt)
        return response.text

    def create_improved_agent_prompt():
        return PromptTemplate.from_template("""
        You are an expert research assistant specializing in finding precise information from government documents and official sources.
        
        Your goal is to retrieve the most accurate, relevant, and authoritative information to answer questions about government policies, regulations, programs, and services.
        
        You have access to the following tools:
        
        {tools}
        
        Previous conversation history:
        {chat_history}
        
        When researching, follow these principles:
        - Prioritize primary sources (official government websites, legislation, policy documents)
        - Consider the recency and applicability of information
        - Look for specific sections, clauses, or paragraphs that directly address the query
        - Gather sufficient context to ensure accurate interpretation
        - Cross-reference information when possible for verification
        
        Use the following format for your responses:
        
        Question: the input question you must answer
        Thought: carefully analyze what information is needed and where to find it
        Action: the action to take, should be one of [{tool_names}]
        Action Input: precise, targeted input to the action that will yield relevant results
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer based on authoritative government information
        Final Answer: the comprehensive answer to the original question, citing specific government sources
        
        Begin!
        
        Question: {input}
        Thought:{agent_scratchpad}
        """)

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        # Use the agent to retrieve context intelligently
        context = await self.agentic_retrieve(prompt)
        full_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAssistant:"
        # Gemini streaming
        stream = self.model.generate_content(full_prompt, stream=True)
        try:
            for chunk in stream:
                if hasattr(chunk, 'text'):
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # Tool implementation methods
    def date_filtered_search(self, query: str, start_date: str, end_date: str) -> str:
        """Search for documents within a specific date range"""
        try:
            # Format dates properly
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Convert to string format that matches our metadata
            start_date_str = start_date_obj.strftime("%Y-%m-%d")
            end_date_str = end_date_obj.strftime("%Y-%m-%d")
            
            # Get documents from ChromaDB with filtering
            collection = self.chroma_client.get_collection("rag_docs")
            
            # First get semantic matches
            docs = collection.query(
                query_texts=[query],
                n_results=10,
                where={"publication_date": {"$gte": start_date_str, "$lte": end_date_str}}
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
                    
            if not results:
                return "No documents found within the specified date range."
                
            return "\n\n---\n\n".join(results)
        except Exception as e:
            logger.error(f"Date filtered search error: {e}")
            return f"Error performing date-filtered search: {str(e)}"

    def agency_filtered_search(self, query: str, agency: str) -> str:
        """Search for documents from a specific agency"""
        try:
            # Get documents from ChromaDB with filtering
            collection = self.chroma_client.get_collection("rag_docs")
            
            # Use contains operator for partial agency name matches
            docs = collection.query(
                query_texts=[query],
                n_results=10,
                where_document={"$contains": agency}
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    # Check if the agency actually appears in the document
                    if agency.lower() in doc.lower():
                        results.append(doc)
                    
            if not results:
                # Try alternative approach with direct metadata match
                docs = collection.query(
                    query_texts=[query],
                    n_results=10,
                    where={"agencies": {"$contains": agency}}
                )
                
                if docs and 'documents' in docs and docs['documents']:
                    for doc in docs['documents'][0]:
                        results.append(doc)
                
            if not results:
                return f"No documents found from agency: {agency}."
                
            return "\n\n---\n\n".join(results)
        except Exception as e:
            logger.error(f"Agency filtered search error: {e}")
            return f"Error performing agency-filtered search: {str(e)}"

    def keyword_search(self, query: str) -> str:
        """Search for documents with exact keyword matches"""
        try:
            # Get documents from ChromaDB with exact matching
            collection = self.chroma_client.get_collection("rag_docs")
            
            # Use contains operator for exact keyword matches in document text
            docs = collection.query(
                query_texts=[query],
                n_results=10,
                where_document={"$contains": query}
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
                    
            if not results:
                return f"No documents found with exact match for: {query}."
                
            return "\n\n---\n\n".join(results)
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return f"Error performing keyword search: {str(e)}"

    async def search_federal_register_api(self, query: str) -> str:
        """Search the Federal Register API directly."""
        try:
            # Build query parameters
            params = {
                "conditions[term]": query,
                "format": "json",
                "per_page": 20
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(BASE_URL, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process and structure the results
                        results = []
                        for item in data.get("results", []):
                            result = {
                                "title": item.get("title"),
                                "type": item.get("type"),
                                "document_number": item.get("document_number"),
                                "publication_date": item.get("publication_date"),
                                "agencies": [
                                    {
                                        "name": agency.get("name"),
                                        "id": agency.get("id"),
                                        "parent_id": agency.get("parent_id"),
                                        "slug": agency.get("slug")
                                    }
                                    for agency in item.get("agencies", [])
                                ],
                                "abstract": item.get("abstract"),
                                "excerpts": item.get("excerpts"),
                                "html_url": item.get("html_url"),
                                "pdf_url": item.get("pdf_url"),
                                "public_inspection_pdf_url": item.get("public_inspection_pdf_url")
                            }
                            results.append(result)
                        
                        # Add pagination info
                        response_data = {
                            "count": data.get("count"),
                            "total_pages": data.get("total_pages"),
                            "next_page_url": data.get("next_page_url"),
                            "results": results
                        }
                        
                        return json.dumps(response_data)
                    else:
                        return json.dumps({
                            "error": f"API request failed with status {response.status}",
                            "status": response.status
                        })
        except Exception as e:
            logger.error(f"Federal Register API search error: {str(e)}")
            return json.dumps({
                "error": str(e),
                "status": "error"
            })

    async def advanced_search(self, query: str, min_confidence: float = 0.7, 
                            max_results: int = 10, include_metadata: bool = True) -> str:
        """Perform advanced search with confidence scoring and metadata filtering."""
        try:
            # Perform semantic search with the query
            results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=max_results * 2).retrieve,
                query
            )
            
            # Filter results by confidence score
            filtered_results = [
                result for result in results 
                if result.score >= min_confidence
            ][:max_results]
            
            # Format results with enhanced metadata
            formatted_results = []
            for result in filtered_results:
                result_dict = {
                    "text": result.text,
                    "score": result.score,
                }
                
                if include_metadata and hasattr(result, "metadata"):
                    metadata = result.metadata
                    result_dict["metadata"] = {
                        "document_number": metadata.get("document_number"),
                        "publication_date": metadata.get("publication_date"),
                        "type": metadata.get("type"),
                        "agencies": metadata.get("agencies", []),
                        "agency_ids": metadata.get("agency_ids", []),
                        "parent_agencies": metadata.get("parent_agencies", []),
                        "agency_slugs": metadata.get("agency_slugs", []),
                        "title": metadata.get("title"),
                        "abstract": metadata.get("abstract"),
                        "has_html_content": metadata.get("has_html_content", False),
                        "has_pdf_content": metadata.get("has_pdf_content", False),
                        "has_public_inspection": metadata.get("has_public_inspection", False)
                    }
                
                formatted_results.append(result_dict)
            
            # Add search statistics
            response_data = {
                "results": formatted_results,
                "total_results": len(results),
                "filtered_results": len(filtered_results),
                "min_confidence": min_confidence,
                "max_results": max_results
            }
            
            return json.dumps(response_data)
        except Exception as e:
            logger.error(f"Advanced search error: {str(e)}")
            return json.dumps({
                "error": str(e),
                "status": "error"
            })

    async def topic_search(self, query: str, topic: str, include_related: bool = True) -> str:
        """Search within specific topics or categories."""
        try:
            # Combine topic and query for better context
            enhanced_query = f"Topic: {topic}. Query: {query}"
            
            # Perform semantic search
            results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=10).retrieve,
                enhanced_query
            )
            
            # Filter and format results
            formatted_results = []
            for result in results:
                if include_related or topic.lower() in result.text.lower():
                    formatted_results.append({
                        "text": result.text,
                        "score": result.score,
                        "topic": topic
                    })
            
            return json.dumps(formatted_results)
        except Exception as e:
            logger.error(f"Topic search error: {str(e)}")
            return f"Error: {str(e)}"

    async def citation_search(self, query: str, citation_type: str, 
                            jurisdiction: Optional[str] = None) -> str:
        """Search for specific types of citations and references."""
        try:
            # Build citation-specific query
            citation_query = f"Citation type: {citation_type}. "
            if jurisdiction:
                citation_query += f"Jurisdiction: {jurisdiction}. "
            citation_query += f"Query: {query}"
            
            # Perform search
            results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=10).retrieve,
                citation_query
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.text,
                    "citation_type": citation_type,
                    "jurisdiction": jurisdiction,
                    "score": result.score
                })
            
            return json.dumps(formatted_results)
        except Exception as e:
            logger.error(f"Citation search error: {str(e)}")
            return f"Error: {str(e)}"

    async def cross_reference_search(self, document_id: str, reference_type: str, 
                                   depth: int = 1) -> str:
        """Find documents that reference or are referenced by a specific document."""
        try:
            # Get the original document
            doc_results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=1).retrieve,
                f"document_id:{document_id}"
            )
            
            if not doc_results:
                return json.dumps({"error": "Document not found"})
            
            original_doc = doc_results[0]
            references = []
            
            # Find references based on type
            if reference_type == "cited_by":
                # Find documents that cite this one
                ref_results = await asyncio.to_thread(
                    self.index.as_retriever(similarity_top_k=10).retrieve,
                    original_doc.text
                )
                references = [r for r in ref_results if r.text != original_doc.text]
            elif reference_type == "cites":
                # Find documents cited by this one
                # This would require parsing citations from the document
                # For now, we'll use a simple text-based approach
                ref_results = await asyncio.to_thread(
                    self.index.as_retriever(similarity_top_k=10).retrieve,
                    f"cited in {original_doc.text}"
                )
                references = [r for r in ref_results if r.text != original_doc.text]
            
            # Format results
            formatted_results = {
                "original_document": {
                    "text": original_doc.text,
                    "score": original_doc.score
                },
                "references": [
                    {
                        "text": ref.text,
                        "score": ref.score,
                        "reference_type": reference_type
                    }
                    for ref in references
                ]
            }
            
            return json.dumps(formatted_results)
        except Exception as e:
            logger.error(f"Cross-reference search error: {str(e)}")
            return f"Error: {str(e)}"

    async def summarize_document(self, query: str) -> str:
        """Generate a concise summary of a document."""
        try:
            # Find the most relevant document
            results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=1).retrieve,
                query
            )
            
            if not results:
                return json.dumps({"error": "No relevant document found"})
            
            document = results[0]
            
            # Generate summary using the LLM
            prompt = f"""Please provide a concise summary of the following document:
            
            {document.text}
            
            Focus on the key points and main arguments."""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return json.dumps({
                "summary": response.text,
                "original_text": document.text,
                "score": document.score
            })
        except Exception as e:
            logger.error(f"Document summarization error: {str(e)}")
            return f"Error: {str(e)}"

    async def compare_documents(self, query: str) -> str:
        """Compare multiple documents for similarities and differences."""
        try:
            # Find relevant documents
            results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=3).retrieve,
                query
            )
            
            if len(results) < 2:
                return json.dumps({"error": "Not enough documents found for comparison"})
            
            # Prepare documents for comparison
            documents = [r.text for r in results]
            
            # Generate comparison using the LLM
            prompt = f"""Please compare the following documents and highlight their similarities and differences:
            
            Document 1:
            {documents[0]}
            
            Document 2:
            {documents[1]}
            
            {'Document 3:' + documents[2] if len(documents) > 2 else ''}
            
            Focus on key points, arguments, and any contradictions or agreements."""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return json.dumps({
                "comparison": response.text,
                "documents": [
                    {
                        "text": doc,
                        "score": results[i].score
                    }
                    for i, doc in enumerate(documents)
                ]
            })
        except Exception as e:
            logger.error(f"Document comparison error: {str(e)}")
            return f"Error: {str(e)}"

    async def ingest_federal_register_documents(self, limit: int = 10) -> List[str]:
        """Fetch, clean, and ingest documents from the Federal Register API into ChromaDB."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(BASE_URL, params={"limit": limit}) as response:
                    response.raise_for_status()
                    data = await response.json()
                    documents = []
                    
                    for item in data.get("results", []):
                        # Extract and structure document information
                        doc_info = {
                            "title": item.get("title", ""),
                            "type": item.get("type", ""),
                            "abstract": item.get("abstract", ""),
                            "document_number": item.get("document_number", ""),
                            "publication_date": item.get("publication_date", ""),
                            "agencies": [
                                {
                                    "name": agency.get("name", ""),
                                    "id": agency.get("id"),
                                    "parent_id": agency.get("parent_id"),
                                    "slug": agency.get("slug")
                                }
                                for agency in item.get("agencies", [])
                            ],
                            "excerpts": item.get("excerpts", ""),
                            "html_url": item.get("html_url", ""),
                            "pdf_url": item.get("pdf_url", ""),
                            "public_inspection_pdf_url": item.get("public_inspection_pdf_url")
                        }
                        
                        # Fetch content from URLs
                        html_content = await self.fetch_url_content(session, doc_info['html_url'])
                        pdf_content = await self.fetch_url_content(session, doc_info['pdf_url'])
                        
                        # Create a structured text representation
                        text = f"""
                        Document Number: {doc_info['document_number']}
                        Title: {doc_info['title']}
                        Type: {doc_info['type']}
                        Publication Date: {doc_info['publication_date']}
                        Agencies: {', '.join([agency['name'] for agency in doc_info['agencies']])}
                        Abstract: {doc_info['abstract']}
                        Excerpts: {doc_info['excerpts']}

                        HTML Content:
                        {html_content}

                        PDF Content:
                        {pdf_content}

                        URL: {doc_info['html_url']}
                        PDF: {doc_info['pdf_url']}
                        """
                        
                        # Create metadata for better retrieval - ensure all values are simple types
                        metadata = {
                            "document_number": doc_info['document_number'],
                            "publication_date": doc_info['publication_date'],
                            "type": doc_info['type'],
                            "agencies": ", ".join([agency['name'] for agency in doc_info['agencies']]),  # Convert list to string
                            "agency_ids": ", ".join([str(agency['id']) for agency in doc_info['agencies'] if agency['id']]),  # Convert list to string
                            "parent_agencies": ", ".join([str(agency['parent_id']) for agency in doc_info['agencies'] if agency['parent_id']]),  # Convert list to string
                            "agency_slugs": ", ".join([agency['slug'] for agency in doc_info['agencies'] if agency['slug']]),  # Convert list to string
                            "has_html_content": str(bool(html_content)),  # Convert boolean to string
                            "has_pdf_content": str(bool(pdf_content)),  # Convert boolean to string
                            "has_public_inspection": str(bool(doc_info['public_inspection_pdf_url'])),  # Convert boolean to string
                            "title": doc_info['title'],
                            "abstract": doc_info['abstract']
                        }
                        
                        # Create document with metadata
                        doc = Document(
                            text=text.strip(),
                            metadata=metadata
                        )
                        documents.append(doc)
                    
                    # Ingest documents into ChromaDB
                    self.index.insert_nodes(documents)
                    logger.info(f"Ingested {len(documents)} documents from Federal Register.")
                    return [doc.text for doc in documents]
                    
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise

    async def document_type_search(self, query: str, doc_type: str) -> str:
        """Search for specific types of Federal Register documents."""
        try:
            # Build type-specific query
            type_query = f"Document type: {doc_type}. Query: {query}"
            
            # Search in ChromaDB
            collection = self.chroma_client.get_collection("rag_docs")
            docs = collection.query(
                query_texts=[type_query],
                n_results=10,
                where={"type": doc_type}
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
            
            return json.dumps({
                "documents": results,
                "type": doc_type,
                "count": len(results)
            })
        except Exception as e:
            logger.error(f"Document type search error: {str(e)}")
            return f"Error: {str(e)}"

    async def public_comment_search(self, query: str, comment_status: str, 
                                  days_remaining: Optional[int] = None) -> str:
        """Search for documents with specific public comment status."""
        try:
            # Build comment-specific query
            comment_query = f"Comment status: {comment_status}. Query: {query}"
            if days_remaining is not None:
                comment_query += f" Days remaining: {days_remaining}"
            
            # Search in ChromaDB
            collection = self.chroma_client.get_collection("rag_docs")
            where_clause = {"comment_status": comment_status}
            if days_remaining is not None:
                where_clause["days_remaining"] = {"$lte": days_remaining}
            
            docs = collection.query(
                query_texts=[comment_query],
                n_results=10,
                where=where_clause
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
            
            return json.dumps({
                "documents": results,
                "comment_status": comment_status,
                "count": len(results)
            })
        except Exception as e:
            logger.error(f"Public comment search error: {str(e)}")
            return f"Error: {str(e)}"

    async def regulatory_impact_search(self, query: str, impact_type: str,
                                     significance_level: Optional[str] = None) -> str:
        """Search for documents based on regulatory impact."""
        try:
            # Build impact-specific query
            impact_query = f"Impact type: {impact_type}. Query: {query}"
            if significance_level:
                impact_query += f" Significance: {significance_level}"
            
            # Search in ChromaDB
            collection = self.chroma_client.get_collection("rag_docs")
            where_clause = {"impact_type": impact_type}
            if significance_level:
                where_clause["significance_level"] = significance_level
            
            docs = collection.query(
                query_texts=[impact_query],
                n_results=10,
                where=where_clause
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
            
            return json.dumps({
                "documents": results,
                "impact_type": impact_type,
                "significance_level": significance_level,
                "count": len(results)
            })
        except Exception as e:
            logger.error(f"Regulatory impact search error: {str(e)}")
            return f"Error: {str(e)}"

    async def document_relationship_search(self, query: str, relationship_type: str,
                                         target_document: Optional[str] = None) -> str:
        """Search for documents based on their relationships to other documents."""
        try:
            # Build relationship-specific query
            rel_query = f"Relationship type: {relationship_type}. Query: {query}"
            if target_document:
                rel_query += f" Target document: {target_document}"
            
            # Search in ChromaDB
            collection = self.chroma_client.get_collection("rag_docs")
            where_clause = {"relationship_type": relationship_type}
            if target_document:
                where_clause["target_document"] = target_document
            
            docs = collection.query(
                query_texts=[rel_query],
                n_results=10,
                where=where_clause
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
            
            return json.dumps({
                "documents": results,
                "relationship_type": relationship_type,
                "target_document": target_document,
                "count": len(results)
            })
        except Exception as e:
            logger.error(f"Document relationship search error: {str(e)}")
            return f"Error: {str(e)}"

    async def agency_hierarchy_search(self, query: str, parent_agency: Optional[str] = None,
                                    include_subagencies: bool = True) -> str:
        """Search for documents based on agency hierarchy."""
        try:
            # Build hierarchy-specific query
            hierarchy_query = f"Query: {query}"
            if parent_agency:
                hierarchy_query += f" Parent agency: {parent_agency}"
            
            # Search in ChromaDB
            collection = self.chroma_client.get_collection("rag_docs")
            where_clause = {}
            if parent_agency:
                if include_subagencies:
                    where_clause["agencies"] = {"$contains": parent_agency}
                else:
                    where_clause["parent_agency"] = parent_agency
            
            docs = collection.query(
                query_texts=[hierarchy_query],
                n_results=10,
                where=where_clause
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
            
            return json.dumps({
                "documents": results,
                "parent_agency": parent_agency,
                "include_subagencies": include_subagencies,
                "count": len(results)
            })
        except Exception as e:
            logger.error(f"Agency hierarchy search error: {str(e)}")
            return f"Error: {str(e)}"

    async def document_timeline_search(self, query: str, timeline_type: str,
                                     include_related: bool = True) -> str:
        """Search for documents based on their timeline in the rulemaking process."""
        try:
            # Build timeline-specific query
            timeline_query = f"Timeline type: {timeline_type}. Query: {query}"
            
            # Search in ChromaDB
            collection = self.chroma_client.get_collection("rag_docs")
            where_clause = {"timeline_type": timeline_type}
            
            docs = collection.query(
                query_texts=[timeline_query],
                n_results=10,
                where=where_clause
            )
            
            results = []
            if docs and 'documents' in docs and docs['documents']:
                for doc in docs['documents'][0]:
                    results.append(doc)
            
            # Get related documents if requested
            related_docs = []
            if include_related and results:
                related_query = f"Related to: {results[0]}"
                related = collection.query(
                    query_texts=[related_query],
                    n_results=5
                )
                if related and 'documents' in related and related['documents']:
                    related_docs = related['documents'][0]
            
            return json.dumps({
                "documents": results,
                "related_documents": related_docs,
                "timeline_type": timeline_type,
                "count": len(results)
            })
        except Exception as e:
            logger.error(f"Document timeline search error: {str(e)}")
            return f"Error: {str(e)}"

    async def analyze_document(self, query: str) -> str:
        """Perform detailed analysis of a document's content, structure, and implications."""
        try:
            # Find the most relevant document
            results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=1).retrieve,
                query
            )
            
            if not results:
                return json.dumps({"error": "No relevant document found"})
            
            document = results[0]
            
            # Generate analysis using the LLM
            prompt = f"""Please analyze the following Federal Register document in detail:
            
            {document.text}
            
            Consider the following aspects:
            1. Key provisions and requirements
            2. Regulatory impact and implications
            3. Implementation timeline and deadlines
            4. Compliance requirements
            5. Stakeholder considerations
            6. Legal and policy implications
            
            Provide a comprehensive analysis."""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return json.dumps({
                "analysis": response.text,
                "document": document.text,
                "score": document.score
            })
        except Exception as e:
            logger.error(f"Document analysis error: {str(e)}")
            return f"Error: {str(e)}"

    async def check_regulatory_compliance(self, query: str) -> str:
        """Check compliance requirements and obligations in regulations."""
        try:
            # Find relevant documents
            results = await asyncio.to_thread(
                self.index.as_retriever(similarity_top_k=3).retrieve,
                query
            )
            
            if not results:
                return json.dumps({"error": "No relevant documents found"})
            
            # Generate compliance analysis using the LLM
            prompt = f"""Please analyze the compliance requirements in the following Federal Register documents:
            
            {[r.text for r in results]}
            
            Consider:
            1. Specific compliance obligations
            2. Deadlines and timelines
            3. Required actions and documentation
            4. Exemptions and exceptions
            5. Enforcement mechanisms
            6. Penalties for non-compliance
            
            Provide a detailed compliance analysis."""
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return json.dumps({
                "compliance_analysis": response.text,
                "documents": [r.text for r in results],
                "scores": [r.score for r in results]
            })
        except Exception as e:
            logger.error(f"Regulatory compliance check error: {str(e)}")
            return f"Error: {str(e)}"

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        messages = self.memory.chat_memory.messages
        history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.memory.clear()