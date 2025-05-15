import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import logging
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Any, Union, Callable
from datetime import datetime
from ..config import get_settings
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
            
            # Tool for federal register API searches when local data might be insufficient
            StructuredTool.from_function(
                func=self.search_federal_register_api,
                name="search_federal_register_api",
                description="Search the Federal Register API directly. Use this when the information might be very recent or not in the local database.",
                args_schema=SearchQuery
            )
        ]
        
        # Create the agent prompt with required variables
        agent_prompt = PromptTemplate.from_template("""
        You are an intelligent research assistant that helps find accurate information from government documents.
        
        You have access to the following tools:
        
        {tools}
        
        Use the following format for your responses:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought:{agent_scratchpad}
        """)
        
        # Create the agent
        agent = create_react_agent(self.llm, self.tools, agent_prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    # [Rest of your existing methods remain unchanged...]
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
            # Run the agent synchronously using asyncio.to_thread to prevent blocking
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"input": prompt}
            )
            # Return the agent's output
            return result.get("output", "No relevant information found.")
        except Exception as e:
            logger.error(f"Agent retrieval error: {e}")
            # Fallback to traditional retrieval if agent fails
            retriever = self.index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(prompt)
            return "\n".join([n.get_content() for n in nodes])

    async def generate(self, prompt: str, **kwargs) -> str:
        # Use the agent to retrieve context intelligently
        context = await self.agentic_retrieve(prompt)
        full_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAssistant:"
        # Call Gemini API
        response = await asyncio.to_thread(self.model.generate_content, full_prompt)
        return response.text

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
        """Search directly in the Federal Register API"""
        try:
            encoded_query = quote(query)
            search_url = f"{BASE_URL}?conditions[term]={encoded_query}&per_page=5"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    if response.status != 200:
                        return f"API search failed with status: {response.status}"
                    
                    data = await response.json()
                    results = []
                    
                    for item in data.get("results", []):
                        doc_info = {
                            "title": item.get("title", ""),
                            "document_number": item.get("document_number", ""),
                            "publication_date": item.get("publication_date", ""),
                            "agencies": [agency.get("name", "") for agency in item.get("agencies", [])],
                            "abstract": item.get("abstract", ""),
                            "html_url": item.get("html_url", "")
                        }
                        
                        summary = f"""
                        Document: {doc_info['title']}
                        Document Number: {doc_info['document_number']}
                        Publication Date: {doc_info['publication_date']}
                        Agencies: {', '.join(doc_info['agencies'])}
                        Abstract: {doc_info['abstract']}
                        URL: {doc_info['html_url']}
                        """
                        results.append(summary)
                    
                    if not results:
                        return f"No documents found in Federal Register API for: {query}."
                        
                    return "\n\n---\n\n".join(results)
                    
        except Exception as e:
            logger.error(f"Federal Register API search error: {e}")
            return f"Error searching Federal Register API: {str(e)}"
    
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
                            "agencies": [agency.get("name", "") for agency in item.get("agencies", [])],
                            "excerpts": item.get("excerpts", ""),
                            "html_url": item.get("html_url", ""),
                            "pdf_url": item.get("pdf_url", "")
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
                        Agencies: {', '.join(doc_info['agencies'])}
                        Abstract: {doc_info['abstract']}
                        Excerpts: {doc_info['excerpts']}

                        HTML Content:
                        {html_content}

                        PDF Content:
                        {pdf_content}

                        URL: {doc_info['html_url']}
                        PDF: {doc_info['pdf_url']}
                """
                        # Create metadata for better retrieval
                        # Convert the agencies list to a comma-separated string for ChromaDB compatibility
                        agencies_str = ", ".join(doc_info['agencies']) if doc_info['agencies'] else ""
                        
                        metadata = {
                            "document_number": doc_info['document_number'],
                            "publication_date": doc_info['publication_date'],
                            "type": doc_info['type'],
                            "agencies": agencies_str,  # Store as string instead of list
                            "has_html_content": bool(html_content),
                            "has_pdf_content": bool(pdf_content)
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