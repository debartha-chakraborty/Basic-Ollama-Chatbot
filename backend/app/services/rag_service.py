import os
import json
import logging
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional, List, Dict
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

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalregister.gov/api/v1/documents"

class RAGService:
    def __init__(self):
        self.settings = get_settings()
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chromadb_path)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_client.get_or_create_collection("rag_docs"))
        
        # Use a local embedding model instead of the default OpenAI one
        # Use a very lightweight model that's widely available
        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        
        # Pass the embed_model explicitly
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=embed_model
        )
        
        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("Initialized RAGService with Gemini and ChromaDB.")

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

    async def generate(self, prompt: str, **kwargs) -> str:
        # Retrieve context from vector DB
        retriever = self.index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(prompt)
        context = "\n".join([n.get_content() for n in nodes])
        full_prompt = f"Context:\n{context}\n\nUser: {prompt}\nAssistant:"
        # Call Gemini API
        response = await asyncio.to_thread(self.model.generate_content, full_prompt)
        return response.text

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        retriever = self.index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(prompt)
        context = "\n".join([n.get_content() for n in nodes])
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