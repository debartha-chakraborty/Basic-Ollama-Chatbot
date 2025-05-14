import httpx
import json
import logging
from typing import AsyncGenerator, Optional
from config import get_settings

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://{self.settings.ollama_host}:{self.settings.ollama_port}"
        self.client = httpx.AsyncClient(timeout=60.0)
        self.default_model = "gemma2:2b"  # Single default model
        logger.info(f"Initialized Ollama service with model: {self.default_model}")

    async def generate(self, prompt: str, model: str = "gemma2:2b", **kwargs):
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, **kwargs}
        
        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            # Handle potential multi-JSON responses
            try:
                return response.json().get("response", "")
            except json.JSONDecodeError:
                # Fallback: manually parse line-delimited JSON
                full_response = ""
                for line in response.text.splitlines():
                    if line.strip():
                        data = json.loads(line)
                        full_response += data.get("response", "")
                return full_response
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def generate_stream(self, prompt: str, model: str = "gemma2:2b", **kwargs):
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": True, **kwargs}
        
        try:
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                
                buffer = ""
                async for chunk in response.aiter_text():
                    for line in chunk.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield f"data: {json.dumps({'text': data['response']})}\n\n"
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON chunk: {line}")
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"