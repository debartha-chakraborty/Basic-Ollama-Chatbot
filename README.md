# Ollama FastAPI Streamlit Application

A dockerized application featuring a FastAPI backend that connects to Ollama for AI model inference, with a Streamlit frontend for user interaction.

## Features

- **FastAPI Backend**: RESTful API service that handles requests to Ollama
- **Ollama Integration**: Connects to Ollama for text generation using various models
- **Streamlit Frontend**: User-friendly web interface for interacting with AI models
- **Docker Integration**: Fully containerized setup for easy deployment
- **Streaming Support**: Real-time streamed responses from AI models

## Prerequisites

- Docker
- Docker Compose
- At least 16GB RAM (for running larger models)
- Internet connection (for pulling Docker images and models)

## Getting Started

1. Clone this repository:

   ```
   git clone <repository-url>
   cd ollama-fastapi-streamlit
   ```

2. Start the application using Docker Compose:

   ```
   docker-compose up -d
   ```

3. Access the services:
   - Streamlit UI: http://localhost:8501
   - FastAPI Backend: http://localhost:8000
   - FastAPI Docs: http://localhost:8000/docs

## Available Models

By default, no models are pre-loaded. You should download at least one model using Ollama. You can:

1. Connect to the Ollama container:

   ```
   docker exec -it ollama-fastapi-streamlit_ollama_1 /bin/bash
   ```

2. Pull a model (e.g., Llama2):
   ```
   ollama pull llama2
   ```

## API Endpoints

The backend provides the following API endpoints:

- `GET /api/v1/models`: List available models
- `POST /api/v1/generate`: Generate text (non-streaming)
- `POST /api/v1/generate/stream`: Generate text with streaming response

## Configuration

The application can be configured using environment variables:

**Backend**:

- `OLLAMA_HOST`: Hostname for Ollama (default: "ollama")
- `OLLAMA_PORT`: Port for Ollama (default: 11434)

**Frontend**:

- `BACKEND_URL`: URL for the backend service (default: "http://backend:8000")

## Project Structure

```
ollama-fastapi-streamlit/
├── backend/              # FastAPI backend service
│   ├── app/              # Application code
│   │   ├── main.py       # Main FastAPI application
│   │   ├── routers/      # API route definitions
│   │   ├── services/     # Service layer for Ollama
│   │   └── config.py     # Configuration settings
│   ├── Dockerfile        # Docker config for backend
│   └── requirements.txt  # Python dependencies
├── frontend/             # Streamlit frontend
│   ├── app.py            # Streamlit application
│   ├── Dockerfile        # Docker config for frontend
│   └── requirements.txt  # Python dependencies
└── docker-compose.yml    # Docker compose configuration
```

## Customization

- **Adding more models**: Pull additional models using Ollama
- **Adjusting parameters**: Modify the Streamlit UI to add additional parameters
- **Extending the API**: Add new endpoints in the backend for additional functionality

## Troubleshooting

- **Connectivity issues**: Ensure all containers are running with `docker-compose ps`
- **Model loading errors**: Check Ollama logs with `docker-compose logs ollama`
- **Backend errors**: Check backend logs with `docker-compose logs backend`
