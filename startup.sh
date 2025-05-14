#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting services...${NC}"

# Create required directories if they don't exist
mkdir -p ollama
if [ ! -f ollama/Dockerfile ]; then
  echo -e "${YELLOW}Creating Ollama Dockerfile...${NC}"
  cat > ollama/Dockerfile <<EOF
FROM ollama/ollama:latest

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose the default Ollama port
EXPOSE 11434

# Default command to run Ollama
CMD ["ollama", "serve"]
EOF
fi

# Make sure we have access to the directory structure
mkdir -p backend/app/routers backend/app/services frontend

# Stop any running containers
echo -e "${YELLOW}Stopping any existing containers...${NC}"
docker-compose down

# Start everything
echo -e "${YELLOW}Starting containers...${NC}"
docker-compose up -d

# Function to check service status with retries
check_service() {
  local service_name=$1
  local max_attempts=$2
  local sleep_time=$3
  local endpoint=$4
  
  echo -e "${YELLOW}Checking $service_name status...${NC}"
  
  for ((i=1; i<=max_attempts; i++)); do
    echo -e "Attempt $i/$max_attempts..."
    
    # Check if container exists and is running
    if [ ! "$(docker ps -q -f name=$service_name)" ]; then
      echo -e "${RED}$service_name container is not running${NC}"
      docker-compose logs $service_name | tail -n 50
      return 1
    fi
    
    # If endpoint is specified, check it
    if [ ! -z "$endpoint" ]; then
      # Get container IP
      container_ip=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker-compose ps -q $service_name))
      
      # Try accessing the endpoint
      if docker exec $(docker-compose ps -q $service_name) curl -s -f $endpoint >/dev/null 2>&1; then
        echo -e "${GREEN}$service_name is running and endpoint $endpoint is accessible${NC}"
        return 0
      fi
    else
      echo -e "${GREEN}$service_name container is running${NC}"
      return 0
    fi
    
    echo -e "${YELLOW}Waiting $sleep_time seconds before next attempt...${NC}"
    sleep $sleep_time
  done
  
  echo -e "${RED}Failed to verify $service_name after $max_attempts attempts${NC}"
  return 1
}

# Wait for services to initialize
sleep 10

# Check Ollama
check_service "ollama" 5 5 "http://localhost:11434/" || {
  echo -e "${RED}Ollama service check failed. Check the logs above.${NC}"
  exit 1
}

# Check Backend
check_service "backend" 5 5 "http://localhost:8000/health" || {
  echo -e "${RED}Backend service check failed. Check the logs above.${NC}"
  exit 1
}

# Check Frontend
check_service "frontend" 5 5 || {
  echo -e "${RED}Frontend service check failed. Check the logs above.${NC}"
  exit 1
}

echo -e "${GREEN}All services are running!${NC}"

# Check if any models are available
echo -e "${YELLOW}Checking available models...${NC}"
models=$(docker exec -it $(docker-compose ps -q ollama) ollama list 2>/dev/null)

if [ -z "$models" ] || echo "$models" | grep -q "no models" ; then
  echo -e "${YELLOW}No models found. Pulling llama2...${NC}"
  docker exec -it $(docker-compose ps -q ollama) ollama pull llama2
else
  echo -e "${GREEN}Available models:${NC}"
  echo "$models"
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "Access Streamlit UI at: ${GREEN}http://localhost:8501${NC}"
echo -e "Access FastAPI docs at: ${GREEN}http://localhost:8000/docs${NC}"