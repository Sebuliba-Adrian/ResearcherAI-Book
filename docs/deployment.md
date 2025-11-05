---
layout: default
title: Deployment & CI/CD
---

# Deployment & CI/CD


Building the system was fun. Deploying it reliably? That's where the real engineering happens.


## Deployment Options

I designed three deployment paths:

1. **Local Development**: Instant startup, no Docker
2. **Docker Compose**: Full stack on one machine
3. **Kubernetes** (future): Multi-node, auto-scaling

## Docker Containerization

### Multi-Service Architecture

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Main application
  rag-multiagent:
    build: .
    container_name: rag-multiagent
    depends_on:
      - neo4j
      - qdrant
      - kafka
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - USE_NEO4J=true
      - USE_QDRANT=true
      - USE_KAFKA=true
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=research_password
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    volumes:
      - ./sessions:/app/sessions
      - ./logs:/app/logs
    networks:
      - rag-network

  # Scheduler for automated collection
  rag-scheduler:
    build: .
    command: python -m agents.scheduler
    depends_on:
      - rag-multiagent
    environment:
      - SCHEDULE_INTERVAL=3600
    networks:
      - rag-network

  # Neo4j graph database
  neo4j:
    image: neo4j:5.13
    container_name: rag-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/research_password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - ./volumes/neo4j:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rag-network

  # Qdrant vector database
  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: rag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./volumes/qdrant:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rag-network

  # Zookeeper for Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    container_name: rag-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - ./volumes/zookeeper:/var/lib/zookeeper
    networks:
      - rag-network

  # Kafka event streaming
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    container_name: rag-kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "9094:9094"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: INSIDE://kafka:9092,OUTSIDE://localhost:9094
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: INSIDE
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    volumes:
      - ./volumes/kafka:/var/lib/kafka
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server=localhost:9092"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rag-network

  # Kafka UI for monitoring
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: rag-kafka-ui
    depends_on:
      - kafka
    ports:
      - "8081:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge

volumes:
  neo4j-data:
  qdrant-data:
  zookeeper-data:
  kafka-data:
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs sessions volumes

# Expose port (if running web API)
EXPOSE 8000

# Default command
CMD ["python", "main.py"]
```

## GitHub Actions CI/CD

### Test Workflow

```yaml
# .github/workflows/ci.yml
name: CI - Code Quality

on:
  push:
    branches: [master, main]
  pull_request:
    branches: [master, main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        pip install flake8 black isort mypy

    - name: Lint with flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Check formatting with black
      run: black --check .

    - name: Check imports with isort
      run: isort --check-only .

  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      env:
        USE_NEO4J: false
        USE_QDRANT: false
        USE_KAFKA: false
      run: |
        pytest tests/ --cov=utils --cov=agents --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
```

### Docker Build & Push

```yaml
# .github/workflows/docker.yml
name: Docker Build and Push

on:
  push:
    branches: [master, main]
    tags:
      - 'v*.*.*'
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  validate-docker:
    name: Validate Docker Configuration
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Create .env file
      run: |
        echo "GOOGLE_API_KEY=test-key" > .env
        echo "USE_NEO4J=true" >> .env
        echo "USE_QDRANT=true" >> .env
        echo "USE_KAFKA=false" >> .env

    - name: Validate docker-compose.yml
      run: docker compose config

    - name: Build Docker image (validation only)
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: researcherai:test
        cache-from: type=gha
        cache-to: type=gha,mode=max

  build-and-push:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    needs: validate-docker
    if: github.event_name != 'pull_request' && github.ref == 'refs/heads/master'
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

## Deployment Steps

### 1. Local Development

```bash
# Clone repo
git clone https://github.com/Sebuliba-Adrian/ResearcherAI.git
cd ResearcherAI

# Install dependencies
pip install -r requirements.txt

# Set environment
export GOOGLE_API_KEY="your-key"
export USE_NEO4J=false
export USE_QDRANT=false
export USE_KAFKA=false

# Run
python main.py
```

**Startup time**: < 1 second
**Resource usage**: ~500MB RAM

### 2. Docker Compose (Production)

```bash
# Create .env file
cat > .env << EOF
GOOGLE_API_KEY=your-key-here
USE_NEO4J=true
USE_QDRANT=true
USE_KAFKA=true
NEO4J_PASSWORD=your-secure-password
EOF

# Start all services
docker-compose up -d

# Check health
docker-compose ps

# View logs
docker-compose logs -f rag-multiagent

# Run inside container
docker exec -it rag-multiagent bash
python main.py
```

**Startup time**: ~30 seconds
**Resource usage**: ~4GB RAM (7 containers)

## Configuration Management

### Environment Variables

```bash
# API Keys
GOOGLE_API_KEY=          # Required

# Backend Selection
USE_NEO4J=true           # false for NetworkX
USE_QDRANT=true          # false for FAISS
USE_KAFKA=true           # false for sync

# Neo4j (if USE_NEO4J=true)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Qdrant (if USE_QDRANT=true)
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Kafka (if USE_KAFKA=true)
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Scheduler (optional)
SCHEDULE_INTERVAL=3600   # 1 hour
```

### YAML Configuration

```yaml
# config/settings.yaml
data_sources:
  arxiv:
    enabled: true
    max_results: 10
  semantic_scholar:
    enabled: true
    max_results: 10
  pubmed:
    enabled: true
    max_results: 10

llm:
  model: "gemini-2.0-flash-exp"
  temperature: 0.7
  max_tokens: 2048

retrieval:
  vector_top_k: 10
  graph_depth: 2
  chunk_size: 400
  chunk_overlap: 50

production:
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
  token_budget:
    per_request: 10000
    per_user: 100000
  cache:
    ttl: 3600
    max_size: 1000
```

## Health Checks

All services have health checks:

```python
# utils/health.py
def check_health() -> Dict[str, str]:
    """Check health of all services"""

    status = {}

    # Check Neo4j
    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        status["neo4j"] = "healthy"
    except:
        status["neo4j"] = "unhealthy"

    # Check Qdrant
    try:
        response = requests.get("http://qdrant:6333/health")
        status["qdrant"] = "healthy" if response.ok else "unhealthy"
    except:
        status["qdrant"] = "unhealthy"

    # Check Kafka
    try:
        admin = KafkaAdminClient(bootstrap_servers="kafka:9092")
        admin.list_topics()
        status["kafka"] = "healthy"
    except:
        status["kafka"] = "unhealthy"

    return status
```

## Monitoring & Logs

### Structured Logging

```python
# utils/logger.py
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger("researcherai")
handler = logging.FileHandler("logs/app.log")
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Log Aggregation

```yaml
# docker-compose.yml (add to services)
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./logs:/var/log
      - ./promtail-config.yaml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Scaling Considerations

### Horizontal Scaling

With Kafka, we can scale agents independently:

```yaml
# docker-compose.scale.yml
services:
  rag-multiagent:
    deploy:
      replicas: 3  # Run 3 instances

  rag-scheduler:
    deploy:
      replicas: 2
```

### Load Balancing

```nginx
# nginx.conf
upstream backend {
    least_conn;
    server rag-multiagent-1:8000;
    server rag-multiagent-2:8000;
    server rag-multiagent-3:8000;
}

server {
    listen 80;

    location /api {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## What I Learned

**✅ Wins**

1. **Docker Compose simplifies deployment**: One command, full stack
2. **Health checks prevent silent failures**: Catch issues early
3. **GitHub Actions automates everything**: Push → Test → Build → Deploy
4. **Structured logging helps debugging**: JSON logs are easy to parse

**🤔 Challenges**

1. **Docker build time**: 71 minutes initially (optimized to 5 min with caching)
2. **GHCR permissions**: Required package deletion after repo recreation
3. **Volume permissions**: Kafka/Zookeeper needed specific UIDs
4. **Container networking**: DNS resolution took trial and error

**💡 Insights**

> Good deployment should be boring - automated, repeatable, reliable.

> Health checks are critical - fail fast, recover automatically.

> Caching is essential for CI/CD speed.

## Next: Monitoring

Deployment done! Now let's make sure we can observe what's happening in production.


  <a href="testing">← Back: Testing</a>
  <a href="monitoring">Next: Monitoring & Operations →</a>

