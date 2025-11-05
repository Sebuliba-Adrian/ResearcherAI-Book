---
title: "Observability & LLM Monitoring"
---

# Chapter 7A: Observability & LLM Monitoring

Once ResearcherAI is deployed to production, we need visibility into what's happening. This chapter covers comprehensive observability for our multi-agent system, with special focus on LLM-specific monitoring using LangFuse and LangSmith - tools designed specifically for tracking language model applications.

## Why LLM-Specific Observability Matters

Traditional monitoring (CPU, memory, response times) isn't enough for LLM applications. We need to track:

- **LLM API costs**: Each Gemini API call costs money
- **Token usage**: Input/output tokens determine costs
- **Prompt quality**: Are our prompts effective?
- **Model responses**: What is the LLM actually generating?
- **Conversation flows**: How do multi-turn conversations perform?
- **Agent coordination**: Which agents are being called and why?
- **Retrieval quality**: Are we finding relevant documents?
- **Latency breakdown**: Where is time being spent?

### The Problem

Without LLM observability, you're flying blind:

```python
# What you see:
response = llm.generate(prompt)
# Cost: ???
# Tokens: ???
# Quality: ???
# Why this response: ???
```

With proper observability:

```python
# What you see:
response = llm.generate(prompt)
# Cost: $0.0023
# Input tokens: 450
# Output tokens: 180
# Latency: 1.2s
# Model: gemini-1.5-flash
# Prompt template: research_query_v3
# Retrieved docs: 5 (avg relevance: 0.87)
# User feedback: 👍
```

## LLM Monitoring Tools Overview

### LangFuse (Open Source)

**What it does:**
- Traces all LLM calls with full context
- Tracks costs and token usage automatically
- Provides prompt management and versioning
- Supports user feedback collection
- Self-hosted or cloud-hosted
- Free open-source version available

**Best for:**
- Cost-conscious teams
- Self-hosted requirements
- Open-source stack preference
- Full data control

### LangSmith (LangChain's Platform)

**What it does:**
- Deep integration with LangChain
- Automatic tracing of chains and agents
- Prompt playground for testing
- Dataset management for evaluation
- Production monitoring dashboards
- Commercial product by LangChain

**Best for:**
- LangChain-heavy applications
- Teams already using LangChain
- Rapid development and iteration
- Managed solution preference

### Our Approach

For Researcher AI, I'll show you how to implement both:
1. **LangFuse** for production monitoring (self-hosted)
2. **LangSmith** integration for LangChain components
3. **Prometheus + Grafana** for infrastructure metrics
4. **Jaeger** for distributed tracing
5. **Loki** for centralized logging

## Setting Up LangFuse

### Step 1: Deploy LangFuse with Docker Compose

Create `docker-compose.langfuse.yml`:

```yaml
version: '3.8'

services:
  langfuse-server:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      # Database
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-db:5432/langfuse

      # Security
      NEXTAUTH_SECRET: ${LANGFUSE_SECRET}  # Generate with: openssl rand -base64 32
      NEXTAUTH_URL: http://localhost:3000
      SALT: ${LANGFUSE_SALT}  # Generate with: openssl rand -base64 32

      # Features
      TELEMETRY_ENABLED: "false"  # Disable telemetry for privacy

      # Optional: S3 for media storage
      # S3_ENDPOINT: https://nyc3.digitaloceanspaces.com
      # S3_ACCESS_KEY_ID: ${S3_ACCESS_KEY}
      # S3_SECRET_ACCESS_KEY: ${S3_SECRET_KEY}
      # S3_BUCKET_NAME: langfuse-media
    depends_on:
      - langfuse-db
    restart: unless-stopped

  langfuse-db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
    volumes:
      - langfuse-db-data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  langfuse-db-data:
```

Deploy:

```bash
# Set secrets
export LANGFUSE_SECRET=$(openssl rand -base64 32)
export LANGFUSE_SALT=$(openssl rand -base64 32)

# Start LangFuse
docker-compose -f docker-compose.langfuse.yml up -d

# Check status
docker-compose -f docker-compose.langfuse.yml ps

# Access UI
open http://localhost:3000
```

### Step 2: Create API Keys

```bash
# 1. Open LangFuse UI: http://localhost:3000
# 2. Sign up for an account
# 3. Create a new project: "ResearcherAI"
# 4. Generate API keys: Settings → API Keys
# 5. Save:
#    - Public Key: pk-lf-...
#    - Secret Key: sk-lf-...
```

### Step 3: Integrate LangFuse in Application

Install LangFuse SDK:

```bash
pip install langfuse
```

Update `agents/reasoner.py`:

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import google.generativeai as genai

class ReasoningAgent:
    def __init__(self):
        # Initialize LangFuse
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        )

        # Initialize Gemini
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @observe(name="reasoning-agent")
    def reason(self, query: str, context: dict) -> dict:
        """
        Perform reasoning with full LangFuse tracing.
        """
        # Start a new trace
        trace = self.langfuse.trace(
            name="research-query",
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            metadata={
                "query": query,
                "retrieved_docs": len(context.get("documents", [])),
                "graph_nodes": len(context.get("graph_data", {}).get("nodes", []))
            }
        )

        # 1. Prepare prompt (tracked as generation)
        generation = trace.generation(
            name="prepare-prompt",
            model="gemini-1.5-flash",
            model_parameters={
                "temperature": 0.7,
                "max_output_tokens": 2048
            },
            input=query,
            metadata={
                "prompt_template": "research_assistant_v3",
                "context_docs": len(context.get("documents", []))
            }
        )

        prompt = self._build_prompt(query, context)
        generation.update(
            prompt=prompt,
            metadata={"prompt_tokens": self._count_tokens(prompt)}
        )

        # 2. Call LLM
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Track successful generation
            generation.end(
                output=response_text,
                metadata={
                    "output_tokens": self._count_tokens(response_text),
                    "finish_reason": "stop",
                    "safety_ratings": self._parse_safety_ratings(response)
                },
                usage={
                    "input": self._count_tokens(prompt),
                    "output": self._count_tokens(response_text),
                    "total": self._count_tokens(prompt) + self._count_tokens(response_text),
                    "unit": "TOKENS"
                },
                level="DEFAULT",
                status_message="Success"
            )

        except Exception as e:
            # Track errors
            generation.end(
                level="ERROR",
                status_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
            raise

        # 3. Post-process and return
        result = {
            "answer": response_text,
            "sources": context.get("documents", []),
            "metadata": {
                "trace_id": trace.id,
                "model": "gemini-1.5-flash",
                "tokens": self._count_tokens(prompt) + self._count_tokens(response_text)
            }
        }

        trace.update(
            output=result,
            metadata={"success": True}
        )

        return result

    def _count_tokens(self, text: str) -> int:
        """Estimate tokens (Gemini uses ~1 token per 4 chars)"""
        return len(text) // 4

    def _parse_safety_ratings(self, response) -> dict:
        """Extract safety ratings from Gemini response"""
        try:
            return {
                rating.category.name: rating.probability.name
                for rating in response.candidates[0].safety_ratings
            }
        except:
            return {}

    @observe(name="build-prompt")
    def _build_prompt(self, query: str, context: dict) -> str:
        """Build prompt with context"""
        langfuse_context.update_current_observation(
            input={"query": query, "context_keys": list(context.keys())},
            metadata={"template": "research_assistant_v3"}
        )

        documents = context.get("documents", [])
        graph_data = context.get("graph_data", {})

        prompt = f"""You are a research assistant helping with academic research.

Query: {query}

Retrieved Documents ({len(documents)}):
{self._format_documents(documents)}

Knowledge Graph:
{self._format_graph(graph_data)}

Provide a comprehensive answer based on the above context.
"""

        langfuse_context.update_current_observation(
            output=prompt,
            metadata={"prompt_length": len(prompt)}
        )

        return prompt
```

### Step 4: Track User Feedback

```python
# In your API endpoint
@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Collect user feedback on LLM responses.
    """
    langfuse.score(
        trace_id=feedback.trace_id,
        name="user-feedback",
        value=1 if feedback.is_helpful else 0,
        comment=feedback.comment
    )

    return {"status": "success"}
```

### Step 5: View Traces in LangFuse

Access the LangFuse UI to see:

1. **Traces Dashboard**
   - All LLM calls
   - Response times
   - Token usage
   - Costs

2. **Prompt Management**
   - Version your prompts
   - A/B test different versions
   - Track performance by prompt

3. **User Feedback**
   - See which responses users liked
   - Identify problematic queries
   - Iterate on prompts

4. **Cost Analysis**
   - Daily/weekly/monthly costs
   - Cost per user
   - Cost per model
   - Trend analysis

## Setting Up LangSmith

### Step 1: Create LangSmith Account

```bash
# 1. Sign up at https://smith.langchain.com/
# 2. Create a new project: "ResearcherAI"
# 3. Get API key from Settings
```

### Step 2: Configure Environment

```bash
# Add to .env
LANGSMITH_API_KEY=your-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ResearcherAI
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Step 3: Integrate with LangChain Code

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import LangChainTracer

class OrchestratorAgent:
    def __init__(self):
        # LangSmith automatically traces when env vars are set
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Create chain with prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research coordination agent."),
            ("user", "{query}")
        ])

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            metadata={"agent": "orchestrator"}  # Tagged in LangSmith
        )

    async def orchestrate(self, query: str) -> dict:
        """
        Orchestrate multi-agent workflow.
        LangSmith automatically traces all chain calls.
        """
        # This call is automatically tracked in LangSmith
        result = await self.chain.ainvoke(
            {"query": query},
            config={
                "metadata": {
                    "user_query": query,
                    "session_id": "session-123"
                },
                "tags": ["production", "research-query"]
            }
        )

        return result
```

### Step 4: Add Custom Evaluations

```python
from langsmith import Client
from langsmith.evaluation import evaluate

# Initialize LangSmith client
client = Client()

# Define evaluation dataset
dataset_name = "research-queries-test-set"
dataset = client.create_dataset(dataset_name)

# Add test cases
client.create_examples(
    dataset_id=dataset.id,
    inputs=[
        {"query": "What are the latest advances in RAG systems?"},
        {"query": "Explain knowledge graph embeddings"},
        {"query": "How does vector search work?"}
    ],
    outputs=[
        {"expected_topics": ["retrieval", "generation", "embeddings"]},
        {"expected_topics": ["knowledge graphs", "embeddings", "representations"]},
        {"expected_topics": ["similarity search", "embeddings", "indexing"]}
    ]
)

# Define evaluator
def relevance_evaluator(run, example):
    """Check if response covers expected topics"""
    response = run.outputs.get("answer", "")
    expected = example.outputs.get("expected_topics", [])

    # Simple keyword matching (use LLM for better evaluation)
    covered = sum(1 for topic in expected if topic.lower() in response.lower())
    score = covered / len(expected) if expected else 0

    return {
        "key": "topic_coverage",
        "score": score,
        "comment": f"Covered {covered}/{len(expected)} expected topics"
    }

# Run evaluation
results = evaluate(
    lambda inputs: orchestrator.orchestrate(inputs["query"]),
    data=dataset_name,
    evaluators=[relevance_evaluator],
    experiment_prefix="production-eval"
)

print(f"Average topic coverage: {results['topic_coverage']}")
```

## Comprehensive Observability Stack

Now let's add traditional monitoring alongside LLM-specific tools.

### Architecture

```
Application
  ↓
┌─────────────────────────────────────┐
│ LangFuse (LLM Traces)               │
│ - All LLM calls                     │
│ - Token usage                       │
│ - Costs                             │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Prometheus (Metrics)                │
│ - Request rates                     │
│ - Error rates                       │
│ - Response times                    │
│ - Resource usage                    │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Jaeger (Distributed Tracing)        │
│ - Request flows                     │
│ - Service dependencies              │
│ - Bottleneck identification         │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Loki (Logs)                         │
│ - Application logs                  │
│ - Error logs                        │
│ - Audit logs                        │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Grafana (Visualization)             │
│ - Dashboards                        │
│ - Alerts                            │
│ - Unified view                      │
└─────────────────────────────────────┘
```

### Deploy Observability Stack

`docker-compose.observability.yml`:

```yaml
version: '3.8'

services:
  # Prometheus - Metrics collection
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  # Grafana - Visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
      - loki
    restart: unless-stopped

  # Jaeger - Distributed tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"  # UI
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
    restart: unless-stopped

  # Loki - Log aggregation
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
      - loki-data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    restart: unless-stopped

  # Promtail - Log shipper
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
  loki-data:
```

### Prometheus Configuration

`prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # ResearcherAI application
  - job_name: 'researcherai'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'

  # Neo4j
  - job_name: 'neo4j'
    static_configs:
      - targets: ['neo4j:2004']

  # Qdrant
  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'

  # Kafka (if using Kafka Exporter)
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka-exporter:9308']

  # Node Exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Instrument Application with Prometheus

```python
# utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

# Define metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'type']  # type: input or output
)

llm_cost_total = Counter(
    'llm_cost_total',
    'Total LLM costs in USD',
    ['model']
)

llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

active_sessions = Gauge(
    'active_sessions',
    'Number of active user sessions'
)

query_requests_total = Counter(
    'query_requests_total',
    'Total research queries',
    ['status']
)

# Metrics endpoint
from fastapi import Response

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

Update agents to record metrics:

```python
# agents/reasoner.py
from utils.metrics import (
    llm_requests_total,
    llm_tokens_total,
    llm_cost_total,
    llm_latency_seconds
)
import time

class ReasoningAgent:
    def reason(self, query: str, context: dict) -> dict:
        start_time = time.time()
        model = "gemini-1.5-flash"

        try:
            # Make LLM call
            response = self.model.generate_content(prompt)

            # Calculate metrics
            input_tokens = self._count_tokens(prompt)
            output_tokens = self._count_tokens(response.text)
            total_tokens = input_tokens + output_tokens

            # Gemini pricing (example)
            cost = (input_tokens * 0.00001875 + output_tokens * 0.000075) / 1000

            # Record metrics
            llm_requests_total.labels(model=model, status='success').inc()
            llm_tokens_total.labels(model=model, type='input').inc(input_tokens)
            llm_tokens_total.labels(model=model, type='output').inc(output_tokens)
            llm_cost_total.labels(model=model).inc(cost)

            # Record latency
            latency = time.time() - start_time
            llm_latency_seconds.labels(model=model).observe(latency)

            return {
                "answer": response.text,
                "metadata": {
                    "tokens": total_tokens,
                    "cost": cost,
                    "latency": latency
                }
            }

        except Exception as e:
            llm_requests_total.labels(model=model, status='error').inc()
            latency = time.time() - start_time
            llm_latency_seconds.labels(model=model).observe(latency)
            raise
```

### Create Grafana Dashboards

`grafana/dashboards/researcherai.json`:

```json
{
  "dashboard": {
    "title": "ResearcherAI - LLM Monitoring",
    "panels": [
      {
        "title": "LLM Requests per Minute",
        "targets": [
          {
            "expr": "rate(llm_requests_total[1m])",
            "legendFormat": "{{model}} - {{status}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Total LLM Cost (Last 24h)",
        "targets": [
          {
            "expr": "sum(increase(llm_cost_total[24h]))",
            "legendFormat": "Total Cost"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Token Usage by Model",
        "targets": [
          {
            "expr": "sum by (model) (rate(llm_tokens_total[5m]))",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "LLM Latency (95th percentile)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_latency_seconds_bucket[5m]))",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(llm_requests_total{status=\"error\"}[5m]) / rate(llm_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Sessions",
        "targets": [
          {
            "expr": "active_sessions",
            "legendFormat": "Sessions"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

## Alerting

### Configure Prometheus Alerts

`prometheus/alerts.yml`:

```yaml
groups:
  - name: llm_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighLLMErrorRate
        expr: |
          rate(llm_requests_total{status="error"}[5m])
          / rate(llm_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High LLM error rate detected"
          description: "LLM error rate is {{ $value | humanizePercentage }}"

      # High cost
      - alert: HighDailyCost
        expr: |
          sum(increase(llm_cost_total[1h])) * 24 > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Daily LLM cost projection exceeds $100"
          description: "Projected daily cost: ${{ $value | humanize }}"

      # High latency
      - alert: HighLLMLatency
        expr: |
          histogram_quantile(0.95,
            rate(llm_latency_seconds_bucket[5m])
          ) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile LLM latency > 5s"
          description: "LLM latency: {{ $value | humanizeDuration }}"

      # Service down
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.instance }} has been down for 1 minute"
```

### Configure Alertmanager

`alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack'

receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'email'
    email_configs:
      - to: 'ops@researcherai.com'
        from: 'alerts@researcherai.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alerts@researcherai.com'
        auth_password: '${SMTP_PASSWORD}'
```

## Production Monitoring Checklist

### ✅ LLM-Specific Monitoring

- [ ] LangFuse deployed and tracking all LLM calls
- [ ] Token usage tracked per model
- [ ] Costs calculated and monitored
- [ ] Prompt versions managed
- [ ] User feedback collection enabled
- [ ] LangSmith integrated for LangChain components
- [ ] Evaluation datasets created
- [ ] Cost alerts configured

### ✅ Infrastructure Monitoring

- [ ] Prometheus collecting metrics
- [ ] Grafana dashboards created
- [ ] Alert rules defined
- [ ] Alertmanager configured
- [ ] Slack/email notifications set up
- [ ] Jaeger tracing distributed requests
- [ ] Loki aggregating logs

### ✅ Application Monitoring

- [ ] Request rates tracked
- [ ] Error rates tracked
- [ ] Latency percentiles (p50, p95, p99)
- [ ] Resource usage (CPU, memory)
- [ ] Database query performance
- [ ] Cache hit rates
- [ ] Queue depths (Kafka)

### ✅ Business Metrics

- [ ] Active users
- [ ] Queries per user
- [ ] Cost per query
- [ ] User satisfaction (feedback)
- [ ] Popular query types
- [ ] Retention metrics

## Debugging in Production

### Trace a Specific Request

1. **In LangFuse:**
   ```
   - Find trace by trace_id
   - See full prompt and response
   - Check token usage and cost
   - View all sub-spans (retrieval, reasoning, etc.)
   ```

2. **In Jaeger:**
   ```
   - Find trace by trace_id
   - See timing breakdown
   - Identify slow services
   - Check error stack traces
   ```

3. **In Grafana:**
   ```
   - Filter logs by trace_id
   - See all log messages
   - Check correlated metrics
   ```

### Investigate High Costs

```promql
# Top 10 expensive queries (last hour)
topk(10,
  sum by (user_id) (increase(llm_cost_total[1h]))
)

# Cost by model
sum by (model) (increase(llm_cost_total[24h]))

# Cost trend over time
sum(rate(llm_cost_total[5m])) * 86400  # Daily projection
```

### Identify Slow Queries

```promql
# Slowest endpoints
topk(10,
  histogram_quantile(0.95,
    sum by (endpoint, le) (rate(request_duration_seconds_bucket[5m]))
  )
)

# LLM latency breakdown
histogram_quantile(0.95,
  sum by (model, le) (rate(llm_latency_seconds_bucket[5m]))
)
```

## Cost Optimization Strategies

### 1. Model Selection

```python
# Use cheaper models for simple tasks
class ModelSelector:
    def select_model(self, complexity: str) -> str:
        if complexity == "simple":
            return "gemini-1.5-flash"  # Cheaper, faster
        elif complexity == "medium":
            return "gemini-1.5-pro"
        else:
            return "gemini-1.5-pro-002"  # Most capable

    def estimate_complexity(self, query: str) -> str:
        # Simple heuristic
        if len(query) < 50 and "?" in query:
            return "simple"
        elif "explain" in query.lower() or "analyze" in query.lower():
            return "medium"
        else:
            return "complex"
```

### 2. Prompt Optimization

```python
# Track and optimize prompts
@observe(name="prompt-optimization")
def optimize_prompt(query: str, context: dict) -> str:
    """
    Use shorter prompts when possible.
    LangFuse tracks performance by prompt version.
    """
    # Version 1: Very detailed (expensive)
    if os.getenv("PROMPT_VERSION") == "v1":
        return f"""[Very long detailed instructions...]
        Query: {query}
        Context: {json.dumps(context, indent=2)}
        [More detailed instructions...]"""

    # Version 2: Concise (cheaper, test if effective)
    elif os.getenv("PROMPT_VERSION") == "v2":
        return f"""Answer this research query concisely:
        {query}

        Context: {self._format_context_briefly(context)}"""

    # A/B test in LangFuse to see which performs better
```

### 3. Caching

```python
from functools import lru_cache
import hashlib

class CachedLLM:
    def __init__(self):
        self.cache = {}

    def generate(self, prompt: str, **kwargs) -> str:
        # Hash prompt for cache key
        cache_key = hashlib.md5(
            (prompt + str(kwargs)).encode()
        ).hexdigest()

        # Check cache
        if cache_key in self.cache:
            llm_requests_total.labels(
                model="cached",
                status="cache_hit"
            ).inc()
            return self.cache[cache_key]

        # Call LLM
        response = self.model.generate_content(prompt)

        # Cache response
        self.cache[cache_key] = response.text

        return response.text
```

### 4. Batch Processing

```python
async def process_batch(queries: List[str]) -> List[str]:
    """
    Process multiple queries in parallel to reduce overhead.
    """
    tasks = [
        process_query(query)
        for query in queries
    ]

    results = await asyncio.gather(*tasks)

    # Track batch efficiency
    batch_size = len(queries)
    langfuse.track_event(
        name="batch_processed",
        metadata={
            "batch_size": batch_size,
            "efficiency": batch_size / sum(r["latency"] for r in results)
        }
    )

    return results
```

## Next Steps

Congratulations! You now have comprehensive observability:

- ✅ LangFuse for LLM-specific monitoring
- ✅ LangSmith for LangChain integration
- ✅ Prometheus for metrics collection
- ✅ Grafana for visualization
- ✅ Jaeger for distributed tracing
- ✅ Loki for log aggregation
- ✅ Alerting for proactive monitoring
- ✅ Cost optimization strategies

In the next chapter, we'll automate everything with **CI/CD Pipelines**:
- GitHub Actions workflows
- Automated testing
- Security scanning
- Automated deployments
- GitOps with ArgoCD

Let's ensure every code change is automatically tested and deployed safely!
