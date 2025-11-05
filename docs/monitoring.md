---
layout: default
title: Monitoring & Operations
---

# Monitoring & Operations


The system is deployed. But how do you know it's working? Welcome to the world of observability.


## Monitoring Philosophy

I follow the three pillars of observability:

1. **Metrics**: What's happening? (numbers, gauges, counters)
2. **Logs**: Why did it happen? (structured events)
3. **Traces**: How did it happen? (request flows)

## Apache Airflow for ETL

One of my favorite additions was Apache Airflow for orchestrating data collection.

### Why Airflow?

**Before Airflow:**
```python
# Sequential, slow, no visibility
for source in sources:
    papers = collect(source)  # Blocks
```
- ❌ Sequential execution (19-38 seconds)
- ❌ No retry logic
- ❌ No monitoring
- ❌ Can't see progress

**With Airflow:**
```python
# Parallel, fast, observable
with DAG("research_paper_etl"):
    arxiv = PythonOperator(task_id="arxiv", ...)
    pubmed = PythonOperator(task_id="pubmed", ...)
    semantic = PythonOperator(task_id="semantic", ...)
    # All run in parallel!
```
- ✅ Parallel execution (5-10 seconds)
- ✅ Automatic retries with backoff
- ✅ Visual DAG at http://localhost:8080
- ✅ Real-time progress tracking

### Airflow Setup

```bash
# airflow/setup_airflow.sh
#!/bin/bash

# Create directories
mkdir -p dags logs plugins config

# Initialize Airflow
export AIRFLOW_HOME=$(pwd)
airflow db init

# Create admin user
airflow users create \
    --username airflow \
    --password airflow \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start webserver and scheduler
airflow webserver -p 8080 &
airflow scheduler &

echo "Airflow running at http://localhost:8080"
echo "Username: airflow / Password: airflow"
```

### Research Paper ETL DAG

```python
# airflow/dags/research_paper_etl.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from agents.data_collector import DataCollectorAgent

default_args = {
    'owner': 'researcherai',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
}

dag = DAG(
    'research_paper_etl',
    default_args=default_args,
    description='Collect research papers from multiple sources',
    schedule_interval=timedelta(hours=6),  # Run every 6 hours
    catchup=False,
)

collector = DataCollectorAgent()

def collect_from_source(source, query, **context):
    """Collect papers from a single source"""
    papers = collector.collect(query, sources=[source], max_per_source=10)
    return len(papers)

# Create tasks for each source
arxiv_task = PythonOperator(
    task_id='collect_arxiv',
    python_callable=collect_from_source,
    op_kwargs={'source': 'arxiv', 'query': 'machine learning'},
    dag=dag,
)

semantic_task = PythonOperator(
    task_id='collect_semantic_scholar',
    python_callable=collect_from_source,
    op_kwargs={'source': 'semantic_scholar', 'query': 'machine learning'},
    dag=dag,
)

pubmed_task = PythonOperator(
    task_id='collect_pubmed',
    python_callable=collect_from_source,
    op_kwargs={'source': 'pubmed', 'query': 'machine learning'},
    dag=dag,
)

# Tasks run in parallel (no dependencies)
# If you wanted sequential: arxiv_task >> semantic_task >> pubmed_task
```

### Airflow Benefits



**Performance:**
- Sequential: 19-38 seconds
- Parallel: 5-10 seconds
- **3-4x faster** ⚡

**Reliability:**
- Automatic retries with exponential backoff
- Failure notifications
- Task-level success/failure tracking

**Visibility:**
- Real-time DAG status
- Task logs in web UI
- Historical run data
- Performance metrics

**Scalability:**
- Add Celery workers for more parallelism
- Queue management
- Priority scheduling



## Kafka Event Monitoring

Kafka UI gives visibility into the event stream.

### Kafka UI Dashboard

Access at **http://localhost:8081**:

- 📊 **Topics**: View all 16 event topics
- 📩 **Messages**: See event payloads in real-time
- 👥 **Consumers**: Track consumer groups
- 📈 **Metrics**: Throughput, lag, partition status

### Key Metrics to Monitor

```python
# utils/kafka_metrics.py
from kafka import KafkaConsumer, KafkaProducer

def get_kafka_metrics():
    """Get Kafka cluster metrics"""

    metrics = {
        "topics": [],
        "total_messages": 0,
        "consumer_lag": {},
    }

    # Get topic info
    admin = KafkaAdminClient(bootstrap_servers="kafka:9092")
    topic_metadata = admin.list_topics()

    for topic in topic_metadata:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers="kafka:9092",
            auto_offset_reset='earliest',
        )

        # Get partition info
        partitions = consumer.partitions_for_topic(topic)
        messages = 0

        for partition in partitions:
            tp = TopicPartition(topic, partition)
            consumer.assign([tp])
            consumer.seek_to_end(tp)
            messages += consumer.position(tp)

        metrics["topics"].append({
            "name": topic,
            "partitions": len(partitions),
            "messages": messages,
        })
        metrics["total_messages"] += messages

    return metrics
```

## Neo4j Browser

Access at **http://localhost:7474**:

### Useful Cypher Queries

```cypher
// View all papers
MATCH (p:Paper)
RETURN p
LIMIT 25

// Find most cited authors
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
RETURN a.name, count(p) as paper_count
ORDER BY paper_count DESC
LIMIT 10

// Explore topics
MATCH (p:Paper)-[:IS_ABOUT]->(t:Topic)
RETURN t.name, count(p) as paper_count
ORDER BY paper_count DESC

// Find collaborations
MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
WHERE a1.name < a2.name
RETURN a1.name, a2.name, count(p) as collaborations
ORDER BY collaborations DESC
LIMIT 10

// Graph statistics
MATCH (n)
RETURN labels(n)[0] as type, count(n) as count
```

### Graph Visualization

The Neo4j Browser provides interactive graph visualization:

1. Run a query
2. Switch to graph view
3. Explore relationships visually
4. Expand nodes to discover connections

## Qdrant Dashboard

Access at **http://localhost:6333/dashboard**:

### Collection Metrics

```python
# utils/qdrant_metrics.py
from qdrant_client import QdrantClient

def get_qdrant_metrics():
    """Get Qdrant vector database metrics"""

    client = QdrantClient(host="qdrant", port=6333)

    collections = client.get_collections().collections
    metrics = {"collections": []}

    for collection in collections:
        info = client.get_collection(collection.name)

        metrics["collections"].append({
            "name": collection.name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "status": info.status,
        })

    return metrics
```

## Application Metrics

### Custom Metrics Dashboard

I built a simple metrics endpoint:

```python
# api/metrics.py
from fastapi import APIRouter
from utils.health import check_health
from utils.kafka_metrics import get_kafka_metrics
from utils.qdrant_metrics import get_qdrant_metrics

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""

    return {
        "health": check_health(),
        "kafka": get_kafka_metrics() if USE_KAFKA else None,
        "qdrant": get_qdrant_metrics() if USE_QDRANT else None,
        "neo4j": get_neo4j_metrics() if USE_NEO4J else None,
    }

@router.get("/metrics/prometheus")
async def prometheus_metrics():
    """Expose metrics in Prometheus format"""

    metrics = []

    # Example metrics
    metrics.append("# HELP papers_collected Total papers collected")
    metrics.append("# TYPE papers_collected counter")
    metrics.append(f"papers_collected {get_total_papers()}")

    metrics.append("# HELP graph_nodes Current number of graph nodes")
    metrics.append("# TYPE graph_nodes gauge")
    metrics.append(f"graph_nodes {get_graph_node_count()}")

    metrics.append("# HELP vector_embeddings Current number of vector embeddings")
    metrics.append("# TYPE vector_embeddings gauge")
    metrics.append(f"vector_embeddings {get_vector_count()}")

    return "\n".join(metrics)
```

## Performance Monitoring

### Key Metrics to Track



**Data Collection:**
- Papers collected per source
- Collection duration
- Error rate by source
- Deduplication rate

**Query Performance:**
- Query response time (p50, p95, p99)
- Vector search latency
- Graph traversal time
- LLM generation time

**Resource Usage:**
- CPU utilization
- Memory usage
- Disk I/O
- Network throughput

**Database Performance:**
- Neo4j query time
- Qdrant search latency
- Kafka producer/consumer lag
- Connection pool stats



### Response Time Tracking

```python
# utils/monitoring.py
import time
from functools import wraps

def track_duration(metric_name: str):
    """Decorator to track function duration"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start

                # Log metric
                logger.info(f"{metric_name}", extra={
                    "duration_ms": duration * 1000,
                    "status": "success"
                })

                return result
            except Exception as e:
                duration = time.time() - start

                logger.error(f"{metric_name}", extra={
                    "duration_ms": duration * 1000,
                    "status": "error",
                    "error": str(e)
                })

                raise

        return wrapper
    return decorator

# Usage
@track_duration("data_collection")
def collect_data(query, sources):
    # ... implementation ...

@track_duration("query_answering")
def answer_question(question):
    # ... implementation ...
```

## Alerting

### Basic Alert Rules

```python
# utils/alerts.py
import smtplib
from email.mime.text import MIMEText

class AlertManager:
    def __init__(self, smtp_config):
        self.smtp_config = smtp_config

    def send_alert(self, subject: str, message: str, severity: str = "warning"):
        """Send email alert"""

        msg = MIMEText(message)
        msg['Subject'] = f"[{severity.upper()}] {subject}"
        msg['From'] = self.smtp_config['from']
        msg['To'] = self.smtp_config['to']

        with smtplib.SMTP(self.smtp_config['server']) as server:
            server.send_message(msg)

    def check_and_alert(self):
        """Check metrics and send alerts"""

        health = check_health()

        # Alert on unhealthy services
        for service, status in health.items():
            if status == "unhealthy":
                self.send_alert(
                    f"{service} is unhealthy",
                    f"Service {service} failed health check",
                    severity="critical"
                )

        # Alert on high error rates
        error_rate = get_error_rate()
        if error_rate > 0.05:  # 5%
            self.send_alert(
                "High error rate detected",
                f"Error rate: {error_rate:.2%}",
                severity="warning"
            )
```

## Cost Monitoring

Track LLM API costs:

```python
# utils/cost_tracking.py
class CostTracker:
    def __init__(self):
        self.costs = {
            "gemini-flash": 0.35 / 1_000_000,  # $ per token
        }
        self.usage = {}

    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Record API usage"""

        total_tokens = input_tokens + output_tokens
        cost = total_tokens * self.costs.get(model, 0)

        if model not in self.usage:
            self.usage[model] = {"tokens": 0, "cost": 0, "calls": 0}

        self.usage[model]["tokens"] += total_tokens
        self.usage[model]["cost"] += cost
        self.usage[model]["calls"] += 1

    def get_summary(self) -> Dict:
        """Get cost summary"""

        total_cost = sum(u["cost"] for u in self.usage.values())
        total_tokens = sum(u["tokens"] for u in self.usage.values())

        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "by_model": self.usage,
        }
```

## Operational Runbooks

### Common Issues and Solutions



**Issue: Kafka consumers lagging**

```bash
# Check consumer group lag
docker exec rag-kafka kafka-consumer-groups \
    --bootstrap-server localhost:9092 \
    --describe --group researcherai

# Solution: Scale up consumers or increase retention
docker-compose up -d --scale rag-multiagent=3
```





**Issue: Neo4j running out of memory**

```bash
# Check memory usage
docker stats rag-neo4j

# Solution: Increase heap size in docker-compose.yml
environment:
  - NEO4J_dbms_memory_heap_initial__size=2G
  - NEO4J_dbms_memory_heap_max__size=4G
```





**Issue: Qdrant search slow**

```bash
# Check collection stats
curl http://localhost:6333/collections/papers

# Solution: Rebuild with better indexing
# Set appropriate index parameters in Qdrant config
```



## Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup Neo4j
docker exec rag-neo4j neo4j-admin dump \
    --database=neo4j \
    --to=/tmp/neo4j-backup.dump

docker cp rag-neo4j:/tmp/neo4j-backup.dump \
    $BACKUP_DIR/neo4j.dump

# Backup Qdrant
docker exec rag-qdrant tar czf /tmp/qdrant-backup.tar.gz \
    /qdrant/storage

docker cp rag-qdrant:/tmp/qdrant-backup.tar.gz \
    $BACKUP_DIR/qdrant.tar.gz

# Backup sessions
tar czf $BACKUP_DIR/sessions.tar.gz sessions/

echo "Backup completed: $BACKUP_DIR"
```

### Restore Procedure

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR=$1

# Stop services
docker-compose down

# Restore Neo4j
docker run --rm \
    -v $BACKUP_DIR:/backup \
    -v neo4j-data:/data \
    neo4j:5.13 \
    neo4j-admin load --from=/backup/neo4j.dump --database=neo4j

# Restore Qdrant
tar xzf $BACKUP_DIR/qdrant.tar.gz -C volumes/qdrant/

# Restore sessions
tar xzf $BACKUP_DIR/sessions.tar.gz

# Start services
docker-compose up -d

echo "Restore completed from: $BACKUP_DIR"
```

## What I Learned

**✅ Wins**

1. **Airflow transformed ETL**: 3-4x faster, visible, reliable
2. **Kafka UI is invaluable**: Real-time event visibility
3. **Health checks catch issues early**: Failed fast, recovered automatically
4. **Structured logs simplify debugging**: JSON logs are easy to parse and query

**🤔 Challenges**

1. **Too many dashboards**: Neo4j, Qdrant, Kafka UI, Airflow...
2. **Alert fatigue**: Had to tune thresholds carefully
3. **Cost tracking complexity**: Multiple models, different pricing
4. **Backup size**: Neo4j dumps can get large quickly

**💡 Insights**

> Observability is not optional for production systems.

> The best monitoring is proactive - catch issues before users do.

> Start simple with monitoring, add complexity as needed.

> Good logs are worth their weight in gold when debugging.

## Conclusion

We've come full circle:
- ✅ Planned a production-grade system
- ✅ Designed a scalable architecture
- ✅ Built 6 specialized agents
- ✅ Created a beautiful React frontend
- ✅ Achieved 96.60% test coverage
- ✅ Deployed with Docker and CI/CD
- ✅ Added comprehensive monitoring

**ResearcherAI is production-ready!** 🎉


  <a href="deployment">← Back: Deployment</a>
  <a href="/">Return to Home →</a>


---


<h3>Thank you for reading!</h3>
<p>I hope this tutorial helped you understand not just what I built, but how and why I made each decision.</p>
<p>If you have questions or want to share what you're building:</p>
<p>
  [GitHub](https://github.com/Sebuliba-Adrian/ResearcherAI) |
  [Issues](https://github.com/Sebuliba-Adrian/ResearcherAI/issues)
</p>

