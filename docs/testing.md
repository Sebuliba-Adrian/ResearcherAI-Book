---
layout: default
title: Testing Strategy
---

# Testing Strategy


Testing isn't just about catching bugs - it's about confidence. I wanted to achieve >90% coverage so I could deploy without fear.


## Testing Goals

**Targets I set:**
- ✅ 90%+ test coverage
- ✅ Fast test suite (< 2 minutes)
- ✅ Test both dev and prod backends
- ✅ Integration tests for full workflows
- ✅ CI/CD pipeline automation

**Results achieved:**
- 🎯 **96.60% coverage** (exceeded goal!)
- ⚡ **90 seconds** total test time
- ✅ **291 tests passing**
- ✅ **Zero flaky tests**

## Test Structure

```
tests/
├── utils/                      # 73 tests, 97.32% coverage
│   ├── test_cache.py           # Caching tests
│   ├── test_circuit_breaker.py # Circuit breaker tests
│   ├── test_model_selector.py  # Model selection tests
│   └── test_token_budget.py    # Token budget tests
├── agents/                     # 218 tests, 94.04% coverage
│   ├── test_data_collector.py  # 50 tests, 99.47% coverage
│   ├── test_knowledge_graph.py # 64 tests, 99.16% coverage
│   ├── test_vector_agent.py    # 52 tests, 97.49% coverage
│   ├── test_reasoner.py        # 37 tests, 100.00% coverage
│   └── test_scheduler.py       # 15 tests, 75.49% coverage
├── integration/
│   ├── test_dev_mode.py        # NetworkX + FAISS
│   ├── test_prod_mode.py       # Neo4j + Qdrant
│   └── test_kafka_events.py    # Event system
└── conftest.py                 # Shared fixtures
```

## Unit Testing Strategy

### Testing Dual Backends

The dual-backend architecture required testing both implementations:

```python
# tests/agents/test_knowledge_graph.py
import pytest
from agents.knowledge_graph import KnowledgeGraphAgent
from agents.backends.networkx_backend import NetworkXBackend
from agents.backends.neo4j_backend import Neo4jBackend

@pytest.fixture
def nx_graph():
    """NetworkX backend for fast tests"""
    backend = NetworkXBackend()
    return KnowledgeGraphAgent(backend)

@pytest.fixture
def neo4j_graph():
    """Neo4j backend for integration tests"""
    backend = Neo4jBackend("bolt://localhost:7687", "neo4j", "password")
    yield KnowledgeGraphAgent(backend)
    backend.clear()  # Cleanup

# Test both backends with same test
@pytest.mark.parametrize("graph_fixture", ["nx_graph", "neo4j_graph"])
def test_add_node(graph_fixture, request):
    graph = request.getfixturevalue(graph_fixture)

    # Add node
    graph.add_node("paper:1", "Paper", {"title": "Test"})

    # Verify
    nodes = graph.get_nodes(label="Paper")
    assert len(nodes) == 1
    assert nodes[0]["title"] == "Test"
```

**Benefits:**
- Same test logic for both backends
- Catches backend-specific bugs
- Verifies abstraction works

### Mocking External APIs

API calls are slow and unreliable in tests. I mocked them:

```python
# tests/agents/test_data_collector.py
import pytest
from unittest.mock import patch, MagicMock
from agents.data_collector import DataCollectorAgent

@pytest.fixture
def mock_arxiv_api():
    with patch('arxiv.Search') as mock:
        # Create mock result
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/1234.5678"
        mock_result.title = "Test Paper"
        mock_result.summary = "Abstract"
        mock_result.authors = [MagicMock(name="John Doe")]
        mock_result.published = MagicMock(isoformat=lambda: "2025-01-01")

        mock.return_value.results.return_value = [mock_result]
        yield mock

def test_collect_arxiv(mock_arxiv_api):
    agent = DataCollectorAgent()

    papers = agent.collect("transformers", sources=["arxiv"])

    assert len(papers) == 1
    assert papers[0].title == "Test Paper"
    assert papers[0].source == "arxiv"
```

### Testing Circuit Breakers

Circuit breakers prevent cascade failures:

```python
# tests/utils/test_circuit_breaker.py
def test_circuit_opens_after_failures():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    def flaky_function():
        raise Exception("API Error")

    # First 3 calls fail
    for i in range(3):
        with pytest.raises(Exception):
            breaker.call(flaky_function)

    # Circuit should be open now
    assert breaker.state == "OPEN"

    # 4th call should fail immediately
    with pytest.raises(CircuitBreakerOpen):
        breaker.call(flaky_function)

def test_circuit_recovers_after_timeout():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

    # Open the circuit
    for i in range(3):
        with pytest.raises(Exception):
            breaker.call(lambda: raise_exception())

    assert breaker.state == "OPEN"

    # Wait for recovery timeout
    time.sleep(0.2)

    # Should be in HALF_OPEN state
    breaker.call(lambda: "success")
    assert breaker.state == "CLOSED"
```

### Testing Token Budgets

Prevent runaway costs:

```python
# tests/utils/test_token_budget.py
def test_per_request_limit():
    budget = TokenBudget(per_request=1000, per_user=10000)

    # Within limit - should pass
    budget.check("user1", 500)

    # Exceeds limit - should raise
    with pytest.raises(BudgetExceeded):
        budget.check("user1", 2000)

def test_per_user_limit():
    budget = TokenBudget(per_request=1000, per_user=5000)

    # Use up budget
    budget.record("user1", 3000)
    budget.record("user1", 1500)

    # Total: 4500, next call would exceed
    with pytest.raises(BudgetExceeded):
        budget.check("user1", 600)
```

## Integration Testing

### Development Mode Full Pipeline

```python
# tests/integration/test_dev_mode.py
import os
os.environ["USE_NEO4J"] = "false"
os.environ["USE_QDRANT"] = "false"
os.environ["USE_KAFKA"] = "false"

from agents.orchestrator import OrchestratorAgent

def test_full_workflow():
    """Test complete workflow with in-memory backends"""

    # Create orchestrator
    orch = OrchestratorAgent("test_session", {})

    # 1. Collect data
    result = orch.collect_data("transformers", max_per_source=2)
    assert result["papers_collected"] > 0
    assert result["graph_stats"]["nodes_added"] > 0
    assert result["vector_stats"]["chunks_added"] > 0

    # 2. Ask question
    answer = orch.ask("What are transformers in deep learning?")
    assert len(answer) > 0
    assert "transformer" in answer.lower()

    # 3. Get stats
    stats = orch.get_stats()
    assert stats["papers_collected"] == result["papers_collected"]
    assert stats["graph"]["nodes"] == result["graph_stats"]["nodes_added"]

    # 4. Save session
    orch.save_session()
    assert os.path.exists(f"sessions/{orch.session_name}.json")

    # 5. Load session
    orch2 = OrchestratorAgent("test_session", {})
    stats2 = orch2.get_stats()
    assert stats2["papers_collected"] == stats["papers_collected"]

    orch.close()
    orch2.close()
```

**Results:**
- ✅ 297 nodes, 524 edges in graph
- ✅ 72 vector embeddings
- ✅ 9 papers collected
- ⚡ 90 seconds total time

### Production Mode with Docker

```python
# tests/integration/test_prod_mode.py
import os
os.environ["USE_NEO4J"] = "true"
os.environ["USE_QDRANT"] = "true"
os.environ["USE_KAFKA"] = "true"

@pytest.mark.docker
def test_production_workflow():
    """Test with Neo4j, Qdrant, Kafka"""

    orch = OrchestratorAgent("prod_test", {})

    # Collect data
    result = orch.collect_data("machine learning", max_per_source=5)

    # Verify Neo4j
    assert result["graph_stats"]["nodes_added"] > 100
    assert result["graph_stats"]["edges_added"] > 100

    # Verify Qdrant
    assert result["vector_stats"]["chunks_added"] > 50

    # Verify Kafka events
    events = get_kafka_events("data.collection.completed")
    assert len(events) > 0

    orch.close()
```

**Results:**
- ✅ 1,119 nodes, 1,105 edges in Neo4j
- ✅ 61 vectors in Qdrant
- ✅ 14 papers from 5 sources
- ✅ 16 Kafka topics operational

### Testing Kafka Events

```python
# tests/integration/test_kafka_events.py
from utils.event_bus import KafkaEventBus

def test_event_publishing():
    bus = KafkaEventBus("localhost:9092")

    # Publish event
    bus.publish("test.event", {"data": "hello"})

    # Subscribe and receive
    received = []
    def handler(event):
        received.append(event)

    bus.subscribe("test.event", handler)

    # Wait for message
    time.sleep(1)

    assert len(received) == 1
    assert received[0]["data"] == "hello"
```

## Test Coverage

Using pytest-cov for coverage reporting:

```bash
# Run tests with coverage
pytest tests/ \
    --cov=utils \
    --cov=agents \
    --cov-report=html \
    --cov-report=term

# Results
utils/cache.py                97.32%
utils/circuit_breaker.py      96.15%
utils/model_selector.py       98.21%
utils/token_budget.py         97.50%

agents/data_collector.py      99.47%
agents/knowledge_graph.py     99.16%
agents/vector_agent.py        97.49%
agents/reasoner.py           100.00%
agents/scheduler.py           75.49%

TOTAL                         96.60%
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
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

    - name: Run unit tests
      run: |
        export USE_NEO4J=false USE_QDRANT=false USE_KAFKA=false
        pytest tests/utils tests/agents \
          --cov=utils --cov=agents \
          --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  integration:
    runs-on: ubuntu-latest

    services:
      neo4j:
        image: neo4j:5.13
        env:
          NEO4J_AUTH: neo4j/password
        ports:
          - 7687:7687

      qdrant:
        image: qdrant/qdrant:v1.7.0
        ports:
          - 6333:6333

    steps:
    - uses: actions/checkout@v4

    - name: Run integration tests
      run: |
        export USE_NEO4J=true USE_QDRANT=true
        pytest tests/integration
```

**Results:**
- ✅ Runs on every push
- ✅ Unit tests: < 2 minutes
- ✅ Integration tests: < 5 minutes
- ✅ Coverage uploaded to Codecov

## Performance Testing

I also added basic performance tests:

```python
# tests/performance/test_benchmarks.py
import pytest
import time

@pytest.mark.benchmark
def test_data_collection_speed():
    """Collection should be < 2 minutes for 10 papers"""
    orch = OrchestratorAgent("perf_test", {})

    start = time.time()
    result = orch.collect_data("AI", max_per_source=10)
    duration = time.time() - start

    assert duration < 120  # 2 minutes
    assert result["papers_collected"] >= 10

@pytest.mark.benchmark
def test_query_response_time():
    """Queries should respond in < 5 seconds"""
    orch = OrchestratorAgent("perf_test", {})

    # Collect some data first
    orch.collect_data("transformers", max_per_source=5)

    # Time query
    start = time.time()
    answer = orch.ask("What are transformers?")
    duration = time.time() - start

    assert duration < 5.0
    assert len(answer) > 0
```

## What I Learned

**✅ Wins**

1. **Dual backend testing**: Caught several backend-specific bugs
2. **Mocking saved time**: Tests run in 90s vs 10+ minutes
3. **High coverage = confidence**: 96.60% means I can deploy fearlessly
4. **CI/CD automation**: Catches issues before merging

**🤔 Challenges**

1. **Testing async Kafka**: Required careful timing
2. **Mock complexity**: Some tests had too many mocks
3. **Flaky network tests**: Fixed with better retry logic
4. **Docker overhead**: Integration tests are slower

**💡 Insights**

> Good tests are an investment that pays off every deployment.

> Mock external dependencies, but test real integrations too.

> High coverage isn't everything, but it's a good proxy for quality.

> Fast tests = more frequent testing = better code.

## Next: Deployment

Tests passing? Time to ship! Let's deploy this to production.


  <a href="frontend">← Back: Frontend</a>
  <a href="deployment">Next: Deployment →</a>

