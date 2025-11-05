---
layout: default
title: Backend Development
---

# Backend Development


Time to get our hands dirty with code! This chapter walks through building each agent, integrating data sources, and connecting everything together.


## Project Structure

First, I organized the codebase for clarity:

```
ResearcherAI/
├── agents/
│   ├── __init__.py
│   ├── data_collector.py      # 7 data sources
│   ├── knowledge_graph.py     # Neo4j/NetworkX
│   ├── vector_agent.py        # Qdrant/FAISS
│   ├── reasoner.py            # Gemini LLM
│   ├── scheduler.py           # Automated collection
│   └── orchestrator.py        # Coordinator
├── utils/
│   ├── cache.py               # Dual-tier caching
│   ├── circuit_breaker.py     # Fault tolerance
│   ├── model_selector.py      # Dynamic model selection
│   ├── token_budget.py        # Cost controls
│   └── event_bus.py           # Kafka/sync events
├── config/
│   └── settings.yaml          # Configuration
├── tests/
│   ├── utils/                 # 73 util tests
│   └── agents/                # 218 agent tests
├── airflow/
│   └── dags/                  # ETL workflows
├── main.py                    # CLI entry point
├── docker-compose.yml         # Infrastructure
└── requirements.txt           # Dependencies
```

##Building the Data Collector Agent

This was the first agent I built because everything else depends on having data.



### Design Goals

1. Support multiple data sources with same interface
2. Handle rate limiting and errors gracefully
3. Deduplicate papers across sources
4. Publish collection events to Kafka



### The Base Structure

```python
# agents/data_collector.py

from typing import List, Dict
from dataclasses import dataclass
import time

@dataclass
class Paper:
    """Represents a research paper"""
    id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: str
    source: str
    url: str
    citations: int = 0

class DataCollectorAgent:
    def __init__(self, event_bus=None, cache=None):
        self.event_bus = event_bus
        self.cache = cache
        self.sources = {
            "arxiv": self._collect_arxiv,
            "semantic_scholar": self._collect_semantic_scholar,
            "pubmed": self._collect_pubmed,
            "zenodo": self._collect_zenodo,
            "web": self._collect_web,
            "huggingface": self._collect_huggingface,
            "kaggle": self._collect_kaggle,
        }

    def collect(self, query: str, sources: List[str] = None,
                max_per_source: int = 10) -> Dict:
        """Collect papers from specified sources"""

        if sources is None:
            sources = list(self.sources.keys())

        # Check cache first
        cache_key = f"collect:{query}:{':'.join(sources)}"
        if self.cache and (cached := self.cache.get(cache_key)):
            return cached

        # Publish start event
        if self.event_bus:
            self.event_bus.publish("data.collection.started", {
                "query": query,
                "sources": sources,
            })

        all_papers = []
        errors = []

        for source in sources:
            try:
                papers = self.sources[source](query, max_per_source)
                all_papers.extend(papers)
            except Exception as e:
                errors.append({"source": source, "error": str(e)})

        # Deduplicate by title similarity
        unique_papers = self._deduplicate(all_papers)

        result = {
            "papers": unique_papers,
            "count": len(unique_papers),
            "sources": sources,
            "errors": errors,
        }

        # Cache result
        if self.cache:
            self.cache.set(cache_key, result, ttl=3600)

        # Publish completion event
        if self.event_bus:
            self.event_bus.publish("data.collection.completed", result)

        return result
```



### Implementing arXiv Source

```python
import arxiv

def _collect_arxiv(self, query: str, max_results: int) -> List[Paper]:
    """Collect papers from arXiv"""

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        paper = Paper(
            id=f"arxiv:{result.entry_id.split('/')[-1]}",
            title=result.title,
            abstract=result.summary,
            authors=[a.name for a in result.authors],
            published_date=result.published.isoformat(),
            source="arxiv",
            url=result.entry_id,
        )
        papers.append(paper)

    return papers
```

### Implementing Semantic Scholar

```python
import requests

def _collect_semantic_scholar(self, query: str, max_results: int) -> List[Paper]:
    """Collect from Semantic Scholar API"""

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,citationCount,url",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    papers = []

    for item in data.get("data", []):
        paper = Paper(
            id=f"s2:{item['paperId']}",
            title=item["title"],
            abstract=item.get("abstract", ""),
            authors=[a["name"] for a in item.get("authors", [])],
            published_date=f"{item.get('year', 'Unknown')}-01-01",
            source="semantic_scholar",
            url=item.get("url", ""),
            citations=item.get("citationCount", 0),
        )
        papers.append(paper)

    return papers
```

### Rate Limiting & Retry Logic

APIs have rate limits. I added decorators to handle this:

```python
from functools import wraps
import time

def rate_limit(calls: int, period: int):
    """Rate limit decorator"""
    timestamps = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old timestamps
            timestamps[:] = [t for t in timestamps if now - t < period]

            # Wait if rate limit exceeded
            if len(timestamps) >= calls:
                sleep_time = period - (now - timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                timestamps.pop(0)

            timestamps.append(time.time())
            return func(*args, **kwargs)

        return wrapper
    return decorator

def retry(max_attempts: int = 3, backoff: float = 2.0):
    """Retry decorator with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = backoff ** attempt
                    print(f"Attempt {attempt + 1} failed, retrying in {wait}s...")
                    time.sleep(wait)

        return wrapper
    return decorator

# Apply to API calls
@rate_limit(calls=10, period=60)  # 10 calls per minute
@retry(max_attempts=3, backoff=2.0)
def _collect_semantic_scholar(self, query, max_results):
    # ... implementation ...
```

### Deduplication

Papers from different sources might be duplicates. I used fuzzy title matching:

```python
from difflib import SequenceMatcher

def _deduplicate(self, papers: List[Paper]) -> List[Paper]:
    """Remove duplicate papers based on title similarity"""

    unique = []
    seen_titles = []

    for paper in papers:
        # Check if similar title already seen
        is_duplicate = False
        for seen in seen_titles:
            similarity = SequenceMatcher(None, paper.title.lower(),
                                       seen.lower()).ratio()
            if similarity > 0.85:  # 85% similar = duplicate
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(paper)
            seen_titles.append(paper.title)

    return unique
```

## Building the Knowledge Graph Agent

Next, I needed to organize papers into a knowledge graph.



### Design Goals

1. Abstract backend (Neo4j or NetworkX)
2. Extract entities automatically (papers, authors, topics)
3. Create relationships (AUTHORED, CITES, IS_ABOUT)
4. Support graph queries and visualization



### Backend Abstraction

```python
from abc import ABC, abstractmethod

class GraphBackend(ABC):
    """Abstract graph database interface"""

    @abstractmethod
    def add_node(self, node_id: str, label: str, properties: Dict):
        pass

    @abstractmethod
    def add_edge(self, from_id: str, to_id: str, relationship: str):
        pass

    @abstractmethod
    def query(self, pattern: str) -> List[Dict]:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, depth: int) -> List[Dict]:
        pass

    @abstractmethod
    def visualize(self) -> Dict:
        pass
```

### NetworkX Backend (Development)

```python
import networkx as nx

class NetworkXBackend(GraphBackend):
    """In-memory graph using NetworkX"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, label: str, properties: Dict):
        self.graph.add_node(node_id, label=label, **properties)

    def add_edge(self, from_id: str, to_id: str, relationship: str):
        self.graph.add_edge(from_id, to_id, relationship=relationship)

    def query(self, pattern: str) -> List[Dict]:
        # Simple pattern matching for common queries
        if pattern.startswith("MATCH (p:Paper)"):
            return [{"node_id": n, **self.graph.nodes[n]}
                    for n in self.graph.nodes
                    if self.graph.nodes[n].get("label") == "Paper"]

    def get_neighbors(self, node_id: str, depth: int) -> List[Dict]:
        # BFS to depth
        neighbors = []
        visited = {node_id}
        queue = [(node_id, 0)]

        while queue:
            current, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue

            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbors.append({
                        "node_id": neighbor,
                        **self.graph.nodes[neighbor]
                    })
                    queue.append((neighbor, current_depth + 1))

        return neighbors

    def visualize(self) -> Dict:
        return {
            "nodes": [{"id": n, **self.graph.nodes[n]}
                     for n in self.graph.nodes],
            "edges": [{"source": u, "target": v,
                      **self.graph[u][v][k]}
                     for u, v, k in self.graph.edges(keys=True)]
        }
```

### Neo4j Backend (Production)

```python
from neo4j import GraphDatabase

class Neo4jBackend(GraphBackend):
    """Persistent graph using Neo4j"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_node(self, node_id: str, label: str, properties: Dict):
        with self.driver.session() as session:
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=node_id, props=properties
            )

    def add_edge(self, from_id: str, to_id: str, relationship: str):
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                MERGE (a)-[r:{relationship}]->(b)
                """,
                from_id=from_id, to_id=to_id
            )

    def query(self, cypher: str) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]

    def get_neighbors(self, node_id: str, depth: int) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start {id: $node_id})-[*1..{depth}]-(neighbor)
                RETURN DISTINCT neighbor
                """,
                node_id=node_id, depth=depth
            )
            return [dict(record["neighbor"]) for record in result]

    def visualize(self) -> Dict:
        with self.driver.session() as session:
            # Get all nodes
            nodes_result = session.run("MATCH (n) RETURN n")
            nodes = [dict(record["n"]) for record in nodes_result]

            # Get all edges
            edges_result = session.run(
                "MATCH (a)-[r]->(b) RETURN a.id as source, b.id as target, type(r) as type"
            )
            edges = [dict(record) for record in edges_result]

            return {"nodes": nodes, "edges": edges}
```

### Entity Extraction

When papers are added, I extract entities using simple heuristics:

```python
class KnowledgeGraphAgent:
    def add_papers(self, papers: List[Paper]) -> Dict:
        """Add papers to graph with entity extraction"""

        nodes_added = 0
        edges_added = 0

        for paper in papers:
            # Add paper node
            self.backend.add_node(
                node_id=paper.id,
                label="Paper",
                properties={
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "published_date": paper.published_date,
                    "source": paper.source,
                    "url": paper.url,
                }
            )
            nodes_added += 1

            # Add author nodes and AUTHORED edges
            for author in paper.authors:
                author_id = f"author:{author.lower().replace(' ', '_')}"
                self.backend.add_node(
                    node_id=author_id,
                    label="Author",
                    properties={"name": author}
                )
                nodes_added += 1

                self.backend.add_edge(author_id, paper.id, "AUTHORED")
                edges_added += 1

            # Extract topics from title and abstract
            topics = self._extract_topics(paper.title + " " + paper.abstract)
            for topic in topics:
                topic_id = f"topic:{topic.lower().replace(' ', '_')}"
                self.backend.add_node(
                    node_id=topic_id,
                    label="Topic",
                    properties={"name": topic}
                )
                nodes_added += 1

                self.backend.add_edge(paper.id, topic_id, "IS_ABOUT")
                edges_added += 1

        return {
            "nodes_added": nodes_added,
            "edges_added": edges_added,
        }

    def _extract_topics(self, text: str) -> List[str]:
        """Simple keyword extraction for topics"""
        # In production, use NER or topic modeling
        # For now, extract common AI/ML terms

        keywords = [
            "transformer", "attention", "neural network", "deep learning",
            "machine learning", "nlp", "computer vision", "reinforcement learning",
            "gan", "lstm", "bert", "gpt", "diffusion", "embedding"
        ]

        text_lower = text.lower()
        return [k for k in keywords if k in text_lower]
```

## Building the Vector Agent

For semantic search, I needed vector embeddings.



### Design Goals

1. Abstract backend (Qdrant or FAISS)
2. Generate embeddings automatically
3. Chunk long documents intelligently
4. Fast similarity search



### Backend Abstraction

```python
class VectorBackend(ABC):
    @abstractmethod
    def add_vectors(self, vectors: List[Tuple[str, List[float], Dict]]):
        """Add vectors with IDs and metadata"""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def delete(self, vector_ids: List[str]):
        """Delete vectors by ID"""
        pass
```

### Document Chunking

Long documents need to be split into chunks:

```python
def chunk_text(text: str, max_words: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""

    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

        # Stop if we've covered the whole text
        if i + max_words >= len(words):
            break

    return chunks
```

### Embedding Generation

```python
from sentence_transformers import SentenceTransformer

class VectorAgent:
    def __init__(self, backend: VectorBackend):
        self.backend = backend
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, documents: List[Paper]) -> Dict:
        """Add documents as vector embeddings"""

        vectors = []

        for doc in documents:
            # Combine title and abstract
            text = f"{doc.title}. {doc.abstract}"

            # Chunk text
            chunks = chunk_text(text, max_words=400, overlap=50)

            # Generate embeddings
            embeddings = self.encoder.encode(chunks)

            # Add to vectors list
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{doc.id}:chunk:{i}"
                vectors.append((
                    vector_id,
                    embedding.tolist(),
                    {
                        "paper_id": doc.id,
                        "chunk_index": i,
                        "text": chunk,
                        "title": doc.title,
                    }
                ))

        # Bulk insert
        self.backend.add_vectors(vectors)

        return {"chunks_added": len(vectors)}

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search for similar documents"""

        # Encode query
        query_vector = self.encoder.encode(query).tolist()

        # Search
        results = self.backend.search(query_vector, top_k)

        return results
```

### FAISS Backend (Development)

```python
import faiss
import numpy as np

class FAISSBackend(VectorBackend):
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = {}  # id -> metadata
        self.id_to_index = {}  # id -> index position

    def add_vectors(self, vectors: List[Tuple[str, List[float], Dict]]):
        for vec_id, vector, metadata in vectors:
            # Add to FAISS index
            index_pos = self.index.ntotal
            self.index.add(np.array([vector], dtype=np.float32))

            # Store metadata
            self.metadata[vec_id] = metadata
            self.id_to_index[vec_id] = index_pos

    def search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        query = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Find vector ID by index position
            vec_id = next(k for k, v in self.id_to_index.items() if v == idx)
            results.append({
                "id": vec_id,
                "score": float(1 / (1 + dist)),  # Convert distance to similarity
                "metadata": self.metadata[vec_id],
            })

        return results
```

### Qdrant Backend (Production)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantBackend(VectorBackend):
    def __init__(self, host: str, port: int, collection: str = "papers"):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection

        # Create collection if not exists
        try:
            self.client.get_collection(collection)
        except:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def add_vectors(self, vectors: List[Tuple[str, List[float], Dict]]):
        points = [
            PointStruct(
                id=vec_id,
                vector=vector,
                payload=metadata
            )
            for vec_id, vector, metadata in vectors
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload,
            }
            for hit in results
        ]
```

## Building the Reasoning Agent

The brain of the system - uses LLM to answer questions.

```python
import google.generativeai as genai

class ReasoningAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.conversation_history = []

    def answer(self, question: str, context: Dict, history: List = None) -> str:
        """Generate answer using RAG context"""

        # Build prompt
        prompt = self._build_prompt(question, context, history)

        # Generate answer
        response = self.model.generate_content(prompt)

        # Store in history
        self.conversation_history.append({
            "question": question,
            "answer": response.text,
            "timestamp": time.time(),
        })

        # Keep only last 5 turns
        self.conversation_history = self.conversation_history[-5:]

        return response.text

    def _build_prompt(self, question: str, context: Dict, history: List) -> str:
        """Build RAG prompt with context and history"""

        prompt_parts = []

        # System instruction
        prompt_parts.append(
            "You are a research assistant. Answer the question using the "
            "provided papers and context. Cite sources using paper IDs."
        )

        # Add conversation history
        if history:
            prompt_parts.append("\n## Conversation History:")
            for turn in history[-3:]:  # Last 3 turns
                prompt_parts.append(f"Q: {turn['question']}")
                prompt_parts.append(f"A: {turn['answer']}\n")

        # Add vector search results
        if context.get("vector_results"):
            prompt_parts.append("\n## Relevant Paper Excerpts:")
            for i, result in enumerate(context["vector_results"][:5], 1):
                prompt_parts.append(f"{i}. {result['metadata']['title']}")
                prompt_parts.append(f"   {result['metadata']['text']}")
                prompt_parts.append(f"   (Source: {result['metadata']['paper_id']})\n")

        # Add graph context
        if context.get("graph_context"):
            prompt_parts.append("\n## Related Entities:")
            prompt_parts.append(f"Papers: {len(context['graph_context']['papers'])}")
            prompt_parts.append(f"Authors: {', '.join(context['graph_context']['authors'][:5])}")
            prompt_parts.append(f"Topics: {', '.join(context['graph_context']['topics'][:5])}")

        # Add question
        prompt_parts.append(f"\n## Question:\n{question}")

        prompt_parts.append("\n## Answer:")

        return "\n".join(prompt_parts)
```

## Orchestrating Everything

The OrchestratorAgent ties it all together:

```python
class OrchestratorAgent:
    def __init__(self, session_name: str, config: Dict):
        self.session_name = session_name
        self.config = config

        # Initialize agents
        self.data_collector = DataCollectorAgent(event_bus, cache)
        self.graph_agent = KnowledgeGraphAgent(graph_backend)
        self.vector_agent = VectorAgent(vector_backend)
        self.reasoning_agent = ReasoningAgent(api_key)

        # Load or create session
        self.session = self._load_or_create_session()

    def collect_data(self, query: str, sources: List[str] = None,
                    max_per_source: int = 10) -> Dict:
        """Orchestrate data collection workflow"""

        # 1. Collect papers
        result = self.data_collector.collect(query, sources, max_per_source)

        # 2. Add to graph
        graph_stats = self.graph_agent.add_papers(result["papers"])

        # 3. Add to vectors
        vector_stats = self.vector_agent.add_documents(result["papers"])

        # 4. Update session
        self.session["papers_collected"] += result["count"]

        return {
            "papers_collected": result["count"],
            "graph_stats": graph_stats,
            "vector_stats": vector_stats,
            "errors": result["errors"],
        }

    def ask(self, question: str) -> str:
        """Orchestrate RAG query workflow"""

        # 1. Vector search for relevant chunks
        vector_results = self.vector_agent.search(question, top_k=10)

        # 2. Graph traversal for related entities
        paper_ids = [r["metadata"]["paper_id"] for r in vector_results[:5]]
        graph_context = self.graph_agent.get_context(paper_ids)

        # 3. Build context
        context = {
            "vector_results": vector_results,
            "graph_context": graph_context,
        }

        # 4. Generate answer
        answer = self.reasoning_agent.answer(
            question,
            context,
            history=self.session.get("conversations", [])
        )

        # 5. Store conversation
        self.session.setdefault("conversations", []).append({
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
        })

        return answer
```

## What I Learned

**✅ Wins**

1. **Abstractions paid off**: Switching backends is trivial
2. **Events enable parallelism**: 3x faster collection
3. **Caching is essential**: 40% cost reduction
4. **Good error handling early**: Saved hours of debugging

**🤔 Challenges**

1. **Rate limiting APIs**: Required careful retry logic
2. **Deduplication is hard**: 85% similarity threshold was trial-and-error
3. **Entity extraction**: Simple heuristics work surprisingly well
4. **Prompt engineering**: Took many iterations to get citations right

**💡 Insights**

> Good abstractions make testing and development much faster.

> Real-world APIs are messy - build resilience from the start.

> Simple heuristics (like keyword extraction) can be good enough for v1.

## Next: Frontend

Backend done! Now let's build a beautiful UI to interact with our system.


  <a href="architecture">← Back: Architecture</a>
  <a href="frontend">Next: Frontend Development →</a>

