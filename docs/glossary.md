---
layout: default
title: Glossary
---

# Glossary: AI/ML Terms for Web Developers


A comprehensive reference of all AI/ML terms used in this book, explained in plain English with web development analogies.


## A

**Agent**
- **Definition:** A software component that can act autonomously to achieve goals
- **In This Book:** Specialized modules (DataCollector, ReasoningAgent, etc.)
- **Web Dev Analogy:** Like a microservice with specific responsibilities
- **Example:** DataCollectorAgent fetches papers from multiple sources

**Airflow (Apache Airflow)**
- **Definition:** Platform for programmatically authoring, scheduling, and monitoring workflows
- **Purpose:** Automate ETL processes, run tasks in parallel, retry on failures
- **Web Dev Analogy:** Like GitHub Actions or Jenkins for data pipelines
- **Example:** Schedule paper collection every 6 hours automatically

**API (Application Programming Interface)**
- **Definition:** Interface for programs to communicate
- **In AI Context:** LLM APIs (OpenAI, Google Gemini) are called like REST APIs
- **Cost Model:** Usually pay-per-token (pay-per-use)
- **Example:** `openai.ChatCompletion.create(...)`

**Attention Mechanism**
- **Definition:** Technique allowing models to focus on relevant parts of input
- **Foundation:** Core of transformer architecture
- **Web Dev Analogy:** Like SQL's WHERE clause - filtering what's relevant
- **Paper:** "Attention is All You Need" (Vaswani et al., 2017)

---

## B

**Batch Processing**
- **Definition:** Processing multiple items together for efficiency
- **In AI:** Process 100 documents at once instead of one-by-one
- **Benefit:** Faster, more efficient use of GPU/API
- **Example:** `model.encode(all_documents)` vs looping

**BERT (Bidirectional Encoder Representations from Transformers)**
- **Definition:** Pre-trained language model from Google
- **Innovation:** Understands context from both directions (bidirectional)
- **Use Case:** Base for many embedding models
- **Fun Fact:** Revolutionized NLP in 2018

**Bottleneck**
- **Definition:** The slowest part of your system limiting overall performance
- **In AI Systems:** Often the LLM API call or database query
- **Solution:** Caching, parallel processing, faster models
- **Example:** If data collection takes 60s and LLM takes 2s, collection is the bottleneck

---

## C

**Caching**
- **Definition:** Storing results to avoid recomputing
- **In AI:** Cache LLM responses, embeddings, search results
- **Benefit:** 40-70% cost reduction in this project
- **Example:** If user asks same question, return cached answer

**Chain-of-Thought Prompting**
- **Definition:** Asking LLM to "think step by step"
- **Benefit:** Better reasoning on complex problems
- **Example:** "Let's solve this step by step: First..."
- **Paper:** Wei et al., 2022

**Chunk/Chunking**
- **Definition:** Breaking large text into smaller pieces
- **Why:** LLMs have token limits, smaller chunks improve retrieval
- **Strategy:** 400-500 words with 50-word overlap
- **Web Dev Analogy:** Like pagination

**Circuit Breaker**
- **Definition:** Pattern that stops calling failing services
- **Purpose:** Prevent cascade failures in distributed systems
- **Example:** After 5 API failures, stop calling for 60 seconds
- **Benefit:** 94% error reduction

**Cosine Similarity**
- **Definition:** Measure of similarity between two vectors
- **Range:** -1 (opposite) to 1 (identical), typically 0 to 1 for embeddings
- **Formula:** `cos(θ) = (A · B) / (||A|| ||B||)`
- **Example:** "dog" vs "puppy" might be 0.87 (very similar)

**Context Window**
- **Definition:** Maximum amount of text an LLM can process at once
- **Size:** GPT-4: ~8K tokens, Gemini 2.0: ~1M tokens
- **Web Dev Analogy:** Like browser memory limits or max payload size
- **Implication:** Affects how much context you can provide

**Cypher**
- **Definition:** Query language for Neo4j graph database
- **Like:** SQL, but for graph relationships
- **Example:** `MATCH (p:Paper)-[:AUTHORED]->(a:Author) RETURN a`
- **Use:** Query knowledge graphs

---

## D

**DAG (Directed Acyclic Graph)**
- **Definition:** Graph with directed edges and no cycles
- **In Airflow:** Represents workflow dependencies
- **Web Dev Analogy:** Like npm dependency tree
- **Example:** Task B depends on Task A, Task C depends on B

**Dense Retrieval**
- **Definition:** Search using vector embeddings (semantic search)
- **Pros:** Finds similar meaning, not just keywords
- **Cons:** Computationally expensive
- **Contrast:** Sparse retrieval (keyword matching)

**Docker**
- **Definition:** Platform for containerizing applications
- **Benefit:** "Works on my machine" → "Works everywhere"
- **In This Project:** 7-service stack (Neo4j, Qdrant, Kafka, etc.)
- **File:** `docker-compose.yml`

**Dual-Backend Architecture**
- **Definition:** Two implementations of same interface
- **Example:** FAISS (dev) and Qdrant (prod) behind same API
- **Benefit:** Fast development, scalable production
- **Pattern:** Common in software engineering

---

## E

**Embedding**
- **Definition:** Numerical representation of text (array of floats)
- **Size:** Typically 384, 768, or 1536 dimensions
- **Purpose:** Enable mathematical comparison of text
- **Example:** "dog" → [0.23, -0.45, 0.67, ...] (384 numbers)

**Embedding Model**
- **Definition:** Neural network that converts text to embeddings
- **Popular:** `all-MiniLM-L6-v2` (384-dim), `text-embedding-ada-002` (1536-dim)
- **Pre-trained:** Don't need to train yourself
- **Example:** `SentenceTransformer("all-MiniLM-L6-v2")`

**Entity Extraction**
- **Definition:** Identifying entities (people, places, topics) in text
- **In This Project:** Extract authors, papers, topics from documents
- **Methods:** Rule-based, NER models, LLM-based
- **Example:** "John Smith wrote about AI" → Extract: Person="John Smith", Topic="AI"

**ETL (Extract, Transform, Load)**
- **Extract:** Get data from sources (APIs, databases)
- **Transform:** Clean, format, deduplicate
- **Load:** Save to destination database
- **Tool:** Apache Airflow for automation

**Event-Driven Architecture**
- **Definition:** Systems communicate via events (pub/sub)
- **Benefit:** Decoupling, scalability, parallel processing
- **Tool:** Apache Kafka
- **Web Dev Analogy:** Like WebSockets, but for services

---

## F

**FAISS (Facebook AI Similarity Search)**
- **Definition:** Library for efficient similarity search
- **Use Case:** In-memory vector database
- **Speed:** Very fast for search
- **Limitation:** Not persistent, single-machine
- **Best For:** Development, testing

**Few-Shot Learning**
- **Definition:** Learning from few examples (vs zero-shot or fine-tuning)
- **In Prompts:** Giving LLM 2-3 examples of desired output
- **Example:** "Q: ... A: ..., Q: ... A: ..., Q: [your question]"
- **Benefit:** Better results without training

**Fine-Tuning**
- **Definition:** Further training a pre-trained model on your data
- **Cost:** Expensive ($1000s)
- **Benefit:** Better accuracy on domain-specific tasks
- **Alternative:** Prompt engineering, RAG (cheaper)

---

## G

**Gemini (Google Gemini)**
- **Definition:** Google's LLM family
- **Models:** Gemini 2.0 Flash (fast, cheap), Gemini Pro (balanced)
- **Cost:** $0.35 per 1M tokens (Flash)
- **Context:** Up to 1M tokens
- **Used In:** This project for reasoning

**GPU (Graphics Processing Unit)**
- **Definition:** Specialized processor for parallel computations
- **In AI:** Much faster than CPU for neural networks
- **Not Required:** For inference with APIs (they have GPUs)
- **Required For:** Training your own models

**Graph Database**
- **Definition:** Database storing data as nodes and edges
- **Example:** Neo4j, Amazon Neptune
- **Query Language:** Cypher (Neo4j)
- **Best For:** Relationships, multi-hop queries
- **Use Case:** Knowledge graphs

**GraphRAG**
- **Definition:** RAG + knowledge graphs
- **Benefit:** Semantic search (vectors) + relationship queries (graphs)
- **Example:** Find papers (vector) + their authors (graph)
- **Paper:** Emerging research area

---

## H

**Hallucination**
- **Definition:** When LLM generates false or nonsensical information
- **Cause:** Model training artifacts, lack of knowledge, over-generalization
- **Mitigation:** RAG (provide context), citations, verification
- **Example:** LLM cites papers that don't exist

**Hybrid Search**
- **Definition:** Combining multiple search strategies
- **Example:** Vector search + keyword search
- **Or:** Vector search + graph traversal
- **Benefit:** Best of both worlds

**Hyperparameter**
- **Definition:** Configuration setting for ML models
- **Examples:** Learning rate, batch size, temperature
- **For LLMs:** Temperature, top-p, max tokens
- **Tuning:** Adjusting for better results

---

## I

**Index**
- **In Vector DBs:** Data structure for fast similarity search
- **Types:** Flat (exact), HNSW (approximate), IVF (clustered)
- **Trade-off:** Speed vs accuracy vs memory
- **Example:** FAISS IndexFlatL2 (exact, slow), IndexHNSW (approximate, fast)

**Inference**
- **Definition:** Using a trained model to make predictions
- **Contrast:** Training (learning from data)
- **In LLMs:** Generating text given a prompt
- **Example:** Calling `model.predict()` or LLM API

**In-Memory Database**
- **Definition:** Database stored in RAM (not disk)
- **Speed:** Very fast
- **Limitation:** Lost on restart, size limited by RAM
- **Examples:** FAISS, NetworkX (used for development)

---

## J

**JSON (JavaScript Object Notation)**
- **Usage:** Format for API requests/responses, configuration
- **In AI:** Store metadata, event payloads, session data
- **Example:** `{"query": "AI", "results": 10}`

---

## K

**Kafka (Apache Kafka)**
- **Definition:** Distributed event streaming platform
- **Purpose:** Pub/sub messaging, event logs, data pipelines
- **Benefit:** Decoupling, scalability, event replay
- **Web Dev Analogy:** Industrial-strength message queue
- **Port:** Usually 9092

**Knowledge Graph**
- **Definition:** Graph representing entities and relationships
- **Structure:** Nodes (things), Edges (relationships), Properties (attributes)
- **Example:** (Alice)-[AUTHORED]->(Paper1)-[CITES]->(Paper2)
- **Query:** Cypher, SPARQL
- **Tool:** Neo4j, NetworkX

**KNN (K-Nearest Neighbors)**
- **Definition:** Find K most similar items
- **In Vector Search:** Find 10 most similar documents
- **Parameter K:** How many results to return
- **Example:** `search(query, k=10)` returns top 10 matches

---

## L

**LangChain / LangGraph**
- **LangChain:** Framework for building LLM applications
- **LangGraph:** Extension for multi-agent workflows
- **Purpose:** Simplify common LLM patterns (RAG, agents, chains)
- **Used In:** This project for orchestration

**Latency**
- **Definition:** Time delay from request to response
- **In AI:** LLM API calls: 1-5 seconds typical
- **Optimization:** Caching, faster models, parallel processing
- **User Experience:** Sub-second ideal, less than 3s acceptable

**LLM (Large Language Model)**
- **Definition:** Neural network trained on massive text data
- **Examples:** GPT-4, Gemini, Claude, Llama
- **Capability:** Generate text, answer questions, summarize, translate
- **Web Dev Analogy:** Super-powered autocomplete with understanding

**LlamaIndex**
- **Definition:** Framework for building RAG applications
- **Features:** Document loading, indexing, retrieval, integration
- **Benefit:** Handles chunking, embedding, retrieval automatically
- **Used In:** This project for RAG patterns

---

## M

**Microservices**
- **Definition:** Architecture with small, independent services
- **Benefit:** Scalability, maintainability, fault isolation
- **In AI:** Multi-agent systems follow microservices patterns
- **Communication:** REST APIs, message queues

**Model**
- **General:** Trained neural network
- **LLM:** Text generation model (GPT, Gemini)
- **Embedding Model:** Text → vector converter
- **Selection:** Choose based on speed, cost, accuracy needs

**Multi-Agent System**
- **Definition:** Multiple specialized AI agents working together
- **Example:** DataCollector, GraphAgent, VectorAgent, ReasoningAgent
- **Coordination:** Orchestrator agent
- **Benefit:** Separation of concerns, parallel execution

---

## N

**Neo4j**
- **Definition:** Graph database (most popular)
- **Query Language:** Cypher
- **Web UI:** Neo4j Browser (port 7474)
- **Use Case:** Production knowledge graphs
- **Alternative:** NetworkX (in-memory, development)

**Neural Network**
- **Definition:** Computing system inspired by biological brains
- **Structure:** Layers of interconnected "neurons"
- **Training:** Learn patterns from data
- **In AI:** Foundation of LLMs, embeddings, etc.

**NetworkX**
- **Definition:** Python library for graph manipulation
- **Type:** In-memory, not persistent
- **Speed:** Fast for small graphs
- **Use Case:** Development, testing, prototyping
- **Alternative:** Neo4j (production)

**NLP (Natural Language Processing)**
- **Definition:** Field of AI focused on human language
- **Tasks:** Translation, summarization, sentiment analysis
- **Foundation:** Transformers, LLMs
- **Pre-LLM:** Rule-based, statistical models

**Node**
- **In Graphs:** Entity (person, paper, topic)
- **Properties:** Attributes (name, title, date)
- **In Neo4j:** `(:Label {property: value})`
- **Example:** `(:Paper {title: "Attention Is All You Need"})`

---

## O

**Observability**
- **Definition:** Ability to understand system internal state from outputs
- **Three Pillars:** Metrics, Logs, Traces
- **Tools:** Prometheus, Grafana, Airflow UI, Kafka UI
- **Importance:** Essential for debugging production systems

**Orchestrator/Orchestration**
- **Definition:** Component coordinating multiple agents/services
- **Pattern:** Common in microservices, multi-agent systems
- **Example:** OrchestratorAgent coordinates all other agents
- **Benefit:** Central control, workflow management

---

## P

**Parallel Processing**
- **Definition:** Executing multiple tasks simultaneously
- **Benefit:** Faster completion (3x in this project)
- **Example:** Fetch from arXiv and PubMed at same time
- **Tools:** Python `asyncio`, Airflow DAGs, Kafka consumers

**Persistent Storage**
- **Definition:** Data saved to disk (survives restarts)
- **Contrast:** In-memory (lost on restart)
- **Examples:** Neo4j, Qdrant (persistent), FAISS, NetworkX (in-memory)
- **Trade-off:** Speed vs persistence

**Pipeline**
- **Definition:** Series of data processing steps
- **Example:** ETL pipeline, ML pipeline, RAG pipeline
- **Tool:** Apache Airflow for orchestration
- **Visualization:** DAG (directed acyclic graph)

**Prompt**
- **Definition:** Text input to an LLM
- **Components:** System prompt (role), user prompt (question), context
- **Engineering:** Crafting prompts for better results
- **Example:** "You are a helpful assistant. Answer based on: \{context\}"

**Prompt Engineering**
- **Definition:** Skill of crafting effective LLM prompts
- **Techniques:** Few-shot, chain-of-thought, role-playing
- **Importance:** Can improve results without fine-tuning
- **Iterative:** Takes experimentation

**Pub/Sub (Publish-Subscribe)**
- **Definition:** Messaging pattern where senders don't know receivers
- **Flow:** Publisher → Topic → Subscribers
- **Benefit:** Decoupling, scalability
- **Tool:** Apache Kafka
- **Web Dev Analogy:** Event emitters

---

## Q

**Qdrant**
- **Definition:** Vector database (open source)
- **Features:** REST API, dashboard, persistence
- **Speed:** Fast similarity search
- **Use Case:** Production vector search
- **Alternative:** FAISS (development), Pinecone (managed)

**Query**
- **General:** Request for information
- **In DBs:** SQL query, Cypher query, vector search
- **In AI:** User question to LLM
- **Example:** "What are transformers in AI?"

---

## R

**RAG (Retrieval-Augmented Generation)**
- **Definition:** LLM + retrieval from your data
- **Steps:** 1) Retrieve relevant docs, 2) Add to prompt, 3) Generate answer
- **Benefit:** LLM can answer about YOUR data (not just training data)
- **Example:** "What did John say in meeting?" → Search notes → LLM answers

**RDF (Resource Description Framework)**
- **Definition:** Standard for representing information (triples)
- **Structure:** Subject-Predicate-Object
- **Example:** (Alice, AUTHORED, Paper1)
- **Use Case:** Semantic web, knowledge graphs
- **Query Language:** SPARQL

**Reasoning Agent**
- **Definition:** AI agent that generates answers using LLM
- **Input:** Question + context (from retrieval)
- **Output:** Answer with citations
- **Model:** Gemini 2.0 Flash (in this project)

**Relationship**
- **In Graphs:** Connection between nodes (edge)
- **Types:** AUTHORED, CITES, IS_ABOUT, etc.
- **Direction:** Can be unidirectional or bidirectional
- **Example:** (Alice)-[AUTHORED]->(Paper1)

**Retrieval**
- **Definition:** Finding relevant information
- **Methods:** Vector search, keyword search, graph traversal
- **In RAG:** First step (before generation)
- **Goal:** Find context for LLM

---

## S

**Scaling**
- **Vertical:** More powerful hardware (bigger server)
- **Horizontal:** More servers (distribute load)
- **In AI:** Add more agents, Kafka consumers, API workers
- **Benefit:** Handle more users, faster processing

**Semantic Search**
- **Definition:** Search by meaning (not exact keywords)
- **Technology:** Vector embeddings + similarity search
- **Example:** "ML tutorial" finds "Machine Learning Guide"
- **Contrast:** Keyword search (exact match only)

**Sentence Transformers**
- **Definition:** Library for creating sentence embeddings
- **Models:** all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
- **Output:** 384 or 768-dimensional vectors
- **Usage:** `SentenceTransformer("all-MiniLM-L6-v2").encode(text)`

**Similarity Score**
- **Definition:** Measure of how similar two vectors are
- **Range:** 0 (different) to 1 (identical)
- **Calculation:** Cosine similarity, dot product
- **Example:** "dog" vs "puppy" = 0.87 (similar)

**Sparse Retrieval**
- **Definition:** Keyword-based search (TF-IDF, BM25)
- **Pros:** Fast, interpretable
- **Cons:** Misses synonyms and semantic meaning
- **Contrast:** Dense retrieval (embeddings)

---

## T

**Temperature**
- **Definition:** LLM creativity parameter
- **Range:** 0.0 (deterministic) to 1.0 (creative)
- **Use Cases:** 0 for facts, 0.7 for creative writing
- **Effect:** Higher = more random/creative outputs

**Token**
- **Definition:** Piece of text (roughly a word)
- **Examples:** "Hello" = 1 token, "world" = 1 token, "!" = 1 token
- **Importance:** LLM APIs charge per token
- **Estimation:** ~750 words = 1000 tokens

**Token Budget**
- **Definition:** Limit on LLM API usage
- **Purpose:** Prevent runaway costs
- **Example:** Max 10K tokens per request, 100K per user per day
- **Implementation:** Middleware that tracks usage

**Transformer**
- **Definition:** Neural network architecture using attention
- **Innovation:** Parallel processing, long-range dependencies
- **Foundation:** All modern LLMs (GPT, BERT, Gemini)
- **Paper:** "Attention is All You Need" (2017)

---

## U

**Unstructured Data**
- **Definition:** Data without predefined format (text, images, audio)
- **Contrast:** Structured data (databases, spreadsheets)
- **Challenge:** Hard to query with SQL
- **Solution:** Embeddings, LLMs

---

## V

**Vector**
- **Definition:** Array of numbers
- **In AI:** Numerical representation of text/data
- **Example:** [0.2, 0.8, -0.3, 0.5] (4-dimensional)
- **Purpose:** Enable mathematical operations on text

**Vector Database**
- **Definition:** Database optimized for vector similarity search
- **Examples:** FAISS, Qdrant, Pinecone, Weaviate
- **Query:** Find K nearest neighbors (KNN)
- **Use Case:** Semantic search in RAG systems

**Vector Search**
- **Definition:** Finding similar vectors by distance/similarity
- **Algorithm:** KNN (k-nearest neighbors)
- **Distance Metrics:** Cosine similarity, Euclidean distance, dot product
- **Example:** Find 10 documents most similar to query

**Versioning**
- **Models:** Track which model version in production
- **Data:** Track dataset versions for reproducibility
- **Prompts:** Version control for prompt templates
- **Tools:** Git, DVC (Data Version Control)

---

## W

**Workflow**
- **Definition:** Sequence of automated tasks
- **Tool:** Apache Airflow
- **Representation:** DAG (directed acyclic graph)
- **Example:** Collect → Clean → Index → Notify

---

## Z

**Zero-Shot Learning**
- **Definition:** LLM performs task without examples
- **Example:** "Translate to French: Hello" (no examples given)
- **Contrast:** Few-shot (2-3 examples), fine-tuning (many examples)
- **Capability:** Modern LLMs are good at zero-shot

---

## Quick Reference: Beginner Traps



**Common Mistakes:**

1. **Not chunking documents** → Token limit errors
2. **Using expensive models for everything** → High costs
3. **No caching** → Repeated expensive API calls
4. **Trusting LLM outputs blindly** → Hallucinations
5. **Ignoring latency** → Poor user experience
6. **No error handling** → System crashes
7. **Not monitoring costs** → Surprise bills
8. **Poor prompt engineering** → Bad results

**Best Practices:**

✅ Start with cheap models, upgrade if needed
✅ Cache aggressively
✅ Chunk long documents
✅ Verify LLM outputs
✅ Set token budgets
✅ Monitor everything
✅ Test prompts iteratively
✅ Handle errors gracefully



---


  <a href="primer">← Primer for Web Developers</a>
  <a href="/">Introduction →</a>

