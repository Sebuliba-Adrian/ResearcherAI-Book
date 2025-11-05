---
layout: default
title: Primer for Web Developers
---

# Chapter 0: AI/ML Primer for Web Developers


Coming from web development? Welcome! This chapter explains all the AI/ML concepts you'll encounter in this book using analogies and examples from traditional web development. No prior AI knowledge required.


## Introduction: From Web Apps to AI Apps

If you've built traditional web applications, you already understand:
- **Databases** (SQL, MongoDB) store and retrieve data
- **APIs** connect systems and fetch data
- **Frontend** displays information to users
- **Backend** handles business logic

AI applications add new capabilities:
- **Understanding natural language** (not just exact matches)
- **Finding similar content** (semantic search vs keyword search)
- **Generating human-like responses** (not just templated responses)
- **Discovering relationships** in data automatically

This book teaches you how to build these AI capabilities into your applications.

---

## Part 1: Core AI Concepts for Web Developers

### 1.1 What is Machine Learning? (The Database That Learns)

**Traditional Programming:**
```javascript
// You write explicit rules
function isSpam(email) {
  if (email.includes("viagra") || email.includes("lottery")) {
    return true;
  }
  return false;
}
```

**Machine Learning:**
```javascript
// The system learns patterns from examples
const model = trainSpamDetector(10000_labeled_emails);
const isSpam = model.predict(new_email); // Learns patterns you didn't code
```

**Key Insight:** Instead of writing rules, you provide examples and the system learns the patterns.

**Web Dev Analogy:**
- **Traditional:** Writing SQL queries by hand
- **ML:** The database figures out the best query optimization automatically

### 1.2 What are Large Language Models (LLMs)?

Think of LLMs like **super-powered autocomplete** that understands context.

**Traditional Autocomplete:**
```javascript
// Simple pattern matching
userTypes("hello w") → suggests "world"
```

**LLM (like ChatGPT, Gemini):**
```javascript
// Understands context and meaning
userAsks("What are transformers in deep learning?")
→ LLM generates coherent, contextual explanation
```

**How They Work (Simplified):**
1. **Training:** Read billions of text documents (like Wikipedia, books, code)
2. **Learning:** Find patterns in how words relate to each other
3. **Prediction:** Given some text, predict what comes next (but context-aware)

**Web Dev Analogy:**
- **Traditional:** Template engines (fill in blanks: `Hello {{name}}`)
- **LLM:** Dynamic content generation that understands what you're asking for

**Popular LLMs:**
- **OpenAI GPT-4** - Most capable, expensive ($30 per 1M tokens)
- **Google Gemini** - Fast, affordable ($0.35 per 1M tokens) ← *We use this*
- **Anthropic Claude** - Good at long documents
- **Llama** - Open source, self-hostable

**Token = Word-ish:** "Hello world" ≈ 2 tokens. A typical paragraph ≈ 100 tokens.

### 1.3 What is RAG? (Making LLMs Smarter with Your Data)

**Problem:** LLMs don't know about YOUR data (your documents, your company info)

**Solution:** RAG = **Retrieval-Augmented Generation**

**How It Works:**

```javascript
// Traditional LLM (doesn't know about your data)
answer = llm.ask("What did John say in yesterday's meeting?")
// → "I don't have access to specific meeting information"

// RAG Approach
1. Find relevant documents: docs = search("meeting notes + John")
2. Give context to LLM: answer = llm.ask(question, context=docs)
// → "John mentioned the Q4 roadmap includes feature X..."
```

**Web Dev Analogy:**
- **Traditional LLM:** A chatbot with hardcoded responses
- **RAG:** A chatbot that can search your database first, then respond with actual data

**Three Steps of RAG:**
1. **Retrieval:** Find relevant documents/data
2. **Augmentation:** Add that data to your question
3. **Generation:** LLM generates answer using the context

**Example Flow:**
```
User: "What are transformers in AI?"
   ↓
1. Search vector DB for papers about transformers
   ↓
2. Find 5 relevant paper excerpts
   ↓
3. Give excerpts + question to LLM
   ↓
4. LLM generates answer with citations
```

---

## Part 2: Databases for AI (Not Your Father's SQL)

### 2.1 Vector Databases (Finding "Similar" Things)

**The Problem with Traditional Search:**

```javascript
// SQL/MongoDB (exact matches only)
SELECT * FROM articles WHERE title LIKE '%machine learning%'
// Finds: "Machine Learning Basics" ✓
// Misses: "Neural Networks Tutorial" ✗ (same topic, different words!)
```

**Vector Database Solution:**

Converts text into numbers (vectors) that capture MEANING:

```javascript
// Convert text to vectors (numbers)
"machine learning" → [0.2, 0.8, 0.1, ...] (384 numbers)
"neural networks"  → [0.3, 0.7, 0.2, ...] (similar numbers!)
"cooking recipes"  → [0.9, 0.1, 0.8, ...] (different numbers)

// Search by similarity (not exact match)
results = vectorDB.search("AI tutorials", topK=10)
// Finds: "Machine Learning", "Neural Networks", "Deep Learning"
// All semantically similar, even with different words!
```

**Web Dev Analogy:**
- **Traditional:** `string.includes("keyword")` - exact match
- **Vector DB:** Fuzzy matching that understands synonyms and concepts

**How Vectors Work:**

Think of vectors as **GPS coordinates for meaning**:
- Words with similar meanings → nearby coordinates
- Words with different meanings → far apart coordinates

```
"king" and "queen" → Close together (both royalty)
"king" and "pizza" → Far apart (unrelated concepts)
```

**Popular Vector Databases:**
- **FAISS** (Facebook) - In-memory, fast, free ← *We use this for dev*
- **Qdrant** - Production-ready, REST API ← *We use this for prod*
- **Pinecone** - Managed cloud service
- **Weaviate** - Good for hybrid search

### 2.2 Embeddings (Turning Words into Numbers)

**What Are Embeddings?**

A way to convert text into numbers that computers can compare.

```javascript
// Traditional (can't compare meaning)
"dog" === "puppy" // false (different strings)

// Embeddings (can compare meaning)
embed("dog")   → [0.8, 0.2, 0.6, ...]
embed("puppy") → [0.7, 0.3, 0.5, ...] // Similar numbers!
similarity(embed("dog"), embed("puppy")) // 0.87 (very similar!)

embed("car") → [0.1, 0.9, 0.2, ...]  // Different numbers
similarity(embed("dog"), embed("car")) // 0.12 (not similar)
```

**Web Dev Analogy:**
- Like CSS colors: "red" and "#FF0000" represent the same thing
- Embeddings: Text → Array of numbers that represent meaning

**How We Generate Embeddings:**

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model (someone already trained this for you!)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to 384-dimensional vector
text = "I love programming"
embedding = model.encode(text)  # [0.234, -0.567, 0.890, ...] (384 numbers)
```

**Key Points:**
- **384 dimensions** = 384 numbers per piece of text
- **Pre-trained models** = You don't need to train them yourself
- **Fast** = Generating embeddings is quick (milliseconds)

### 2.3 Knowledge Graphs (The Database That Shows Relationships)

**Problem with Traditional Databases:**

```sql
-- Traditional relational DB
Papers (id, title, abstract)
Authors (id, name)
Paper_Authors (paper_id, author_id)

-- Query: "Find colleagues of Alice" (papers co-authored)
-- Requires complex JOINs, slow for deep relationships
```

**Knowledge Graph Solution:**

Stores data as **nodes** (things) and **edges** (relationships):

```
(Alice) -[AUTHORED]-> (Paper1)
(Bob)   -[AUTHORED]-> (Paper1)  // Alice and Bob are colleagues!
(Paper1) -[CITES]-> (Paper2)
(Paper2) -[IS_ABOUT]-> (Topic: "AI")
```

**Web Dev Analogy:**
- **SQL Tables:** Like separate spreadsheets with foreign keys
- **Knowledge Graph:** Like a mind map where everything connects visually

**Why Knowledge Graphs for AI?**

1. **Multi-hop queries** (follow relationships):
   ```cypher
   // Find papers by colleagues of colleagues (2 hops)
   MATCH (alice:Author)-[:AUTHORED]->(:Paper)<-[:AUTHORED]-(colleague)
         (colleague)-[:AUTHORED]->(:Paper)<-[:AUTHORED]-(colleague2)
         (colleague2)-[:AUTHORED]->(paper:Paper)
   RETURN paper
   ```

2. **Discover hidden connections**:
   ```
   Alice wrote Paper1 → Paper1 cites Paper2 → Paper2 is about "Transformers"
   // Insight: Alice's work relates to Transformers!
   ```

**Tools:**
- **Neo4j** - Production graph database ← *We use this*
- **NetworkX** - Python library for graphs ← *We use this for dev*
- **Amazon Neptune** - Managed graph database
- **ArangoDB** - Multi-model (graph + document)

**Example in ResearcherAI:**

```python
# Find related papers (graph traversal)
MATCH (paper:Paper {id: "arxiv:1234"})
      -[:IS_ABOUT]->(topic:Topic)
      <-[:IS_ABOUT]-(related:Paper)
RETURN related

// Finds papers about the same topics (semantic relationship)
```

### 2.4 Hybrid Search (Best of Both Worlds)

**The Power of Combining Vectors + Graphs:**

```javascript
// Step 1: Vector search (find semantically similar content)
const similarPapers = vectorDB.search("deep learning", topK=20);

// Step 2: Graph traversal (find related entities)
const paperIds = similarPapers.map(p => p.id);
const relatedAuthors = graphDB.query(`
  MATCH (p:Paper)-[:AUTHORED]->(a:Author)
  WHERE p.id IN ${paperIds}
  RETURN a
`);

// Step 3: Combine results
// Now we have papers (from vector search) + their authors (from graph)
```

**Why Both?**

| Capability | Vector DB | Graph DB |
|------------|-----------|----------|
| "Find similar papers" | ✅ Excellent | ❌ Poor |
| "Who authored this?" | ❌ Poor | ✅ Excellent |
| "Papers citing this" | ❌ Poor | ✅ Excellent |
| Fuzzy matching | ✅ Yes | ❌ No |

**Web Dev Analogy:**
- **Vector DB:** Like Elasticsearch (full-text search with relevance)
- **Graph DB:** Like JOIN queries (relationships between records)
- **Hybrid:** Using both for comprehensive results

---

## Part 3: Architectural Patterns

### 3.1 Multi-Agent Systems (Microservices for AI)

**Traditional Monolith:**
```javascript
// One big function does everything
async function handleUserQuery(question) {
  const papers = await fetchPapers(question);      // Slow
  const processed = await processPapers(papers);   // Slow
  const answer = await generateAnswer(question, processed); // Slow
  return answer; // Total: VERY SLOW
}
```

**Multi-Agent Approach:**
```javascript
// Specialized agents (like microservices)
class DataCollectorAgent {
  async collect(query) { /* Only fetches papers */ }
}

class KnowledgeGraphAgent {
  async addPapers(papers) { /* Only builds graph */ }
}

class VectorAgent {
  async indexPapers(papers) { /* Only creates embeddings */ }
}

class OrchestratorAgent {
  async handleQuery(question) {
    // Coordinate agents (can run in parallel!)
    const [papers, graphData, vectors] = await Promise.all([
      dataCollector.collect(question),
      graphAgent.buildGraph(question),
      vectorAgent.search(question)
    ]);
    return reasoningAgent.answer(question, {papers, graphData, vectors});
  }
}
```

**Benefits:**
- **Parallel execution** (faster)
- **Easier to test** (test each agent separately)
- **Easier to update** (change one agent without touching others)
- **Scales independently** (add more of the slow agents)

**Web Dev Analogy:**
- **Monolith:** One Express.js app doing everything
- **Multi-Agent:** Microservices architecture with API gateway

### 3.2 Event-Driven Architecture (Pub/Sub for AI)

**Synchronous (Traditional):**
```javascript
// Agent A must wait for Agent B
const papers = await dataCollector.collect(query);  // Wait...
const graph = await graphAgent.addPapers(papers);   // Wait...
const vectors = await vectorAgent.index(papers);    // Wait...
// Total time: 10s + 5s + 3s = 18 seconds
```

**Event-Driven (Kafka):**
```javascript
// Agent A publishes event, doesn't wait
eventBus.publish("papers.collected", papers);  // Instant!

// Agents B and C subscribe and run in parallel
graphAgent.on("papers.collected", (papers) => {
  // Process in parallel (5s)
});

vectorAgent.on("papers.collected", (papers) => {
  // Process in parallel (3s)
});
// Total time: 10s + max(5s, 3s) = 15 seconds (faster!)
```

**Web Dev Analogy:**
- **Synchronous:** REST API calls (wait for response)
- **Event-Driven:** WebSockets / Message Queue (fire and forget)

**Kafka = Industrial-Strength Message Queue**

```javascript
// Publish event
producer.send({
  topic: "data.collection.completed",
  value: JSON.stringify({ papers, query, timestamp })
});

// Subscribe to event
consumer.subscribe(["data.collection.completed"]);
consumer.on("message", (event) => {
  const { papers } = JSON.parse(event.value);
  processPapers(papers);
});
```

**Benefits:**
- **Decoupling** (agents don't need to know about each other)
- **Event replay** (can re-process events for debugging)
- **Scaling** (add more consumers to process events faster)
- **Reliability** (events persist even if agents crash)

### 3.3 ETL (Extract, Transform, Load) with Airflow

**What is ETL?**

```javascript
// E = Extract (get data from sources)
const arxivPapers = await fetchFromArxiv();
const pubmedPapers = await fetchFromPubMed();

// T = Transform (clean and format)
const cleaned = papers.map(cleanData);
const deduplicated = removeDuplicates(cleaned);

// L = Load (save to database)
await database.insert(deduplicated);
```

**Problem:** Running ETL manually is tedious and error-prone.

**Solution:** Apache Airflow (scheduled workflows)

```python
# Define workflow as code
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    "paper_collection",
    schedule_interval="@daily",  # Run every day
)

# Tasks run in order or parallel
collect_arxiv = PythonOperator(
    task_id="collect_arxiv",
    python_callable=fetch_arxiv_papers,
    dag=dag
)

collect_pubmed = PythonOperator(
    task_id="collect_pubmed",
    python_callable=fetch_pubmed_papers,
    dag=dag
)

# Both run in parallel, then merge
[collect_arxiv, collect_pubmed] >> merge_papers
```

**Web Dev Analogy:**
- **Manual ETL:** Running npm scripts manually
- **Airflow:** GitHub Actions / Jenkins (automated workflows)

**Benefits:**
- **Scheduled execution** (runs automatically)
- **Retries** (automatically retries if failures)
- **Monitoring** (web UI shows progress)
- **Dependencies** (Task B waits for Task A)

---

## Part 4: Making Sense of the Jargon

### 4.1 Common Terms Explained



**Tokens:**
- **What:** Pieces of text (roughly words)
- **Example:** "Hello world" = 2 tokens
- **Why Care:** LLMs charge per token
- **Web Dev Analogy:** Like counting characters for SMS messages

**Context Window:**
- **What:** How much text an LLM can "remember" at once
- **Example:** GPT-4 can handle ~8,000 tokens (≈6,000 words)
- **Why Care:** Limits how much data you can send
- **Web Dev Analogy:** Like browser memory limits

**Fine-Tuning:**
- **What:** Training an existing model on your specific data
- **Example:** Make GPT-4 better at medical terminology
- **Why Care:** Expensive but improves accuracy
- **Web Dev Analogy:** Like customizing a framework for your needs

**Prompt:**
- **What:** The text you send to an LLM
- **Example:** "Explain quantum physics in simple terms"
- **Why Care:** Better prompts = better answers
- **Web Dev Analogy:** Like crafting good search queries

**Temperature:**
- **What:** How "creative" vs "deterministic" the LLM is
- **Range:** 0.0 (always same answer) to 1.0 (more random)
- **Example:** 0 for facts, 0.7 for creative writing
- **Web Dev Analogy:** Like randomness in game algorithms

**Hallucination:**
- **What:** When LLM makes up false information
- **Example:** Cites papers that don't exist
- **Why Care:** Need to verify LLM outputs
- **Web Dev Analogy:** Like a bug that returns incorrect data

**Embeddings (Revisited):**
- **What:** Text converted to numbers
- **Size:** Usually 384, 768, or 1536 dimensions
- **Why Care:** Enables semantic search
- **Web Dev Analogy:** Like hashing, but preserves similarity

**Chunking:**
- **What:** Breaking long documents into smaller pieces
- **Example:** 500-word chunks with 50-word overlap
- **Why Care:** LLMs have token limits
- **Web Dev Analogy:** Like pagination

**Similarity Score:**
- **What:** How similar two pieces of text are
- **Range:** 0.0 (completely different) to 1.0 (identical)
- **Example:** "dog" vs "puppy" = 0.87
- **Web Dev Analogy:** Like Levenshtein distance, but semantic

**Cypher:**
- **What:** Query language for Neo4j (like SQL for graphs)
- **Example:** `MATCH (p:Paper) RETURN p`
- **Why Care:** How you query knowledge graphs
- **Web Dev Analogy:** Like SQL, but for relationships

**DAG (Directed Acyclic Graph):**
- **What:** Workflow where tasks have dependencies
- **Example:** Task B depends on Task A completing
- **Why Care:** How Airflow organizes workflows
- **Web Dev Analogy:** Like npm dependency tree



### 4.2 Architecture Terms



**Circuit Breaker:**
- **What:** Stops calling a failing service
- **Example:** If API fails 5 times, stop trying for 60s
- **Why:** Prevents cascade failures
- **Web Dev Analogy:** Like rate limiting

**Token Budget:**
- **What:** Limit on LLM API spending
- **Example:** Max $100/day, max 10K tokens per request
- **Why:** Prevents runaway costs
- **Web Dev Analogy:** Like API rate limits

**Dual Backend:**
- **What:** Two implementations of same interface
- **Example:** FAISS (dev) vs Qdrant (prod)
- **Why:** Fast dev, scalable prod
- **Web Dev Analogy:** SQLite (dev) vs PostgreSQL (prod)

**Orchestrator:**
- **What:** Coordinates multiple agents
- **Example:** Decides which agent runs when
- **Why:** Central control point
- **Web Dev Analogy:** API Gateway / Load Balancer

**Retrieval Strategy:**
- **What:** How you find relevant documents
- **Types:** Dense (vectors), Sparse (keywords), Hybrid (both)
- **Why:** Affects search quality
- **Web Dev Analogy:** Different database indexes



---

## Part 5: Practical Examples

### 5.1 Building Your First RAG System (Simplified)

Let's build a simple "Ask questions about documents" system:

```python
# Step 1: Install libraries
# pip install sentence-transformers faiss-cpu openai

# Step 2: Load documents
documents = [
    "Python is a programming language.",
    "JavaScript runs in browsers.",
    "Databases store data."
]

# Step 3: Convert to embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)  # [[0.2, 0.8, ...], ...]

# Step 4: Store in vector database (FAISS)
import faiss
import numpy as np

dimension = 384  # all-MiniLM-L6-v2 outputs 384 dimensions
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))  # Add embeddings to index

# Step 5: Search
question = "How do I store information?"
question_embedding = model.encode([question])
distances, indices = index.search(np.array(question_embedding), k=2)

# Results: indices = [2, 1] (document 2 is most similar)
print(documents[indices[0][0]])  # "Databases store data."

# Step 6: Use LLM to generate answer
import openai
context = "\n".join([documents[i] for i in indices[0]])
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Answer using the context provided."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
)
print(response.choices[0].message.content)
# "You can store information using databases."
```

**That's it!** You've built a basic RAG system in 30 lines of code.

### 5.2 Common Pitfalls for Beginners



**Pitfall 1: Using LLMs for Everything**
```javascript
// ❌ Bad: Use LLM for simple tasks
const answer = await llm.ask("What is 2 + 2?"); // Costs money!

// ✅ Good: Use logic when possible
const answer = 2 + 2; // Free!
```

**Pitfall 2: Not Chunking Long Documents**
```python
# ❌ Bad: Send 10,000-word document to LLM
answer = llm.ask(long_document + question)  # Exceeds token limit!

# ✅ Good: Chunk and search relevant parts
chunks = chunk_document(long_document, chunk_size=500)
relevant = vector_search(question, chunks, top_k=3)
answer = llm.ask(relevant + question)  # Only sends relevant parts
```

**Pitfall 3: Ignoring Costs**
```python
# ❌ Bad: No limits
for question in user_questions:  # User asks 10,000 questions!
    answer = expensive_llm.ask(question)  # $$$$$

# ✅ Good: Budget and cache
@cache(ttl=3600)  # Cache for 1 hour
@token_budget(max=10000)  # Limit spending
def ask_question(question):
    return llm.ask(question)
```

**Pitfall 4: Trusting LLM Outputs Blindly**
```python
# ❌ Bad: Assume LLM is always right
answer = llm.ask("What is the cure for cancer?")
return answer  # Might hallucinate!

# ✅ Good: Verify and cite sources
context = search_papers("cancer treatment")
answer = llm.ask(question, context=context)
citations = extract_citations(answer)
return {"answer": answer, "sources": citations}  # Verifiable
```



---

## Part 6: Learning Path

### 6.1 What to Learn First

If you're completely new, follow this order:

**Week 1-2: Foundations**
1. **Embeddings & Vector Search**
   - Play with: Sentence Transformers
   - Build: Simple semantic search (see example above)
   - Resource: Sentence-BERT paper (cited in bibliography)

2. **LLM Basics**
   - Play with: ChatGPT, Google Gemini API
   - Build: Simple Q&A bot
   - Resource: OpenAI Cookbook

**Week 3-4: RAG Systems**
3. **Basic RAG**
   - Play with: LlamaIndex
   - Build: "Chat with your documents" app
   - Resource: LlamaIndex documentation

4. **Vector Databases**
   - Play with: FAISS (local)
   - Build: Scaled semantic search
   - Resource: FAISS GitHub examples

**Week 5-6: Advanced Concepts**
5. **Knowledge Graphs**
   - Play with: NetworkX
   - Build: Simple relationship graph
   - Resource: Neo4j tutorial

6. **Multi-Agent Systems**
   - Play with: LangGraph
   - Build: Two agents working together
   - Resource: LangGraph documentation

**Week 7-8: Production Patterns**
7. **Event-Driven Architecture**
   - Play with: Python message queues
   - Build: Producer-consumer pattern
   - Resource: Kafka documentation

8. **Deployment**
   - Play with: Docker
   - Build: Containerized RAG app
   - Resource: Docker tutorial

### 6.2 Recommended Free Resources

**Interactive Tutorials:**
- **fast.ai** - Practical deep learning for coders
- **Hugging Face Course** - NLP and transformers
- **LangChain Academy** - Building with LLMs
- **Neo4j GraphAcademy** - Knowledge graphs

**YouTube Channels:**
- **Sentdex** - Python & AI tutorials
- **3Blue1Brown** - Visual math explanations
- **Two Minute Papers** - Latest AI research explained

**Practice Projects:**
1. Build a semantic search for your bookmark collection
2. Create a Q&A bot for your company's documentation
3. Make a recommendation system using embeddings
4. Build a simple knowledge graph of your notes

---

## Part 7: How This Book is Structured for You

Now that you understand the basics, here's how to read this book:

**Chapter 1: Planning**
- NOW YOU KNOW: Requirements, tech selection
- YOU'LL LEARN: How to plan an AI project
- ASSUMES: Web dev knowledge, general programming

**Chapter 2: Architecture**
- NOW YOU KNOW: Multi-agent, RAG, dual backends
- YOU'LL LEARN: How to design scalable AI systems
- ASSUMES: Everything from this primer

**Chapters 3-7: Implementation**
- NOW YOU KNOW: All concepts explained above
- YOU'LL LEARN: Actual code implementation
- ASSUMES: Python basics, Docker basics

**Chapter 8: Conclusion**
- NOW YOU KNOW: Complete working system
- YOU'LL LEARN: Production best practices
- ASSUMES: You've read previous chapters

---

## Key Takeaways for Web Developers



**What's Familiar:**
- ✅ Databases (just different types)
- ✅ APIs (calling LLM APIs like any REST API)
- ✅ Microservices (multi-agent is similar)
- ✅ Message queues (Kafka is pub/sub)
- ✅ Docker (same containerization)
- ✅ CI/CD (same deployment patterns)

**What's New:**
- 📚 **Vectors** instead of just strings
- 📚 **Embeddings** for semantic understanding
- 📚 **Graphs** for relationship queries
- 📚 **LLM APIs** for generation
- 📚 **Prompt engineering** as a skill

**Cost Consciousness:**
- Traditional apps: Fixed infrastructure cost
- AI apps: Pay-per-token (like pay-per-API-call)
- **Must implement:** Caching, budgets, smart model selection

**Testing Challenges:**
- Traditional: Deterministic outputs (same input = same output)
- AI: Non-deterministic (LLMs can give different answers)
- **Must implement:** Prompt testing, output validation



---

## You're Ready!

You now understand all the core concepts needed to follow this book. When you encounter technical terms in later chapters, refer back to this primer.

Remember: **AI/ML is just new tools in your toolkit.** You already know how to build applications - this book teaches you how to add AI capabilities to them.

**Next Chapter:** [Introduction](/) - Understanding the problem ResearcherAI solves

---

## Quick Reference Card

Keep this handy while reading:

| Term | Simple Definition | Like... |
|------|------------------|---------|
| **LLM** | AI that generates text | Super autocomplete |
| **RAG** | LLM + your data | Chatbot + database search |
| **Vector** | Numbers representing meaning | GPS coordinates for words |
| **Embedding** | Text → Vector conversion | Turning text into coordinates |
| **Vector DB** | Database for semantic search | Fuzzy matching database |
| **Knowledge Graph** | Database of relationships | Mind map as a database |
| **Multi-Agent** | Multiple AI services | Microservices for AI |
| **Kafka** | Event streaming | Industrial message queue |
| **Airflow** | Workflow automation | Cron jobs with dependencies |
| **Token** | Piece of text (~word) | SMS character counting |
| **Prompt** | Text sent to LLM | Query to AI |
| **Hallucination** | LLM making up facts | AI bug/lie |

---


  <a href="00-frontmatter.html">← Front Matter</a>
  <a href="/">Next: Introduction →</a>

