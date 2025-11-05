---
slug: /
title: Home
---

# Building ResearcherAI: A Journey from Idea to Production


Welcome! I'm Adrian, and this is the story of how I built **ResearcherAI** - a production-ready multi-agent system that helps researchers discover, analyze, and synthesize academic papers using AI.


## What You'll Learn

This isn't just another tutorial. It's a real journey through building a complex system from scratch, complete with the challenges I faced, the decisions I made, and the lessons I learned along the way.



### 📋 Table of Contents

1. **[Planning & Requirements](planning)**
   - How I came up with the idea
   - Defining the problem and goals
   - Choosing the tech stack
   - Planning the architecture

2. **[Architecture Design](architecture)**
   - Multi-agent system patterns
   - Dual-backend strategy (dev vs production)
   - Knowledge graphs + vector search
   - Event-driven architecture with Kafka

3. **[Backend Development](backend)**
   - Building 6 specialized agents
   - Integrating 7 data sources
   - LangGraph + LlamaIndex orchestration
   - Conversation memory and sessions

4. **[Frontend Development](frontend)**
   - React + TypeScript setup
   - Glassmorphism design system
   - Framer Motion animations
   - API integration layer

5. **[Testing Strategy](testing)**
   - Achieving 96.60% test coverage
   - Unit, integration, and E2E tests
   - Testing dual backends
   - CI/CD with GitHub Actions

6. **[Deployment](deployment)**
   - Docker containerization
   - Multi-service orchestration
   - GitHub Actions workflows
   - Production considerations

7. **[Monitoring & Operations](monitoring)**
   - Apache Airflow for ETL
   - Kafka UI for event monitoring
   - Neo4j and Qdrant dashboards
   - Performance optimization



## The Journey at a Glance



### 🎯 What I Built

A sophisticated research assistant that:
- **Collects** papers from 7 sources (arXiv, PubMed, Semantic Scholar, etc.)
- **Analyzes** them using knowledge graphs and vector embeddings
- **Answers** questions with context-aware AI reasoning
- **Scales** from laptop development to production deployment

### 📊 By the Numbers

<span class="badge">96.60% Test Coverage</span>
<span class="badge">291 Tests Passing</span>
<span class="badge">6 Specialized Agents</span>
<span class="badge">7 Data Sources</span>
<span class="badge">2 Deployment Modes</span>

### ⚡ Key Features

- **Dual-Backend Architecture**: Switch between dev (NetworkX+FAISS) and prod (Neo4j+Qdrant) with zero code changes
- **Event-Driven**: Kafka-based async communication between agents
- **Production-Grade**: 40-70% cost reduction with intelligent caching and model selection
- **Modern UI**: React frontend with glassmorphism design
- **Fully Tested**: Comprehensive test suite with 96.60% coverage



## Why I Built This

I was frustrated with how hard it was to keep up with research papers in AI/ML. There are thousands published every month, and it's impossible to read them all. I wanted a tool that could:

1. **Find** relevant papers across multiple sources automatically
2. **Understand** the relationships between papers, authors, and concepts
3. **Answer** my questions by synthesizing information from multiple sources
4. **Remember** context from our previous conversations

But I also wanted to learn best practices for building production systems with:
- Multi-agent architectures
- Knowledge graphs and vector databases
- Event-driven patterns
- Comprehensive testing
- Modern frontend development

So I built ResearcherAI.

## The Tech Stack

Here's what I used and why:



| Component | Technology | Why I Chose It |
|-----------|-----------|----------------|
| **LLM** | Gemini 2.0 Flash | Fast, accurate, and cost-effective |
| **Orchestration** | LangGraph | Perfect for multi-agent workflows |
| **RAG** | LlamaIndex | Best-in-class for retrieval augmentation |
| **Graph DB** | Neo4j / NetworkX | Dual backend for flexibility |
| **Vector DB** | Qdrant / FAISS | Fast semantic search |
| **Events** | Kafka | Async, scalable communication |
| **ETL** | Apache Airflow | Production-grade workflow automation |
| **Frontend** | React + TypeScript | Type-safe, modern UI |
| **Styling** | Tailwind CSS | Rapid development with utility classes |
| **Animations** | Framer Motion | Smooth, professional animations |
| **Containers** | Docker | Consistent deployment |
| **CI/CD** | GitHub Actions | Automated testing and deployment |



## The Development Process

Building ResearcherAI took about **3 weeks** of focused work. Here's how I approached it:

**Week 1: Foundation**
- Researched multi-agent patterns and RAG architectures
- Set up basic agent structure with LangGraph
- Integrated first 3 data sources (arXiv, Semantic Scholar, PubMed)
- Got basic knowledge graph working with NetworkX

**Week 2: Production Features**
- Added Neo4j and Qdrant for production deployment
- Implemented Kafka event system
- Built conversation memory and session management
- Added comprehensive error handling and retry logic
- Integrated Apache Airflow for automated collection

**Week 3: Frontend & Testing**
- Built React frontend with glassmorphism design
- Wrote comprehensive test suite (291 tests, 96.60% coverage)
- Set up CI/CD with GitHub Actions
- Docker containerization and orchestration
- Documentation and polish

## What Makes This Different

Most tutorials show you the happy path. I'm going to show you:

- ✅ **Real Challenges**: The bugs I hit and how I fixed them
- ✅ **Design Decisions**: Why I chose certain patterns over others
- ✅ **Trade-offs**: What I compromised on and why
- ✅ **Production Patterns**: Circuit breakers, token budgets, caching
- ✅ **Testing Strategy**: How I achieved 96.60% coverage
- ✅ **Cost Optimization**: Techniques that reduced costs by 40-70%

## Who This Is For

This tutorial is for you if:

- You're building AI/ML applications and want to go beyond prototypes
- You're interested in multi-agent systems and RAG architectures
- You want to learn production-grade patterns for LLM applications
- You're building research tools or knowledge management systems
- You love building things and learning by doing

**Prerequisites**:
- Python experience (intermediate level)
- Basic understanding of APIs and databases
- Familiarity with React is helpful but not required
- Docker knowledge is helpful for deployment

## Ready to Start?

Let's dive in! Click below to start with the planning phase, where I'll walk you through how I went from idea to requirements.


  
  <a href="planning">Next: Planning & Requirements →</a>


---


<p>Built with ❤️ by Adrian Sebuliba</p>
<p>
  <a href="https://github.com/Sebuliba-Adrian/ResearcherAI">View on GitHub</a> |
  <a href="https://github.com/Sebuliba-Adrian/ResearcherAI/issues">Report an Issue</a>
</p>

