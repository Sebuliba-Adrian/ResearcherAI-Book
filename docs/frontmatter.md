---
layout: default
title: Front Matter
---



# Building Production-Grade Multi-Agent RAG Systems

## A Comprehensive Guide to ResearcherAI

### From Concept to Deployment


<img src="assets/images/cover.png" alt="Book Cover" width="400" />


**Author:** Adrian Sebuliba

**Version:** 2.0

**Publication Date:** November 2025

---

## About This Book

This book chronicles the complete journey of building **ResearcherAI**, a production-ready multi-agent system for research paper analysis. Unlike typical tutorials that show only the happy path, this book shares the real challenges, design decisions, trade-offs, and lessons learned from building a system that achieves:

- 96.60% test coverage
- Dual-backend architecture (development & production)
- Event-driven design with Apache Kafka
- 7 data sources integrated
- Modern React frontend with glassmorphism design
- Complete CI/CD pipeline
- Production-grade patterns (circuit breakers, token budgets, intelligent caching)

Whether you're building AI applications, multi-agent systems, or simply want to learn production engineering practices, this book provides practical, battle-tested insights you can apply immediately.

---

## Who This Book Is For

**Primary Audience:**
- Software engineers building LLM-powered applications
- ML engineers implementing RAG systems
- System architects designing multi-agent platforms
- DevOps engineers deploying AI services

**Prerequisites:**
- Intermediate Python programming
- Basic understanding of APIs and databases
- Familiarity with Docker (helpful but not required)
- General software engineering principles

**What You'll Learn:**
- Multi-agent architecture patterns
- RAG (Retrieval-Augmented Generation) implementation
- Knowledge graphs with Neo4j and NetworkX
- Vector databases with Qdrant and FAISS
- Event-driven architecture with Kafka
- Apache Airflow for ETL orchestration
- React frontend development
- Testing strategies for AI systems
- Production deployment with Docker
- Cost optimization techniques (40-70% reduction)
- Monitoring and observability

---

## How to Use This Book

**Linear Reading:** The chapters build on each other and are best read in sequence, especially if you're new to multi-agent systems.

**Reference Guide:** Each chapter is self-contained enough to serve as a reference for specific topics (testing, deployment, monitoring).

**Hands-On Learning:** All code is available on GitHub. Follow along by building the system yourself.

**Academic Study:** Extensive references are provided for deeper exploration of concepts.

---

## Code Repository

All source code, configuration files, and documentation are available at:

**GitHub:** https://github.com/Sebuliba-Adrian/ResearcherAI

**License:** MIT License (see repository for details)

---

## Conventions Used in This Book

**Code Blocks:**
```python
# Python code examples look like this
def example_function():
    pass
```

**Terminal Commands:**
```bash
$ command to execute
output from command
```

**Important Notes:**
> Important concepts and warnings are highlighted in blockquotes.

**File Paths:**
- `relative/path/to/file.py`
- `/absolute/path/to/file.py`

**References:**
Academic papers and technical resources are cited inline with footnotes and compiled in the bibliography at the end of each chapter.

---

## Acknowledgments

This project stands on the shoulders of giants. Special thanks to:

- The **LangChain** and **LangGraph** teams for pioneering multi-agent frameworks
- **LlamaIndex** contributors for excellent RAG tools
- **Google** for the Gemini API
- **Neo4j**, **Qdrant**, **Apache Kafka**, and **Apache Airflow** communities
- The open-source community for countless libraries and tools

Technical reviewers and beta readers provided invaluable feedback that shaped this book.

---

## About the Author

**Adrian Sebuliba** is a software engineer specializing in AI/ML systems and production infrastructure. With experience building scalable systems, he focuses on practical, production-grade patterns that balance innovation with reliability.

Connect:
- GitHub: @Sebuliba-Adrian
- Email: sebuliba.adrian@gmail.com

---

## Feedback and Corrections

Despite careful review, errors may remain. If you find technical inaccuracies, typos, or have suggestions for improvement:

- Open an issue on GitHub: https://github.com/Sebuliba-Adrian/ResearcherAI/issues
- Submit a pull request with corrections
- Email directly with feedback

Your input helps improve this resource for the community.

---

## Copyright and License

**Copyright © 2025 Adrian Sebuliba**

The content of this book is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License** (CC BY-NC-SA 4.0).

The accompanying source code is licensed under the **MIT License**.

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit
- **NonCommercial** — Not for commercial use without permission
- **ShareAlike** — Distribute derivatives under the same license

For commercial licensing inquiries, contact the author.

---





## Table of Contents

**Front Matter**
- About This Book
- Who This Book Is For
- How to Use This Book
- Acknowledgments

**[Introduction: The Problem and the Vision](/)**

**[Chapter 1: Planning & Requirements](planning)**
- 1.1 The Problem
- 1.2 The Vision
- 1.3 Core Requirements
- 1.4 Technology Selection
- 1.5 Architecture Philosophy
- 1.6 The Development Plan
- 1.7 Lessons from Planning
- References

**[Chapter 2: Architecture Design](architecture)**
- 2.1 System Overview
- 2.2 Multi-Agent Architecture
- 2.3 Dual-Backend Strategy
- 2.4 Event-Driven Architecture
- 2.5 RAG Pipeline Design
- 2.6 Session Management
- 2.7 Production-Grade Patterns
- 2.8 Airflow Integration
- 2.9 Architecture Evolution
- References

**[Chapter 3: Backend Development](backend)**
- 3.1 Project Structure
- 3.2 Data Collector Agent
- 3.3 Knowledge Graph Agent
- 3.4 Vector Agent
- 3.5 Reasoning Agent
- 3.6 Orchestration
- 3.7 Lessons Learned
- References

**[Chapter 4: Frontend Development](frontend)**
- 4.1 Design Vision
- 4.2 Technology Stack
- 4.3 Glassmorphism Design System
- 4.4 Page Components
- 4.5 API Integration
- 4.6 TypeScript Types
- 4.7 Lessons Learned
- References

**[Chapter 5: Testing Strategy](testing)**
- 5.1 Testing Philosophy
- 5.2 Test Structure
- 5.3 Unit Testing
- 5.4 Integration Testing
- 5.5 Coverage Analysis
- 5.6 CI/CD Integration
- 5.7 Performance Testing
- 5.8 Lessons Learned
- References

**[Chapter 6: Deployment & CI/CD](deployment)**
- 6.1 Deployment Options
- 6.2 Docker Containerization
- 6.3 GitHub Actions Pipelines
- 6.4 Configuration Management
- 6.5 Health Checks
- 6.6 Scaling Considerations
- 6.7 Lessons Learned
- References

**[Chapter 7: Monitoring & Operations](monitoring)**
- 7.1 Observability Pillars
- 7.2 Apache Airflow Monitoring
- 7.3 Kafka Event Monitoring
- 7.4 Database Dashboards
- 7.5 Application Metrics
- 7.6 Alerting
- 7.7 Cost Monitoring
- 7.8 Backup and Recovery
- 7.9 Lessons Learned
- References

**[Chapter 8: Conclusion & Future Work](conclusion)**
- 8.1 What We Built
- 8.2 Key Takeaways
- 8.3 Future Enhancements
- 8.4 Final Thoughts

**[Bibliography](bibliography)**

---


<p><em>Ready to begin? Turn to the <a href="/">Introduction</a></em></p>

