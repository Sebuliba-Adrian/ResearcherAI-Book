# ResearcherAI Book Diagrams

This directory contains all architectural diagrams for the ResearcherAI book.

## Diagram Sources

All diagrams are created using Excalidraw for a professional, hand-drawn aesthetic perfect for publication.

### Editing Diagrams

1. Go to https://excalidraw.com
2. Open the corresponding `.excalidraw` file
3. Edit as needed
4. Export as SVG (File > Export image > SVG)
5. Save to this directory with same name
6. Commit both .excalidraw and .svg files

## Diagram List

1. **multi-agent-overview.svg** - Complete system architecture with 6 agents
2. **langgraph-workflow.svg** - LangGraph 8-node workflow with conditional routing
3. **llamaindex-pipeline.svg** - LlamaIndex RAG dual-pipeline architecture
4. **framework-integration.svg** - LangGraph + LlamaIndex integration layers
5. **multimodal-architecture.svg** - Multi-modal RAG system architecture

## Usage in Documentation

Diagrams are referenced in markdown files using relative paths:

```markdown
![Multi-Agent Architecture](/img/diagrams/multi-agent-overview.svg)
```
