# Neo4j Vectorless RAG

This directory is an enhanced, production-ready adaptation of the [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) vectorless RAG architecture. Instead of relying solely on holding the hierarchical JSON tree in memory during inference, this version acts as an **Agentic Retriever** over a **Neo4j Graph Database**.

## Why this is better than the original method:
1. **Persistent Memory & Scalability**: The original method generates a single flat JSON for a document. Loading hundreds of these JSONs for cross-document query reasoning would overwhelm LLM context windows and crash traditional workflows. By persisting the document trees into a Neo4j Graph, we can load millions of documents. The LLM then just starts at the roots and navigates down to find relevant leaf nodes across the *entire* database, seamlessly.
2. **Graph Traversal & Relationships**: If you want to connect sections across different documents (e.g. cross-referencing citations), you simply create a Neo4j `[:REFERENCES]` edge. The LLM can traverse these edges, giving you a true "reasoning" knowledge graph that basic JSON tree search cannot replicate.
3. **Stand-alone Execution**: Everything you need to generate the PageIndex trees from PDFs/Markdown **and** ingest them into Neo4j is co-located in this directory and governed by `uv` for lightning-fast package management.

## Graph Representation

<img width="1360" height="682" alt="image" src="https://github.com/user-attachments/assets/9e1b9fb9-cbf3-4827-afb1-850f21048195" />

The graph above demonstrates how the hierarchical structure of a PDF (e.g., `earthmover.pdf`) is persisted in Neo4j. 
- The central **Document node** connects to all top-level chapters (e.g., *PRELIMINARIES*, *INTRODUCTION*) via `[:HAS_SECTION]` relationships.
- Those topics recursively connect to their sub-sections via `[:HAS_SUBSECTION]` relationships.
- During retrieval, the LLM starts at the center and explicitly chooses which paths to walk down based on the user's query, ignoring irrelevant branches and saving massive amounts of context tokens.
## Setup
We strictly use `uv` for python package management as per local best practices.

1. **Install Requirements via uv**
   ```bash
   uv sync
   ```

2. **Environment Variables**
   Create a `.env` file in this directory with your credentials:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   GROQ_API_KEY=your_groq_api_key
   ```
   *(Defaults to `groq/llama-3.3-70b-versatile` through LiteLLM).*

## Usage

### 1. Document Parsing (Original PageIndex)
First, generate the `_structure.json` file for your PDF/Markdown.
```bash
uv run python main.py --pdf_path /path/to/my_document.pdf
```
*(The JSON will be stored in `./results/`)*

### 2. Neo4j Ingestion
Ingest the generated structure into your graph database:
```bash
uv run python -m src.database.ingest --json_path ./results/my_document_structure.json
```

### 3. Agentic Graph Retrieval
Ask a natural language question against your running Neo4j database:
```bash
uv run python -m src.agent.retriever --doc_name my_document_structure.json --query "What is the main topic of section 2?"
```

The script will ask the LLM to identify relevant root nodes, then iteratively query Neo4j for their children, drilling down until it finds the specific text or pages that contain the answer.
