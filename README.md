# Domain_aware_RAG

A domain-aware Retrieval-Augmented Generation (RAG) prototype for knowledge-intensive documents. This repository combines ontology-driven query understanding, hybrid retrieval, hierarchical document structure, and grounded answer generation in a Streamlit-based interface.

## Overview

This project is designed to support domain-specific policies and documents by using:

- Domain ontologies for query expansion, answer behavior, and retrieval hints
- A hybrid search stack that blends semantic embeddings with text-based matching
- Hierarchical document ingestion to preserve parent/child context
- Grounded answer generation that returns responses only from retrieved evidence

## Architecture

The repository is organized around the following core components:

- **Streamlit UI** (`domain_aware_rag/merged_ui.py`)
  - Provides domain selection, search chat, ontology manager, and PDF/document upload.
  - Sends user input through ontology-aware retrieval and answer generation.

- **Ontology manager and loader** (`domain_aware_rag/ontology/registry.py`, `domain_aware_rag/ontology/loader.py`)
  - Loads JSON-based domain ontologies from `ontologies/`.
  - Persists active domain state and exposes domain-specific prompts and retrieval hints.

- **Query expansion and validation** (`domain_aware_rag/retrieval/query_selector.py`, `domain_aware_rag/retrieval/query_validation.py`)
  - Builds ontology-aware search variants using anchor taxonomy and example queries.
  - Validates whether user queries are on-topic before retrieval.

- **Hybrid retrieval** (`domain_aware_rag/retrieval/hybrid_retrieval.py`)
  - Combines semantic embeddings, BM25-style text search, and metadata filtering.
  - Ranks passages using entity-aware weighting and domain hints.

- **Hierarchical ingestion and context storage** (`domain_aware_rag/ingestion/ingest_optimized.py`)
  - Stores content as structured nodes with parent/child relationships.
  - Retrieves leaf nodes plus parent context to preserve document structure.

- **Answer generation** (`domain_aware_rag/generation/answer_generator.py`)
  - Synthesizes answers from retrieved evidence and domain-specific prompts.
  - Grounds responses in the selected ontology and returned text.

- **Ingestion orchestrator** (`domain_aware_rag/ingestion/multi_doc_orchestrator.py`)
  - Extracts text and metadata from uploaded documents.
  - Applies NER and hierarchy extraction before ingesting content into the database.

## How this repo is better than generic RAG

This project is not just "query → vector search → answer." It adds:

- Domain-specific query normalization and ontology-based term selection
- Hybrid retrieval to capture both semantic meaning and exact policy matches
- Entity-aware and hierarchy-aware retrieval for better context
- Grounded generation with domain prompts to reduce hallucination

## Running

1. Initialize the database if required:

```powershell
python domain_aware_rag/ingestion/init_db_fixed.py
```

2. Start the Streamlit app:

```powershell
python -m streamlit run domain_aware_rag/merged_ui.py
```

## Ontologies

Domain ontologies are stored in `ontologies/` and are loaded by `domain_aware_rag/ontology/loader.py`.
Each ontology can include:

- `query_expansion`
- `answer_generation`
- `retrieval_hints`
- `entity_catalogue_path`
- `anchor_taxonomy`
- `example_queries`

## Notes

- The active domain selection is persisted by the ontology registry.
- The system supports optional `pgvector` for embeddings if your database has the extension.
- The repository focuses on `Domain_aware_RAG` as the active deliverable.
