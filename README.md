---
title: Arxiv-rag
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Arxiv Research Assistant

ArXiv Research Assistant is an end-to-end RAG pipeline that synthesizes recent deep learning research from ArXiv into conversational answers. It uses hybrid dense and sparse retrieval over a vector database, with a full evaluation harness measuring retrieval quality and answer faithfulness + relevance.

## 1. Architecture Overview

- **Data Source:** Arxiv API
- **Ingestion:** Papers' titles, categories, authors, abstracts, published dates, and entry IDs.
- **Vector Database (Qdrant):** Hybrid dense + sparse retrieval.
- **RAG Pipeline:** Top-k retrieval feeds an LLM via the OpenAI-compatible Groq endpoint.
- **Serving (Streamlit):** User queries answered with retrieved papers as context.

## 2. Tech Stack

- **Qdrant** — vector store with native hybrid search support
- **fastembed** — client-side embedding for pipeline visibility
- **Groq** — fast inference for LLM calls
- **Streamlit** — UI

## 3. How to Run Locally

```
git clone ...
cd arxiv-rag
cp .env.example .env  # fill in your API keys
docker compose up --build
```

## 4. Project Structure

```
pipeline/       # ingestion and RAG query logic
evaluation/     # search metrics and LLM-as-judge harness
clients/        # VectorDB and LLM client wrappers
schemas/        # Pydantic data models
generate_data/  # ground truth generation scripts
```

## Known Limitations and Planned Improvements

- Ground truth uses single-document relevance labels. Graded relevance would improve MAP and NDCG fidelity.
- Ablation study (dense vs sparse vs hybrid, top-k sweep, reranking, judge consistency) is in progress.
```