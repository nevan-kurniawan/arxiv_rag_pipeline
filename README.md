# Arxiv Research assistant
ArXiv Research Assistant is an end-to-end RAG pipeline that synthesizes recent deep learning research from ArXiv into conversational answers. It uses hybrid dense and sparse retrieval over a daily-updated vector database, with a full evaluation harness measuring retrieval quality and answer faithfulness  + relevance.

## 1. Architecture Overview
- Data Source: Arxiv API
- Ingestion: Ingestion from the Arxiv API of the papers' titles, categories, authors, abstracts, published dates, and entry IDs.
- Vector Database (Qdrant): Using Qdrant as the vector database.
- RAG Pipeline: Retrieval of three most relevant papers to the query.
- Serving (Streamlit): Users can query the LLM, with the RAG pipeline providing relevant papers as context for the LLM.

## 2. Tech Stack
- Qdrant — vector store with native hybrid search support
- fastembed — client-side embedding for pipeline visibility
- Groq — fast inference for LLM calls
- Prefect — lightweight pipeline orchestration for daily ingestion
- Streamlit — rapid UI for demonstration

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
- Ground truth uses single-document relevance labels. Planned to add graded relevance, which would improve MAP and NDCG accuracy.
- RAGAS production monitoring planned for v2.

<!-- 1. Ingestion from Arxiv (pipeline/ingestion.py)
    - Schema:
            - schemas/document.py: ArxivDocument:
            - 'title':str
            - 'categories': list[str]
            - 'authors': list[str]
            - 'summary': list[str]
            - 'entry_id': str -> primary key
            - 'published': str
    - Outputs 'data/raw_data/arxiv_data_cache.jsonl'

2. VectorDB ingestion (vecdb_client.py, VectorDBClient)
    - Hybrid search
    - Schema:
        - vectors:
            - 'dense': 'jinaai/jina-embeddings-v2-small-en' embed 'summary'
            - 'sparse': 'Qdrant/bm25' embed 'summary'
            - payload={
                - 'title':str
                - 'categories': list[str]
                - 'authors': list[str]
                - 'summary': str
                - 'entry_id': str -> primary key
                - 'published': str (isoformat)
                - 'id': int
            }

3. Generate search ground truth (generate_data/search_ground_truth.py)
    - Takes 'data/raw_data/arxiv_data_cache.json'
    - Builds prompt from each entry
    - Sends to LLM to receive 5 relevant questions per entry
    - Schema:
        - 'question': str
        - 'entry_id': str -> primary key
        - 'generated_at': str
    - Outputs 'data/ground_truth/ground-truth-data.csv' -> total 500 questions

4. Make subset of search ground truth for evaluation, 1 per entry  (generate_data\response_evaluation_subset_generation.py)
    - Takes 'data/ground_truth/ground-truth-data.csv'
    - Schema:
        - 'question': str
        - 'entry_id': str -> primary key
        - 'generated_at': str
    - Outputs 'data/ground_truth/response-evaluation-subset.csv' -> 100 total questions

5. Search evaluation (evaluation/search_evaluation.py)
    - Takes 'data/ground_truth/ground-truth-data.csv'
    - Calculates hit rate at k, mrr at k, ndcg at k, and map at k.
    - Schema:
        - hit_rate_k_{top_k}: float,
        - mrr_k_{top_k}: float,
        - ndcg_k_{top_k}: float,
        - map_k_{top_k}: float
    - Outputs 'data/evaluation/search_evaluation_results.json' -->