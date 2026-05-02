1. Ingestion from Arxiv (pipeline/ingestion.py)
    - Schema:
        - 'title':str
        - 'categories': list[str]
        - 'authors': list[str]
        - 'summary': list[str]
        - 'entry_id': str -> primary key
        - 'published': str (isoformat)
    - Outputs 'data/raw_data/arxiv_data_cache.json'

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
                - 'summary': list[str]
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

4. Make subset of search ground truth for evaluation, 1 per entry
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
    - Outputs 'data/evaluation/search_evaluation_results.json'