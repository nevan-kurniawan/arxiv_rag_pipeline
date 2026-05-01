from clients.vecdb_client import VectorDBClient
import pandas as pd
import numpy as np
import config

def hit_rate(relevance_matrix: np.ndarray) -> float:
    return np.any(relevance_matrix, axis = 1).mean()

def mrr(relevance_matrix: np.ndarray) -> float:
    hits = np.any(relevance_matrix, axis=1)
    ranks = np.argmax(relevance_matrix, axis=1).astype(float) + 1
    return np.sum((1 / ranks) * hits) / len(relevance_matrix)

def ndcg_at_k(relevance_matrix: np.ndarray) -> float:
    hits = np.any(relevance_matrix, axis=1)
    ranks = np.argmax(relevance_matrix, axis=1) + 1
    dcg = np.where(hits, 1 / np.log2(ranks + 1), 0)
    idcg = 1 / np.log2(2)
    return (dcg / idcg).mean()

def map_at_k(relevance_matrix: np.ndarray) -> float:
    hits = np.any(relevance_matrix, axis=1)
    ranks = np.argmax(relevance_matrix, axis=1) + 1
    map = np.where(hits, 1 / ranks, 0).mean()
    return map

def run_evaluation(ground_truth: pd.DataFrame, search_function: VectorDBClient, top_k:int = 10):
    ground_truth_dict = ground_truth.to_dict(orient='records')
    limit = top_k
    bool_match_list = []
    for entry in ground_truth_dict:
        entry_question = entry['question']
        entry_id = entry['entry_id']
        search = search_function.search(entry_question, limit = limit)
        matches = [result.payload.get('entry_id') == entry_id for result in search.points]
        matches += [False] * (limit - len(matches))
        bool_match_list.append(matches)

    arr = np.array(bool_match_list)
    return {
        f'hit_rate_k_{top_k}': hit_rate(arr),
        f'mrr_k_{top_k}': mrr(arr),
        f'ndcg_k_{top_k}': ndcg_at_k(arr),
        f'map_k_{top_k}': map_at_k(arr)
    }

def main(ground_truth_path:str = str(config.GROUND_TRUTH_DIR / 'ground-truth-data.csv'), top_k:int = 10):
    vecdb_client = VectorDBClient()
    ground_truth = pd.read_csv(ground_truth_path, dtype={'entry_id': str})
    result = run_evaluation(ground_truth, vecdb_client, top_k=top_k)
    
    output_df = pd.DataFrame([result])
    
    output_df.to_json(
        config.EVAL_DIR / 'search_evaluation_results.json',
        orient='records', 
        indent=4
    )

if __name__ == "__main__":
    main()