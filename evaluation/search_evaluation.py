from clients.vecdb_client import VectorDBClient
import numpy as np
import config
from utils.jsonl_utils import load_jsonl
from schemas.document import SearchSyntheticGroundTruth
from pathlib import Path
import json

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

def run_evaluation(ground_truth: list[SearchSyntheticGroundTruth], search_function: VectorDBClient, top_k:int = 10):
    limit = top_k
    bool_match_list = []
    for entry in ground_truth:
        entry_id = entry.entry_id
        for question in entry.question:
            search = search_function.search(question, limit=limit)
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

def main(ground_truth_path: Path = config.GROUND_TRUTH_DIR / 'search-ground-truth-data.jsonl', top_k:int = 10):
    vecdb_client = VectorDBClient()
    ground_truth = load_jsonl(SearchSyntheticGroundTruth, ground_truth_path)
    result = run_evaluation(ground_truth, vecdb_client, top_k=top_k)
    
    output_path = config.EVAL_DIR / 'search_evaluation_results.jsonl'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()