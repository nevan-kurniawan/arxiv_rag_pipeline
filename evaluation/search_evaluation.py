from clients.vecdb_client import VectorDBClient
import numpy as np
import config.paths as paths
from utils.jsonl_utils import load_jsonl
from schemas.document import SearchSyntheticGroundTruth
from pathlib import Path
import json
from typing import Literal
from tqdm import tqdm


def hit_rate(relevance_matrix: np.ndarray) -> float:
    return np.any(relevance_matrix, axis=1).mean()


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


# def map_at_k(relevance_matrix: np.ndarray) -> float:
#     hits = np.any(relevance_matrix, axis=1)
#     ranks = np.argmax(relevance_matrix, axis=1) + 1
#     map = np.where(hits, 1 / ranks, 0).mean()
#     return map


def run_evaluation(
    ground_truth: list[SearchSyntheticGroundTruth],
    search_function: VectorDBClient,
    mode: Literal["dense", "sparse", "cascade", "rrf"],
    prefetch_limit: int = 30,
    top_k: int = 3,
):
    bool_match_list = []
    for entry in ground_truth:
        entry_id = entry.entry_id
        for question in entry.question:
            search = search_function.search(
                question, mode, prefetch_limit=prefetch_limit, top_k=top_k
            )
            matches = [
                result.payload.get("entry_id") == entry_id for result in search.points
            ]
            matches += [False] * (top_k - len(matches))
            bool_match_list.append(matches)

    arr = np.array(bool_match_list)
    return {
        "hit_rate": hit_rate(arr),
        "mrr": mrr(arr),
        "ndcg": ndcg_at_k(arr),
        # "map": map_at_k(arr),
        "mode": mode,
        "prefetch_limit": prefetch_limit,
        "top_k": top_k,
    }


def main(
    prefetch_limit: int,
    top_k: int,
    output_path: Path,
    client: VectorDBClient,
    mode: Literal["dense", "sparse", "cascade", "rrf"],
    ground_truth: list[SearchSyntheticGroundTruth],
):
    vecdb_client = client
    ground_truth = ground_truth
    result = run_evaluation(
        ground_truth,
        vecdb_client,
        mode=mode,
        prefetch_limit=prefetch_limit,
        top_k=top_k,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    output_path = paths.RESULTS_DIR / "search_evaluation_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modes: list[Literal["dense", "sparse", "cascade", "rrf"]] = [
        "dense",
        "sparse",
        "cascade",
        "rrf",
    ]
    k_values = [1, 3, 5, 10, 20]

    client = VectorDBClient()

    completed = set()

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed.add((record["mode"], record["top_k"]))
                    except json.JSONDecodeError:
                        continue

    ground_truth = load_jsonl(
        SearchSyntheticGroundTruth,
        paths.GROUND_TRUTH_DIR / "search-ground-truth-data.jsonl",
    )

    for mode in tqdm(modes):
        for k in tqdm(k_values):
            if (mode, k) not in completed:
                print(f"Running mode={mode}, top_k={k}...")
                main(
                    prefetch_limit=k * 10,
                    top_k=k,
                    output_path=output_path,
                    mode=mode,
                    client=client,
                    ground_truth=ground_truth,
                )
