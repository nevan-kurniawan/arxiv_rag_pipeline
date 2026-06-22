import random
import config.paths as paths
from schemas.document import ArxivDocument
from utils.jsonl_utils import load_jsonl


def main():
    """Subsamples 400 out of the 1000 Arxiv documents for evaluation"""
    data = load_jsonl(ArxivDocument, paths.RAW_DATA / "arxiv_data_cache.jsonl")
    if len(data) < 400:
        raise ValueError(f"Corpus has {len(data)} docs, fewer than 400.")

    rng = random.Random(1)
    sampled = rng.sample(data, 400)

    sample_path = paths.GROUND_TRUTH_DIR / "corpus_raw_subsample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for doc in sampled:
            f.write(doc.model_dump_json() + "\n")
    print(f"Wrote 400 sampled documents to {sample_path}")


if __name__ == "__main__":
    main()
