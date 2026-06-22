import pandas as pd
import config.paths as paths
from utils.jsonl_utils import load_jsonl
from schemas.document import SearchSyntheticGroundTruth


def main():
    """Subsamples 100 out of the 400 evaluation subset for llm-as-a-judge experiment"""
    data = load_jsonl(
        SearchSyntheticGroundTruth,
        paths.GROUND_TRUTH_DIR / "search-ground-truth-data.jsonl",
    )
    df = pd.DataFrame([item.model_dump(mode="json") for item in data])
    unique_papers_df = df.groupby("entry_id").sample(n=1, random_state=1)
    unique_papers_df["question"] = unique_papers_df["question"].str[0:1]
    if len(unique_papers_df) < 100:
        raise ValueError(
            f"Insufficient unique papers: found {len(unique_papers_df)}, require at least 100."
        )

    evaluation_subset = unique_papers_df.sample(n=100, random_state=1)

    evaluation_subset.to_json(
        paths.GROUND_TRUTH_DIR / "response-evaluation-subset.jsonl",
        orient="records",
        lines=True,
    )

    print(
        f"Subset generated successfully. Total unique papers: {len(evaluation_subset)}"
    )


if __name__ == "__main__":
    main()
