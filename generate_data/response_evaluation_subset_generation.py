import pandas as pd
import config

def main():
    df = pd.read_csv(config.GROUND_TRUTH_DIR / 'ground-truth-data.csv')

    unique_papers_df = df.groupby('entry_id').sample(n=1, random_state=42)

    if len(unique_papers_df) < 100:
        raise ValueError(f"Insufficient unique papers: found {len(unique_papers_df)}, require at least 100.")

    evaluation_subset = unique_papers_df.sample(n=100, random_state=42)

    evaluation_subset.to_csv(config.GROUND_TRUTH_DIR / 'response-evaluation-subset.csv', index=False)

    print(f"Subset generated successfully. Total unique papers: {len(evaluation_subset)}")

if __name__ == "__main__":
    main()