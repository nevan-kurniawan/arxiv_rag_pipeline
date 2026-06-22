import json

import matplotlib.pyplot as plt
import pandas as pd

import config.paths as paths

# Retrieval sweep analysis

sweep_path = paths.RESULTS_DIR / "search_evaluation_results.jsonl"
rows = [
    json.loads(line)
    for line in sweep_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(rows) == 20, f"Expected 20 sweep rows, got {len(rows)}"

df = pd.DataFrame(rows)

for metric in ("hit_rate", "mrr", "ndcg"):
    pivot = df.pivot(index="mode", columns="top_k", values=metric)
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.loc[["dense", "sparse", "cascade", "rrf"]]
    print(f"\n{'=' * 50}")
    print(f"  {metric.upper()}")
    print(f"{'=' * 50}")
    print(pivot.round(4).to_string())

# Line charts — one per metric, modes as lines, k on x-axis
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
mode_order = ["dense", "sparse", "cascade", "rrf"]
colors = {
    "dense": "#1f77b4",
    "sparse": "#ff7f0e",
    "cascade": "#2ca02c",
    "rrf": "#d62728",
}

for ax, metric in zip(axes, ("hit_rate", "mrr", "ndcg")):
    for mode in mode_order:
        subset = df[df["mode"] == mode].sort_values("top_k")
        ax.plot(
            subset["top_k"], subset[metric], marker="o", label=mode, color=colors[mode]
        )
    ax.set_title(metric.upper(), fontsize=14)
    ax.set_xlabel("top_k")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle("Retrieval Mode Comparison Across Top-K", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(paths.RESULTS_DIR / "retrieval_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved retrieval chart to {paths.EVAL_DIR / 'retrieval_sweep.png'}")

# llm-as-a-judge analysis

judge_path = paths.RESULTS_DIR / "response_evaluation_results.jsonl"
if judge_path.exists():
    judge_rows = [
        json.loads(line)
        for line in judge_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jdf = pd.DataFrame(judge_rows)

    print(f"\n{'=' * 50}")
    print("  RESPONSE EVALUATION (LLM-as-Judge)")
    print(f"{'=' * 50}")
    print(f"Judged records: {len(jdf)}")

    for metric in ("faithfulness", "relevance"):
        if metric in jdf.columns:
            dist = jdf[metric].value_counts().sort_index()
            mean = jdf[metric].mean()
            print(f"\n{metric.upper()}")
            print(f"  distribution: {dist.to_dict()}")
            print(f"  mean: {mean:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("faithfulness", "relevance")):
        if metric in jdf.columns:
            counts = jdf[metric].value_counts().reindex([1, 2, 3], fill_value=0)
            counts.plot.bar(
                ax=ax, color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="black"
            )
            ax.set_title(f"{metric.capitalize()} Score Distribution", fontsize=13)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_xticklabels(["1 (Low)", "2 (Medium)", "3 (High)"], rotation=0)
    fig.tight_layout()
    fig.savefig(
        paths.RESULTS_DIR / "judge_distributions.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"\nSaved judge chart to {paths.EVAL_DIR / 'judge_distributions.png'}")
else:
    print(f"\nNo judge results at {judge_path} — skipping response eval analysis.")
