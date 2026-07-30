import os
import re

import matplotlib.pyplot as plt
import numpy as np


def extract_mrr_from_report(report_text: str) -> tuple[float, float]:
    match = re.search(r"\|\s*\*\*MRR\*\*\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*\*\*([0-9]*\.?[0-9]+)\*\*\s*\|", report_text)
    if not match:
        raise ValueError("Could not find MRR row in report markdown.")
    return float(match.group(1)), float(match.group(2))


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    report_path = os.path.join(project_root, "reports", "hotpot_qa_visualized_report.md")
    image_path = os.path.join(project_root, "reports", "images_hotpot", "mrr_comparison.png")

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    baseline_mrr, fusion_mrr = extract_mrr_from_report(report_text)

    labels = ["Baseline", "FusionRAG"]
    values = [baseline_mrr, fusion_mrr]

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(x, values, width=width, color=["skyblue", "salmon"])

    ax.set_ylabel("Score")
    ax.set_title("Mean Reciprocal Rank (MRR)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    min_v = min(values)
    max_v = max(values)
    margin = max(0.01, (max_v - min_v) * 2.5)
    y_min = max(0.0, min_v - margin)
    y_max = min(1.0, max_v + margin)
    ax.set_ylim(y_min, y_max)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()

    print(f"Saved: {image_path}")
    print(f"Baseline MRR: {baseline_mrr:.4f}")
    print(f"FusionRAG MRR: {fusion_mrr:.4f}")


if __name__ == "__main__":
    main()
