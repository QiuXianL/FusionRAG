import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    data = {
        "Normal RAG": {"Recall@1": 0.74, "Recall@5": 0.74, "Recall@10": 0.74, "MRR": 0.740},
        "Base Model": {"Recall@1": 0.76, "Recall@5": 0.88, "Recall@10": 0.88, "MRR": 0.794},
        "Finetuned Model": {"Recall@1": 0.78, "Recall@5": 0.84, "Recall@10": 0.90, "MRR": 0.811},
    }

    methods = list(data.keys())
    metrics = ["Recall@1", "Recall@5", "Recall@10", "MRR"]
    values = np.array([[data[m][k] for k in metrics] for m in methods], dtype=float)

    x = np.arange(len(metrics))
    bar_width = 0.16
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * (bar_width * 1.45)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    for i, method in enumerate(methods):
        bars = ax.bar(
            x + offsets[i],
            values[i],
            width=bar_width,
            label=method,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    min_v = float(values.min())
    max_v = float(values.max())
    margin = max(0.02, (max_v - min_v) * 2.2)
    y_min = max(0.0, min_v - margin)
    y_max = min(1.0, max_v + margin)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("QGen Model Comparison (Retrieval Metrics)")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper left")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "reports", "finetuned_model_metrics.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
