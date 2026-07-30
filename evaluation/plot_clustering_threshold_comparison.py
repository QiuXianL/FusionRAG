import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    thresholds = ["阈值 0.85（严格）", "阈值 0.90（适度）"]

    total_questions = np.array([6572, 6572], dtype=float)
    kept = np.array([2648, 3782], dtype=float)
    removed = np.array([3924, 2790], dtype=float)
    reduction_rate = np.array([0.597, 0.425], dtype=float)

    x = np.arange(len(thresholds))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax0 = axes[0]
    bar_width = 0.42
    ax0.bar(x, kept, width=bar_width, color="#4C78A8", edgecolor="white", linewidth=0.8, label="保留问题")
    ax0.bar(x, removed, width=bar_width, bottom=kept, color="#E45756", edgecolor="white", linewidth=0.8, label="移除问题")
    ax0.set_title("问题数量组成（去重前后）")
    ax0.set_xticks(x)
    ax0.set_xticklabels(thresholds)
    ax0.set_ylabel("问题数量")
    ax0.set_ylim(0, max(total_questions) * 1.12)
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(frameon=False, loc="upper right")

    for i in range(len(thresholds)):
        ax0.text(x[i], kept[i] / 2, f"{int(kept[i])}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax0.text(
            x[i],
            kept[i] + removed[i] / 2,
            f"{int(removed[i])}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
        ax0.text(x[i], total_questions[i] + max(total_questions) * 0.02, f"总计 {int(total_questions[i])}", ha="center", va="bottom", fontsize=9)

    ax1 = axes[1]
    ax1.bar(x, reduction_rate, width=0.32, color="#54A24B", edgecolor="white", linewidth=0.8)
    ax1.set_title("整体缩减率（去重强度）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(thresholds)
    ax1.set_ylabel("缩减率")
    min_v = float(reduction_rate.min())
    max_v = float(reduction_rate.max())
    margin = max(0.02, (max_v - min_v) * 2.0)
    ax1.set_ylim(max(0.0, min_v - margin), min(1.0, max_v + margin))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i, v in enumerate(reduction_rate):
        ax1.text(i, v + margin * 0.12, f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("语义聚类去重阈值对比（0.85 vs 0.90）", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "reports", "clustering_threshold_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

