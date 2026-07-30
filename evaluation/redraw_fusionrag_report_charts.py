import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class SectionTable:
    section_name: str
    headers: List[str]
    rows: List[List[str]]


def _strip_pipe_cells(line: str) -> List[str]:
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return parts


def _is_alignment_row(cells: List[str]) -> bool:
    if not cells:
        return False
    for c in cells:
        c = c.replace(":", "").replace("-", "").strip()
        if c != "":
            return False
    return True


def parse_markdown_table(lines: List[str], start_index: int) -> Tuple[Optional[SectionTable], int]:
    i = start_index
    while i < len(lines) and not lines[i].lstrip().startswith("|"):
        if lines[i].strip() == "":
            return None, i + 1
        i += 1

    if i >= len(lines) or not lines[i].lstrip().startswith("|"):
        return None, i

    header_cells = _strip_pipe_cells(lines[i])
    i += 1

    if i >= len(lines):
        return None, i

    align_cells = _strip_pipe_cells(lines[i])
    if _is_alignment_row(align_cells):
        i += 1

    rows: List[List[str]] = []
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith("|"):
            break
        row_cells = _strip_pipe_cells(line)
        if row_cells:
            rows.append(row_cells)
        i += 1

    return SectionTable(section_name="", headers=header_cells, rows=rows), i


def _find_section_index(lines: List[str], pattern: str) -> Optional[int]:
    rx = re.compile(pattern)
    for idx, line in enumerate(lines):
        if rx.search(line):
            return idx
    return None


def _table_to_metrics(table: SectionTable) -> Dict[str, Dict[str, float]]:
    headers = table.headers
    if not headers:
        return {}

    name_col_idx = 0
    metric_cols = [h for h in headers if h in ("Recall@1", "Recall@5", "Recall@10", "MRR")]
    metric_indices = [headers.index(h) for h in metric_cols]

    out: Dict[str, Dict[str, float]] = {}
    for r in table.rows:
        if not r:
            continue
        name = r[name_col_idx].strip()
        out[name] = {}
        for h, j in zip(metric_cols, metric_indices):
            if j >= len(r):
                continue
            try:
                out[name][h] = float(r[j])
            except Exception:
                out[name][h] = 0.0
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_recall(metrics: Dict[str, Dict[str, float]], title: str, out_path: str) -> None:
    methods = list(metrics.keys())
    ks = ["Recall@1", "Recall@5", "Recall@10"]
    ks = [k for k in ks if any(k in metrics[m] for m in methods)]
    if not ks:
        return

    values = np.array([[metrics[m].get(k, 0.0) for k in ks] for m in methods], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(ks))
    bar_width = 0.18
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * (bar_width * 1.35)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for i, m in enumerate(methods):
        ax.bar(
            x + offsets[i],
            values[i],
            width=bar_width,
            label=m,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.8,
        )
        for xi, yi in zip(x + offsets[i], values[i]):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_mrr(metrics: Dict[str, Dict[str, float]], title: str, out_path: str) -> None:
    methods = list(metrics.keys())
    vals = np.array([metrics[m].get("MRR", 0.0) for m in methods], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    ax.bar(methods, vals, width=0.35, color=[colors[i % len(colors)] for i in range(len(methods))], edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MRR")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    report_path = os.getenv("REPORT_PATH", "reports/FusionRAG_vs_NormalRAG_100_report.md")
    assets_dir = os.path.splitext(report_path)[0] + "_assets"
    _ensure_dir(assets_dir)

    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = {
        "overall": r"^###\s*3\.1\s*Overall",
        "hotpot": r"^##\s*3\.2\s*HotpotQA",
        "squad": r"^##\s*3\.3\s*SQuAD",
    }

    extracted: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, pat in sections.items():
        idx = _find_section_index(lines, pat)
        if idx is None:
            continue
        table, _ = parse_markdown_table(lines, idx)
        if not table:
            continue
        extracted[key] = _table_to_metrics(table)

    if not extracted:
        raise RuntimeError(f"Failed to extract any tables from {report_path}")

    if "overall" in extracted:
        plot_recall(extracted["overall"], "Overall Recall@K", os.path.join(assets_dir, "recall_overall.png"))
        plot_mrr(extracted["overall"], "Overall MRR", os.path.join(assets_dir, "mrr_overall.png"))

    if "hotpot" in extracted:
        plot_recall(extracted["hotpot"], "HotpotQA Recall@K", os.path.join(assets_dir, "recall_hotpot.png"))
        plot_mrr(extracted["hotpot"], "HotpotQA MRR", os.path.join(assets_dir, "mrr_hotpot.png"))

    if "squad" in extracted:
        plot_recall(extracted["squad"], "SQuAD Recall@K", os.path.join(assets_dir, "recall_squad.png"))
        plot_mrr(extracted["squad"], "SQuAD MRR", os.path.join(assets_dir, "mrr_squad.png"))

    print(f"Redrawn charts saved to: {assets_dir}")


if __name__ == "__main__":
    main()

