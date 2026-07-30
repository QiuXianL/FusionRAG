import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "#4A4A4A", fs: int = 11) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.8, color="#444444"))


def plot_weighted_fusion(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("检索端：加权融合检索（Weighted Fusion）", pad=12, fontweight="bold")

    _draw_box(ax, 0.04, 0.32, 0.18, 0.36, "① Query 向量化", "#E8F1FB", ec="#4C78A8")
    _draw_box(ax, 0.28, 0.55, 0.24, 0.20, "② 计算 doc_score\n（文档向量相似度）", "#E8F1FB", ec="#4C78A8", fs=10.5)
    _draw_box(ax, 0.28, 0.30, 0.24, 0.20, "③ 计算 max_q_score\n（问题向量最大相似度）", "#FDECEC", ec="#E45756", fs=10.5)
    _draw_box(ax, 0.58, 0.32, 0.17, 0.36, "④ 按chunk聚合\n缺失问题向量时\nmax_q_score ← doc_score", "#F3F3F3", ec="#666666", fs=10)
    _draw_box(ax, 0.80, 0.32, 0.17, 0.36, "⑤ 融合打分\n0.7·doc + 0.3·q\n文档为主，问题为辅", "#EAF7EE", ec="#54A24B", fs=10.5)

    _arrow(ax, 0.22, 0.50, 0.28, 0.64)
    _arrow(ax, 0.22, 0.50, 0.28, 0.40)
    _arrow(ax, 0.52, 0.64, 0.58, 0.54)
    _arrow(ax, 0.52, 0.40, 0.58, 0.46)
    _arrow(ax, 0.75, 0.50, 0.80, 0.50)

    _save(fig, out_path)


def plot_hybrid_rerank(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("检索端：轻量级重排序（Hybrid Rerank）", pad=12, fontweight="bold")

    _draw_box(ax, 0.05, 0.34, 0.18, 0.32, "① Top候选集\n（语义检索）", "#E8F1FB", ec="#4C78A8")
    _draw_box(ax, 0.29, 0.34, 0.18, 0.32, "② 提取关键字符\n去停用符号", "#F3F3F3", ec="#666666")
    _draw_box(ax, 0.53, 0.34, 0.18, 0.32, "③ 计算 Coverage\n字符覆盖率", "#FFF6E6", ec="#F58518")
    _draw_box(ax, 0.77, 0.34, 0.18, 0.32, "④ 分数微调\nNew=0.8·Semantic+0.2·Coverage\n仅当 Semantic≥0.6", "#EAF7EE", ec="#54A24B", fs=10)

    _arrow(ax, 0.23, 0.50, 0.29, 0.50)
    _arrow(ax, 0.47, 0.50, 0.53, 0.50)
    _arrow(ax, 0.71, 0.50, 0.77, 0.50)

    _save(fig, out_path)


def plot_cot_rrf(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("检索端：CoT 查询扩展 + RRF 多查询融合（Super Brain）", pad=12, fontweight="bold")

    _draw_box(ax, 0.04, 0.35, 0.14, 0.30, "① 原始Query", "#E8F1FB", ec="#4C78A8")
    _draw_box(ax, 0.24, 0.68, 0.15, 0.18, "扩展①\n更具体", "#FDECEC", ec="#E45756", fs=10)
    _draw_box(ax, 0.24, 0.45, 0.15, 0.18, "扩展②\n同义泛化", "#FDECEC", ec="#E45756", fs=10)
    _draw_box(ax, 0.24, 0.22, 0.15, 0.18, "扩展③\n桥接视角", "#FDECEC", ec="#E45756", fs=10)
    _draw_box(ax, 0.46, 0.35, 0.18, 0.30, "② 分别检索\n得到多路排序", "#F3F3F3", ec="#666666")
    _draw_box(ax, 0.70, 0.35, 0.26, 0.30, "③ RRF融合\nScore(d)=Σ w(q)/(k+rank)\n原始query权重大，扩展query权重小", "#EAF7EE", ec="#54A24B", fs=10)

    _arrow(ax, 0.18, 0.50, 0.24, 0.77)
    _arrow(ax, 0.18, 0.50, 0.24, 0.54)
    _arrow(ax, 0.18, 0.50, 0.24, 0.31)
    _arrow(ax, 0.39, 0.77, 0.46, 0.58)
    _arrow(ax, 0.39, 0.54, 0.46, 0.50)
    _arrow(ax, 0.39, 0.31, 0.46, 0.42)
    _arrow(ax, 0.64, 0.50, 0.70, 0.50)

    _save(fig, out_path)


def plot_system_architecture(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("系统工程实现与部署形态（Web / CLI）", pad=12, fontweight="bold")

    _draw_box(ax, 0.05, 0.57, 0.15, 0.25, "Web 界面\n(Flask)", "#E8F1FB", ec="#4C78A8")
    _draw_box(ax, 0.05, 0.20, 0.15, 0.25, "CLI 命令行\n(rag_cli)", "#E8F1FB", ec="#4C78A8")
    _draw_box(ax, 0.28, 0.34, 0.28, 0.32, "核心服务\n索引/检索/重排/融合\nDocumentManager / SmartRetriever / QueryOptimizer", "#F3F3F3", ec="#666666", fs=10)
    _draw_box(ax, 0.62, 0.57, 0.20, 0.25, "向量库（本地JSON）\n文档向量+问题向量\n+映射关系", "#EAF7EE", ec="#54A24B", fs=10)
    _draw_box(ax, 0.62, 0.20, 0.20, 0.25, "评测与报告\n(evaluation / reports)", "#EAF7EE", ec="#54A24B", fs=10)
    _draw_box(ax, 0.86, 0.34, 0.11, 0.32, "生成模型\n(可选)\nDeepSeek API", "#FFF6E6", ec="#F58518", fs=10)

    _arrow(ax, 0.20, 0.66, 0.28, 0.53)
    _arrow(ax, 0.20, 0.32, 0.28, 0.47)
    _arrow(ax, 0.56, 0.55, 0.62, 0.67)
    _arrow(ax, 0.56, 0.45, 0.62, 0.33)
    _arrow(ax, 0.82, 0.50, 0.86, 0.50)

    _save(fig, out_path)


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "reports", "report_assets")
    _ensure_dir(out_dir)

    plot_weighted_fusion(os.path.join(out_dir, "weighted_fusion.png"))
    plot_hybrid_rerank(os.path.join(out_dir, "hybrid_rerank.png"))
    plot_cot_rrf(os.path.join(out_dir, "cot_rrf.png"))
    plot_system_architecture(os.path.join(out_dir, "system_architecture.png"))

    print(f"Saved diagrams to: {out_dir}")


if __name__ == "__main__":
    main()

