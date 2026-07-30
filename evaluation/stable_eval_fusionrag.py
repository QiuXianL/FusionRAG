import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.smart_retrieval import SmartRetriever
from evaluation.compare_qgen_experts import CustomDocumentManager


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Metrics:
    recall_1: float
    recall_5: float
    recall_10: float
    mrr: float


def _load_hotpot(num_samples: int) -> List[dict]:
    candidates = [
        f"evaluation/hotpot_{num_samples}_samples.json",
        "evaluation/hotpot_150_samples.json",
        "evaluation/hotpot_50_samples.json",
        "evaluation/hotpot_40_samples.json",
        "evaluation/hotpot_20_samples.json",
        "evaluation/hotpot_10_samples.json",
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if not path:
        raise FileNotFoundError(f"HotpotQA file not found. Tried: {candidates}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = data[:num_samples]
    for s in out:
        s["source"] = s.get("source") or "hotpot_qa"
    return out


def _load_squad(num_samples: int) -> List[dict]:
    path = "evaluation/squad_data.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"SQuAD file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc_to_item = {}
    for item in data:
        doc = item.get("document", "")
        if not doc:
            continue
        if doc not in doc_to_item:
            doc_to_item[doc] = item
    unique_items = list(doc_to_item.values())[:num_samples]
    out = []
    for i, item in enumerate(unique_items):
        out.append(
            {
                "id": f"squad_{i}",
                "question": item.get("question", ""),
                "answer": item.get("answer", "N/A"),
                "content": item.get("document", ""),
                "source": "squad",
            }
        )
    return out


def _attach_doc_index(samples: List[dict]) -> List[dict]:
    out = []
    for i, s in enumerate(samples):
        s2 = dict(s)
        s2["doc_index"] = i
        out.append(s2)
    return out


def _eval_on_indices(retriever: SmartRetriever, samples: List[dict], indices: np.ndarray) -> Tuple[Metrics, Dict[str, Metrics]]:
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    mrr_sum = 0.0

    by_source = {}
    for idx in indices.tolist():
        sample = samples[idx]
        query = sample["question"]
        target_doc_name = f"doc_{sample.get('doc_index', idx)}"
        source = sample.get("source", "unknown")

        results = retriever.retrieve_with_strategy(query, top_k=10)
        found_rank = -1
        for r, res in enumerate(results):
            if res.get("document_name") == target_doc_name:
                found_rank = r
                break

        if source not in by_source:
            by_source[source] = {"n": 0, "c1": 0, "c5": 0, "c10": 0, "mrr": 0.0}
        by_source[source]["n"] += 1

        if found_rank != -1:
            if found_rank < 1:
                correct_1 += 1
                by_source[source]["c1"] += 1
            if found_rank < 5:
                correct_5 += 1
                by_source[source]["c5"] += 1
            if found_rank < 10:
                correct_10 += 1
                by_source[source]["c10"] += 1
            rr = 1.0 / (found_rank + 1)
            mrr_sum += rr
            by_source[source]["mrr"] += rr

    n = len(indices)
    overall = Metrics(
        recall_1=correct_1 / n if n else 0.0,
        recall_5=correct_5 / n if n else 0.0,
        recall_10=correct_10 / n if n else 0.0,
        mrr=mrr_sum / n if n else 0.0,
    )

    by_source_metrics = {}
    for s, agg in by_source.items():
        ns = agg["n"]
        by_source_metrics[s] = Metrics(
            recall_1=agg["c1"] / ns if ns else 0.0,
            recall_5=agg["c5"] / ns if ns else 0.0,
            recall_10=agg["c10"] / ns if ns else 0.0,
            mrr=agg["mrr"] / ns if ns else 0.0,
        )

    return overall, by_source_metrics


def _compute_contributions(retriever: SmartRetriever, samples: List[dict]) -> Dict[str, np.ndarray]:
    n = len(samples)
    c1 = np.zeros(n, dtype=np.int32)
    c5 = np.zeros(n, dtype=np.int32)
    c10 = np.zeros(n, dtype=np.int32)
    rr = np.zeros(n, dtype=np.float32)
    sources = np.array([s.get("source", "unknown") for s in samples], dtype=object)

    for i, sample in enumerate(samples):
        query = sample["question"]
        target_doc_name = f"doc_{sample.get('doc_index', i)}"
        results = retriever.retrieve_with_strategy(query, top_k=10)

        found_rank = -1
        for r, res in enumerate(results):
            if res.get("document_name") == target_doc_name:
                found_rank = r
                break

        if found_rank != -1:
            if found_rank < 1:
                c1[i] = 1
            if found_rank < 5:
                c5[i] = 1
            if found_rank < 10:
                c10[i] = 1
            rr[i] = 1.0 / (found_rank + 1)

    return {"c1": c1, "c5": c5, "c10": c10, "rr": rr, "source": sources}


def _metrics_from_indices(contrib: Dict[str, np.ndarray], indices: np.ndarray) -> Metrics:
    if indices.size == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0)
    c1 = contrib["c1"][indices].mean()
    c5 = contrib["c5"][indices].mean()
    c10 = contrib["c10"][indices].mean()
    mrr = contrib["rr"][indices].mean()
    return Metrics(float(c1), float(c5), float(c10), float(mrr))


def _build_db(
    db_path: str,
    samples: List[dict],
    generator_type: str,
    questions_per_chunk: int,
) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)

    doc_manager = CustomDocumentManager(
        generator_type=generator_type,
        local_generator=None,
        db_path=db_path,
        backup_enabled=False,
    )

    eval_chunk_size = int(os.getenv("EVAL_CHUNK_SIZE", "0")) or None
    eval_chunk_overlap = int(os.getenv("EVAL_CHUNK_OVERLAP", "0")) or None
    if eval_chunk_size is not None and eval_chunk_overlap is not None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        doc_manager.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=eval_chunk_size,
            chunk_overlap=eval_chunk_overlap,
        )

    for s in samples:
        doc_name = f"doc_{s['doc_index']}"
        if hasattr(doc_manager, "_qgen_calls_in_current_doc"):
            doc_manager._qgen_calls_in_current_doc = 0
        doc_manager.add_document_from_text(
            s["content"],
            document_name=doc_name,
            skip_duplicates=False,
            questions_per_chunk=questions_per_chunk,
        )


def _plot_delta_bars(
    labels: List[str],
    deltas: List[Tuple[float, float, float]],
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(labels))
    means = [d[0] for d in deltas]
    lows = [d[0] - d[1] for d in deltas]
    highs = [d[2] - d[0] for d in deltas]
    yerr = np.array([lows, highs])

    ax.bar(x, means, width=0.26, color="#4C78A8", edgecolor="white", linewidth=0.8)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4)
    for xi, yi in zip(x, means):
        ax.text(xi, yi + (0.008 if yi >= 0 else -0.02), f"{yi:+.3f}", ha="center", va="bottom" if yi >= 0 else "top", fontsize=9)

    ax.axhline(0, color="#333333", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ (FusionRAG - Normal RAG)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    mean = float(values.mean()) if values.size else 0.0
    lo = float(np.quantile(values, alpha / 2)) if values.size else 0.0
    hi = float(np.quantile(values, 1 - alpha / 2)) if values.size else 0.0
    return mean, lo, hi


def main() -> None:
    out_report = os.getenv("REPORT_PATH", "reports/FusionRAG_vs_NormalRAG_100_stability_report.md")
    assets_dir = os.path.splitext(out_report)[0] + "_assets"
    os.makedirs(os.path.dirname(out_report) or "reports", exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    eval_chunk_size = os.getenv("EVAL_CHUNK_SIZE", "default")
    eval_chunk_overlap = os.getenv("EVAL_CHUNK_OVERLAP", "default")

    total = int(os.getenv("NUM_SAMPLES", "100"))
    hotpot_n = int(os.getenv("HOTPOT_SAMPLES", str(max(1, total // 2))))
    squad_n = int(os.getenv("SQUAD_SAMPLES", str(max(1, total - hotpot_n))))
    n_boot = int(os.getenv("N_BOOT", "200"))
    seed = int(os.getenv("SEED", "42"))
    api_q = int(os.getenv("API_Q_PER_CHUNK", "3"))
    max_qgen_chunks = int(os.getenv("MAX_QGEN_CHUNKS_PER_DOC", "1"))

    hotpot = _load_hotpot(hotpot_n)
    squad = _load_squad(squad_n)
    samples = _attach_doc_index(hotpot + squad)

    os.environ["MAX_QGEN_CHUNKS_PER_DOC"] = str(max_qgen_chunks)

    stem = os.path.splitext(os.path.basename(out_report))[0]
    normal_db = os.getenv("NORMAL_DB_PATH", f"temp_db_{stem}_Normal_RAG.json")
    fusion_db = os.getenv("FUSION_DB_PATH", f"temp_db_{stem}_FusionRAG.json")

    use_existing = os.getenv("USE_EXISTING_DB", "0") == "1"
    if not use_existing:
        _build_db(normal_db, samples, generator_type="none", questions_per_chunk=0)
        _build_db(fusion_db, samples, generator_type="deepseek", questions_per_chunk=api_q)
    else:
        if not os.path.exists(normal_db):
            raise FileNotFoundError(f"Normal DB not found: {normal_db} (set NORMAL_DB_PATH or USE_EXISTING_DB=0)")
        if not os.path.exists(fusion_db):
            raise FileNotFoundError(f"Fusion DB not found: {fusion_db} (set FUSION_DB_PATH or USE_EXISTING_DB=0)")

    normal_ret = SmartRetriever(db_path=normal_db)
    fusion_ret = SmartRetriever(db_path=fusion_db)

    n = len(samples)
    idx_all = np.arange(n)
    hotpot_idx = np.array([i for i, s in enumerate(samples) if s.get("source") == "hotpot_qa"], dtype=int)
    squad_idx = np.array([i for i, s in enumerate(samples) if s.get("source") == "squad"], dtype=int)

    normal_contrib = _compute_contributions(normal_ret, samples)
    fusion_contrib = _compute_contributions(fusion_ret, samples)

    overall_normal = _metrics_from_indices(normal_contrib, idx_all)
    overall_fusion = _metrics_from_indices(fusion_contrib, idx_all)

    by_source_normal = {}
    by_source_fusion = {}
    if hotpot_idx.size:
        by_source_normal["hotpot_qa"] = _metrics_from_indices(normal_contrib, hotpot_idx)
        by_source_fusion["hotpot_qa"] = _metrics_from_indices(fusion_contrib, hotpot_idx)
    if squad_idx.size:
        by_source_normal["squad"] = _metrics_from_indices(normal_contrib, squad_idx)
        by_source_fusion["squad"] = _metrics_from_indices(fusion_contrib, squad_idx)

    rng = np.random.default_rng(seed)
    delta_r1 = []
    delta_mrr = []
    delta_r1_hotpot = []
    delta_mrr_hotpot = []
    delta_r1_squad = []
    delta_mrr_squad = []

    for _ in range(n_boot):
        if hotpot_idx.size and squad_idx.size:
            samp_hotpot = rng.choice(hotpot_idx, size=hotpot_idx.size, replace=True)
            samp_squad = rng.choice(squad_idx, size=squad_idx.size, replace=True)
            boot_idx = np.concatenate([samp_hotpot, samp_squad])
        else:
            boot_idx = rng.choice(idx_all, size=idx_all.size, replace=True)

        n_o = _metrics_from_indices(normal_contrib, boot_idx)
        f_o = _metrics_from_indices(fusion_contrib, boot_idx)
        delta_r1.append(f_o.recall_1 - n_o.recall_1)
        delta_mrr.append(f_o.mrr - n_o.mrr)

        if hotpot_idx.size:
            nh = _metrics_from_indices(normal_contrib, samp_hotpot)
            fh = _metrics_from_indices(fusion_contrib, samp_hotpot)
            delta_r1_hotpot.append(fh.recall_1 - nh.recall_1)
            delta_mrr_hotpot.append(fh.mrr - nh.mrr)
        if squad_idx.size:
            ns = _metrics_from_indices(normal_contrib, samp_squad)
            fs = _metrics_from_indices(fusion_contrib, samp_squad)
            delta_r1_squad.append(fs.recall_1 - ns.recall_1)
            delta_mrr_squad.append(fs.mrr - ns.mrr)

    delta_r1 = np.array(delta_r1, dtype=float)
    delta_mrr = np.array(delta_mrr, dtype=float)
    delta_r1_hotpot = np.array(delta_r1_hotpot, dtype=float)
    delta_mrr_hotpot = np.array(delta_mrr_hotpot, dtype=float)
    delta_r1_squad = np.array(delta_r1_squad, dtype=float)
    delta_mrr_squad = np.array(delta_mrr_squad, dtype=float)

    ci_r1 = _percentile_ci(delta_r1)
    ci_mrr = _percentile_ci(delta_mrr)
    ci_r1_hotpot = _percentile_ci(delta_r1_hotpot) if delta_r1_hotpot.size else (0.0, 0.0, 0.0)
    ci_mrr_hotpot = _percentile_ci(delta_mrr_hotpot) if delta_mrr_hotpot.size else (0.0, 0.0, 0.0)
    ci_r1_squad = _percentile_ci(delta_r1_squad) if delta_r1_squad.size else (0.0, 0.0, 0.0)
    ci_mrr_squad = _percentile_ci(delta_mrr_squad) if delta_mrr_squad.size else (0.0, 0.0, 0.0)

    _plot_delta_bars(
        labels=["Overall", "HotpotQA", "SQuAD"],
        deltas=[ci_r1, ci_r1_hotpot, ci_r1_squad],
        title="Δ Recall@1 (with 95% bootstrap CI)",
        out_path=os.path.join(assets_dir, "delta_recall1.png"),
    )
    _plot_delta_bars(
        labels=["Overall", "HotpotQA", "SQuAD"],
        deltas=[ci_mrr, ci_mrr_hotpot, ci_mrr_squad],
        title="Δ MRR (with 95% bootstrap CI)",
        out_path=os.path.join(assets_dir, "delta_mrr.png"),
    )

    def _fmt(x: float) -> str:
        return f"{x:.4f}"

    def _line_metrics(m: Metrics) -> str:
        return f"| {_fmt(m.recall_1)} | {_fmt(m.recall_5)} | {_fmt(m.recall_10)} | {_fmt(m.mrr)} |"

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# 🧠 智融检索（FusionRAG） vs Normal RAG 稳定性评测报告（Bootstrap）\n\n")
        f.write(f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. 实验设置\n")
        f.write(f"- 总样本数：{len(samples)}（HotpotQA={len(hotpot)}, SQuAD={len(squad)}）\n")
        f.write(f"- FusionRAG：API_Q_PER_CHUNK={api_q}，MAX_QGEN_CHUNKS_PER_DOC={max_qgen_chunks}\n")
        f.write(f"- 分块参数：EVAL_CHUNK_SIZE={eval_chunk_size}，EVAL_CHUNK_OVERLAP={eval_chunk_overlap}\n")
        f.write(f"- Bootstrap：分层重采样（HotpotQA 与 SQuAD 各自保持样本量不变），重复次数 N_BOOT={n_boot}，随机种子 SEED={seed}\n")
        f.write("- 说明：该方法固定同一批文档库，只通过“重采样 query 集合”估计指标波动，避免因每次重建索引带来的额外噪声与成本。\n\n")

        f.write("## 2. 单次全量结果（原始100%样本）\n")
        f.write("### 2.1 Overall\n\n")
        f.write("| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        f.write(f"| Normal RAG {_line_metrics(overall_normal)}\n".replace("| |", "|"))
        f.write(f"| FusionRAG {_line_metrics(overall_fusion)}\n".replace("| |", "|"))

        if "hotpot_qa" in by_source_normal and "hotpot_qa" in by_source_fusion:
            f.write("\n### 2.2 HotpotQA\n\n")
            f.write("| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            f.write(f"| Normal RAG {_line_metrics(by_source_normal['hotpot_qa'])}\n".replace("| |", "|"))
            f.write(f"| FusionRAG {_line_metrics(by_source_fusion['hotpot_qa'])}\n".replace("| |", "|"))

        if "squad" in by_source_normal and "squad" in by_source_fusion:
            f.write("\n### 2.3 SQuAD\n\n")
            f.write("| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            f.write(f"| Normal RAG {_line_metrics(by_source_normal['squad'])}\n".replace("| |", "|"))
            f.write(f"| FusionRAG {_line_metrics(by_source_fusion['squad'])}\n".replace("| |", "|"))

        f.write("\n## 3. 稳定性分析（Δ=FusionRAG-Normal 的 95% Bootstrap CI）\n\n")
        f.write("| 维度 | Δ Recall@1 (mean) | 95% CI | Δ MRR (mean) | 95% CI |\n")
        f.write("| --- | ---: | :---: | ---: | :---: |\n")
        f.write(f"| Overall | {ci_r1[0]:+.4f} | [{ci_r1[1]:+.4f}, {ci_r1[2]:+.4f}] | {ci_mrr[0]:+.4f} | [{ci_mrr[1]:+.4f}, {ci_mrr[2]:+.4f}] |\n")
        f.write(f"| HotpotQA | {ci_r1_hotpot[0]:+.4f} | [{ci_r1_hotpot[1]:+.4f}, {ci_r1_hotpot[2]:+.4f}] | {ci_mrr_hotpot[0]:+.4f} | [{ci_mrr_hotpot[1]:+.4f}, {ci_mrr_hotpot[2]:+.4f}] |\n")
        f.write(f"| SQuAD | {ci_r1_squad[0]:+.4f} | [{ci_r1_squad[1]:+.4f}, {ci_r1_squad[2]:+.4f}] | {ci_mrr_squad[0]:+.4f} | [{ci_mrr_squad[1]:+.4f}, {ci_mrr_squad[2]:+.4f}] |\n")

        f.write("\n### 3.1 可视化\n\n")
        f.write(f"![Delta Recall@1]({os.path.basename(assets_dir)}/delta_recall1.png)\n\n")
        f.write(f"![Delta MRR]({os.path.basename(assets_dir)}/delta_mrr.png)\n\n")

        f.write("## 4. 结论建议（写进结项报告的表达方式）\n")
        f.write("- 优先看 Overall 与 SQuAD/HotpotQA 的分项是否同向提升；若 HotpotQA 与 SQuAD 出现反向波动，建议在结项中解释为“多跳与单跳任务的问法结构差异”，并用融合权重/生成策略做讨论。\n")
        f.write("- 若 95% CI 跨过 0，建议表述为“总体呈提升趋势，但存在样本波动，后续可通过增大样本与离线更充分的 QGen 提升稳定性”。\n")


if __name__ == "__main__":
    main()

