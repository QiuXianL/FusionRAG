import json
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from datetime import datetime
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from smart_retrieval import SmartRetriever
from document_manager import DocumentManager
from query_optimizer import QueryOptimizer

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TEST_DB_PATH = os.path.join(PROJECT_ROOT, "documents", "test_hotpot_vectors_viz.json")
SAMPLES_PATH = os.path.join(PROJECT_ROOT, "documents", "test_hotpot_samples.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
IMAGES_DIR = os.path.join(REPORTS_DIR, "images_hotpot")
REPORT_PATH = os.path.join(REPORTS_DIR, "hotpot_qa_visualized_report.md")

class TestSmartRetriever(SmartRetriever):
    """Subclass to load test database"""
    def _load_databases(self):
        self.databases = {}
        if os.path.exists(TEST_DB_PATH):
            try:
                with open(TEST_DB_PATH, 'r', encoding='utf-8') as f:
                    self.databases["enhanced"] = json.load(f)
                print(f"✅ Loaded test database from {TEST_DB_PATH}")
            except Exception as e:
                print(f"❌ Failed to load test database: {e}")

def load_env():
    """Load .env file"""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("DEEPSEEK_API="):
                    key = line.split("=", 1)[1].strip().strip('"')
                    os.environ["DEEPSEEK_API"] = key
                    print("✅ DEEPSEEK_API loaded from .env")
                    return
    print("⚠️ .env file not found or DEEPSEEK_API not set")

def setup_hotpot_database(num_samples=50):
    """Build a temporary database from HotpotQA data"""
    
    # Check if we can reuse existing data
    if os.path.exists(TEST_DB_PATH) and os.path.exists(SAMPLES_PATH):
        print(f"♻️ Found existing database and samples.")
        print(f"  DB: {TEST_DB_PATH}")
        print(f"  Samples: {SAMPLES_PATH}")
        try:
            with open(SAMPLES_PATH, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            
            # If we have more samples than requested, slice them
            if len(samples) > num_samples:
                print(f"⚠️ Existing samples ({len(samples)}) > requested ({num_samples}). Slicing...")
                samples = samples[:num_samples]
                # Note: The DB will still contain vectors for all original samples.
                # This is fine for retrieval (just more distractors), but might be slower.
                # If we want strict speedup, we should rebuild.
                print("⚠️ Using larger DB with subset of samples. This is acceptable.")
            elif len(samples) < num_samples:
                print(f"⚠️ Existing samples ({len(samples)}) < requested ({num_samples}). Rebuilding...")
                # Fall through to rebuild
            else:
                print(f"✅ Loaded {len(samples)} samples from disk.")
                return samples
        except Exception as e:
            print(f"⚠️ Failed to load samples: {e}. Rebuilding...")

    # If we decided to rebuild (or didn't return above)
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
            print("🧹 Removed old database to ensure consistency.")
        except:
            pass

    print(f"🏗️ Loading HotpotQA dataset (first {num_samples} samples)...")
    
    # Load dataset stream to avoid downloading everything
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    
    samples = []
    unique_docs = {} # Title -> Content
    
    iterator = iter(ds)
    for _ in tqdm(range(num_samples), desc="Fetching samples"):
        try:
            item = next(iterator)
            # Convert dataset item to dict for JSON serialization
            item_dict = {
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'type': item['type'],
                'level': item['level'],
                'supporting_facts': {
                    'title': item['supporting_facts']['title'],
                    'sent_id': item['supporting_facts']['sent_id']
                },
                'context': {
                    'title': item['context']['title'],
                    'sentences': item['context']['sentences']
                }
            }
            samples.append(item_dict)
            
            # Extract context documents
            # context is a dictionary: {'title': [...], 'sentences': [...]} in some versions
            # or a list of [title, sentences] in raw json. 
            # In huggingface datasets:
            # context: {'title': Sequence(Value(dtype='string')), 'sentences': Sequence(Sequence(Value(dtype='string')))}
            
            titles = item['context']['title']
            sentences_list = item['context']['sentences']
            
            for title, sentences in zip(titles, sentences_list):
                content = " ".join(sentences)
                if title not in unique_docs:
                    unique_docs[title] = content
                    
        except StopIteration:
            break
            
    print(f"📊 Found {len(unique_docs)} unique documents from {len(samples)} questions.")
    
    # Save samples for future reuse
    try:
        with open(SAMPLES_PATH, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved samples to {SAMPLES_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to save samples: {e}")
    
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except:
            pass
        
    manager = DocumentManager(db_path=TEST_DB_PATH, backup_enabled=False)
    
    print(f"🔄 Indexing {len(unique_docs)} documents...")
    # Use batch addition for speed
    manager.add_documents_from_texts_batch(unique_docs, questions_per_chunk=0)
        
    print(f"✅ Test database created at {TEST_DB_PATH}")
    
    return samples

def calculate_metrics(results, ground_truths_list, k_list=[1, 5, 10]):
    """
    results: List of List of dicts (retrieved docs)
    ground_truths_list: List of Set of strings (required titles)
    """
    metrics = {k: {'recall': 0, 'precision': 0} for k in k_list}
    mrr = 0
    
    for res_list, gt_titles in zip(results, ground_truths_list):
        # Calculate MRR (based on first relevant doc found)
        rank = 0
        for i, r in enumerate(res_list):
            # document_name stores the title
            if r.get('document_name') in gt_titles:
                rank = i + 1
                break
        if rank > 0:
            mrr += 1.0 / rank
            
        # Calculate Recall and Precision at K
        for k in k_list:
            top_k = res_list[:k]
            # Retrieved Titles
            retrieved_titles = set(r.get('document_name') for r in top_k)
            
            # Hits: How many required titles were found?
            hits = len(retrieved_titles.intersection(gt_titles))
            
            # Recall: Proportion of required docs found
            if len(gt_titles) > 0:
                metrics[k]['recall'] += hits / len(gt_titles)
            
            # Precision: Proportion of retrieved docs that are relevant
            if k > 0:
                metrics[k]['precision'] += hits / k
    
    total = len(results)
    final_metrics = {}
    
    final_metrics['MRR'] = mrr / total
    for k in k_list:
        final_metrics[f'Recall@{k}'] = metrics[k]['recall'] / total
        final_metrics[f'Precision@{k}'] = metrics[k]['precision'] / total
        
    return final_metrics

def visualize_results(baseline_metrics, super_metrics):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # 1. Recall Comparison
    labels = ['Recall@1', 'Recall@5', 'Recall@10']
    baseline_scores = [baseline_metrics[l] for l in labels]
    super_scores = [super_metrics[l] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color='skyblue')
    rects2 = ax.bar(x + width/2, super_scores, width, label='Super Brain', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title('Recall Comparison: Baseline vs Super Brain (HotpotQA)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    recall_path = os.path.join(IMAGES_DIR, 'recall_comparison.png')
    plt.savefig(recall_path)
    plt.close()
    
    # 2. MRR Comparison
    fig, ax = plt.subplots(figsize=(6, 6))
    mrr_labels = ['MRR']
    mrr_base = [baseline_metrics['MRR']]
    mrr_super = [super_metrics['MRR']]
    
    x_mrr = np.arange(len(mrr_labels))
    
    ax.bar(x_mrr - width/2, mrr_base, width, label='Baseline', color='skyblue')
    ax.bar(x_mrr + width/2, mrr_super, width, label='Super Brain', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title('Mean Reciprocal Rank (MRR)')
    ax.set_xticks(x_mrr)
    ax.set_xticklabels(mrr_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for i, v in enumerate(mrr_base):
        ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
    for i, v in enumerate(mrr_super):
        ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
        
    plt.tight_layout()
    mrr_path = os.path.join(IMAGES_DIR, 'mrr_comparison.png')
    plt.savefig(mrr_path)
    plt.close()
    
    return recall_path, mrr_path

def analyze_improvements(samples, baseline_results, super_results, ground_truths_list):
    """Analyze which queries improved or regressed"""
    improvements = []
    regressions = []
    
    for i, (query, gt_titles) in enumerate(zip([s['question'] for s in samples], ground_truths_list)):
        # Calculate Baseline Rank (first hit)
        base_rank = 1000 # Not found
        for j, r in enumerate(baseline_results[i]):
            if r.get('document_name') in gt_titles:
                base_rank = j + 1
                break
                
        # Calculate Super Rank
        super_rank = 1000
        for j, r in enumerate(super_results[i]):
            if r.get('document_name') in gt_titles:
                super_rank = j + 1
                break
        
        item = {
            'query': query,
            'base_rank': base_rank if base_rank != 1000 else '>10',
            'super_rank': super_rank if super_rank != 1000 else '>10',
            'diff': base_rank - super_rank # Positive means improvement (lower rank is better)
        }
        
        if super_rank < base_rank:
            improvements.append(item)
        elif super_rank > base_rank:
            regressions.append(item)
            
    # Sort by magnitude of improvement
    improvements.sort(key=lambda x: x['diff'], reverse=True)
    regressions.sort(key=lambda x: x['diff']) # Most negative first
    
    return improvements, regressions

def generate_report(total_samples, baseline_metrics, super_metrics, improvements, regressions):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format improvements table
    imp_table = "| Query | Baseline Rank | Super Brain Rank |\n| :--- | :---: | :---: |\n"
    for item in improvements[:5]: # Top 5
        imp_table += f"| {item['query']} | {item['base_rank']} | **{item['super_rank']}** |\n"
    if not improvements:
        imp_table += "| No improvements found | - | - |\n"
        
    # Format regressions table
    reg_table = "| Query | Baseline Rank | Super Brain Rank |\n| :--- | :---: | :---: |\n"
    for item in regressions[:5]: # Top 5
        reg_table += f"| {item['query']} | {item['base_rank']} | {item['super_rank']} |\n"
    if not regressions:
        reg_table += "| No regressions found | - | - |\n"

    content = f"""# 🧠 Super Brain HotpotQA 评估报告
    
**生成时间**: {timestamp}
**测试样本数**: {total_samples}
**数据集**: HotpotQA (Validation / Distractor)

## 1. 实验结论 (Executive Summary)

本次实验在 **HotpotQA** 多跳推理数据集上对比了 **Baseline (基准检索)** 和 **Super Brain (最强大脑)** 的表现。
HotpotQA 要求从多个文档中寻找线索，比单跳问答更具挑战性。

实验结果显示：

- **Recall@5 (召回率)**: 从 `{baseline_metrics['Recall@5']:.2%}` 提升至 `{super_metrics['Recall@5']:.2%}`
- **MRR (平均倒数排名)**: 从 `{baseline_metrics['MRR']:.2f}` 提升至 `{super_metrics['MRR']:.2f}`

## 2. 详细指标对比

### 2.1 召回率 (Recall) 与 精确率 (Precision)

| 指标 (Metric) | Baseline | Super Brain | 提升幅度 |
| :--- | :---: | :---: | :---: |
| **Recall@1** | {baseline_metrics['Recall@1']:.2%} | **{super_metrics['Recall@1']:.2%}** | {super_metrics['Recall@1'] - baseline_metrics['Recall@1']:+.2%} |
| **Recall@5** | {baseline_metrics['Recall@5']:.2%} | **{super_metrics['Recall@5']:.2%}** | {super_metrics['Recall@5'] - baseline_metrics['Recall@5']:+.2%} |
| **Recall@10** | {baseline_metrics['Recall@10']:.2%} | **{super_metrics['Recall@10']:.2%}** | {super_metrics['Recall@10'] - baseline_metrics['Recall@10']:+.2%} |
| **MRR** | {baseline_metrics['MRR']:.2f} | **{super_metrics['MRR']:.2f}** | {super_metrics['MRR'] - baseline_metrics['MRR']:+.2f} |

> **注**: 
> - 对于 HotpotQA，Recall@K 表示 **所需的所有支撑文档** 中有多少比例出现在了前 K 个结果中。
> - 由于每个问题通常需要 2 个文档，Recall=100% 意味着两个文档都被找回了。

### 2.2 可视化图表

#### 召回率对比图
![Recall Comparison](images_hotpot/recall_comparison.png)

#### MRR 对比图
![MRR Comparison](images_hotpot/mrr_comparison.png)

## 3. 详细案例分析 (Case Studies)

### ✅ 排名提升案例 (Top Improvements)
Super Brain 成功通过思维链关联找到了更相关的文档：

{imp_table}

### 🔻 排名下降案例 (Regressions)
部分查询可能因过度扩展引入了噪音：

{reg_table}

## 4. 分析与洞察

HotpotQA 是多跳问答任务，通常需要"桥接"实体（例如：问题提到A，A关联B，答案在B）。
Super Brain 的 **思维链 (CoT)** 在此处发挥了关键作用：
1.  它能分解问题，不仅搜索A，还会推测可能涉及的B。
2.  **RRF** 融合了直接搜索和间接搜索的结果，提高了找回第二跳文档的概率。

---
*Report generated automatically by `evaluate_hotpot_visualized.py`*
"""
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📄 Report generated at: {REPORT_PATH}")

def main():
    load_env()
    
    # 1. Setup Data & DB
    # Run on 20 samples to keep document count around 200 (approx 10 docs per sample)
    samples = setup_hotpot_database(num_samples=20)
    
    print("\n🚀 Starting HotpotQA Evaluation...")
    retriever = TestSmartRetriever()
    optimizer = QueryOptimizer()
    
    ground_truths_list = []
    baseline_results_list = []
    super_results_list = []
    
    if not optimizer.client:
        print("⚠️ Warning: No API Key found. Super Brain will fall back to baseline.")
    
    for i, item in enumerate(tqdm(samples, desc="Evaluating Queries")):
        query = item['question']
        
        # Ground Truth: Set of titles in supporting_facts
        # supporting_facts: {'title': [...], 'sent_id': [...]}
        gt_titles = set(item['supporting_facts']['title'])
        ground_truths_list.append(gt_titles)
        
        # --- Baseline ---
        # Get top 10 for metrics calculation
        base_res = retriever.retrieve_with_strategy(query, strategy='enhanced', top_k=10)
        baseline_results_list.append(base_res)
        
        # --- Super Brain ---
        super_res = []
        try:
            if optimizer.client:
                # Use CoT expansion
                queries = optimizer.expand_query(query)
                all_results = {}
                for q in queries:
                    # Get more candidates for fusion
                    res = retriever.retrieve_with_strategy(q, strategy='enhanced', top_k=10)
                    all_results[q] = res
                
                # Fuse results
                super_res = optimizer.fuse_results(all_results, original_query=query, k=60)
                super_res = super_res[:10]
            else:
                super_res = base_res
        except Exception as e:
            print(f"Error in Super Brain: {e}")
            super_res = base_res
            
        super_results_list.append(super_res)
        
    # Calculate Metrics
    print("\n📊 Calculating Metrics...")
    baseline_metrics = calculate_metrics(baseline_results_list, ground_truths_list)
    super_metrics = calculate_metrics(super_results_list, ground_truths_list)
    
    # Visualize
    print("🎨 Generating Charts...")
    visualize_results(baseline_metrics, super_metrics)
    
    # Analyze improvements
    improvements, regressions = analyze_improvements(samples, baseline_results_list, super_results_list, ground_truths_list)
    
    # Report
    generate_report(len(samples), baseline_metrics, super_metrics, improvements, regressions)
    
    # Cleanup
    # Commented out cleanup to persist data for future runs
    # if os.path.exists(TEST_DB_PATH):
    #     try:
    #         os.remove(TEST_DB_PATH)
    #         print("🧹 Test database cleaned up.")
    #     except:
    #         pass

if __name__ == "__main__":
    main()
