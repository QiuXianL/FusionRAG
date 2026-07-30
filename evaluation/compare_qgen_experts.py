import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 设置国内镜像 (必须在 import datasets/transformers 之前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.document_manager import DocumentManager
from src.smart_retrieval import SmartRetriever

class LocalQGen:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
             print(f"Error loading tokenizer {model_path}: {e}")
             return

        # Determine if we should try 8-bit loading
        load_in_8bit = True
        if device == "cpu" or not torch.cuda.is_available():
            load_in_8bit = False
            
        try:
            print(f"Loading model from {model_path} (8bit={load_in_8bit})...")
            # Try loading with device_map="auto"
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
                    device_map="auto" if torch.cuda.is_available() else None,
                    load_in_8bit=load_in_8bit,
                    low_cpu_mem_usage=True
                )
            except Exception as e_map:
                print(f"Failed to load with device_map='auto' and 8bit={load_in_8bit}: {e_map}. Fallback to manual/no-8bit.")
                # Fallback: try without 8bit if it was enabled
                if load_in_8bit:
                     try:
                        print("Retrying without 8-bit quantization...")
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            low_cpu_mem_usage=True
                        )
                     except Exception as e_no8bit:
                         print(f"Failed without 8bit: {e_no8bit}. Fallback to CPU/manual.")
                         self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True
                         ).to(self.device)
                else:
                    # If 8bit was False and it failed, try CPU/float32
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    ).to(self.device)
            
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            self.model = None

    def generate(self, content, num_questions=1):
        if self.model is None:
            print("Error: Model is not initialized.")
            return []
            
        try:
            # Simple prompt for T5
            input_text = f"generate questions: {content}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_return_sequences=num_questions,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
            
            questions = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            return questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

class CustomDocumentManager(DocumentManager):
    def __init__(self, generator_type="deepseek", local_generator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator_type = generator_type
        self.local_generator = local_generator
        self.captured_questions = []
        self.max_qgen_calls_per_doc = int(os.getenv("MAX_QGEN_CHUNKS_PER_DOC", "0")) or None
        self._qgen_calls_in_current_doc = 0
        eval_chunk_size = int(os.getenv("EVAL_CHUNK_SIZE", "0")) or None
        eval_chunk_overlap = int(os.getenv("EVAL_CHUNK_OVERLAP", "0")) or None
        if eval_chunk_size is not None and eval_chunk_overlap is not None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=eval_chunk_size,
                chunk_overlap=eval_chunk_overlap,
            )

    def _generate_enhanced_questions(self, content: str, num_questions: int = 3) -> List[str]:
        questions = []
        
        # Optimization: Randomly skip 80% of chunks for DeepSeek to speed up evaluation
        # This is acceptable for evaluation purposes as we just need SOME enhanced vectors
        # import random
        # if self.generator_type == "deepseek" and random.random() > 0.2:
        #    return []

        if self.generator_type == "deepseek":
            if self.max_qgen_calls_per_doc is not None and self._qgen_calls_in_current_doc >= self.max_qgen_calls_per_doc:
                return []
            self._qgen_calls_in_current_doc += 1
            # Use original logic
            print("DEBUG: Calling super()._generate_enhanced_questions for DeepSeek")
            questions = super()._generate_enhanced_questions(content, num_questions)
            print(f"DEBUG: DeepSeek returned {len(questions)} questions")
        elif self.generator_type in ["base", "finetuned"]:
            if self.local_generator:
                questions = self.local_generator.generate(content, num_questions)
                print(f"DEBUG: Local Gen returned {len(questions)} questions")
        elif self.generator_type in ["none", "normal", "doc_only"]:
            questions = []
        
        # Capture first 3 examples (only if we have questions)
        if len(self.captured_questions) < 3 and questions:
             self.captured_questions.append({
                 "content_snippet": content[:100] + "...",
                 "questions": questions
             })
        
        return questions

def evaluate_expert(expert_name, model_path, samples, temp_db_path, use_existing_db=True, questions_per_chunk=1, generator_type=None):
    print(f"\nEvaluating Expert: {expert_name}")
    
    # 1. Initialize Generator
    local_gen = None
    if generator_type is None:
        name_to_type = {
            "Normal RAG": "none",
            "DeepSeek API": "deepseek",
            "DeepSeek": "deepseek",
            "智融检索（FusionRAG）": "deepseek",
            "Base Model": "base",
            "Finetuned Model": "finetuned",
        }
        generator_type = name_to_type.get(expert_name, "none")

    if not use_existing_db and generator_type in ["base", "finetuned"]:
        local_gen = LocalQGen(model_path)
    
    # 2. Initialize DocumentManager with Custom Logic
    # We use a temp DB to avoid polluting the main one
    if not use_existing_db and os.path.exists(temp_db_path):
        os.remove(temp_db_path)
    elif use_existing_db and os.path.exists(temp_db_path):
        print(f"Using existing database: {temp_db_path}")

    doc_manager = None
    if not use_existing_db:
        doc_manager = CustomDocumentManager(
            generator_type=generator_type,
            local_generator=local_gen,
            db_path=temp_db_path,
            backup_enabled=False
        )
    
    # 3. Index Documents
    if not use_existing_db:
        print(f"Indexing {len(samples)} documents...")
        # Prepare batch data
        docs = [s['content'] for s in samples]
        
        for i, doc_content in enumerate(docs):
            # We use a unique name to verify retrieval
            doc_name = f"doc_{i}"
            if hasattr(doc_manager, "_qgen_calls_in_current_doc"):
                doc_manager._qgen_calls_in_current_doc = 0
            doc_manager.add_document_from_text(
                doc_content,
                document_name=doc_name,
                skip_duplicates=False,
                questions_per_chunk=questions_per_chunk
            )
    else:
        print("Skipping indexing (using existing DB)...")
        # Need to reconstruct docs list for content matching fallback
        docs = [s['content'] for s in samples]
        
    # 4. Run Retrieval Evaluation
    retriever = SmartRetriever(db_path=temp_db_path)
    
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    mrr_sum = 0

    per_source = {}
    
    print("Running retrieval test...")
    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        query = sample['question']
        target_doc_name = f"doc_{i}"
        source = sample.get("source", "unknown")
        if source not in per_source:
            per_source[source] = {"n": 0, "correct_1": 0, "correct_5": 0, "correct_10": 0, "mrr_sum": 0.0}
        per_source[source]["n"] += 1
        
        results = retriever.retrieve_with_strategy(query, top_k=10)
        
        # Check if target doc is in results
        found_rank = -1
        for rank, res in enumerate(results):
            # Use document name for precise matching if available
            if res.get('document_name') == target_doc_name:
                found_rank = rank
                break
            # Fallback to content matching if name is missing (shouldn't happen with correct setup)
            elif res.get('original_text') and res['original_text'] in docs[i]:
                 found_rank = rank
                 break
        
        if found_rank != -1:
            if found_rank < 1: correct_1 += 1
            if found_rank < 5: correct_5 += 1
            if found_rank < 10: correct_10 += 1
            mrr_sum += 1.0 / (found_rank + 1)
            if found_rank < 1: per_source[source]["correct_1"] += 1
            if found_rank < 5: per_source[source]["correct_5"] += 1
            if found_rank < 10: per_source[source]["correct_10"] += 1
            per_source[source]["mrr_sum"] += 1.0 / (found_rank + 1)
            
    # Capture generated samples before cleanup
    captured_samples = []
    if doc_manager:
        captured_samples = doc_manager.captured_questions
    elif use_existing_db:
        # Try to extract samples from existing DB
        try:
             if 'enhanced' in retriever.databases:
                 data = retriever.databases['enhanced']
                 
                 # Reconstruct chunk-to-questions mapping
                 chunk_to_questions = {}
                 q_idx = 0
                 
                 # vector_types aligns with embeddings. questions list only contains questions.
                 # We need to iterate vector_types to find which chunk each question belongs to.
                 if 'vector_types' in data and 'vector_to_chunk_map' in data and 'questions' in data:
                     for i, v_type in enumerate(data['vector_types']):
                         if v_type == 'question':
                             if q_idx < len(data['questions']):
                                 chunk_idx = data['vector_to_chunk_map'][i]
                                 q_text = data['questions'][q_idx]
                                 
                                 if chunk_idx not in chunk_to_questions:
                                     chunk_to_questions[chunk_idx] = []
                                 chunk_to_questions[chunk_idx].append(q_text)
                                 
                                 q_idx += 1
                 
                 # Now extract samples
                 for chunk_idx, questions in chunk_to_questions.items():
                     if chunk_idx < len(data['documents']) and len(captured_samples) < 3:
                         content = data['documents'][chunk_idx]
                         captured_samples.append({
                             "content_snippet": content[:100] + "...",
                             "questions": questions
                         })
                     if len(captured_samples) >= 3:
                         break
             else:
                 print("Warning: 'enhanced' database not found in retriever.")
        except Exception as e:
            print(f"Warning: Could not extract samples from existing DB: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup model to free memory
    print("Cleaning up resources...")
    if 'retriever' in locals():
        del retriever
    if 'doc_manager' in locals() and doc_manager:
        del doc_manager
    
    if local_gen:
        print("Cleaning up local generator...")
        if hasattr(local_gen, 'model'):
            del local_gen.model
        if hasattr(local_gen, 'tokenizer'):
            del local_gen.tokenizer
        del local_gen
    
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleanup complete.")
        
    metrics = {
        "Recall@1": correct_1 / len(samples),
        "Recall@5": correct_5 / len(samples),
        "Recall@10": correct_10 / len(samples),
        "MRR": mrr_sum / len(samples),
        "generated_samples": captured_samples,
        "questions_per_chunk": questions_per_chunk,
        "generator_type": generator_type,
        "by_source": {
            k: {
                "Recall@1": (v["correct_1"] / v["n"]) if v["n"] else 0.0,
                "Recall@5": (v["correct_5"] / v["n"]) if v["n"] else 0.0,
                "Recall@10": (v["correct_10"] / v["n"]) if v["n"] else 0.0,
                "MRR": (v["mrr_sum"] / v["n"]) if v["n"] else 0.0,
                "n": v["n"],
            }
            for k, v in per_source.items()
        },
    }
    
    return metrics

def load_hotpotqa_dataset(num_samples=50):
    print(f"Loading HotpotQA Dataset from local file...")
    samples = []
    try:
        candidate_paths = [
            f"evaluation/hotpot_{num_samples}_samples.json",
            "evaluation/hotpot_150_samples.json",
            "evaluation/hotpot_50_samples.json",
            "evaluation/hotpot_40_samples.json",
            "evaluation/hotpot_20_samples.json",
            "evaluation/hotpot_10_samples.json",
        ]

        path = None
        for p in candidate_paths:
            if os.path.exists(p):
                path = p
                break

        if not path:
            print(f"Error: evaluation/hotpot_{num_samples}_samples.json not found.")
            return []
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # The data is already in the correct format from our download script
        samples = data[:num_samples]
            
    except Exception as e:
        print(f"Error loading HotpotQA dataset: {e}")
    
    print(f"Loaded {len(samples)} samples from HotpotQA")
    return samples

def load_squad_dataset(num_samples=50):
    print("Loading SQuAD Dataset from local file...")
    path = "evaluation/squad_data.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading SQuAD dataset: {e}")
        return []

    doc_to_item = {}
    for item in data:
        doc = item.get("document", "")
        if not doc:
            continue
        if doc not in doc_to_item:
            doc_to_item[doc] = item

    unique_items = list(doc_to_item.values())[:num_samples]
    samples = []
    for i, item in enumerate(unique_items):
        samples.append({
            "id": f"squad_{i}",
            "question": item.get("question", ""),
            "answer": item.get("answer", "N/A"),
            "content": item.get("document", ""),
            "source": "squad"
        })

    print(f"Loaded {len(samples)} samples from SQuAD")
    return samples

def load_custom_dataset(path, num_samples=50):
    print(f"Loading Custom Dataset from {path}...")
    samples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(samples) >= num_samples:
                    break
                data = json.loads(line)
                # Parse input_text to get content
                input_text = data.get("input_text", "")
                if input_text.startswith("generate questions: "):
                    content = input_text.replace("generate questions: ", "", 1)
                else:
                    content = input_text
                
                samples.append({
                    "id": f"custom_{i}",
                    "question": data.get("target_text", ""),
                    "answer": "N/A",
                    "content": content,
                    "source": "custom"
                })
    except Exception as e:
        print(f"Error loading custom dataset: {e}")
    
    return samples

def run_api_vs_normal(samples):
    use_existing_db = os.getenv("USE_EXISTING_DB", "0") == "1"
    experts = [
        {"name": "Normal RAG", "path": None, "questions_per_chunk": 0, "generator_type": "none"},
        {"name": "智融检索（FusionRAG）", "path": None, "questions_per_chunk": int(os.getenv("API_Q_PER_CHUNK", "3")), "generator_type": "deepseek"},
    ]

    results = {}
    for expert in experts:
        try:
            metrics = evaluate_expert(
                expert["name"],
                expert["path"],
                samples,
                f"temp_db_{expert['name'].replace(' ', '_')}.json",
                use_existing_db=use_existing_db,
                questions_per_chunk=expert["questions_per_chunk"],
                generator_type=expert["generator_type"],
            )
            results[expert["name"]] = metrics
            print(f"Results for {expert['name']}: {metrics}")
        except Exception as e:
            print(f"❌ Failed to evaluate {expert['name']}: {e}")
            results[expert["name"]] = {
                "Recall@1": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0,
                "error": str(e),
                "generated_samples": [],
                "questions_per_chunk": expert.get("questions_per_chunk"),
                "generator_type": expert.get("generator_type"),
            }

    generate_report(results, samples)

def run_expert_comparison(samples):
    experts = [
        {"name": "Normal RAG", "path": None},
        {"name": "Base Model", "path": "google/flan-t5-large"},
        {"name": "Finetuned Model", "path": "output_model/output_model_server"}
    ]

    results = {}
    for expert in experts:
        try:
            metrics = evaluate_expert(
                expert["name"],
                expert["path"],
                samples,
                f"temp_db_{expert['name'].replace(' ', '_')}.json",
                use_existing_db=False
            )
            results[expert["name"]] = metrics
            print(f"Results for {expert['name']}: {metrics}")
        except Exception as e:
            print(f"❌ Failed to evaluate {expert['name']}: {e}")
            results[expert["name"]] = {
                "Recall@1": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0,
                "error": str(e),
                "generated_samples": []
            }

    generate_report(results, samples)

def main():
    print(f"DeepSeek API Key Present: {bool(os.getenv('DEEPSEEK_API'))}")
    mode = "api_vs_normal" if "api_vs_normal" in sys.argv else "expert_comparison"
    if mode == "api_vs_normal":
        total = int(os.getenv("NUM_SAMPLES", "100"))
        hotpot_n = int(os.getenv("HOTPOT_SAMPLES", str(max(1, total // 2))))
        squad_n = int(os.getenv("SQUAD_SAMPLES", str(max(1, total - hotpot_n))))
        hotpot_samples = load_hotpotqa_dataset(num_samples=hotpot_n)
        squad_samples = load_squad_dataset(num_samples=squad_n)
        samples = hotpot_samples + squad_samples
        if not samples:
            print("No samples loaded. Exiting.")
            return
        print(f"Loaded mixed dataset: HotpotQA={len(hotpot_samples)}, SQuAD={len(squad_samples)}, Total={len(samples)}")
    else:
        num_samples = int(os.getenv("NUM_SAMPLES", "150"))
        samples = load_hotpotqa_dataset(num_samples=num_samples)
        if not samples:
            print("No samples loaded. Exiting.")
            return

    if mode == "api_vs_normal":
        run_api_vs_normal(samples)
    else:
        run_expert_comparison(samples)

def generate_report(results, samples):
    # Separate metrics and samples
    metrics_data = {}
    samples_data = {}
    
    for expert, data in results.items():
        metrics_data[expert] = {k: v for k, v in data.items() if k not in ['generated_samples', 'error', 'by_source']}
        samples_data[expert] = data.get('generated_samples', [])

    df = pd.DataFrame(metrics_data).T
    print("\nFinal Results:")
    print(df)
    
    report_path = os.getenv("REPORT_PATH", "reports/expert_comparison_report.md")
    os.makedirs(os.path.dirname(report_path) or "reports", exist_ok=True)
    assets_dir = os.path.splitext(report_path)[0] + "_assets"
    os.makedirs(assets_dir, exist_ok=True)

    is_fusionrag_vs_normal = set(results.keys()) == {"Normal RAG", "智融检索（FusionRAG）"}
    title = "# 🧠 智融检索（FusionRAG） vs Normal RAG 对比报告\n\n" if is_fusionrag_vs_normal else "# 🧠 问题生成专家对比报告 (Expert Comparison Report)\n\n"
    subtitle = "本报告对比了 **智融检索（FusionRAG）** 与 **Normal RAG（仅文档向量）** 的检索性能。\n" if is_fusionrag_vs_normal else "本报告对比了在使用不同'问题生成专家'（Question Generation Experts）进行文档索引时，RAG系统的检索性能和生成质量。\n"

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    def _df_numeric(df_in: pd.DataFrame) -> pd.DataFrame:
        return df_in.map(_safe_float)

    def _save_recall_plot(df_in: pd.DataFrame, filename: str):
        cols = [c for c in ["Recall@1", "Recall@5", "Recall@10"] if c in df_in.columns]
        if not cols:
            return None

        plot_df = _df_numeric(df_in[cols].copy())
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        x = np.arange(len(cols))
        labels = list(plot_df.index)
        width = 0.45 / max(1, len(labels))
        for i, name in enumerate(labels):
            ax.bar(x + (i - (len(labels) - 1) / 2) * width, plot_df.loc[name].values, width=width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(cols)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Recall@K Comparison")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(assets_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def _save_mrr_plot(df_in: pd.DataFrame, filename: str):
        if "MRR" not in df_in.columns:
            return None
        plot_df = _df_numeric(df_in[["MRR"]].copy())
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        labels = list(plot_df.index)
        ax.bar(labels, plot_df["MRR"].values, width=0.4)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("MRR")
        ax.set_title("MRR Comparison")
        fig.tight_layout()
        out_path = os.path.join(assets_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    recall_img = _save_recall_plot(df, "recall_overall.png")
    mrr_img = _save_mrr_plot(df, "mrr_overall.png")
    assets_rel = os.path.basename(assets_dir)

    by_source_tables = {}
    for expert, data in results.items():
        by_source = data.get("by_source", {})
        for source_name in by_source.keys():
            if source_name not in by_source_tables:
                by_source_tables[source_name] = {}
            by_source_tables[source_name][expert] = by_source[source_name]

    source_dfs = {}
    for source_name, expert_map in by_source_tables.items():
        source_dfs[source_name] = pd.DataFrame(expert_map).T

    hotpot_df = source_dfs.get("hotpot_qa")
    squad_df = source_dfs.get("squad")

    hotpot_recall_img = _save_recall_plot(hotpot_df, "recall_hotpot.png") if hotpot_df is not None else None
    hotpot_mrr_img = _save_mrr_plot(hotpot_df, "mrr_hotpot.png") if hotpot_df is not None else None
    squad_recall_img = _save_recall_plot(squad_df, "recall_squad.png") if squad_df is not None else None
    squad_mrr_img = _save_mrr_plot(squad_df, "mrr_squad.png") if squad_df is not None else None
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(title)
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 摘要 (Executive Summary)\n")
        f.write(subtitle)
        if is_fusionrag_vs_normal:
            f.write("\n### 实验设置 (Settings)\n")
            f.write(f"- **样本数 (NUM_SAMPLES)**: {len(samples)}\n")
            f.write(f"- **API每块问题数 (API_Q_PER_CHUNK)**: {os.getenv('API_Q_PER_CHUNK', '3')}\n")
            f.write(f"- **每文档最多QGen块数 (MAX_QGEN_CHUNKS_PER_DOC)**: {os.getenv('MAX_QGEN_CHUNKS_PER_DOC', '0')}（0表示不限制）\n")
            f.write(f"- **评测分块参数 (EVAL_CHUNK_SIZE/EVAL_CHUNK_OVERLAP)**: {os.getenv('EVAL_CHUNK_SIZE', 'default')}/{os.getenv('EVAL_CHUNK_OVERLAP', 'default')}\n")
            f.write("- **对照组**: Normal RAG（questions_per_chunk=0，Doc-only）\n")
            f.write("- **实验组**: 智融检索（FusionRAG）（questions_per_chunk=API_Q_PER_CHUNK，QGen-enhanced）\n")
        
        # Dataset Statistics
        hotpot_count = len([s for s in samples if s.get('source') == 'hotpot_qa'])
        squad_count = len([s for s in samples if s.get('source') == 'squad'])
        f.write("### 数据集统计 (Dataset Statistics)\n")
        f.write(f"- **总样本数**: {len(samples)}\n")
        f.write(f"- **HotpotQA**: {hotpot_count}\n")
        f.write(f"- **SQuAD**: {squad_count}\n\n")
        
        f.write("## 2. 技术实现细节 (Technical Implementation Details)\n")
        f.write("本评估使用了以下核心算法和策略：\n\n")
        
        f.write("### 2.1 增强检索策略 (Enhanced Retrieval Strategy)\n")
        f.write("- **算法**: 加权融合 (Weighted Fusion)\n")
        f.write("- **公式**: `Final Score = 0.7 * Doc_Score + 0.3 * Max_Question_Score`\n")
        f.write("- **说明**: 结合文档本身的语义相似度和生成问题的最大语义相似度。该策略在 `SmartRetriever._enhanced_retrieval` 中实现。\n\n")
        
        f.write("### 2.2 重排序机制 (Reranking Mechanism)\n")
        f.write("- **算法**: 字符覆盖率微调 (Character Coverage Reranking)\n")
        f.write("- **公式**: `New Score = Original_Score * 0.8 + Coverage * 0.2`\n")
        f.write("- **说明**: 在语义检索的基础上，引入查询词在文档中的字符覆盖率作为辅助信号，提升精确匹配能力。仅对原始分数 > 0.6 的候选结果应用。\n\n")
        
        f.write("### 2.3 生成问题优化 (Question Optimization)\n")
        f.write("- **算法**: 贪心语义聚类 (Greedy Semantic Clustering)\n")
        f.write("- **说明**: 在生成多个问题后，计算问题间的余弦相似度矩阵。按问题长度（信息量）降序排列，去除与已保留问题相似度 > 0.90 的冗余问题。该逻辑在 `DocumentManager._optimize_generated_questions` 中实现。\n\n")
        
        f.write("### 2.4 关于 RRF (Reciprocal Rank Fusion)\n")
        f.write("- **说明**: RRF 算法（倒排秩融合）已在系统 CLI 的 `QueryOptimizer` 中实现，用于多路查询扩展（Query Expansion）的融合。但在本次单一查询评估中，我们主要关注底层的向量表示和加权融合效果，未启用多路扩展。\n\n")

        f.write("## 3. 性能指标 (Performance Metrics)\n")
        f.write("### 3.1 Overall（混合数据集整体）\n")
        f.write(df.to_markdown())
        f.write("\n\n## 3.1 指标对比图 (Plots)\n")
        if recall_img:
            f.write(f"![Recall Overall]({assets_rel}/recall_overall.png)\n\n")
        if mrr_img:
            f.write(f"![MRR Overall]({assets_rel}/mrr_overall.png)\n\n")

        if hotpot_df is not None:
            f.write("\n## 3.2 HotpotQA（多跳）指标\n")
            f.write(hotpot_df.to_markdown())
            f.write("\n\n")
            if hotpot_recall_img:
                f.write(f"![Recall HotpotQA]({assets_rel}/recall_hotpot.png)\n\n")
            if hotpot_mrr_img:
                f.write(f"![MRR HotpotQA]({assets_rel}/mrr_hotpot.png)\n\n")

        if squad_df is not None:
            f.write("\n## 3.3 SQuAD（单跳）指标\n")
            f.write(squad_df.to_markdown())
            f.write("\n\n")
            if squad_recall_img:
                f.write(f"![Recall SQuAD]({assets_rel}/recall_squad.png)\n\n")
            if squad_mrr_img:
                f.write(f"![MRR SQuAD]({assets_rel}/mrr_squad.png)\n\n")
        
        f.write("\n\n## 4. 生成问题分析 (Generated Question Analysis)\n")
        f.write("以下是各专家针对相同内容片段生成的问题示例。\n")
        
        for i in range(3): # Show up to 3 examples
            f.write(f"\n### 示例 {i+1}\n")
            
            # Find an expert with samples to use as reference for content snippet
            reference_expert = None
            for expert in results.keys():
                if samples_data[expert] and i < len(samples_data[expert]):
                    reference_expert = expert
                    break
            
            if reference_expert:
                snippet = samples_data[reference_expert][i]['content_snippet']
                f.write(f"**原文片段**: *{snippet}*\n\n")
                
                # Find matching ground truth question
                ground_truth_q = "N/A"
                ground_truth_ans = "N/A"
                # Remove ellipsis from snippet for matching
                clean_snippet = snippet.replace("...", "").strip()
                if len(clean_snippet) > 20:
                     for s in samples:
                         # Use 'content' instead of 'context'
                         if clean_snippet in s['content'] or s['content'] in clean_snippet:
                             ground_truth_q = s['question']
                             ground_truth_ans = s.get('answer', 'N/A')
                             break
                
                f.write(f"**标准参考问题 (Standard Ground Truth)**: *{ground_truth_q}*\n")
                f.write(f"**标准答案 (Standard Answer)**: *{ground_truth_ans}*\n\n")

                f.write("| 专家 (Expert) | 生成的问题 (Generated Questions) |\n")
                f.write("| :--- | :--- |\n")
                
                for expert in results.keys():
                    if samples_data[expert] and i < len(samples_data[expert]):
                        questions = samples_data[expert][i]['questions']
                        q_list = "<br>".join([f"- {q}" for q in questions])
                        f.write(f"| **{expert}** | {q_list} |\n")
                    else:
                        if results[expert].get("generator_type") == "none":
                            error_msg = "Doc-only（未启用问题生成）"
                        else:
                            error_msg = results[expert].get('error', 'No samples generated (Unknown error)')
                        # Highlight known errors
                        if "SafetensorError" in str(error_msg) or "MetadataIncompleteBuffer" in str(error_msg):
                            error_msg += "<br>**(Checkpoints files corrupted/incomplete)**"
                        f.write(f"| **{expert}** | ❌ *{error_msg}* |\n")
                f.write("\n")
            else:
                f.write("*没有可用的示例数据 (No sample data available)*\n")

        f.write("\n## 5. 分析结论 (Analysis)\n")
        if is_fusionrag_vs_normal and not df.empty:
            try:
                r1_gap = float(df.loc["智融检索（FusionRAG）", "Recall@1"]) - float(df.loc["Normal RAG", "Recall@1"])
                mrr_gap = float(df.loc["智融检索（FusionRAG）", "MRR"]) - float(df.loc["Normal RAG", "MRR"])
                f.write(f"- Overall：FusionRAG 相对 Normal RAG 的 **Recall@1** 变化为 {r1_gap:+.4f}，**MRR** 变化为 {mrr_gap:+.4f}。\n")
            except Exception:
                f.write("- Overall：指标差异计算失败，请检查表格数据。\n")
            if hotpot_df is not None:
                try:
                    r1_gap = float(hotpot_df.loc["智融检索（FusionRAG）", "Recall@1"]) - float(hotpot_df.loc["Normal RAG", "Recall@1"])
                    mrr_gap = float(hotpot_df.loc["智融检索（FusionRAG）", "MRR"]) - float(hotpot_df.loc["Normal RAG", "MRR"])
                    f.write(f"- HotpotQA：FusionRAG 相对 Normal RAG 的 **Recall@1** 变化为 {r1_gap:+.4f}，**MRR** 变化为 {mrr_gap:+.4f}。\n")
                except Exception:
                    f.write("- HotpotQA：指标差异计算失败。\n")
            if squad_df is not None:
                try:
                    r1_gap = float(squad_df.loc["智融检索（FusionRAG）", "Recall@1"]) - float(squad_df.loc["Normal RAG", "Recall@1"])
                    mrr_gap = float(squad_df.loc["智融检索（FusionRAG）", "MRR"]) - float(squad_df.loc["Normal RAG", "MRR"])
                    f.write(f"- SQuAD：FusionRAG 相对 Normal RAG 的 **Recall@1** 变化为 {r1_gap:+.4f}，**MRR** 变化为 {mrr_gap:+.4f}。\n")
                except Exception:
                    f.write("- SQuAD：指标差异计算失败。\n")
        elif not df.empty and 'Recall@5' in df.columns:
            best_expert = df['Recall@5'].astype(float).idxmax()
            f.write(f"基于 **Recall@5** 指标，表现最好的专家是 **{best_expert}**。\n")
        else:
            f.write("无法计算分析结论，请检查评估结果。\n")
        
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()
