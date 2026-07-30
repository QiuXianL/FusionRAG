import sys
import os
import json
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.smart_retrieval import SmartRetriever

# Helper to load dataset
def load_hotpotqa_dataset(num_samples=150):
    path = f"evaluation/hotpot_{num_samples}_samples.json"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {path}")
    return data[:num_samples]

def evaluate_expert_from_db(expert_name, db_path, samples):
    print(f"\n==========================================")
    print(f"Evaluating Expert: {expert_name}")
    print(f"DB Path: {db_path}")
    
    if not os.path.exists(db_path):
        print("❌ DB not found. Skipping.")
        return None
        
    try:
        # Check DB content
        with open(db_path, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
        
        doc_sources = db_data.get('document_sources', [])
        num_docs = len(doc_sources)
        print(f"📚 DB contains {num_docs} documents (from document_sources).")
        
        # Initialize Retriever
        retriever = SmartRetriever(db_path=db_path)
        
        correct_1 = 0
        correct_5 = 0
        correct_10 = 0
        mrr_sum = 0
        
        # Determine how many samples we can actually evaluate
        print(f"Running retrieval test on {len(samples)} samples...")
        
        evaluated_count = 0
        
        for i, sample in tqdm(enumerate(samples), total=len(samples)):
            query = sample['question']
            target_doc_name = f"doc_{i}"
            
            # Retrieve
            try:
                results = retriever.retrieve_with_strategy(query, top_k=10)
            except Exception as e:
                # Might happen if DB is empty or corrupted
                results = []
            
            # Check correctness
            found_rank = -1
            for rank, res in enumerate(results):
                if res.get('document_name') == target_doc_name:
                    found_rank = rank
                    break
            
            if found_rank != -1:
                if found_rank < 1: correct_1 += 1
                if found_rank < 5: correct_5 += 1
                if found_rank < 10: correct_10 += 1
                mrr_sum += 1.0 / (found_rank + 1)
            
            evaluated_count += 1
            
        # Calculate metrics
        metrics = {
            "Recall@1": correct_1 / len(samples),
            "Recall@5": correct_5 / len(samples),
            "Recall@10": correct_10 / len(samples),
            "MRR": mrr_sum / len(samples),
            "Indexed_Docs": num_docs
        }
        
        print(f"Results for {expert_name}:")
        print(f"  Recall@1: {metrics['Recall@1']:.4f}")
        print(f"  Recall@5: {metrics['Recall@5']:.4f}")
        print(f"  Recall@10: {metrics['Recall@10']:.4f}")
        print(f"  MRR: {metrics['MRR']:.4f}")
        print(f"  (Based on {num_docs} indexed docs vs {len(samples)} target samples)")
        
        return metrics

    except Exception as e:
        print(f"❌ Error evaluating {expert_name}: {e}")
        return None

def main():
    samples = load_hotpotqa_dataset(num_samples=150)
    if not samples:
        return

    experts = [
        {"name": "Normal RAG", "file": "temp_db_Normal_RAG.json"},
        {"name": "Base Model", "file": "temp_db_Base_Model.json"},
        {"name": "Finetuned Model", "file": "temp_db_Finetuned_Model.json"}
    ]
    
    results = {}
    
    for expert in experts:
        metrics = evaluate_expert_from_db(expert["name"], expert["file"], samples)
        if metrics:
            results[expert["name"]] = metrics
            
    # Print summary
    print("\n\n================ SUMMARY ================")
    print(f"{'Expert':<20} | {'R@1':<8} | {'R@5':<8} | {'MRR':<8} | {'Docs'}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<20} | {m['Recall@1']:.4f}   | {m['Recall@5']:.4f}   | {m['MRR']:.4f}   | {m['Indexed_Docs']}")

if __name__ == "__main__":
    main()
