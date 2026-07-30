import json
import os
import numpy as np
import torch
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SQUAD_PATH = os.path.join(BASE_DIR, "squad_data.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "e5-base-v2")

# Mock Clustering Logic (Extracted from DocumentManager for isolation)
def cluster_questions(questions: List[str], embeddings: np.ndarray, threshold=0.85) -> List[str]:
    if not questions or len(questions) < 2:
        return questions
        
    try:
        # Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norm + 1e-10)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Sort questions by length (descending) to prefer longer, more detailed questions
        indexed_questions = [(i, len(q)) for i, q in enumerate(questions)]
        sorted_indices = [x[0] for x in sorted(indexed_questions, key=lambda x: x[1], reverse=True)]
        
        keep_indices = []
        removed_indices = set()
        
        for i in sorted_indices:
            if i in removed_indices:
                continue
            
            keep_indices.append(i)
            
            # Check against other questions
            for j in sorted_indices:
                if j != i and j not in removed_indices:
                    if similarity_matrix[i][j] > threshold:
                        removed_indices.add(j)
        
        return [questions[i] for i in keep_indices]
        
    except Exception as e:
        print(f"Clustering error: {e}")
        return questions

class ClusteringEvaluator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load Model
        if os.path.exists(MODEL_PATH):
            print(f"Loading local model from {MODEL_PATH}")
            self.model = SentenceTransformer(MODEL_PATH, device=self.device)
        else:
            print(f"Local model not found, loading from HuggingFace")
            self.model = SentenceTransformer("intfloat/e5-base-v2", device=self.device)

    def load_data(self):
        with open(SQUAD_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Group questions by document
        docs_map = {}
        for item in data:
            doc = item['document']
            if doc not in docs_map:
                docs_map[doc] = []
            docs_map[doc].append(item['question'])
            
        return docs_map

    def run_experiment(self):
        report_lines = []
        def log(msg=""):
            print(msg)
            report_lines.append(msg)

        docs_map = self.load_data()
        log(f"Loaded {len(docs_map)} documents for evaluation.")
        
        thresholds = [0.85, 0.90]
        results = {t: {'total_retained': 0, 'total_reduction': 0, 'docs_reduced': 0} for t in thresholds}
        total_questions_start = 0

        log("\n--- Starting Comparative Clustering Evaluation ---\n")
        
        # Limit to first 100 docs for faster iteration if needed, or run all
        # For this report, let's run all but print less details
        processed_count = 0
        
        for i, (doc, questions) in enumerate(docs_map.items()):
            processed_count += 1
            augmented_questions = questions + questions[:2] # Add 2 duplicates
            total_questions_start += len(augmented_questions)
            
            # 1. Encode questions once
            embeddings = self.model.encode(augmented_questions, normalize_embeddings=True)
            
            # 2. Test both thresholds
            for t in thresholds:
                optimized_questions = cluster_questions(augmented_questions, embeddings, threshold=t)
                
                after_count = len(optimized_questions)
                reduction = len(augmented_questions) - after_count
                
                results[t]['total_retained'] += after_count
                results[t]['total_reduction'] += reduction
                if reduction > 0:
                    results[t]['docs_reduced'] += 1
            
            # Print sample for the first few docs to show difference
            if i < 5:
                log(f"Doc {i+1} ({len(augmented_questions)} qs):")
                for t in thresholds:
                    retained = results[t]['total_retained'] # This is cumulative, careful
                    # Recalculate just for this doc for printing
                    opt_q = cluster_questions(augmented_questions, embeddings, threshold=t)
                    log(f"  Thresh {t}: -> {len(opt_q)} qs (-{len(augmented_questions)-len(opt_q)})")

        log("\n--- Final Comparative Results ---")
        log(f"Total Initial Questions: {total_questions_start}")
        log(f"{'Threshold':<10} | {'Retained':<10} | {'Reduction':<10} | {'Reduction %':<12} | {'Docs Triggered':<15}")
        log("-" * 70)
        
        for t in thresholds:
            retained = results[t]['total_retained']
            reduction = results[t]['total_reduction']
            red_pct = (reduction / total_questions_start) * 100
            docs_trig = results[t]['docs_reduced']
            log(f"{t:<10} | {retained:<10} | {reduction:<10} | {red_pct:<11.1f}% | {docs_trig:<15}")
            
        log("\nDone.")

        # Save report
        report_path = os.path.join(PROJECT_ROOT, "reports", "clustering_impact_report.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    evaluator = ClusteringEvaluator()
    evaluator.run_experiment()
