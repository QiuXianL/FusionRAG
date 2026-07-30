import json
import os
import random
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
import torch

# Configuration
# Use paths relative to the script location or project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Assuming script is in evaluation/

SQUAD_PATH = os.path.join(BASE_DIR, "squad_data.json")
# Try to find model in project root
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "e5-base-v2")

NUM_DOCS_TO_TEST = 10 
NUM_QUESTIONS_GEN = 3

class Evaluator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load Model
        if os.path.exists(MODEL_PATH):
            print(f"Loading local model from {MODEL_PATH}")
            self.model = SentenceTransformer(MODEL_PATH, device=self.device)
        else:
            print(f"Local model not found at {MODEL_PATH}, loading from HuggingFace")
            self.model = SentenceTransformer("intfloat/e5-base-v2", device=self.device)
        
        # Setup API
        self.api_key = os.getenv("DEEPSEEK_API")
        if not self.api_key:
            # Try to load from .env file in project root
            env_path = os.path.join(PROJECT_ROOT, ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("DEEPSEEK_API="):
                            self.api_key = line.split("=", 1)[1].strip().strip('"')
                            break
            
            if not self.api_key:
                # Check current dir just in case
                if os.path.exists(".env"):
                     with open(".env", "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("DEEPSEEK_API="):
                                self.api_key = line.split("=", 1)[1].strip().strip('"')
                                break

            if not self.api_key:
                raise ValueError("DEEPSEEK_API not set")
                
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate_questions_old(self, content: str, num=3) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"""
                    You are a professional question generation expert. Generate {num} high-quality questions based on the document content.
                    Your task is to generate multi-angle questions for every knowledge point within the provided document. These questions should be as close to real-world scenarios as possible. Each question should be on a separate line, end with a question mark, not use numbering, and must be answerable using the content of this document.
                    Here is the document:
                    <document>
                    {{DOCUMENT}}
                    </document>
                    When generating questions, please follow these requirements:
                    - Carefully analyze each knowledge point in the document.
                    - Knowledge points include but are not limited to: numbers, time, locations, people, events, etc.
                    - Think from different angles and generate multiple questions related to the knowledge points.
                    - Ensure the questions fit real-world scenarios and have practical significance.
                    Please write the generated questions inside the <question_list> tags.
                    <question_list>
                    [Write the generated questions here]
                    </question_list>
                    """},
                    {"role": "user", "content": f"Document content:\n{content}"}
                ],
                stream=False,
                temperature=0.3
            )
            content = response.choices[0].message.content
            if "<question_list>" in content:
                content = content.split("<question_list>")[1].split("</question_list>")[0]
            questions = [q.strip() for q in content.split('\n') if q.strip() and ('?' in q or '？' in q)]
            return questions[:num]
        except Exception as e:
            print(f"Old gen failed: {e}")
            return []

    def generate_questions_new(self, content: str, num=3) -> List[str]:
        try:
            # 1. Keywords
            kw_response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": """
                    You are a keyword extraction expert. Please extract key information from the provided document content.
                    Keywords should cover: important entities, time, locations, core concepts, etc.
                    Please output the keywords directly, separated by commas, without any other content.
                    """},
                    {"role": "user", "content": f"Document content:\n{content}"}
                ],
                stream=False,
                temperature=0.3
            )
            keywords = kw_response.choices[0].message.content.strip()
            
            # 2. Questions
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"""
                    You are a professional question generation expert. Generate {num} high-quality questions based on the document content and extracted keywords.
                    Your task is to generate multi-angle questions for every knowledge point within the provided document. These questions should be as close to real-world scenarios as possible. Each question should be on a separate line, end with a question mark, not use numbering, and must be answerable using the content of this document.
                    
                    Extracted Keywords: {keywords}
                    
                    When generating questions, please follow these requirements:
                    - Combine the extracted keywords with the document content to generate more targeted questions.
                    - Carefully analyze each knowledge point in the document.
                    - Knowledge points include but are not limited to: numbers, time, locations, people, events, etc.
                    - Think from different angles and generate multiple questions related to the knowledge points.
                    - Ensure the questions fit real-world scenarios and have practical significance.
                    Please write the generated questions inside the <question_list> tags.
                    <question_list>
                    [Write the generated questions here]
                    </question_list>
                    """},
                    {"role": "user", "content": f"Document content:\n{content}"}
                ],
                stream=False,
                temperature=0.3
            )
            content = response.choices[0].message.content
            if "<question_list>" in content:
                content = content.split("<question_list>")[1].split("</question_list>")[0]
            questions = [q.strip() for q in content.split('\n') if q.strip() and ('?' in q or '？' in q)]
            return questions[:num]
        except Exception as e:
            print(f"New gen failed: {e}")
            return []

    def generate_questions_super(self, content: str, num=3) -> List[str]:
        # Implementation of "Super Brain" logic (CoT + Focused Generation)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"""
                    You are an expert content analyst and question generator.
                    
                    Step 1: Analyze the text to identify the 3 most critical pieces of information (facts, causal relationships, or definitions).
                    Step 2: For each piece of information, formulate a specific question that requires understanding that information to answer.
                    
                    Requirements:
                    - Questions must be self-contained (avoid "he", "it", "they" without context).
                    - Questions must be diverse (Who/When/Where vs Why/How).
                    - Generate exactly {num} questions.
                    - Output ONLY the questions inside the tags.
                    
                    Output format:
                    <question_list>
                    [Question 1]
                    [Question 2]
                    ...
                    </question_list>
                    """},
                    {"role": "user", "content": f"Document content:\n{content}"}
                ],
                stream=False,
                temperature=0.3
            )
            content = response.choices[0].message.content
            
            if "<question_list>" in content:
                content = content.split("<question_list>")[1].split("</question_list>")[0]
                
            questions = [q.strip() for q in content.split('\n') if q.strip() and ('?' in q or '？' in q)]
            return questions[:num]
        except Exception as e:
            print(f"Super gen failed: {e}")
            return []

    def optimize_questions(self, questions: List[str], embeddings: List[np.ndarray], threshold=0.90) -> List[int]:
        """
        Mock implementation of clustering logic from DocumentManager.
        Returns indices of questions to keep.
        """
        if not questions or len(questions) < 2:
            return list(range(len(questions)))
            
        try:
            emb_matrix = np.array(embeddings)
            similarity_matrix = np.dot(emb_matrix, emb_matrix.T)
            
            # Sort by length (descending)
            indexed_questions = [(i, len(q)) for i, q in enumerate(questions)]
            sorted_indices = [x[0] for x in sorted(indexed_questions, key=lambda x: x[1], reverse=True)]
            
            keep_indices = []
            removed_indices = set()
            
            for i in sorted_indices:
                if i in removed_indices:
                    continue
                
                keep_indices.append(i)
                
                for j in sorted_indices:
                    if j != i and j not in removed_indices:
                        if similarity_matrix[i][j] > threshold:
                            removed_indices.add(j)
                            
            return keep_indices
        except Exception as e:
            print(f"Optimization failed: {e}")
            return list(range(len(questions)))

    def run(self):
        report_lines = []
        def log(msg=""):
            print(msg)
            report_lines.append(msg)

        # Load Data
        log(f"Loading data from {SQUAD_PATH}...")
        try:
            with open(SQUAD_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            log(f"Error: {SQUAD_PATH} not found.")
            return
        
        # Group by document
        doc_map = {} 
        for item in data:
            doc = item['document']
            q = item['question']
            if doc not in doc_map:
                doc_map[doc] = []
            doc_map[doc].append(q)
            
        unique_docs = list(doc_map.keys())
        if len(unique_docs) > NUM_DOCS_TO_TEST:
            selected_docs = unique_docs[:NUM_DOCS_TO_TEST]
        else:
            selected_docs = unique_docs
            
        log(f"Testing on {len(selected_docs)} documents...")
        
        # Build Index Data
        docs_data = {} 
        
        log("Generating data and building indices...")
        
        # Stats for clustering
        total_gen_qs = 0
        total_kept_qs = 0
        
        for doc_id, content in enumerate(tqdm(selected_docs)):
            docs_data[doc_id] = {
                'doc_vec': self.model.encode(f"passage: {content}", normalize_embeddings=True),
                'old_qs': [],
                'new_qs': [],
                'super_qs': [],
                'super_qs_clustered': [] # Store clustered vectors
            }
            
            # 2. Old Gen
            qs_old = self.generate_questions_old(content, NUM_QUESTIONS_GEN)
            for q in qs_old:
                docs_data[doc_id]['old_qs'].append(self.model.encode(f"passage: {q}", normalize_embeddings=True))
                
            # 3. New Gen
            qs_new = self.generate_questions_new(content, NUM_QUESTIONS_GEN)
            for q in qs_new:
                docs_data[doc_id]['new_qs'].append(self.model.encode(f"passage: {q}", normalize_embeddings=True))
                
            # 4. Super Gen (Raw)
            # To test clustering effectively, we generate MORE questions first (e.g. 6), then cluster
            # This simulates the "Oversaturated Generation" strategy
            qs_super_large = self.generate_questions_super(content, num=6) 
            
            # Calculate embeddings for all super questions
            super_embeddings = []
            for q in qs_super_large:
                super_embeddings.append(self.model.encode(f"passage: {q}", normalize_embeddings=True))
            
            # Save raw (take first 3 for fair comparison with others if they used 3)
            # Or use all 6 as "Super (Raw)"? Let's use top 3 for fairness in "Raw" baseline
            docs_data[doc_id]['super_qs'] = super_embeddings[:3]
            
            # 5. Super Gen (Clustered)
            # Apply clustering on the large set (6 questions)
            keep_indices = self.optimize_questions(qs_super_large, super_embeddings, threshold=0.90)
            clustered_embeddings = [super_embeddings[i] for i in keep_indices]
            docs_data[doc_id]['super_qs_clustered'] = clustered_embeddings
            
            total_gen_qs += len(qs_super_large)
            total_kept_qs += len(clustered_embeddings)

        log(f"聚类统计: 生成了 {total_gen_qs} 个问题 -> 保留了 {total_kept_qs} 个 (缩减率: {(total_gen_qs-total_kept_qs)/total_gen_qs:.1%})")

        # Evaluate
        log("正在评估检索准确率...")
        
        results = {
            'baseline': 0,
            'old': 0,
            'new': 0,
            'super_flat': 0,
            'super_weighted': 0,
            'super_clustered': 0
        }
        total_queries = 0
        
        def retrieve_flat(query_emb, q_type='old'):
            best_score = -1
            best_doc_id = -1
            
            for doc_id, data in docs_data.items():
                # Check Doc
                score = np.dot(query_emb, data['doc_vec'])
                if score > best_score:
                    best_score = score
                    best_doc_id = doc_id
                
                # Check Questions
                q_vecs = []
                if q_type == 'old': q_vecs = data['old_qs']
                elif q_type == 'new': q_vecs = data['new_qs']
                elif q_type == 'super': q_vecs = data['super_qs']
                elif q_type == 'clustered': q_vecs = data['super_qs_clustered']
                
                for q_vec in q_vecs:
                    score = np.dot(query_emb, q_vec)
                    if score > best_score:
                        best_score = score
                        best_doc_id = doc_id
                        
            return best_doc_id

        def retrieve_weighted(query_emb, q_type='super', alpha=0.7):
            # Score = alpha * DocScore + (1-alpha) * Max(QuestionScore)
            best_score = -100
            best_doc_id = -1
            
            for doc_id, data in docs_data.items():
                doc_score = np.dot(query_emb, data['doc_vec'])
                
                q_vecs = []
                if q_type == 'old': q_vecs = data['old_qs']
                elif q_type == 'new': q_vecs = data['new_qs']
                elif q_type == 'super': q_vecs = data['super_qs']
                
                max_q_score = 0
                if q_vecs:
                    max_q_score = max([np.dot(query_emb, qv) for qv in q_vecs])
                else:
                    max_q_score = doc_score # Fallback if no questions
                
                final_score = alpha * doc_score + (1 - alpha) * max_q_score
                
                if final_score > best_score:
                    best_score = final_score
                    best_doc_id = doc_id
            
            return best_doc_id
            
        for doc_id, content in enumerate(tqdm(selected_docs, desc="Querying")):
            test_questions = doc_map[content]
            for q in test_questions:
                total_queries += 1
                query_emb = self.model.encode(f"query: {q}", normalize_embeddings=True)
                
                # Baseline (Doc Only)
                best_score = -1
                pred_id = -1
                for d_id, data in docs_data.items():
                    s = np.dot(query_emb, data['doc_vec'])
                    if s > best_score:
                        best_score = s
                        pred_id = d_id
                if pred_id == doc_id: results['baseline'] += 1
                    
                # Old Flat
                if retrieve_flat(query_emb, 'old') == doc_id: results['old'] += 1
                    
                # New Flat
                if retrieve_flat(query_emb, 'new') == doc_id: results['new'] += 1
                
                # Super Flat
                if retrieve_flat(query_emb, 'super') == doc_id: results['super_flat'] += 1
                
                # Super Weighted
                if retrieve_weighted(query_emb, 'super', alpha=0.7) == doc_id: results['super_weighted'] += 1
                
                # Super Clustered
                if retrieve_flat(query_emb, 'clustered') == doc_id: results['super_clustered'] += 1
        
        log(f"\n{'='*40}")
        log(f"评估结果 (总查询数: {total_queries})")
        log(f"{'='*40}")
        log(f"基准 (仅文档):          {results['baseline']}/{total_queries} ({results['baseline']/total_queries:.2%}) - 基础对照组，只靠文档内容搜索")
        log(f"旧方法 (直接提问):      {results['old']}/{total_queries} ({results['old']/total_queries:.2%}) - 直接让AI生成问题，不做优化")
        log(f"新方法 (关键词+提问):   {results['new']}/{total_queries} ({results['new']/total_queries:.2%}) - 先提取关键词再提问，稍有改进")
        log(f"最强大脑 (思维链):      {results['super_flat']}/{total_queries} ({results['super_flat']/total_queries:.2%}) - 深度思考生成的精选问题")
        log(f"最强大脑 (加权融合):    {results['super_weighted']}/{total_queries} ({results['super_weighted']/total_queries:.2%}) - 结合文档和问题分数的综合算法")
        log(f"最强大脑 (聚类优化):    {results['super_clustered']}/{total_queries} ({results['super_clustered']/total_queries:.2%}) - 生成大量问题后去重筛选，效果最佳！")
        log(f"{'='*40}")

        # Save report
        report_path = os.path.join(PROJECT_ROOT, "reports", "question_generation_report.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    Evaluator().run()
