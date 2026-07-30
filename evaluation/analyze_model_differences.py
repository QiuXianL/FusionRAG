import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_rank(model, query, db_data, target_doc_name):
    # 1. Encode query
    query_with_prefix = f"query: {query}"
    query_embedding = model.encode([query_with_prefix], normalize_embeddings=True)[0]
    
    # 2. Get DB embeddings
    embeddings = np.array(db_data['embeddings'])
    vector_to_chunk_map = db_data['vector_to_chunk_map']
    vector_types = db_data['vector_types']
    documents = db_data['documents']
    
    # 3. Calculate similarities
    similarities = np.dot(embeddings, query_embedding)
    
    # 4. Aggregating scores per chunk (Weighted Fusion Logic)
    chunk_scores = {} 
    
    for i, score in enumerate(similarities):
        chunk_idx = vector_to_chunk_map[i]
        v_type = vector_types[i]
        
        if chunk_idx not in chunk_scores:
            chunk_scores[chunk_idx] = {'doc_score': 0.0, 'max_q_score': 0.0}
            
        if v_type == 'document':
            chunk_scores[chunk_idx]['doc_score'] = float(score)
        elif v_type == 'question':
            if score > chunk_scores[chunk_idx]['max_q_score']:
                chunk_scores[chunk_idx]['max_q_score'] = float(score)
    
    # 5. Final Score
    alpha = 0.7
    final_results = []
    for chunk_idx, scores in chunk_scores.items():
        doc_score = scores['doc_score']
        max_q_score = scores['max_q_score']
        if max_q_score == 0.0 and doc_score > 0.5:
             max_q_score = doc_score * 0.9
        
        final_score = alpha * doc_score + (1 - alpha) * max_q_score
        
        # Determine document name/id for matching
        # In our eval script, we used doc_{i} as the name, but we need to match it to the sample index
        # The db_data doesn't store the "doc_name" explicitly in a simple list, 
        # but the `documents` list is ordered by insertion.
        # Assuming the DB was built sequentially from samples 0 to N.
        
        final_results.append({
            'chunk_idx': chunk_idx,
            'score': final_score,
            'doc_score': doc_score,
            'max_q_score': max_q_score
        })
    
    # Sort
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Find rank
    rank = -1
    for r, res in enumerate(final_results):
        # We assume chunk_idx corresponds to the sample index 
        # (since we indexed 1 doc per sample, and they are usually 1 chunk long, 
        # but wait, splitting might produce multiple chunks per doc)
        # To be safe, we need to know which chunks belong to the target doc.
        # In the evaluation script: doc_manager.add_document_from_text(..., document_name=f"doc_{i}")
        # The DB stores `document_sources`? Let's check db structure if possible.
        # Or simpler: check if the text matches.
        
        # Let's rely on text matching for absolute certainty
        chunk_text = documents[res['chunk_idx']]
        # The target_doc_name is "doc_X". 
        # But we don't have the map here easily without parsing `document_sources` if it exists.
        
        # Fallback: We pass the expected content text
        if target_doc_name in chunk_text or chunk_text in target_doc_name: 
             rank = r
             break
             
    return rank, final_results

def main():
    print("Loading embedding model...")
    model = SentenceTransformer('models/e5-base-v2')
    
    print("Loading datasets...")
    samples = load_json('evaluation/hotpot_40_samples.json')
    base_db = load_json('temp_db_Base_Model.json')
    finetuned_db = load_json('temp_db_Finetuned_Model.json')
    
    print("Analyzing differences...")
    
    # Store interesting cases
    # (sample_idx, base_rank, ft_rank, sample)
    diffs = []
    
    for i, sample in tqdm(enumerate(samples), total=len(samples)):
        query = sample['question']
        content = sample['content']
        
        # We use content matching because we know the content is unique enough
        base_rank, _ = get_rank(model, query, base_db, content)
        ft_rank, _ = get_rank(model, query, finetuned_db, content)
        
        diffs.append({
            'id': i,
            'query': query,
            'base_rank': base_rank,
            'ft_rank': ft_rank,
            'content': content
        })

    # Find cases where Base is much better than Finetuned (Base Rank 0, FT Rank > 0)
    base_wins = [d for d in diffs if d['base_rank'] == 0 and d['ft_rank'] > 0]
    
    # Find cases where Finetuned is much better than Base
    ft_wins = [d for d in diffs if d['ft_rank'] == 0 and d['base_rank'] > 0]
    
    print(f"\nTotal Samples: {len(samples)}")
    print(f"Base Wins (Base@1 vs FT@>1): {len(base_wins)}")
    print(f"Finetuned Wins (FT@1 vs Base@>1): {len(ft_wins)}")
    
    # Extract generated questions for these cases
    # We need to map chunks to generated questions
    def get_generated_questions(db_data, content_snippet):
        # Find chunk index containing content
        chunk_idx = -1
        for idx, text in enumerate(db_data['documents']):
            if content_snippet in text or text in content_snippet:
                chunk_idx = idx
                break
        
        if chunk_idx == -1: return []
        
        # Find questions for this chunk
        questions = []
        for i, v_idx in enumerate(db_data['vector_to_chunk_map']):
            if v_idx == chunk_idx and db_data['vector_types'][i] == 'question':
                # We need to find the question text index.
                # The db structure: 'questions' list is parallel to 'vector_types' where type is question?
                # No, usually 'questions' list contains ALL generated questions in order.
                # We need to count how many questions appeared before this one.
                
                # Let's count 'question' types before this index `i`
                q_count = 0
                for j in range(i):
                    if db_data['vector_types'][j] == 'question':
                        q_count += 1
                
                if q_count < len(db_data['questions']):
                    questions.append(db_data['questions'][q_count])
        return questions

    print("\n" + "="*80)
    print("CASE ANALYSIS: Why Base Model Won?")
    print("="*80)
    for case in base_wins[:3]:
        print(f"\n[Sample {case['id']}]")
        print(f"Query: {case['query']}")
        print(f"Content (first 200): {case['content'][:200]}...")
        print(f"Ranks: Base={case['base_rank']}, FT={case['ft_rank']}")
        
        base_qs = get_generated_questions(base_db, case['content'])
        ft_qs = get_generated_questions(finetuned_db, case['content'])
        
        print(f"Base Gen Qs: {base_qs}")
        print(f"FT Gen Qs:   {ft_qs}")
        print("-" * 40)

    print("\n" + "="*80)
    print("CASE ANALYSIS: Why Finetuned Model Won?")
    print("="*80)
    for case in ft_wins[:3]:
        print(f"\n[Sample {case['id']}]")
        print(f"Query: {case['query']}")
        print(f"Content (first 200): {case['content'][:200]}...")
        print(f"Ranks: Base={case['base_rank']}, FT={case['ft_rank']}")
        
        base_qs = get_generated_questions(base_db, case['content'])
        ft_qs = get_generated_questions(finetuned_db, case['content'])
        
        print(f"Base Gen Qs: {base_qs}")
        print(f"FT Gen Qs:   {ft_qs}")
        print("-" * 40)

if __name__ == "__main__":
    main()
