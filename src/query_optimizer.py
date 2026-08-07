import os
from typing import List, Dict, Any
from openai import OpenAI

class QueryOptimizer:
    """
    Super Brain Query Optimizer
    Implements:
    1. Query Expansion (CoT based)
    2. Reciprocal Rank Fusion (RRF)
    """
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API")
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
            except Exception as e:
                print(f"⚠️ QueryOptimizer init failed: {e}")

    def expand_query(self, query: str) -> List[str]:
        """
        Generate variations of the query using Chain of Thought.
        Returns original query + 3 variations.
        """
        if not self.client:
            return [query]
            
        try:
            # CoT Prompt for Query Expansion
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": """
                    You are an expert search query optimizer.
                    Your goal is to maximize the retrieval recall by generating diverse search queries based on the user's input.
                    
                    Step 1: Analyze the user's intent and potential ambiguity.
                    Step 2: Generate 3 distinct search queries:
                       - Query 1: A specific, detailed version of the original query.
                       - Query 2: A generalized or conceptual version (using synonyms).
                       - Query 3: A related question that implies the answer to the original query.
                    
                    Output ONLY the 3 queries, one per line. Do not number them.
                    """},
                    {"role": "user", "content": f"User Query: {query}"}
                ],
                temperature=0.5,
                max_tokens=200,
                timeout=30,
                stream=False
            )
            
            content = response.choices[0].message.content.strip()
            variations = [q.strip() for q in content.split('\n') if q.strip()]
            
            # Ensure we don't have empty strings and limit to 3
            variations = [v for v in variations if len(v) > 2][:3]
            
            # Return original + variations (unique)
            all_queries = [query] + variations
            return list(dict.fromkeys(all_queries)) # Remove duplicates while preserving order
            
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return [query]

    def fuse_results(self, results_dict: Dict[str, List[Dict]], original_query: str = None, k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) with Weighting
        results_dict: {query_string: [results]}
        original_query: The user's original query string (to give it higher weight)
        """
        fused_scores = {} # doc_content -> score
        doc_map = {} # doc_content -> full_result_object
        
        # Weight configuration
        ORIGINAL_QUERY_WEIGHT = 3.0
        
        for query, results in results_dict.items():
            is_original = (query == original_query) if original_query else False
            weight = ORIGINAL_QUERY_WEIGHT if is_original else 1.0
            
            for rank, result in enumerate(results):
                doc_content = result['original_text']
                
                # Store the result object if not seen
                if doc_content not in doc_map:
                    doc_map[doc_content] = result
                
                # RRF formula: weight * (1 / (k + rank))
                score = weight * (1.0 / (k + rank + 1))
                
                if doc_content in fused_scores:
                    fused_scores[doc_content] += score
                else:
                    fused_scores[doc_content] = score
                    
        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format output
        final_results = []
        for doc_content, score in sorted_docs:
            result = doc_map[doc_content].copy()
            result['similarity_score'] = score # Update score to RRF score
            result['match_type'] = 'fused_multi_query'
            final_results.append(result)
            
        return final_results
