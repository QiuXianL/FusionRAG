import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
from reranker import Reranker, get_reranker

class SmartRetriever:
    """智能检索器，使用增强检索策略：
    - enhanced: 使用文档+问题向量检索
    - 支持 Cross-Encoder 重排序
    """

    def __init__(self, model_path=None, db_path=None, reranker: Optional[Reranker] = None):
        # 使用脚本所在目录的上级目录（项目根目录）来解析相对路径
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(_base, "models", "e5-base-v2")
        if db_path is None:
            db_path = os.path.join(_base, "documents", "document_vectors_enhanced.json")
        self.model_path = model_path
        self.db_path = db_path
        self.model = self._load_model()
        self.databases = {}
        self._load_databases()

        # 初始化重排序器
        if reranker is not None:
            self.reranker = reranker
        else:
            self.reranker = get_reranker()
    
    def _load_model(self):
        """加载模型"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if os.path.exists(self.model_path):
            print(f"💾 使用本地模型: {self.model_path}")
            return SentenceTransformer(self.model_path, device=device)
        else:
            print("⚠️ 本地模型不存在，从网络下载...")
            return SentenceTransformer("intfloat/e5-base-v2", device=device)
    
    def _load_databases(self):
        """加载增强向量数据库"""
        db_files = [
            ("enhanced", self.db_path),
        ]
        
        for db_name, file_path in db_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.databases[db_name] = json.load(f)
                    print(f"✅ 加载数据库: {db_name} ({file_path})")
                except Exception as e:
                    print(f"❌ 加载数据库失败: {db_name} - {e}")
    
    def retrieve_with_strategy(self, query: str, strategy: str = "auto", top_k: int = 3, rerank: bool = True, return_all: bool = False) -> List[Dict]:
        """使用增强检索策略进行检索"""
        # 所有策略都使用增强检索
        return self._enhanced_retrieval(query, top_k, rerank=rerank, return_all=return_all)
    
    def _choose_best_strategy(self, query: str) -> str:
        """智能选择最佳检索策略（现在只返回enhanced）"""
        return "enhanced"
    
    def rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        重排序 (Rerank) 机制
        使用 Cross-Encoder 模型对检索结果进行精确重排序。
        当本地模型不可用时，回退到 API 重排序或改进的关键词覆盖重排序。
        """
        if not results:
            return results

        try:
            # 使用新的 Reranker 进行重排序（不指定 top_k，保持候选集完整）
            reranked = self.reranker.rerank(query, results, top_k=None)
            return reranked
        except Exception as e:
            print(f"⚠️ 重排序失败，使用原始排序: {e}")
            return results

    def _get_document_name(self, chunk_index: int, document_sources: List[Dict]) -> str:
        """根据chunk_index查找文档名称"""
        if not document_sources:
            return None
        
        for source in document_sources:
            start, end = source.get('chunk_range', [-1, -1])
            if start <= chunk_index <= end:
                return source.get('name')
        return None

    def _enhanced_retrieval(self, query: str, top_k: int, rerank: bool = True, return_all: bool = False) -> List[Dict]:
        """增强检索：使用加权融合策略 (Super Brain)
        最终分数 = 0.7 * 文档相似度 + 0.3 * 最大问题相似度
        """
        if "enhanced" not in self.databases:
            raise ValueError("增强检索数据库不存在，请先运行 '更新数据库' 操作生成数据库")
        
        db = self.databases["enhanced"]
        documents = db['documents']
        questions = db.get('questions', [])
        embeddings = np.array(db['embeddings'])
        vector_types = db['vector_types']
        vector_to_chunk_map = db['vector_to_chunk_map']
        document_sources = db.get('document_sources', [])
        
        # 1. 生成查询向量
        query_with_prefix = f"query: {query}"
        query_embedding = self.model.encode([query_with_prefix], normalize_embeddings=True)
        
        # 2. 计算所有向量的相似度
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # 3. 按切片聚合分数
        chunk_scores = {} # chunk_index -> {'doc_score': -1, 'max_q_score': -1}
        
        for i, score in enumerate(similarities):
            chunk_idx = vector_to_chunk_map[i]
            v_type = vector_types[i]
            
            if chunk_idx not in chunk_scores:
                chunk_scores[chunk_idx] = {'doc_score': 0.0, 'max_q_score': 0.0} # Default to 0.0 for calculations
                
            if v_type == 'document':
                chunk_scores[chunk_idx]['doc_score'] = float(score)
            elif v_type == 'question':
                if score > chunk_scores[chunk_idx]['max_q_score']:
                    chunk_scores[chunk_idx]['max_q_score'] = float(score)
        
        # 4. 计算加权最终分数
        final_results = []
        alpha = 0.7
        
        for chunk_idx, scores in chunk_scores.items():
            doc_score = scores['doc_score']
            max_q_score = scores['max_q_score']
            
            # 如果没有问题向量，max_q_score 为 0.0
            # 增加鲁棒性：如果没有问题，使用 doc_score 作为 fallback，避免不公平惩罚
            if max_q_score == 0.0 and doc_score > 0.5:
                 max_q_score = doc_score * 0.9 # 稍微降权
            
            final_score = alpha * doc_score + (1 - alpha) * max_q_score
            
            final_results.append({
                'chunk_index': chunk_idx,
                'final_score': final_score,
                'doc_score': doc_score,
                'max_q_score': max_q_score,
                'original_text': documents[chunk_idx],
                'document_name': self._get_document_name(chunk_idx, document_sources)
            })
            
        # 5. 排序并返回 Top K 候选集 (扩大范围以供重排序)
        final_results.sort(key=lambda x: x['final_score'], reverse=True)

        # 获取候选集进行重排序（候选池折中：4倍 Top K，兼顾召回与速度）
        candidate_k = min(len(final_results), top_k * 4)
        candidates = final_results[:candidate_k]

        # 格式化向量检索候选集（去重）
        seen_texts = set()
        seen_chunks = set()
        formatted_candidates = []
        for res in candidates:
            text_key = res['original_text'][:100]  # 前100字作为去重指纹
            if text_key in seen_texts:
                continue  # 跳过重复块
            seen_texts.add(text_key)
            seen_chunks.add(res['chunk_index'])
            formatted_candidates.append({
                'original_text': res['original_text'],
                'document_name': res['document_name'],
                'match_type': 'weighted_fusion',
                'similarity_score': res['final_score'],
                'chunk_index': res['chunk_index'],
                'vector_index': -1,
                'details': {
                    'doc_score': res['doc_score'],
                    'q_score': res['max_q_score']
                }
            })

        # 5.5. 关键词检索补充（弥补向量检索对精确名称匹配的不足）
        keyword_candidates = self._keyword_retrieval(query, documents, top_k * 2, seen_chunks)
        formatted_candidates.extend(keyword_candidates)

        # 6. 应用重排序 (Rerank) - 支持延迟重排：rerank=False 时先跳过，
        #    由调用方对多查询 RRF 融合后的结果统一重排一次，避免 N 次重复重排
        if rerank:
            reranked_results = self.rerank_results(query, formatted_candidates)
        else:
            reranked_results = formatted_candidates

        # 7. 最终去重（以防重排序后仍有重复）
        seen = set()
        final = []
        for r in reranked_results:
            key = r['original_text'][:100]
            if key not in seen:
                seen.add(key)
                final.append(r)

        # return_all=True（多查询融合路径）：返回完整候选集（含关键词命中的正确段落），
        # 避免截断 top_k 时关键词候选被向量高分无关项挤出；由调用方融合后统一重排
        return final[:top_k] if not return_all else final

    def _keyword_retrieval(self, query: str, documents: list, top_k: int,
                           exclude_chunks: set) -> List[Dict]:
        """关键词检索：找出包含查询关键词的文档块，弥补向量检索的精确匹配不足"""
        # 过滤停用词
        stop_chars = set(" ?,.？，。!！的了是在有和与及或之吗呢吧啊")
        query_chars = [c for c in query if c not in stop_chars]
        if not query_chars:
            return []

        # 对每个文档块计算关键词匹配分数
        scored = []
        for chunk_idx, text in enumerate(documents):
            if chunk_idx in exclude_chunks:
                continue

            # 字符覆盖率
            match_count = sum(1 for c in query_chars if c in text)
            char_score = match_count / len(query_chars) if query_chars else 0

            # Bigram 连续匹配
            bigram_total = max(len(query_chars) - 1, 1)
            bigram_match = 0
            for i in range(len(query_chars) - 1):
                if query_chars[i] + query_chars[i + 1] in text:
                    bigram_match += 1
            bigram_score = bigram_match / bigram_total

            # 综合评分
            score = char_score * 0.6 + bigram_score * 0.4

            if score > 0.3:  # 至少 30% 匹配才纳入
                scored.append((chunk_idx, score, text))

        # 按分数排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 格式化结果
        results = []
        for chunk_idx, score, text in scored[:top_k]:
            results.append({
                'original_text': text,
                'document_name': None,
                'match_type': 'keyword',
                'similarity_score': score * 0.85,  # 关键词高置信度，给较高初始分让重排序器裁决
                'chunk_index': chunk_idx,
                'vector_index': -1,
                'details': {
                    'doc_score': score,
                    'q_score': 0.0,
                    'keyword_match': True,
                }
            })

        return results
    
    def _perform_retrieval(self, query: str, database: Dict, top_k: int, filter_type: str = None) -> List[Dict]:
        """执行检索"""
        documents = database['documents']
        questions = database.get('questions', [])
        embeddings = np.array(database['embeddings'])
        vector_types = database['vector_types']
        vector_to_chunk_map = database['vector_to_chunk_map']
        
        # 过滤向量类型
        if filter_type:
            valid_indices = [i for i, vtype in enumerate(vector_types) if vtype == filter_type]
            embeddings = embeddings[valid_indices]
            filtered_vector_types = [vector_types[i] for i in valid_indices]
            filtered_vector_to_chunk_map = [vector_to_chunk_map[i] for i in valid_indices]
            filtered_original_indices = valid_indices
            print(f"🔍 过滤类型: {filter_type}, 过滤后向量数量: {len(valid_indices)}")
        else:
            filtered_vector_types = vector_types
            filtered_vector_to_chunk_map = vector_to_chunk_map
            filtered_original_indices = list(range(len(vector_types)))
            print(f"🔍 不过滤类型, 总向量数量: {len(vector_types)}")
        
        # 如果没有可用的向量，返回空结果
        if len(embeddings) == 0:
            print(f"⚠️ 没有可用的{filter_type}向量，返回空结果")
            return []
        
        # 生成查询向量 - 添加e5-base-v2要求的"query: "前缀
        query_with_prefix = f"query: {query}"
        query_embedding = self.model.encode([query_with_prefix], normalize_embeddings=True)
        
        # 计算相似度
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            original_idx = filtered_original_indices[idx]
            chunk_idx = filtered_vector_to_chunk_map[idx]
            vector_type = filtered_vector_types[idx]
            
            result = {
                'original_text': documents[chunk_idx],
                'match_type': vector_type,
                'similarity_score': float(similarities[idx]),
                'chunk_index': int(chunk_idx),
                'vector_index': int(original_idx)
            }
            
            # 如果是问题向量，添加对应的问题文本
            if vector_type == 'question' and questions:
                question_text = self._get_question_for_vector(original_idx, vector_types, questions)
                if question_text:
                    result['question_text'] = question_text
            
            results.append(result)
        
        return results
    
    def _get_question_for_vector(self, vector_index: int, vector_types: list, questions: list) -> str:
        """根据向量索引获取对应的问题文本"""
        try:
            # 计算这是第几个问题向量
            question_count = 0
            for i in range(vector_index + 1):
                if vector_types[i] == 'question':
                    if i == vector_index:
                        # 找到了当前向量，返回对应的问题
                        if question_count < len(questions):
                            return questions[question_count]
                        break
                    question_count += 1
            return None
        except (IndexError, ValueError):
            return None

# def main():
#     """主函数，用于测试检索功能"""
#     retriever = SmartRetriever()
    
#     # 测试查询
#     query = "南京航空航天大学什么时候成立的？"
#     print(f"🔍 测试查询: {query}")
    
#     # 执行增强检索
#     results = retriever.retrieve_with_strategy(query, strategy="enhanced", top_k=5)
#     print(f"\n📊 检索结果: {len(results)} 个")
    
#     # 显示结果
#     for i, result in enumerate(results, 1):
#         print(f"\n📄 结果 {i}:")
#         print(f"  📊 相似度: {result['similarity_score']:.4f}")
#         print(f"  📝 匹配类型: {result['match_type']}")
#         print(f"  📃 内容: {result['original_text'][:100]}...")

# if __name__ == "__main__":
#     main()  