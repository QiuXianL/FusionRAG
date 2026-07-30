#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重排序 (Reranker) 模块
支持多种重排序策略：
1. cross-encoder: 使用本地交叉编码器模型（BGE-Reranker）
2. api: 使用 DeepSeek API 进行相关性评分
3. hybrid: 结合本地模型和 API 的优势
"""

import os
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

# 确保加载 .env 中的环境变量
try:
    from dotenv import load_dotenv
    # 从当前文件向上查找项目根目录的 .env
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass


class Reranker:
    """重排序器，使用交叉编码器对检索结果进行精确重排序"""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        use_api_fallback: bool = False,
        use_api_rerank: bool = True,
        api_key: str = None,
    ):
        """
        初始化重排序器

        Args:
            model_name: 交叉编码器模型名称，默认使用 BGE-Reranker-v2-m3（多语言，中文友好）
            device: 设备 ('cuda', 'cpu')，None 则自动选择
            use_api_fallback: 当本地模型不可用时，是否自动回退到 API（会增加延迟和费用）
            use_api_rerank: 是否允许使用 API 进行重排序
            api_key: DeepSeek API key，None 则从环境变量 DEEPSEEK_API 读取
        """
        self.use_api_fallback = use_api_fallback
        self.use_api_rerank = use_api_rerank
        self.api_key = api_key or os.getenv("DEEPSEEK_API")

        # 默认使用 BGE-Reranker-v2-m3，多语言模型，中文效果好
        if model_name is None:
            model_name = os.getenv(
                "RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            )

        self.model_name = model_name
        self.model = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 尝试加载本地模型
        self._load_model()

    def _load_model(self):
        """加载交叉编码器模型"""
        # 检查是否有本地模型路径
        local_paths = [
            os.path.join("models", "bge-reranker-v2-m3"),
            os.path.join("models", "bge-reranker-base"),
            os.path.join("models", "bge-reranker-large"),
        ]

        # 优先使用本地模型
        for local_path in local_paths:
            if os.path.exists(local_path) and os.path.isdir(local_path):
                try:
                    from sentence_transformers import CrossEncoder

                    self.model = CrossEncoder(
                        local_path,
                        device=self.device,
                    )
                    self.model_name = local_path
                    print(f"✅ 加载本地重排序模型: {local_path} (设备: {self.device})")
                    return
                except Exception as e:
                    print(f"⚠️ 加载本地模型失败 ({local_path}): {e}")

        # 检查是否设置了离线模式
        if os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
            print("⚠️ TRANSFORMERS_OFFLINE=1，跳过从 HuggingFace 下载重排序模型")
            self._print_fallback_message()
            self.model = None
            return

        # 尝试从网络加载
        try:
            from sentence_transformers import CrossEncoder

            # 设置较短超时避免长时间卡住
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")

            print(f"📥 正在加载重排序模型: {self.model_name} ...")
            print("   (首次加载需要下载约1-2GB，请耐心等待)")
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
            print(f"✅ 重排序模型加载成功: {self.model_name} (设备: {self.device})")
        except Exception as e:
            print(f"⚠️ 无法加载交叉编码器模型: {e}")
            self._print_fallback_message()
            self.model = None

    def _print_fallback_message(self):
        """打印回退方案信息"""
        if self.use_api_rerank and self.api_key:
            print("💡 将使用 DeepSeek API 进行重排序")
        elif self.use_api_fallback and self.api_key:
            print("💡 将使用 DeepSeek API 回退方案进行重排序")
        else:
            print("💡 将使用改进的关键词覆盖回退方案进行重排序")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        对检索结果进行重排序

        Args:
            query: 用户查询
            documents: 检索结果列表，每个结果需包含 'original_text' 字段
            top_k: 返回前 k 个结果，None 则返回全部
            score_threshold: 最低相关性分数阈值

        Returns:
            重排序后的结果列表，每个结果新增 'rerank_score' 字段
        """
        if not documents:
            return documents

        if self.model is not None:
            return self._cross_encoder_rerank(query, documents, top_k, score_threshold)
        elif self.use_api_rerank and self.api_key:
            return self._api_rerank(query, documents, top_k, score_threshold)
        else:
            if self.use_api_fallback and self.api_key:
                # 用户允许回退到 API
                return self._api_rerank(query, documents, top_k, score_threshold)
            print("⚠️ 无可用重排序器，使用改进的关键词回退方案")
            return self._fallback_rerank(query, documents, top_k)

    def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int],
        score_threshold: float,
    ) -> List[Dict]:
        """使用交叉编码器进行重排序"""
        # 构建 (query, document) 对
        pairs = [(query, doc["original_text"]) for doc in documents]

        # 批量预测相关性分数
        try:
            scores = self.model.predict(
                pairs,
                batch_size=min(len(pairs), 32),
                show_progress_bar=False,
                convert_to_tensor=True,
            )

            # 转换为 numpy 以便处理
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            else:
                scores = np.array(scores)

            # 将分数归一化到 [0, 1] 范围（如果模型输出不是 sigmoid 的话）
            # BGE-Reranker 模型输出的是 logits，需要 sigmoid
            if scores.max() > 1.0 or scores.min() < 0.0:
                scores = 1.0 / (1.0 + np.exp(-scores))

            scores = scores.flatten()

        except Exception as e:
            print(f"⚠️ 交叉编码器预测失败: {e}")
            return self._fallback_rerank(query, documents, top_k)

        # 更新文档分数
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])
            # 用重排序分数更新相似度分数
            doc["similarity_score"] = float(scores[i])
            if "details" not in doc:
                doc["details"] = {}
            doc["details"]["rerank_score"] = float(scores[i])
            doc["details"]["rerank_method"] = "cross-encoder"

        # 按重排序分数降序排列
        documents.sort(key=lambda x: x["similarity_score"], reverse=True)

        # 过滤低分结果
        if score_threshold > 0:
            documents = [d for d in documents if d["similarity_score"] >= score_threshold]

        # 截断到 top_k
        if top_k is not None:
            documents = documents[:top_k]

        return documents

    def _api_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int],
        score_threshold: float,
    ) -> List[Dict]:
        """使用 DeepSeek API 进行重排序"""
        from openai import OpenAI

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )

            # 构建文档列表供 API 评分
            docs_text = []
            for i, doc in enumerate(documents):
                text = doc["original_text"][:500]  # 截断过长文本
                docs_text.append(f"[{i+1}] {text}")

            docs_list = "\n\n".join(docs_text)

            prompt = f"""你的任务是对以下文档与用户查询的相关性进行评分。

用户查询: {query}

请对以下 {len(documents)} 个文档逐一评分（0-10分），分数越高表示越相关：
{docs_list}

请严格按以下格式输出（每行一个分数，不要其他内容）：
文档1: X.X
文档2: X.X
...
文档{len(documents)}: X.X"""

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                stream=False,
            )

            content = response.choices[0].message.content.strip()

            # 解析分数
            scores = []
            for line in content.split("\n"):
                line = line.strip()
                if ":" in line:
                    try:
                        score_str = line.split(":", 1)[1].strip()
                        score = float(score_str) / 10.0  # 归一化到 [0, 1]
                        scores.append(min(max(score, 0.0), 1.0))
                    except (ValueError, IndexError):
                        scores.append(0.5)  # 解析失败给默认分
                else:
                    scores.append(0.5)

            # 确保分数数量匹配
            while len(scores) < len(documents):
                scores.append(0.5)
            scores = scores[: len(documents)]

        except Exception as e:
            print(f"⚠️ API 重排序失败: {e}")
            return self._fallback_rerank(query, documents, top_k)

        # 融合分数: 60% API 相关性 + 40% 原始向量检索分数
        for i, doc in enumerate(documents):
            original_score = doc.get("similarity_score", 0.5)
            api_score = float(scores[i])

            # 加权融合，保留原始检索信号
            blended_score = api_score * 0.6 + original_score * 0.4

            doc["rerank_score"] = api_score
            doc["similarity_score"] = blended_score
            if "details" not in doc:
                doc["details"] = {}
            doc["details"]["rerank_score"] = api_score
            doc["details"]["original_score"] = original_score
            doc["details"]["rerank_method"] = "api"

        # 按融合分数降序排列
        documents.sort(key=lambda x: x["similarity_score"], reverse=True)

        # 过滤低分结果
        if score_threshold > 0:
            documents = [d for d in documents if d["similarity_score"] >= score_threshold]

        # 截断到 top_k
        if top_k is not None:
            documents = documents[:top_k]

        return documents

    def _fallback_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int],
    ) -> List[Dict]:
        """
        无模型时的回退重排序策略。
        使用改进的字符覆盖率 + 语义分数融合。
        """
        try:
            # 过滤非关键字符
            ignored_chars = set(" ?,.？，。!！的了是在有和与及或")
            clean_query = [c for c in query if c not in ignored_chars]

            if not clean_query:
                if top_k is not None:
                    return documents[:top_k]
                return documents

            for doc in documents:
                text = doc["original_text"]
                original_score = doc.get("similarity_score", 0.5)

                # 字符匹配覆盖率
                match_count = sum(1 for c in clean_query if c in text)
                coverage = match_count / len(clean_query)

                # 计算连续匹配奖励（Bigram 匹配）
                bigram_matches = 0
                total_bigrams = max(len(clean_query) - 1, 1)
                for j in range(len(clean_query) - 1):
                    bigram = clean_query[j] + clean_query[j + 1]
                    if bigram in text:
                        bigram_matches += 1
                bigram_score = bigram_matches / total_bigrams

                # 融合分数: 70% 语义 + 20% 字符覆盖 + 10% Bigram 匹配
                new_score = (
                    original_score * 0.70
                    + coverage * 0.20
                    + bigram_score * 0.10
                )

                doc["rerank_score"] = float(new_score)
                doc["similarity_score"] = float(new_score)
                if "details" not in doc:
                    doc["details"] = {}
                doc["details"]["rerank_score"] = float(new_score)
                doc["details"]["rerank_method"] = "fallback"
                doc["details"]["keyword_coverage"] = coverage
                doc["details"]["bigram_score"] = bigram_score

            documents.sort(key=lambda x: x["similarity_score"], reverse=True)

            if top_k is not None:
                documents = documents[:top_k]

            return documents

        except Exception as e:
            print(f"⚠️ 回退重排序失败: {e}")
            if top_k is not None:
                return documents[:top_k]
            return documents

    def compute_pairwise_scores(
        self,
        query: str,
        documents: List[Dict],
    ) -> np.ndarray:
        """
        计算 query 与每个文档的相关性分数（不修改原始文档）

        Args:
            query: 用户查询
            documents: 文档列表

        Returns:
            numpy array of scores
        """
        if self.model is None:
            # 回退到简单分数
            return np.array([d.get("similarity_score", 0.5) for d in documents])

        pairs = [(query, doc["original_text"]) for doc in documents]
        try:
            scores = self.model.predict(
                pairs,
                batch_size=min(len(pairs), 32),
                show_progress_bar=False,
                convert_to_tensor=True,
            )
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            scores = np.array(scores).flatten()

            # Sigmoid normalization for logit outputs
            if scores.max() > 1.0 or scores.min() < 0.0:
                scores = 1.0 / (1.0 + np.exp(-scores))

            return scores
        except Exception as e:
            print(f"⚠️ 评分计算失败: {e}")
            return np.array([d.get("similarity_score", 0.5) for d in documents])

    @property
    def is_available(self) -> bool:
        """检查重排序器是否可用"""
        if self.model is not None:
            return True
        if self.use_api_rerank and self.api_key:
            return True
        return False

    @property
    def method(self) -> str:
        """当前使用的重排序方法"""
        if self.model is not None:
            return "cross-encoder"
        elif self.use_api_rerank and self.api_key:
            return "api"
        return "fallback"


# 全局重排序器单例
_global_reranker: Optional[Reranker] = None


def get_reranker(
    model_name: str = None,
    force_reload: bool = False,
) -> Reranker:
    """
    获取全局重排序器单例

    Args:
        model_name: 模型名称，None 使用默认
        force_reload: 是否强制重新加载

    Returns:
        Reranker 实例
    """
    global _global_reranker
    if _global_reranker is None or force_reload:
        _global_reranker = Reranker(model_name=model_name)
    return _global_reranker
