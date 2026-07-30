# 算法原理与实现细节 (Algorithm Overview)

本文档详细记录了本项目 ("最强大脑" RAG 系统) 中所使用的所有核心算法、技术原理及实现细节。
旨在为开发者提供清晰的技术蓝图，每次新增或修改算法时必须更新此文档。

---

## 1. 数据处理与索引构建 (Indexing)

### 1.1 文本分块 (Chunking)
使用 `RecursiveCharacterTextSplitter` 进行智能分块，确保语义的完整性。
*   **Chunk Size**: 300 字符
*   **Chunk Overlap**: 30 字符
*   **分隔符优先级**: `["\n\n", "\n", "。", "？", "！", "……", "．", ".", "?", "!", "\r"]`
*   **目的**: 将长文档切分为适合向量化的短文本，同时保留上下文连贯性。

### 1.2 "最强大脑" 问题生成 (Super Brain Question Generation)
为了增强检索的语义匹配能力，我们不仅仅索引文档本身，还为每个文档块生成潜在的用户问题（Reverse-HyDE 思想）。

*   **实现位置**: `src/document_manager.py` -> `_generate_enhanced_questions`
*   **算法流程**:
    1.  **关键词提取**: 使用 LLM 提取文档块中的关键实体、时间、地点和核心概念。
    2.  **思维链 (CoT) 生成**:
        *   **Prompt**: 要求 LLM 分析文本和关键词，识别最关键的信息点（事实、因果关系、定义）。
        *   **策略**: 针对每个信息点，生成一个需要理解该信息才能回答的具体问题。
        *   **多样性要求**: 强制生成不同类型的问题（Who/When/Where vs Why/How）。
    3.  **质量过滤**:
        *   长度限制: 5-50 字符。
        *   格式检查: 必须包含且仅包含一个问号。

### 1.3 向量存储结构
*   **混合存储**: 数据库同时存储 "Document Chunk" (文档块) 和 "Generated Question" (生成问题) 的向量。
*   **映射关系**: 每个 Document Chunk 对应多个 Generated Question 向量，但在检索评分时会聚合计算。

---

## 2. 检索策略 (Retrieval Strategy)

### 2.1 查询扩展 (Query Expansion) - CoT
为了解决用户查询语义单一或模糊的问题，使用思维链技术生成多视角查询。

*   **实现位置**: `src/query_optimizer.py` -> `expand_query`
*   **Prompt 策略**:
    *   **Step 1**: 分析用户意图和潜在歧义。
    *   **Step 2**: 生成 3 个不同维度的变体：
        1.  **具体化 (Specific)**: 补充细节的查询。
        2.  **概念化 (Generalized)**: 使用同义词或更宽泛概念的查询。
        3.  **关联问题 (Related)**: 暗示答案的关联问题。

### 2.2 双路增强检索 (Dual-Path Enhanced Retrieval)
对于每一个查询（原始查询 + 扩展查询），执行以下检索逻辑：

*   **实现位置**: `src/smart_retrieval.py` -> `_enhanced_retrieval`
*   **评分公式**:
    $$ Score = \alpha \times S_{doc} + (1-\alpha) \times S_{max\_question} $$
    *   $S_{doc}$: 查询向量与文档块向量的相似度。
    *   $S_{max\_question}$: 查询向量与该文档块下所有生成问题向量的最大相似度。
    *   $\alpha$: 权重系数 (当前为 0.7)，即 70% 看文档本身，30% 看生成的问题。
*   **Fallback 机制**: 如果没有生成问题向量，仅使用文档分数，并进行轻微降权 ($0.9 \times S_{doc}$) 以保持公平。

### 2.3 加权倒数排名融合 (Weighted RRF)
将多个查询视角（原始 + 扩展）的检索结果合并为一个最终列表。

*   **实现位置**: `src/query_optimizer.py` -> `fuse_results`
*   **核心思想**: 给予原始查询更高的权重，防止扩展查询引入的噪声掩盖了精确匹配的结果。
*   **算法公式**:
    $$ RRF\_Score(d) = \sum_{q \in Queries} Weight(q) \times \frac{1}{k + Rank(d, q)} $$
    *   $k$: 平滑常数 (60)。
    *   $Weight(q)$:
        *   **原始查询**: 3.0 (高权重，保护首位命中率)。
        *   **扩展查询**: 1.0 (辅助权重，提升召回率)。

---

## 3. 重排序 (Reranking)

### 3.1 字符级覆盖率重排序 (Character-level Coverage Rerank)
为了弥补向量检索在"精确匹配"上的不足（有时向量相似但关键实体缺失），引入基于字面匹配的轻量级重排序。

*   **实现位置**: `src/smart_retrieval.py` -> `rerank_results`
*   **算法流程**:
    1.  **清洗**: 移除查询中的停用词和标点。
    2.  **计算覆盖率**: 统计查询中的有效字符在文档中出现的比例 ($Coverage$).
    3.  **加权融合**:
        $$ Final\_Score = 0.8 \times S_{semantic} + 0.2 \times Coverage $$
    4.  **触发条件**: 仅对原始相似度 > 0.6 的结果应用此重排序，避免将完全不相关的文档强行提权。

---

## 4. 评估体系 (Evaluation)

### 4.1 核心指标
*   **Recall@K**: 正确文档出现在前 K 个结果中的概率 (关注召回能力)。
*   **MRR (Mean Reciprocal Rank)**: 正确文档排名的倒数均值 (关注排名质量)。

### 4.2 自动化测试
*   **脚本**: `evaluation/evaluate_visualized.py`
*   **流程**:
    1.  自动构建包含 20-50 个样本的测试数据库。
    2.  执行 Baseline vs Super Brain 对比测试。
    3.  使用 `matplotlib` 生成可视化图表。
    4.  输出 Markdown 格式的详细报告。

---
*Last Updated: 2026-01-11*
