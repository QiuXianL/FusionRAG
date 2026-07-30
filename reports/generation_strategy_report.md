# RAG 问题生成与优化策略报告

**日期**: 2026-01-09  
**执行者**: RAG Optimization Assistant

## 1. 核心策略：过饱和生成与语义过滤
我们的系统采用**"广撒网，精过滤"** (Oversaturated Generation & Semantic Filtering) 的策略来平衡检索的召回率与索引的效率。

**您的理解完全正确**：
> "在初始阶段生成10个问题，试图全面，然后再聚类阶段去重。"

## 2. 详细工作流程

该流程在 `src/document_manager.py` 的 `_create_document_vectors` 方法中实现：

### 阶段一：全面覆盖 (Generation Phase)
*   **动作**: 对每个文档块 (Chunk)，调用 DeepSeek API。
*   **参数**: `questions_per_chunk = 10` (默认值)。
*   **目的**: 
    *   利用大模型的发散思维，从不同角度（时间、地点、因果、定义等）对文档内容进行提问。
    *   通过生成大量问题，最大化覆盖用户潜在的查询方式，避免漏掉生僻的问法。
*   **代码参考**:
    ```python
    # src/document_manager.py
    questions = self._generate_enhanced_questions(chunk, num_questions=questions_per_chunk)
    ```

### 阶段二：语义去重 (Optimization Phase)
*   **动作**: 对生成的 10 个问题计算向量，并应用语义聚类。
*   **参数**: `threshold = 0.90` (相似度阈值)。
*   **逻辑**:
    1.  计算所有问题的两两余弦相似度。
    2.  优先保留**长度较长**的问题（假设其包含更多限定词和细节，信息熵更高）。
    3.  如果两个问题相似度 > 0.90，则视为重复，移除较短的那个。
*   **目的**:
    *   去除“语义完全一致但表述微调”的冗余问题（如 "Beyonce 是哪年生的？" vs "Beyonce 出生于哪一年？"）。
    *   防止高度相似的向量挤占向量数据库空间和检索结果的 Top-K 槽位。
*   **代码参考**:
    ```python
    # src/document_manager.py
    final_questions, final_embeddings = self._optimize_generated_questions(temp_questions, temp_embeddings)
    ```

## 3. 策略优势分析

| 策略对比 | 仅生成少量问题 (N=3) | **当前策略 (N=10 + 聚类)** |
| :--- | :--- | :--- |
| **覆盖面** | 低。可能只覆盖了最核心的事实，漏掉边缘细节。 | **高**。大概率覆盖多种问法和细节。 |
| **索引效率** | 高。向量少，查询快。 | **中**。虽然生成了 10 个，但通过聚类通常压缩回 4-6 个高质量问题。 |
| **抗噪能力** | 差。如果生成的 3 个问题质量不高，该文档块就很难被召回。 | **强**。即使有几个质量差的问题，也有其他问题作为补充，且聚类倾向于保留信息量大的问题。 |

## 4. 结论
这一流程是目前 RAG 系统中提升召回率的最佳实践之一。它承认了 LLM 生成具有随机性和冗余性的特点，并利用向量计算的确定性来对齐进行后处理，从而得到一组**既全面又精简**的索引项。
