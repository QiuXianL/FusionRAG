# 项目修改日志 (Project Changelog)

本文档记录了项目的主要技术变更、优化原理及对应的评估报告。

## 📅 2026-01-09: "最强大脑" (Super Brain) 架构升级

本次更新主要集中在 RAG 系统的检索精度与索引质量优化，引入了思维链（CoT）和多路召回融合技术。

### 1. 检索端：多视角思维链与融合 (Super Brain Retrieval)
- **🛠️ 修改内容**:
    - 新增核心模块 `src/query_optimizer.py`。
    - 在 `src/web_app.py` (Web端) 和 `src/rag_cli.py` (命令行端) 中完全集成 `QueryOptimizer`。
    - 搜索策略新增 `auto` 和 `enhanced` 模式，自动启用该优化。

- **💡 原理**:
    1.  **思维链扩展 (CoT Expansion)**: 系统不再直接搜索用户的问题，而是利用大模型“思考”用户意图，将其扩展为三个维度的查询：
        *   **具体化查询**: 补充细节。
        *   **广义化查询**: 使用同义词或上位概念。
        *   **关联性查询**: 猜测用户可能需要的背景知识。
    2.  **倒数排名融合 (RRF)**: 并行执行上述所有查询，通过 RRF 算法将多路结果融合。RRF 能有效降低单一查询的噪声，让在多个视角下都相关的文档排在前面。

- **📄 对应报告**: [super_brain_evaluation_report.md](super_brain_evaluation_report.md)

---

### 2. 索引端：智能问题生成与聚类优化 (Smart Indexing)
- **🛠️ 修改内容**:
    - 重构 `src/document_manager.py` 中的问题生成逻辑。
    - 引入 `_optimize_generated_questions` 方法，实现基于向量相似度的聚类去重。

- **💡 原理**:
    1.  **CoT 问题生成**: 提示词升级为 Chain-of-Thought 模式（先提取关键事实 -> 再针对性提问），相比旧版直接生成，问题逻辑性更强。
    2.  **语义聚类去重 (Semantic Clustering)**:
        *   策略：**过饱和生成 -> 聚类筛选**。
        *   过程：先对每个文档块生成过量问题（如6个），计算它们之间的余弦相似度。如果两个问题相似度超过 0.9，则视为重复，系统会优先保留**更长、信息量更大**的那个。
        *   效果：在保证知识点全覆盖的同时，防止向量库因重复问题而膨胀。

- **📄 对应报告**: 
    - [question_generation_report.md](question_generation_report.md) (生成策略对比)
    - [clustering_optimization_report.md](clustering_optimization_report.md) (聚类去重效果)

---

### 3. 排序端：混合重排序策略 (Hybrid Reranking)
- **🛠️ 修改内容**:
    - 优化 `src/smart_retrieval.py` 中的 `rerank_results` 方法。

- **💡 原理**:
    - 纯向量检索有时会忽略精确的关键词（如特定型号、人名）。
    - 新算法引入 **字面匹配覆盖率 (Character Coverage)**。
    - **最终得分公式**: `Final Score = 0.8 * Semantic_Score + 0.2 * Coverage_Score`。
    - 这确保了结果既要在语义上相关，又要尽可能包含用户的查询关键词。

- **📄 对应报告**: (包含在整体检索效果评估中)

---

### 4. 基础设施：评估体系与报告管理
- **🛠️ 修改内容**:
    - 建立 `evaluation/` 目录，包含 `evaluate_super_brain.py` 等自动化测试脚本。
    - 规范化报告输出目录为 `reports/`。
    - 实现了报告的中文化与可视化（Markdown 表格）。

- **💡 原理**:
    - 数据驱动优化（Data-Driven Optimization）。通过 SQUAD 格式的问答对数据集，计算 Recall@5（前5名召回率）指标，量化 "最强大脑" 相比 "基准方法" 的提升幅度。

- **📄 对应报告**: 所有上述报告均由此体系生成。
