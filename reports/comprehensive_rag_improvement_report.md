# 🧠 用于大模型 RAG 文档检索的嵌入方法研究：项目总报告（事无巨细版）

**用途**：本文件作为结项阶段的“全栈技术说明 + 实验复盘 + 创新点总结”统一稿，直接满足“把创新算法和各种算法写在 md 报告里”的要求。  
**仓库路径**：`E:\program\RAG - 当前使用\`  
**原始综合报告生成时间**：2026-01-13（本次为扩写与补全）  

---

## 0. 一句话概括（答辩开场版）

RAG 的上限取决于检索。本项目从“嵌入向量如何组织”和“检索如何融合与重排”两个层面系统改造 RAG：把**文档向量**与**生成问题向量**一起入库，再用**加权融合（0.7/0.3）+ 轻量重排序（覆盖率 0.8/0.2）+ CoT 查询扩展 + RRF 融合**，提升多跳问题的召回与排序鲁棒性，并建立可复现实验与报告体系。

---

## 1. 仓库结构与核心文件（哪里看实现）

### 1.1 src：算法实现

- 索引构建（分块、向量化、问题生成、聚类去重、数据库结构维护）：[document_manager.py](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py)
- 检索算法（加权融合 + rerank）：[smart_retrieval.py](file:///E:/program/RAG%20-%20当前使用/src/smart_retrieval.py)
- Super Brain（CoT 扩展 + RRF）：[query_optimizer.py](file:///E:/program/RAG%20-%20当前使用/src/query_optimizer.py)

### 1.2 evaluation：评测与对比实验

- Super Brain 检索评测与报告生成：[evaluate_super_brain.py](file:///E:/program/RAG%20-%20当前使用/evaluation/evaluate_super_brain.py)
- 问题生成策略对比（旧/新/CoT）与聚类模拟：[evaluate_question_generation.py](file:///E:/program/RAG%20-%20当前使用/evaluation/evaluate_question_generation.py)
- HotpotQA 样本抽取与清洗：[download_hotpot.py](file:///E:/program/RAG%20-%20当前使用/evaluation/download_hotpot.py)
- 不同问题生成器（Normal/Base/Finetuned）对比：[compare_qgen_experts.py](file:///E:/program/RAG%20-%20当前使用/evaluation/compare_qgen_experts.py)

### 1.3 reports：阶段性报告（可作为结项附件素材）

- V2（聚类 + 重排）技术说明：[RAG_Optimization_Report_V2.md](file:///E:/program/RAG%20-%20当前使用/reports/RAG_Optimization_Report_V2.md)
- 问题生成策略结果（含数值）：[question_generation_report.md](file:///E:/program/RAG%20-%20当前使用/reports/question_generation_report.md)
- 项目变更与总结脉络：[project_changelog.md](file:///E:/program/RAG%20-%20当前使用/reports/project_changelog.md)

---

## 2. 问题背景：RAG 检索为什么会“看起来很像但就是不对”

即使用最强的 embedding，也常见三类失败：

1. **Query 语义锚点不足**：用户问题短、口语、缺少实体/时间/地点等关键锚点
2. **多跳（桥接证据）缺失**：HotpotQA 这类任务需要先找 A 再由 A 找 B；单次检索不一定能抓到“桥梁文档”
3. **关键词约束弱化**：向量检索重语义轻字面，可能把“Java 安装”误排到“Python 安装”前面

本项目每个创新模块都明确对应其中一个失败点。

---

## 3. 你的“嵌入方法研究”核心：向量库里到底存了什么

### 3.1 不是只存文档 embedding：还存“生成问题 embedding”

你把向量分成两类：

- `document`：每个文档块 1 条向量
- `question`：每个文档块生成 N 条“潜在查询问题”，每条问题也向量化入库

这使得检索可以从两个角度命中同一证据块：
- 用户 Query 与文档块语义接近（document 命中）
- 用户 Query 与“生成问题”语义接近（question 命中）

### 3.2 数据库 JSON 结构（Schema 级解释）

增强库（`documents/document_vectors_enhanced.json` 以及 evaluation 的 `temp_db_*.json`）至少包含：

- `documents`: 文档块文本列表
- `embeddings`: 所有向量（文档 + 问题）拼接后的数组
- `questions`: 问题文本列表（仅对应 question 向量）
- `vector_types`: 与 embeddings 对齐，标记每个向量是 `document` 还是 `question`
- `vector_to_chunk_map`: 与 embeddings 对齐，把每个向量映射回哪个 chunk（用于检索端聚合）
- `document_sources`: 文档级元信息（name、chunk_range、参数等），用于从 chunk 反查文档名与评测定位

对应实现：
- 生成问题、聚类去重、向量创建：  
  - [document_manager.py:L284-L481](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py#L284-L481)

---

## 4. 算法全解（从索引到检索，逐一对齐代码）

### 4.1 基础嵌入：E5 的 query/passage 前缀

你严格遵循 E5 的输入域约定：
- Query：`query: {query}`
- 文档/问题（passage）：`passage: {text}`

代码位置：
- 索引端给 chunk 和问题加 `passage:`：[document_manager.py:L432-L452](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py#L432-L452)
- 检索端给 query 加 `query:`：[smart_retrieval.py:L127-L133](file:///E:/program/RAG%20-%20当前使用/src/smart_retrieval.py#L127-L133)

为什么重要：
- 这是 E5 训练时的对齐格式，不按这个来相似度分布会变差，影响召回与排名稳定性。

---

### 4.2 生成式索引（Generative Indexing）：让文档“对齐用户问法”

#### 4.2.1 动机

很多 Query 并不是文档原句，只存文档向量会存在“问法 → 文档措辞”的语义鸿沟。  
生成式索引的做法：把每段文档块转成“用户可能会问的问题”，把这些问题也向量化入库。

#### 4.2.2 流程（对应 DocumentManager）

对每个 chunk：
1. 生成文档向量：`passage: chunk`
2. 调用大模型生成问题列表（DeepSeek API）
3. 对每个问题生成向量：`passage: question`
4. 聚类去重（下一节）
5. 写入库：维护 `vector_types` 与 `vector_to_chunk_map`

实现入口：[document_manager.py:L420-L481](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py#L420-L481)

#### 4.2.3 你不是“直接生成”：而是关键词抽取 + CoT 约束生成

你在生成问题前增加了关键词抽取（实体/时间/地点/概念），再用 CoT 风格提示生成问题：
- 关键词抽取：[document_manager.py:L262-L282](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py#L262-L282)
- CoT 生成问题（Step1 分析、Step2 提问）：[document_manager.py:L306-L331](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py#L306-L331)

这解决了“直接生成法”常见的泛化与幻觉问题。

#### 4.2.4 质量过滤（你在代码里做了强约束）

解析后过滤规则：
- 长度 5–200
- 至少一个中/英文问号

位置：[document_manager.py:L347-L358](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py#L347-L358)

---

### 4.3 智能聚类去重（Smart Clustering）：贪心语义去重

#### 4.3.1 动机：重复问题会膨胀索引并稀释分数

同一 chunk 的问题经常是同义改写，全部入库会导致：
- 向量库膨胀（存储/检索变慢）
- 得分被重复向量稀释
- TopK 缺乏多样性

#### 4.3.2 算法细节（与你的实现等价）

输入：问题 `questions` 与其向量 `embeddings`（已归一化）  
步骤：
1. 相似度矩阵：`S = E · E^T`
2. 按长度降序排序（优先保留信息量大者）
3. 贪心遍历：保留 i，同时移除所有 `S[i,j] > 0.90` 的 j

阈值：`0.90`（只去掉极度相似改写，避免过度删减）  
实现位置：[document_manager.py:L364-L418](file:///E:/program/RAG%20-%20当前使用/src/document_manager.py#L364-L418)

复杂度：
- n 通常很小（每块 3–10），O(n^2) 的矩阵与贪心扫描非常可控。

---

### 4.4 加权融合检索（Weighted Fusion）：文档分 + 问题分的 chunk 级融合

#### 4.4.1 动机：问题向量有收益也有噪声

只用问题向量会被生成噪声拖垮；只用文档向量又丢失“问法对齐”的优势。  
因此你实现了“以文档为主、问题为辅”的加权融合：

`Final Score = 0.7 * Doc_Score + 0.3 * Max_Question_Score`

#### 4.4.2 实现过程（smart_retrieval.py）

1. 查询编码：`query: query`
2. 对库中所有向量算相似度
3. 以 chunk 聚合：`doc_score`、`max_q_score`
4. 计算融合分数（alpha=0.7）
5. 取 top_k*3 候选进入 rerank

实现位置：[smart_retrieval.py:L112-L199](file:///E:/program/RAG%20-%20当前使用/src/smart_retrieval.py#L112-L199)

#### 4.4.3 鲁棒性补丁（非常关键）

如果某个 chunk 没有问题向量（生成失败或 questions_per_chunk=0），你做了回填：
- 当 `max_q_score == 0.0 且 doc_score > 0.5`  
  使用 `max_q_score = doc_score * 0.9`（轻微降权）

位置：[smart_retrieval.py:L158-L163](file:///E:/program/RAG%20-%20当前使用/src/smart_retrieval.py#L158-L163)

---

### 4.5 轻量级重排序（Hybrid Rerank）：字符覆盖率微调

#### 4.5.1 动机：向量相近不等于关键词约束满足

你引入了字符覆盖率作为轻量 reranker（中文无需分词）：

- 清洗 Query（忽略常见停用字符）
- 覆盖率 = Query 关键字符在文档中出现比例
- 融合：`new_score = 0.8 * semantic + 0.2 * coverage`
- 仅在 `semantic > 0.6` 时触发

实现位置：[smart_retrieval.py:L55-L99](file:///E:/program/RAG%20-%20当前使用/src/smart_retrieval.py#L55-L99)

---

### 4.6 Super Brain：CoT 查询扩展 + RRF 多查询融合

这是你应对多跳/桥接证据的核心模块。

#### 4.6.1 CoT 查询扩展（expand_query）

对原始 Query 生成 3 个变体：
- 更具体版本
- 更抽象/同义版本
- 隐含答案关联的相关问题（桥接视角）

实现位置：[query_optimizer.py:L21-L66](file:///E:/program/RAG%20-%20当前使用/src/query_optimizer.py#L21-L66)

#### 4.6.2 RRF 融合（fuse_results）

对多查询的结果做 Reciprocal Rank Fusion：

`Score(doc) = Σ weight(q) * 1 / (k + rank_q(doc))`

你实现的关键点：
- 原查询权重 3.0，扩展查询权重 1.0
- k 默认 60（平滑 top 位尖峰）

实现位置：[query_optimizer.py:L67-L109](file:///E:/program/RAG%20-%20当前使用/src/query_optimizer.py#L67-L109)

---

## 5. 评测体系与实验脚本（你怎么证明有效）

### 5.1 数据集与指标

- 数据集：SQuAD（单跳）、HotpotQA（多跳）
- 指标：Recall@K、MRR

### 5.2 代表性评测脚本解读

- Super Brain 的 SQuAD 评测脚本会：
  - 构建临时库（默认 questions_per_chunk=0 降低成本）
  - Baseline：原查询检索
  - Super：扩展查询 + RRF 融合
  - 输出 Recall@5 与报告
  - 见：[evaluate_super_brain.py](file:///E:/program/RAG%20-%20当前使用/evaluation/evaluate_super_brain.py)

- 问题生成策略对比脚本会：
  - 对比旧/新/CoT 生成方式
  - 评估聚类去重前后的效果趋势
  - 见：[evaluate_question_generation.py](file:///E:/program/RAG%20-%20当前使用/evaluation/evaluate_question_generation.py)

---

## 6. 核心结果（保留原综合报告数据，并解释口径）

### 6.1 HotpotQA（多跳推理）

| 指标 | Baseline | Super Brain | 变化 |
| :--- | :---: | :---: | :---: |
| Recall@1 | 45.00% | 47.50% | +2.5% |
| Recall@5 | 87.50% | 87.50% | 0 |
| Recall@10 | 100.00% | 100.00% | 0 |
| MRR | 0.95 | 0.97 | +0.02 |

### 6.2 SQuAD（单跳事实）

| 指标 | Baseline | Super Brain | 变化 |
| :--- | :---: | :---: | :---: |
| Recall@1 | 84.00% | 82.00% | -2.0% |
| Recall@5 | 96.00% | 96.00% | 0 |
| Recall@10 | 100.00% | 100.00% | 0 |
| MRR | 0.89 | 0.88 | -0.01 |

### 6.3 口径说明（答辩必讲）

不同脚本可能使用不同“命中判定”：
- substring 匹配 vs document_name 精确匹配
- 是否分块、是否启用问题向量、是否启用扩展查询

因此项目结论来自“多数据集 + 多脚本 + 多指标”的交叉验证趋势，而不是依赖单一数字。

---

## 7. 误差分析与经验（为什么有时会波动）

1. 生成问题风格不同（关键词导向 vs 语义抽象）会影响“命中锚点”
2. 生成问题数量越多覆盖越强，但噪声也更大（聚类去重与融合权重就是为此设计）
3. 单跳任务更偏字面精确性，多跳任务更依赖扩展查询与多视角融合

---

## 8. 结论与下一步（结项落点）

### 8.1 已完成的可交付成果

- 可运行的软件原型：索引（文档+问题）、检索（融合+重排）、Super Brain（扩展+融合）
- 系统化创新算法：生成式索引、贪心聚类去重、加权融合、覆盖率重排、CoT+RRF
- 可复现实验体系：evaluation 脚本 + reports 输出

### 8.2 后续改进方向

- 离线索引与增量更新（解决生成式索引的时间/成本问题）
- 引入 Cross-Encoder 精排，进一步提升 Recall@1 与 MRR
- 训练/蒸馏本地 QGen 模型，降低 API 依赖并统一生成风格
