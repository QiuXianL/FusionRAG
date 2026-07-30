# 🧠 问题生成专家对比报告 (Expert Comparison Report)

**日期**: 2026-01-23 22:33:52

## 1. 摘要 (Executive Summary)
本报告对比了在使用不同'问题生成专家'（Question Generation Experts）进行文档索引时，RAG系统的检索性能和生成质量。
### 数据集统计 (Dataset Statistics)
- **总样本数**: 50
- **HotpotQA**: 0
- **SQuAD**: 0

## 2. 性能指标 (Performance Metrics)
|                 |   Recall@1 |   Recall@5 |   Recall@10 |      MRR |
|:----------------|-----------:|-----------:|------------:|---------:|
| Normal RAG      |       0.74 |       0.74 |        0.74 | 0.74     |
| Base Model      |       0.76 |       0.88 |        0.88 | 0.794    |
| Finetuned Model |       0.78 |       0.84 |        0.9  | 0.810833 |

## 3. 生成问题分析 (Generated Question Analysis)
以下是各专家针对相同内容片段生成的问题示例。

### 示例 1
**原文片段**: *Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American si...*

**标准参考问题 (Standard Ground Truth)**: *When did Beyonce start becoming popular?*
**标准答案 (Standard Answer)**: *N/A*

| 专家 (Expert) | 生成的问题 (Generated Questions) |
| :--- | :--- |
| **Normal RAG** | ❌ *No samples generated (Unknown error)* |
| **Base Model** | - What is Beyonce's nationality? |
| **Finetuned Model** | - What is the nationality of Beyoncé Giselle Knowles-Carter? |


### 示例 2
**原文片段**: *. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a ...*

**标准参考问题 (Standard Ground Truth)**: *When did Beyonce start becoming popular?*
**标准答案 (Standard Answer)**: *N/A*

| 专家 (Expert) | 生成的问题 (Generated Questions) |
| :--- | :--- |
| **Normal RAG** | ❌ *No samples generated (Unknown error)* |
| **Base Model** | - What city was Whitney Houston born in? |
| **Finetuned Model** | - Where did Kelly Allen rise to fame? |


### 示例 3
**原文片段**: *. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl group...*

**标准参考问题 (Standard Ground Truth)**: *When did Beyonce start becoming popular?*
**标准答案 (Standard Answer)**: *N/A*

| 专家 (Expert) | 生成的问题 (Generated Questions) |
| :--- | :--- |
| **Normal RAG** | ❌ *No samples generated (Unknown error)* |
| **Base Model** | - Who was the manager of The Knowles? |
| **Finetuned Model** | - Who was Mathew Knowles father? |


## 4. 分析结论 (Analysis)
基于 **Recall@5** 指标，表现最好的专家是 **Base Model**。
