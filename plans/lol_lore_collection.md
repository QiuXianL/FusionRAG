# 方案：获取英雄联盟宇宙背景故事

## 数据源分析

英雄联盟官方**没有**专门的背景故事 API，但有两个可靠数据源：

| 数据源 | 内容 | 格式 | 可靠性 |
|--------|------|------|--------|
| **Data Dragon**（官方） | 173 个英雄的 `lore`（简短背景）+ `blurb`（一句话介绍） | JSON | ⭐⭐⭐ 最稳定 |
| Universe 官网 | 完整短篇故事、漫画、CG | HTML 网页 | ⭐⭐ 需爬虫解析 |

## 建议方案

### 第一步：通过 Data Dragon 获取所有英雄背景

Riot 官方静态数据 CDN，无需 API Key：

```
# 获取最新版本号
GET https://ddragon.leagueoflegends.com/api/versions.json

# 获取所有英雄数据（含 lore 字段）
GET https://ddragon.leagueoflegends.com/cdn/{version}/data/zh_CN/champion.json
```

每个英雄数据结构：
```json
{
  "id": "Aatrox",      // 英文名
  "name": "暗裔剑魔",   // 中文名
  "title": "亚托克斯",  // 称号
  "blurb": "...",      // 一句话简介
  "lore": "...",       // 完整背景故事（中文）
  "tags": ["Fighter", "Tank"]
}
```

### 第二步：格式化为 RAG 文档

将每个英雄的背景故事整理为结构化文本块，字段含：

- 英雄中文名 + 英文名
- 称号
- 定位（tags）
- 一句话简介（blurb）
- 完整背景故事（lore）

最终输出一份文本文件，可直接通过 RAG 的文档管理功能导入。

### 第三步：导入现有 RAG 数据库

通过 DocumentManager 将故事文件添加到向量数据库，之后即可用中文搜索英雄背景。

## 涉及的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/fetch_lol_lore.py` | **新增** | 从 Data Dragon 拉取数据并格式化 |
| `documents/lol_champions.txt` | **新增** | 格式化后的英雄背景故事文本 |
| 现有 RAG 代码 | **不改动** | 通过文档管理功能导入即可 |

## 不做的事情

- 不爬 Universe 官网（反爬风险，且 Data Dragon 已覆盖核心 lore）
- 不修改 RAG 检索/排序逻辑
- 不处理皮肤背景故事（数据量过大，先做英雄本体）

## 数据规模预估

- 173 个英雄 × 平均 300-800 字中文背景故事 ≈ 约 10-15 万字
- 按当前分块策略（chunk_size=300）约生成 300-500 个文档块

## 验证方式

1. 运行脚本查看输出文件 `documents/lol_champions.txt`
2. 通过 Web 界面"文档管理 → 添加文档"导入
3. 搜索 "亚托克斯的背景故事是什么？" 验证检索效果

---

## 完成报告

### 实际改动

| 文件 | 操作 | 说明 |
|------|------|------|
| [scripts/fetch_lol_lore.py](scripts/fetch_lol_lore.py) | **新增** | 从 Data Dragon 官方 CDN 拉取英雄数据 |
| [documents/lol_champions.txt](documents/lol_champions.txt) | **新增** | 233 个英雄的中文背景故事，约 4 万字 |

### 验证结果

- 脚本运行成功，版本 16.15.1
- 获取到 233 个英雄（比预估的 173 多，因 Data Dragon 包含所有 variant 和重做的英雄）
- 输出格式：每个英雄包含中文名、英文名、称号、定位、背景故事
- 可通过 RAG Web 界面直接导入

### 与计划差异

1. **数据量小于预估**：计划预估 10-15 万字，实际约 4 万字。原因是 Data Dragon 中 `lore` 和 `blurb` 字段已合并，每个英雄的背景故事仅约 130-185 字（精炼版），而非 Universe 官网的完整长篇故事。
2. **英雄数量多于预估**：233 vs 173，因为 Data Dragon 包含所有已发布过的英雄 variant。
3. **如需完整长篇故事**，需要后续从 Universe 官网（`universe.leagueoflegends.com`）爬取补充。
