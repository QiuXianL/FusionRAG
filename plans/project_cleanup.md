# 方案：项目清理 & 准备发布到 Gitee/GitHub

## 当前问题

项目根目录和子目录中存在大量临时文件、重复内容、无用数据，总计约 **500MB+ 垃圾文件**。

## 清理计划

### 一、删除项

#### 1.1 根目录临时文件（15 个 temp_db_*.json + 其他）

| 文件 | 大小 | 原因 |
|------|------|------|
| `temp.py` | 1KB | 临时实验脚本 |
| `temp_db_*.json` (15个) | ~450MB | 评测生成的临时向量库 |
| `prompt.txt` | 17KB | 与项目无关的前端设计文档 |
| `24-李秋贤-成果资料.rar` | 1.5MB | 已解压，压缩包冗余 |
| `.cursorrules` | 5KB | 编辑器个人配置，不属于项目 |

#### 1.2 重复的代码副本

| 目录 | 原因 |
|------|------|
| `upload_package/` | 与 `src/` 内容重复，打包产物 |
| `24-李秋贤-成果资料/` | 含截图 + 代码副本。**截图保留移到 `docs/screenshots/`**，代码删除 |
| `src/documents/search_history.json` | 放错位置的历史记录，删除 |

#### 1.3 运行时产物

| 目录/文件 | 原因 |
|-----------|------|
| `__pycache__/` | Python 字节码缓存 |
| `evaluation/__pycache__/` | 同上 |
| `evaluation/temp_db_DeepSeek.json` | 16MB 临时评测数据 |
| `documents/*.backup_*` (3个) | 数据库自动备份，合计 ~3MB |

#### 1.4 reports/ 去重

| 文件 | 原因 |
|------|------|
| `FusionRAG_vs_NormalRAG_100_report - 副本.md` | Windows 副本文件 |
| `*_assets/` 目录中过时的中间产物 | 只保留最终报告引用的 |

保留的报告：`algorithm_overview.md`、`project_changelog.md`、`comprehensive_rag_improvement_report.md`、`项目研究总结报告.md` 等核心文档。

### 二、新增 .gitignore

需要忽略的内容：

```
# 敏感信息
.env

# Python
__pycache__/
*.pyc
*.pyo

# 模型文件（太大，需单独下载）
models/
output_model/

# 临时文件
temp_*
*.backup_*

# IDE/编辑器
.claude/
.cursorrules
.vscode/

# 运行时数据
documents/search_history.json
documents/document_vectors_enhanced.json

# 打包产物
upload_package/
*.rar
```

### 三、新增 .env.example

去掉真实 API Key，保留模板供使用者参考。

### 四、清理后文件树

```
FusionRAG/
├── .env.example              # 环境变量模板
├── .gitignore
├── README.md
├── src/                      # 核心代码（9个文件）
│   ├── rag.py
│   ├── rag_cli.py
│   ├── web_app.py
│   ├── smart_retrieval.py
│   ├── query_optimizer.py
│   ├── document_manager.py
│   ├── history_manager.py
│   ├── reranker.py
│   ├── template.html
│   └── requirements.txt
├── scripts/
│   └── fetch_lol_lore.py
├── documents/
│   ├── document.txt          # 原始文档
│   └── lol_champions.txt     # LoL 背景故事
├── evaluation/
│   ├── stable_eval_fusionrag.py
│   ├── evaluate_*.py
│   └── *.json                # 评测数据集
├── training_package/
│   └── ...
├── reports/                  # 整理后保留的报告
├── docs/
│   └── screenshots/          # 从 24-李秋贤-成果资料 移过来的截图
├── plans/
├── memory/
└── models/                   # .gitignore 中忽略，需单独说明如何下载
```

### 五、不做的事

- 不删除 `evaluation/` 中的评测脚本和数据集
- 不删除 `training_package/` 中的微调代码
- 不删除 `documents/document.txt`（原始文档）
- 不修改代码逻辑

---

## 完成报告

### 实际删除

| 类别 | 内容 | 大小 |
|------|------|------|
| 临时向量库 | 15 个 `temp_db_*.json` | ~450MB |
| 重复代码 | `upload_package/`、`24-李秋贤-成果资料/`（代码部分） | ~5MB |
| Python 缓存 | `__pycache__/` × 3 | ~100KB |
| 备份文件 | `documents/*.backup_*` × 3 | ~3MB |
| 评测临时数据 | `evaluation/temp_db_DeepSeek.json` | 16MB |
| 无用文件 | `temp.py`、`prompt.txt`、`.cursorrules`、`24-李秋贤-成果资料.rar` | ~2MB |
| reports 冗余 | 重复报告、PDF/DOCX、_assets 中间产物 | ~30MB |
| src 冗余 | `template_*_backup.html`、`__pycache__/`、`env.example` | ~1MB |

**合计清理约 500MB+**

### 实际新增

| 文件 | 说明 |
|------|------|
| `.gitignore` | 排除 .env、models/、output_model/、临时文件等 |
| `.env.example` | 环境变量模板 |
| `docs/screenshots/` | 从成果资料中保留的 7 张截图 |

### 最终结构（69MB，不含模型）

```
FusionRAG/
├── .env.example
├── .gitignore
├── src/                # 核心代码 12 个文件
├── scripts/            # 1 个数据采集脚本
├── documents/          # 原始文档 + LoL 故事 + 向量库
├── evaluation/         # 评测脚本 + 数据集
├── training_package/   # 微调代码
├── reports/            # 16 个核心报告
├── docs/screenshots/   # 7 张界面截图
├── plans/              # 方案文档
└── memory/             # 项目记忆
```

### 发布前需知

- `models/e5-base-v2/` (419MB) 和 `output_model/` (8.8GB) 已加入 .gitignore，不会上传
- 需要在 README 中说明如何下载/放置模型
- `.env` 不会上传，使用者需根据 `.env.example` 自行创建
- `documents/document_vectors_enhanced.json` 已 ignore，使用者需自行构建数据库

### 与计划差异

- 额外清理了 `src/template_cinematic_backup.html`、`src/template_matrix_backup.html`（计划中未列出）
- `documents/test_hotpot_*.json` 保留在原地（评测脚本引用该路径）
