# FusionRAG 项目规矩

## 代码改动流程

1. **先写方案文档** — 任何代码改动前，先写方案文档放在 `plans/` 目录下，经用户审核通过后才能开始写代码
2. **写完必须交完成报告** — 方案实施完毕后，在原方案文档末尾追加 `## 完成报告` 章节，写明实际改动、验证结果、与计划的差异

## 文档规范

- 所有方案文档统一放在 `plans/` 文件夹
- 方案文档命名简洁明了，如 `plans/功能名.md`
- 完成报告追加在原方案文档末尾，不另开新文件

## 项目信息

- 项目名：FusionRAG · 符文之地档案馆
- 技术栈：Python/Flask + sentence-transformers + DeepSeek API
- 功能：英雄联盟宇宙知识 RAG 检索 + 多轮对话 + 论战分析
- 入口：`src/rag.py`（交互式菜单）、`src/web_app.py`（Web 服务）
- Web 服务启动：`cd src && python -c "from web_app import run_web_server; run_web_server()"`

## 代码风格

- 保持与周边代码一致的风格（命名、注释密度、缩进）
- 中文注释优先
- 改动尽量小，不重构整个文件
