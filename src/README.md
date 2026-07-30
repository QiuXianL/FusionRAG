# RAG智能问答系统

一个基于检索增强生成（RAG）的智能问答系统，支持中文文档处理和本地化AI模型。

## 核心功能

- 🧠 **智能检索**：基于e5-base-v2模型的向量检索
- 🌐 **双模式运行**：支持命令行和Web界面
- 🚀 **GPU加速**：自动检测并使用GPU加速
- 📝 **中文优化**：专为中文文档设计
- 💾 **文档管理**：支持上传、更新、删除文档
- 📚 **搜索历史**：记录和查询历史问答
- 🤖 **AI增强**：集成DeepSeek API生成智能回答

## 技术栈

- **后端**: Flask + Python 3.8+
- **AI模型**: SentenceTransformer (e5-base-v2)
- **深度学习**: PyTorch
- **文本处理**: LangChain
- **前端**: HTML/CSS/JavaScript

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制`env.example`为`.env`，填入你的DeepSeek API密钥：

```bash
copy env.example .env  # Windows
# 或
cp env.example .env    # Linux/Mac
```

### 3. 准备文档

将知识库文档放入`documents/document.txt`

### 4. 运行程序

```bash
python rag.py
```

### 5. 访问Web界面

浏览器自动打开 `http://localhost:5000`

## 项目结构

```
RAG/
├── rag.py                 # 主程序入口
├── web_app.py            # Web服务器
├── rag_cli.py            # 命令行界面
├── smart_retrieval.py    # 智能检索引擎
├── document_manager.py   # 文档管理
├── history_manager.py    # 历史记录管理
├── template.html         # Web界面模板
├── requirements.txt      # 依赖包
├── documents/            # 文档存储
├── models/               # AI模型
├── INSTALL.md           # 详细安装指南
├── ENV_SETUP_GUIDE.md   # 环境配置指南
└── README.md            # 本文件
```

## 使用说明

### Web界面模式

运行程序后选择Web模式，支持：
- 文档上传和管理
- 向量数据库更新
- 智能问答
- 搜索历史查看

### 命令行模式

运行程序后选择CLI模式，支持：
- 交互式问答
- 批量文档处理
- 系统配置

## 文档说明

- **INSTALL.md**: 详细的安装和环境配置说明
- **ENV_SETUP_GUIDE.md**: 环境变量配置指南
- **env.example**: 环境变量配置模板

## 注意事项

- 首次运行会自动下载AI模型（约500MB）
- 建议使用虚拟环境运行
- GPU加速需要安装CUDA版本的PyTorch
- DeepSeek API需要网络连接

## 许可证

MIT License

