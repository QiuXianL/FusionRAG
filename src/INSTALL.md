# RAG智能问答系统 - 安装指南

## 环境要求

- Python 3.8+
- 推荐使用虚拟环境
- 支持 Windows/Linux/macOS

## 快速安装

### 1. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv rag_env

# 激活虚拟环境
# Windows
rag_env\Scripts\activate
# Linux/macOS
source rag_env/bin/activate
```

### 2. 安装基础依赖

```bash
# 安装基础依赖（CPU版本）
pip install -r requirements.txt
```

## GPU加速安装（可选）

如果你有NVIDIA GPU并想使用GPU加速，请按以下步骤操作：

### 1. 检查CUDA版本

```bash
nvidia-smi
```

### 2. 安装GPU版本的PyTorch

根据你的CUDA版本选择对应的安装命令：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装GPU版本的FAISS

```bash
# 卸载CPU版本
pip uninstall faiss-cpu

# 安装GPU版本
pip install faiss-gpu
```

## 依赖包说明

### 核心依赖
- **Flask**: Web框架，提供Web界面
- **sentence-transformers**: 文本向量化模型
- **torch**: PyTorch深度学习框架
- **langchain**: 文本处理和分块工具

### 可选依赖
- **faiss-gpu**: GPU加速的向量检索（需要NVIDIA GPU）
- **transformers**: Hugging Face模型库
- **scikit-learn**: 机器学习工具

## 常见问题

### 1. 安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2. CUDA版本不匹配
```bash
# 检查PyTorch是否支持CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果不支持，重新安装对应版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 内存不足
```bash
# 使用CPU版本
pip install faiss-cpu

# 或者减少模型大小
# 在代码中使用较小的模型
```

## 验证安装

```bash
# 运行测试
python -c "
import torch
import sentence_transformers
import flask
import langchain
print('✅ 所有依赖安装成功！')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
```

## 开发环境

如果需要开发环境，可以安装额外的开发工具：

```bash
# 代码格式化
pip install black flake8

# 测试框架
pip install pytest

# 类型检查
pip install mypy
```

## 注意事项

1. **虚拟环境**: 强烈建议使用虚拟环境，避免依赖冲突
2. **GPU驱动**: 使用GPU时需要安装正确的NVIDIA驱动
3. **模型下载**: 首次运行时会自动下载模型文件，需要网络连接
4. **磁盘空间**: 模型文件较大，确保有足够的磁盘空间

## 技术支持

如果遇到安装问题，请检查：
1. Python版本是否符合要求
2. 网络连接是否正常
3. 系统权限是否足够
4. 依赖包版本是否兼容 