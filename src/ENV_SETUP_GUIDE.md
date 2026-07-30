# 环境变量配置指南

## 🔑 API密钥配置

本项目需要DeepSeek API密钥用于生成问题和AI回答。

### 方法1：使用.env文件（推荐）

1. **复制示例文件**
   ```bash
   # Windows
   copy env.example .env
   
   # Linux/Mac
   cp env.example .env
   ```

2. **编辑.env文件**
   ```bash
   notepad .env  # Windows
   nano .env     # Linux/Mac
   ```

3. **填写API密钥**
   ```env
   DEEPSEEK_API=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

4. **保存文件**

### 方法2：设置系统环境变量

#### Windows (PowerShell)
```powershell
$env:DEEPSEEK_API="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

或永久设置（需要重启终端）：
```powershell
[System.Environment]::SetEnvironmentVariable('DEEPSEEK_API', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', 'User')
```

#### Linux/Mac (bash/zsh)
```bash
export DEEPSEEK_API="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

或永久设置（添加到 ~/.bashrc 或 ~/.zshrc）：
```bash
echo 'export DEEPSEEK_API="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

## 📦 安装依赖

确保安装了python-dotenv：

```bash
pip install python-dotenv
```

或安装全部依赖：

```bash
pip install -r requirements.txt
```

## ✅ 验证配置

运行任何使用API的脚本时，会自动检查并显示API密钥状态：

```bash
python hotpotqa_adapter.py
```

应该看到：
```
✅ 已加载.env文件
✅ 已检测到DEEPSEEK_API密钥: sk-1234567...wxyz
```

如果看到：
```
⚠️  未检测到DEEPSEEK_API密钥
💡 如需生成额外问题，请在.env文件中添加: DEEPSEEK_API=your_api_key
```

说明配置未成功，请检查上述步骤。

## 🔐 获取DeepSeek API密钥

1. 访问 [DeepSeek平台](https://platform.deepseek.com/)
2. 注册/登录账号
3. 进入控制台 → API密钥
4. 创建新的API密钥
5. 复制密钥（只显示一次，请妥善保存）

## 📝 .env文件格式

```env
# 注释行以#开头
DEEPSEEK_API=your_api_key_here

# 不要添加引号
# ✅ 正确
DEEPSEEK_API=sk-1234567890abcdef

# ❌ 错误
DEEPSEEK_API="sk-1234567890abcdef"
DEEPSEEK_API='sk-1234567890abcdef'
```

## 🛡️ 安全注意事项

1. **不要提交.env文件到Git**
   - .env文件已在.gitignore中
   - 只提交env.example

2. **不要分享API密钥**
   - API密钥相当于密码
   - 不要在代码、截图中暴露

3. **定期轮换密钥**
   - 建议定期更换API密钥
   - 如怀疑泄露，立即删除旧密钥

## 🐛 常见问题

### Q: 运行时提示"未安装python-dotenv"？
A: 运行 `pip install python-dotenv`

### Q: .env文件在哪里？
A: 在项目根目录（与rag.py同级）

### Q: 修改.env后没有生效？
A: 重启Python程序/终端即可

### Q: 可以不用.env文件吗？
A: 可以，使用方法2设置系统环境变量

### Q: 需要同时设置.env和环境变量吗？
A: 不需要，选择一种方式即可

## 📚 相关文件

- `env.example` - 配置文件示例
- `.env` - 实际配置文件（需要自己创建）
- `requirements.txt` - Python依赖包列表
- `hotpotqa_adapter.py` - 使用API的脚本示例

---

**配置完成后，就可以使用AI生成问题功能了！** 🎉

