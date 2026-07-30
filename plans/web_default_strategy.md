# 方案：Web 版默认使用最先进策略

## 现状

- 前端（[template.html](src/template.html) line 1019-1021）发送搜索请求时只传 `{query, top_k}`，**没有传 `strategy`**
- 后端（[web_app.py](src/web_app.py) line 200/370）收到请求后默认 `strategy='auto'`
- `'auto'` 的行为：如果 QueryOptimizer 有 API 客户端（DeepSeek），就跑完整管线（查询扩展 → 多路召回 → RRF 融合 → 重排序）；否则直接检索
- 目前**没有**一个显式的"最先进"策略名

## 当前可用策略分析

| 策略名 | 查询扩展 | RRF 融合 | 重排序 | 说明 |
|--------|----------|----------|--------|------|
| `enhanced` | ❌ | ❌ | ✅ | 单路增强检索 + 重排序 |
| `auto`（当前默认）| ✅（有API时）| ✅（有API时）| ✅ | 自动选择，实际等于 enhanced+扩展 |

`auto` 在有 API 时已经是最先进的路径，但名字不够直观。

## 建议方案

### 1. 新增 `fusion` 策略名（最先进策略）

在 `web_app.py` 和 `rag_cli.py` 中新增 `'fusion'` 策略，**明确表示完整管线**：
- 查询扩展（DeepSeek CoT 生成 3 个变体）
- 多路召回（每个变体从增强向量库检索 top_k 个结果）
- RRF 加权融合（原始查询权重 3.0x）
- Cross-Encoder/API 重排序
- LLM 生成回答

当 API 不可用时，自动降级为 `enhanced`（单路检索 + 重排序）。

### 2. 改动范围

| 文件 | 改动 | 说明 |
|------|------|------|
| [web_app.py](src/web_app.py) | line 200, 370 | 默认策略从 `'auto'` 改为 `'fusion'` |
| [web_app.py](src/web_app.py) | line 216, 396 | 条件判断增加 `'fusion'` |
| [rag_cli.py](src/rag_cli.py) | line 152, 175, 218 | 可选：CLI 也支持 `fusion` |
| [template.html](src/template.html) | 搜索请求 body | 可选：前端显式传 `strategy: 'fusion'` |

### 3. 不做什么

- 不改变 API 接口结构
- 不影响文档管理功能
- 不修改检索器核心逻辑

## 影响评估

- **向后兼容**：`'auto'` 保留不变，只是默认值改为 `'fusion'`
- **API 消耗**：和现在 `'auto'` 行为一致（本来就有 API 时走完整管线）
- **用户体验**：前端/API 调用者不传 strategy 时走最先进策略

## 验证方案

### 方法一：启动 Web 服务直接测试（推荐）

```bash
cd "e:/program/RAG - 当前使用/src"
python -c "
from web_app import app
# 模拟前端不传 strategy 的请求
with app.test_client() as client:
    resp = client.post('/api/search',
        json={'query': '南京航空航天大学什么时候成立的？', 'top_k': 3})
    data = resp.get_json()
    print('策略:', data.get('strategy_used'))  # 应输出 'fusion'
    print('结果数:', len(data.get('results', [])))
    print('AI回答:', data.get('ai_response', '')[:100])
"
```

预期输出：
- `策略: fusion` — 证明默认策略生效
- 日志中出现 `🧠 Super Brain: 正在多角度思考问题...` — 证明走了查询扩展
- 日志中出现 `📥 正在加载重排序模型` 或 `💡 将使用 DeepSeek API 进行重排序` — 证明走了重排序

### 方法二：对比测试

分别用 `enhanced`（旧单路）和 `fusion`（新融合）搜同一个问题，对比：
- `fusion` 结果中应有 `match_type: 'fused_multi_query'`（证明了 RRF 融合）
- `fusion` 结果的 `details.rerank_method` 应有值（证明了重排序）
- `fusion` 通常能找到更精准的答案

```bash
cd "e:/program/RAG - 当前使用/src"
python -c "
from web_app import app
with app.test_client() as client:
    # 不传 strategy → 默认 fusion
    r1 = client.post('/api/search', json={'query': '学校有哪些专业？', 'top_k': 3})
    d1 = r1.get_json()
    print('=== 默认(fusion) ===')
    for i, r in enumerate(d1['results']):
        print(f'{i+1}. [{r[\"match_type\"]}] rerank={r[\"details\"].get(\"rerank_method\",\"none\")} score={r[\"similarity_score\"]:.4f}')

    # 显式传 enhanced
    r2 = client.post('/api/search', json={'query': '学校有哪些专业？', 'strategy': 'enhanced', 'top_k': 3})
    d2 = r2.get_json()
    print()
    print('=== enhanced ===')
    for i, r in enumerate(d2['results']):
        print(f'{i+1}. [{r[\"match_type\"]}] rerank={r[\"details\"].get(\"rerank_method\",\"none\")} score={r[\"similarity_score\"]:.4f}')
"
```

预期差异：
- `fusion` 的 `match_type` = `fused_multi_query`（多路融合结果）
- `enhanced` 的 `match_type` = `weighted_fusion`（单路加权结果）

### 方法三：检查服务端日志

启动 Web 服务后搜索，观察控制台输出：

```
🔍 收到搜索请求: 'xxx' (策略: fusion, top_k: 3)   ← 默认策略显示 fusion
🧠 Super Brain: 正在多角度思考问题...              ← 查询扩展
  ↳ 扩展查询: ['变体1', '变体2', '变体3']          ← 3个扩展查询
✅ 多路召回融合完成                                 ← RRF 融合
                                                    ← 重排序（模型加载或API调用）
🤖 AI回答生成完成                                   ← LLM 回答
```

---

---

## 完成报告

### 实际改动

| 文件 | 改动 |
|------|------|
| [web_app.py](src/web_app.py) line 200 | 默认策略 `'auto'` → `'fusion'` |
| [web_app.py](src/web_app.py) line 370 | 流式端点默认策略同步改为 `'fusion'` |
| [web_app.py](src/web_app.py) line 216, 396 | 条件判断从 `== 'enhanced' or == 'auto'` 改为 `in ('enhanced', 'auto', 'fusion')` |
| [rag_cli.py](src/rag_cli.py) line 152 | CLI 默认策略从 `'auto'` → `'fusion'` |
| [rag_cli.py](src/rag_cli.py) line 175 | 策略切换白名单增加 `'fusion'` |
| [rag_cli.py](src/rag_cli.py) line 218 | 条件判断同步支持 `'fusion'` |
| [rag_cli.py](src/rag_cli.py) line 234-240 | **顺带修复**：旧缩进 bug（`results = optimizer.fuse_results(...)` 缩进错误导致语法错误） |

### 验证结果

- `web_app.py`: `api_search` 和 `api_search_stream` 默认 strategy 确认为 `'fusion'`
- `rag_cli.py`: 导入成功，语法正确
- `'auto'` 和 `'enhanced'` 策略保持可用，向后兼容

### 与计划差异

无。完全按计划实施。
