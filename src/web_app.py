import json
import os
import threading
import webbrowser
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response
from openai import OpenAI
from smart_retrieval import SmartRetriever
from document_manager import DocumentManager
from history_manager import HistoryManager
from query_optimizer import QueryOptimizer

# Flask应用
app = Flask(__name__)

# HTML模板文件路径
# 使用相对于脚本的绝对路径，确保在任何目录下运行都能找到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(BASE_DIR, 'template.html')

# 全局检索器
retriever = None

# 全局文档管理器
document_manager = None

# 全局历史记录管理器
history_manager = None

def log_message(message, level="INFO"):
    """格式化日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def extract_answer_content(response_text):
    """从AI回答中提取<回答>标签之间的内容"""
    try:
        # 查找<回答>和</回答>标签
        start_tag = "<回答>"
        end_tag = "</回答>"
        
        start_index = response_text.find(start_tag)
        end_index = response_text.find(end_tag)
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # 提取标签之间的内容
            content = response_text[start_index + len(start_tag):end_index].strip()
            return content
        else:
            # 如果没有找到标签，返回原始内容
            return response_text.strip()
    except Exception as e:
        log_message(f"⚠️ 提取回答内容时出错: {e}", "WARN")
        return response_text.strip()

def init_retriever():
    """初始化检索器"""
    global retriever
    if retriever is None:
        log_message("🔧 正在初始化智能检索器...")
        try:
            retriever = SmartRetriever()
            log_message("✅ 智能检索器初始化成功", "SUCCESS")
            
            # 显示数据库信息
            databases = retriever.databases if retriever else {}
            log_message(f"📊 加载的数据库: {list(databases.keys())}")
            
            for db_name, db_data in databases.items():
                doc_count = len(db_data.get('documents', []))
                vector_count = len(db_data.get('embeddings', []))
                log_message(f"  - {db_name}: {doc_count}个文档, {vector_count}个向量")
                
        except Exception as e:
            log_message(f"❌ 智能检索器初始化失败: {e}", "ERROR")
            raise

def init_document_manager():
    """初始化文档管理器"""
    global document_manager
    if document_manager is None:
        log_message("🔧 正在初始化文档管理器...")
        try:
            document_manager = DocumentManager()
            log_message("✅ 文档管理器初始化成功", "SUCCESS")
        except Exception as e:
            log_message(f"❌ 文档管理器初始化失败: {e}", "ERROR")
            raise

def init_history_manager():
    """初始化历史记录管理器"""
    global history_manager
    if history_manager is None:
        log_message("🔧 正在初始化历史记录管理器...")
        try:
            history_manager = HistoryManager()
            log_message("✅ 历史记录管理器初始化成功", "SUCCESS")
        except Exception as e:
            log_message(f"❌ 历史记录管理器初始化失败: {e}", "ERROR")
            raise

@app.before_request
def log_request():
    """记录请求信息"""
    if request.endpoint and not request.endpoint.startswith('static'):
        client_ip = request.remote_addr
        method = request.method
        path = request.path
        log_message(f"📨 {method} {path} - 来自 {client_ip}")

@app.after_request
def log_response(response):
    """记录响应信息"""
    if request.endpoint and not request.endpoint.startswith('static'):
        status = response.status_code
        method = request.method
        path = request.path
        
        if status == 200:
            log_message(f"✅ {method} {path} - 响应: {status}")
        else:
            log_message(f"⚠️ {method} {path} - 响应: {status}", "WARN")
    
    return response

@app.route('/')
def index():
    """返回主页面"""
    log_message("🏠 访问主页")
    try:
        with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            html_content = f.read()
        log_message("✅ 主页模板加载成功")
        return html_content
    except FileNotFoundError:
        log_message(f"❌ 模板文件 {TEMPLATE_FILE} 未找到", "ERROR")
        return '''
        <html>
        <head><title>错误</title></head>
        <body>
            <h1>❌ 模板文件未找到</h1>
            <p>请确保 template.html 文件存在于当前目录中。</p>
        </body>
        </html>
        ''', 404

@app.route('/api/status')
def api_status():
    """获取数据库状态"""
    log_message("📊 检查系统状态...")
    try:
        # 初始化检索器
        init_retriever()
        
        # 检查数据库状态
        databases = retriever.databases if retriever else {}
        
        # 获取GPU状态
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "无GPU"
        
        log_message(f"🖥️ GPU状态: {'可用' if gpu_available else '不可用'} - {gpu_name}")
        
        # 计算文档数量
        doc_count = 0
        vector_count = 0
        if databases:
            for db_name, db_data in databases.items():
                doc_count = max(doc_count, len(db_data.get('documents', [])))
                vector_count = max(vector_count, len(db_data.get('embeddings', [])))
        
        log_message(f"📚 数据库状态: {len(databases)}个数据库, {doc_count}个文档, {vector_count}个向量")
        
        return jsonify({
            'status': len(databases) > 0,
            'databases': list(databases.keys()),
            'doc_count': doc_count,
            'vector_count': vector_count,
            'gpu_available': gpu_available,
            'gpu_name': gpu_name,
            'device': 'cuda' if gpu_available else 'cpu'
        })
    except Exception as e:
        log_message(f"❌ 状态检查失败: {e}", "ERROR")
        return jsonify({'status': False, 'error': str(e)})

@app.route('/api/search', methods=['POST'])
def api_search():
    """处理搜索请求"""
    start_time = time.time()
    
    try:
        # 初始化检索器
        init_retriever()
        
        data = request.get_json()
        query = data.get('query', '').strip()
        strategy = data.get('strategy', 'fusion')  # 默认使用最先进的融合策略
        top_k = data.get('top_k', 5)  # 允许前端设置top_k参数，默认为5

        log_message(f"🔍 收到搜索请求: '{query}' (策略: {strategy}, top_k: {top_k})")

        if not query:
            log_message("⚠️ 空查询请求", "WARN")
            return jsonify({'error': '请输入问题'})

        # 使用智能检索器进行检索
        log_message(f"🧠 开始检索...")
        retrieval_start = time.time()

        results = []
        try:
            optimizer = QueryOptimizer()
            if strategy in ('enhanced', 'auto', 'fusion') and optimizer.client:
                log_message(f"🧠 Super Brain: 正在多角度思考问题...")
                queries = optimizer.expand_query(query)
                
                if len(queries) > 1:
                    log_message(f"  ↳ 扩展查询: {queries[1:]}")
                    all_results = {}
                    for q in queries:
                        # 各扩展查询先跳过重排（毫秒级），避免 4 次 × 30 对的重复重排
                        res = retriever.retrieve_with_strategy(q, strategy='enhanced', top_k=top_k, rerank=False, return_all=True)
                        all_results[q] = res

                    # Fuse results
                    results = optimizer.fuse_results(all_results, original_query=query)
                    results = results[:top_k+2] # Take slightly more to allow for reranking drop-offs # Keep a few more for AI context
                    # 融合后统一重排一次（仅约 top_k+2 对，远快于 4 次重排）
                    results = retriever.rerank_results(query, results)
                else:
                    results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=top_k)
            else:
                results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=top_k)
        except Exception as e:
            log_message(f"⚠️ Super Brain 优化失败: {e}", "WARN")
            results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=top_k)
        
        retrieval_time = time.time() - retrieval_start
        log_message(f"⚡ 检索完成，耗时: {retrieval_time:.3f}秒")
        
        if not results:
            log_message("❌ 未找到相关文档")
            return jsonify({'error': '没有找到相关文档'})
        
        log_message(f"✅ 找到 {len(results)} 个相关结果")
        
        # 显示检索结果摘要
        for i, result in enumerate(results, 1):
            similarity = result['similarity_score']
            match_type = result['match_type']
            preview = result['original_text'][:100].replace('\n', ' ')
            
            # 提取详细分数信息（如果存在）
            details = result.get('details', {})
            detail_info = ""
            if details:
                doc_score = details.get('doc_score', 0)
                q_score = details.get('q_score', 0)
                detail_info = f" (文档={doc_score:.4f}, 问题={q_score:.4f})"
                
            log_message(f"  📄 结果{i}: 综合分={similarity:.4f}{detail_info}, 类型={match_type}, 预览={preview}...")
        
        # 生成AI回答
        ai_response = ""
        ai_start = time.time()
        
        try:
            my_api_key = os.getenv("DEEPSEEK_API")
            if my_api_key:
                log_message("🤖 开始生成AI回答...")
                client = OpenAI(api_key=my_api_key, base_url="https://api.deepseek.com")
                
                # 构建上下文
                context = ""
                for i, result in enumerate(results, 1):
                    context += f"参考文档{i}：{result['original_text']}\n\n"
                
                prompt = f"""
                你是符文之地的博学贤者，通晓英雄联盟宇宙的一切传奇。你的任务是基于参考文档和你的知识，回答用户的问题。

                参考文档如下：
                <参考文档>
                {context}
                </参考文档>

                用户问题如下：
                <用户问题>
                {query}
                </用户问题>

                请按照以下要求回答：
                1. 综合分析参考文档中的信息，结合你的背景知识，给出一个明确、有观点的回答。不要回避问题，大胆推理和判断。
                2. 如果文档信息不完整，大胆用你的知识补充，并在文中自然说明哪些推断来自文档、哪些来自背景知识。
                3. 请使用Markdown格式，让回答既有深度又易读。

                当用户要求比较两名角色的实力时，你就是符文之地的论战考据大师。此时必须：
                - 所有论述源于原文+设定，推论分三级：【原作铁证】【合理推演】【仅为猜想】，不把猜想当铁证
                - 按五大维度对比：基础面板（速度、攻防、能量）、能力机制（规则技、克制链）、实战战绩、爆发上限、持续作战
                - 明确对战基准（生死斗/切磋、战场环境、是否允许底牌），区分常态/爆发/底牌形态，禁止跨形态强行对比
                - 两名角色同一套评判标准，不双标，不脑补，信息不足时标注「信息缺失」
                - 给出多场景对局推演结论和胜负概率区间
                4. 回答精炼，正文控制在 600 字以内，避免重复啰嗦。

                请在<回答>标签内写下你的Markdown格式答案。
                <回答>
                [在此给出你作为符文之地贤者的回答]
                </回答>
                """
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    timeout=60,
                    stream=False
                )
                ai_response = response.choices[0].message.content
                
                # 提取<回答>标签之间的内容
                ai_response = extract_answer_content(ai_response)
                
                ai_time = time.time() - ai_start
                log_message(f"🤖 AI回答生成完成，耗时: {ai_time:.3f}秒")
                
            else:
                log_message("⚠️ 未设置DEEPSEEK_API，跳过AI回答生成", "WARN")
                ai_response = "⚠️ 未设置DEEPSEEK_API环境变量，无法提供AI增强回答"
                
        except Exception as e:
            log_message(f"❌ AI回答生成失败: {e}", "ERROR")
            ai_response = f"AI回答生成失败：{str(e)}"
        
        total_time = time.time() - start_time
        log_message(f"🎯 搜索请求完成，总耗时: {total_time:.3f}秒")
        
        # 添加历史记录
        try:
            init_history_manager()
            performance_data = {
                'retrieval_time': float(retrieval_time),
                'ai_time': float(time.time() - ai_start) if my_api_key else 0.0,
                'total_time': float(total_time)
            }
            history_manager.add_history(
                query=query,
                results=results,
                ai_response=ai_response,
                strategy=strategy,
                top_k=top_k,
                performance=performance_data
            )
            log_message("📚 历史记录已保存")
        except Exception as e:
            log_message(f"⚠️ 保存历史记录失败: {e}", "WARN")
        
        return jsonify({
            'results': results,
            'ai_response': ai_response,
            'strategy_used': strategy,
            'query': query,
            'performance': {
                'retrieval_time': float(retrieval_time),
                'ai_time': float(time.time() - ai_start) if my_api_key else 0.0,
                'total_time': float(total_time)
            }
        })
        
    except Exception as e:
        error_time = time.time() - start_time
        log_message(f"❌ 搜索请求失败: {e} (耗时: {error_time:.3f}秒)", "ERROR")
        return jsonify({'error': f'搜索出错：{str(e)}'})

@app.route('/api/search/stream', methods=['POST'])
def api_search_stream():
    """处理流式搜索请求"""
    try:
        # 在请求上下文中获取数据
        data = request.get_json()
        query = data.get('query', '').strip()
        strategy = data.get('strategy', 'fusion')
        top_k = data.get('top_k', 5)

        log_message(f"🔍 收到流式搜索请求: '{query}' (策略: {strategy}, top_k: {top_k})")

        if not query:
            log_message("⚠️ 空查询请求", "WARN")
            return jsonify({'error': '请输入问题'})

        # 初始化检索器
        init_retriever()

        def generate():
            start_time = time.time()

            try:
                # 先发送检索开始信号
                yield f"data: {json.dumps({'type': 'retrieval_start'})}\n\n"

                # 使用智能检索器进行检索
                log_message(f"🧠 开始检索...")
                retrieval_start = time.time()

                results = []
                try:
                    optimizer = QueryOptimizer()
                    if strategy in ('enhanced', 'auto', 'fusion') and optimizer.client:
                        log_message(f"🧠 Super Brain: 正在多角度思考问题...")
                        queries = optimizer.expand_query(query)
                        
                        if len(queries) > 1:
                            log_message(f"  ↳ 扩展查询: {queries[1:]}")
                            all_results = {}
                            for q in queries:
                                # 各扩展查询先跳过重排（毫秒级），避免 4 次 × 30 对的重复重排
                                res = retriever.retrieve_with_strategy(q, strategy='enhanced', top_k=top_k, rerank=False, return_all=True)
                                all_results[q] = res

                            results = optimizer.fuse_results(all_results, original_query=query)
                            results = results[:top_k+2]
                            # 融合后统一重排一次（仅约 top_k+2 对，远快于 4 次重排）
                            results = retriever.rerank_results(query, results)
                        else:
                            results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=top_k)
                    else:
                        results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=top_k)
                except Exception as e:
                    log_message(f"⚠️ Super Brain 优化失败: {e}", "WARN")
                    results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=top_k)
                
                retrieval_time = time.time() - retrieval_start
                log_message(f"⚡ 检索完成，耗时: {retrieval_time:.3f}秒")
                
                if not results:
                    log_message("❌ 未找到相关文档")
                    yield f"data: {json.dumps({'error': '没有找到相关文档'})}\n\n"
                    return
                
                # 发送检索结果
                yield f"data: {json.dumps({'type': 'retrieval_complete', 'results': results})}\n\n"
                
                # 生成AI回答（流式）
                my_api_key = os.getenv("DEEPSEEK_API")
                ai_start = time.time()  # 定义AI开始时间
                cleaned_response = ""  # 初始化AI回答变量
                
                if my_api_key:
                    log_message("🤖 开始生成AI回答...")
                    
                    yield f"data: {json.dumps({'type': 'ai_start'})}\n\n"
                    
                    client = OpenAI(api_key=my_api_key, base_url="https://api.deepseek.com")
                    
                    # 构建上下文
                    context = ""
                    for i, result in enumerate(results, 1):
                        context += f"参考文档{i}：{result['original_text']}\n\n"
                    
                    prompt = f"""
                    你的任务是基于参考文档回答用户的问题。请仔细阅读以下参考文档和用户问题，并根据文档内容给出准确、详细的回答。
                    
                    参考文档如下：
                    <参考文档>
                    {context}
                    </参考文档>
                    
                    用户问题如下：
                    <用户问题>
                    {query}
                    </用户问题>
                    
                    请按照以下要求回答：
                    1. 如果参考文档中没有相关信息，请说明无法从给定文档中找到答案，然后给出你认为的答案。
                    2. 请使用Markdown格式回答，包括适当的标题、列表、代码块等格式。
                    3. 回答要结构清晰，便于阅读。
                    4. 回答精炼，正文控制在 600 字以内，避免重复啰嗦。

                    请在<回答>标签内写下你的Markdown格式答案。
                    <回答>
                    [在此根据文档内容用Markdown格式回答问题]
                    </回答>
                    """
                    
                    # 流式生成AI回答
                    ai_response_chunks = []
                    try:
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1500,
                            timeout=30,
                            stream=True
                        )
                        
                        for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                ai_response_chunks.append(content)
                                yield f"data: {json.dumps({'type': 'ai_chunk', 'content': content})}\n\n"
                        
                        # 完整的AI回答
                        full_ai_response = ''.join(ai_response_chunks)
                        # 提取<回答>标签之间的内容
                        cleaned_response = extract_answer_content(full_ai_response)
                        
                        yield f"data: {json.dumps({'type': 'ai_complete', 'content': cleaned_response})}\n\n"
                        
                        log_message(f"🤖 AI回答生成完成")
                        
                    except Exception as e:
                        log_message(f"❌ AI回答生成失败: {e}", "ERROR")
                        yield f"data: {json.dumps({'type': 'ai_error', 'error': f'AI回答生成失败：{str(e)}'})}\n\n"
                        
                else:
                    log_message("⚠️ 未设置DEEPSEEK_API，跳过AI回答生成", "WARN")
                    yield f"data: {json.dumps({'type': 'ai_error', 'error': '⚠️ 未设置DEEPSEEK_API环境变量，无法提供AI增强回答'})}\n\n"
                
                # 添加历史记录
                try:
                    init_history_manager()
                    ai_time = time.time() - ai_start if my_api_key else 0.0
                    total_time = time.time() - start_time
                    performance_data = {
                        'retrieval_time': float(retrieval_time),
                        'ai_time': float(ai_time),
                        'total_time': float(total_time)
                    }
                    history_manager.add_history(
                        query=query,
                        results=results,
                        ai_response=cleaned_response,
                        strategy=strategy,
                        top_k=top_k,
                        performance=performance_data
                    )
                    log_message("📚 历史记录已保存")
                except Exception as e:
                    log_message(f"⚠️ 保存历史记录失败: {e}", "WARN")
                
                # 发送完成信号
                total_time = time.time() - start_time
                log_message(f"🎯 流式搜索请求完成，总耗时: {total_time:.3f}秒")
                yield f"data: {json.dumps({'type': 'complete', 'total_time': total_time})}\n\n"
                
            except Exception as e:
                error_time = time.time() - start_time
                log_message(f"❌ 流式搜索请求失败: {e} (耗时: {error_time:.3f}秒)", "ERROR")
                yield f"data: {json.dumps({'type': 'error', 'error': f'搜索出错：{str(e)}'})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type'
        })
        
    except Exception as e:
        log_message(f"❌ 流式搜索请求处理失败: {e}", "ERROR")
        return jsonify({'error': f'请求处理失败：{str(e)}'})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """处理多轮对话请求（携带历史记录）"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        history = data.get('history', [])
        top_k = data.get('top_k', 5)

        log_message(f"💬 收到对话消息: '{message[:50]}...' (历史: {len(history)} 轮, top_k: {top_k})")

        if not message:
            return jsonify({'error': '请输入消息'})

        init_retriever()

        def generate():
            start_time = time.time()

            try:
                yield f"data: {json.dumps({'type': 'retrieval_start'})}\n\n"
                retrieval_start = time.time()

                # RAG 检索
                results = []
                try:
                    optimizer = QueryOptimizer()
                    if optimizer.client:
                        queries = optimizer.expand_query(message)
                        if len(queries) > 1:
                            log_message(f"  ↳ 扩展查询: {queries[1:]}")
                            all_results = {}
                            for q in queries:
                                # 各扩展查询先跳过重排，融合后统一重排一次
                                res = retriever.retrieve_with_strategy(q, strategy='enhanced', top_k=top_k, rerank=False, return_all=True)
                                all_results[q] = res
                            results = optimizer.fuse_results(all_results, original_query=message)
                            results = results[:top_k + 2]
                            # 融合后统一重排一次（仅约 top_k+2 对）
                            results = retriever.rerank_results(message, results)
                        else:
                            results = retriever.retrieve_with_strategy(message, strategy='enhanced', top_k=top_k)
                    else:
                        results = retriever.retrieve_with_strategy(message, strategy='enhanced', top_k=top_k)
                except Exception as e:
                    log_message(f"⚠️ 对话检索失败: {e}", "WARN")
                    results = retriever.retrieve_with_strategy(message, strategy='enhanced', top_k=top_k)

                retrieval_time = time.time() - retrieval_start
                log_message(f"⚡ 对话检索完成，耗时: {retrieval_time:.3f}秒")

                if results:
                    yield f"data: {json.dumps({'type': 'retrieval_complete', 'results': results})}\n\n"

                api_key = os.getenv("DEEPSEEK_API")
                if not api_key:
                    yield f"data: {json.dumps({'type': 'ai_error', 'error': '未设置DEEPSEEK_API'})}\n\n"
                    return

                yield f"data: {json.dumps({'type': 'ai_start'})}\n\n"

                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

                # 构建检索上下文
                context = ""
                for i, r in enumerate(results, 1):
                    context += f"参考文档{i}：{r['original_text']}\n\n"

                # 构建知识库概览（动态获取，每次请求实时查询）
                kb_overview = ""
                try:
                    db = retriever.databases.get("enhanced", {})
                    doc_count = len(db.get("documents", []))
                    sources = db.get("document_sources", [])
                    src_names = [s.get("name", "") for s in sources[:3]]
                    if doc_count > 0:
                        kb_overview = f"你目前能访问一座符文之地档案馆，馆藏约 {doc_count} 卷文献"
                        if src_names:
                            kb_overview += f"，主要来源包括：{'、'.join(src_names)}等"
                        kb_overview += "。用户可能询问馆藏内容，请如实根据这些信息回答。"
                except Exception:
                    pass

                # 构建对话消息
                debate_rules = """当用户要求比较两名角色的实力时，你就是符文之地的论战考据大师。此时必须：
- 所有论述源于原文+设定，推论分三级：【原作铁证】【合理推演】【仅为猜想】，不把猜想当铁证
- 按五大维度对比：基础面板（速度、攻防、能量）、能力机制（规则技、克制链）、实战战绩、爆发上限、持续作战
- 明确对战基准（生死斗/切磋、战场环境、是否允许底牌），区分常态/爆发/底牌形态，禁止跨形态强行对比
- 两名角色同一套评判标准，不双标，不脑补，信息不足时标注「信息缺失」但不准以此为由回避结论
- 给出多场景对局推演，最后必须用「我认为X更强」或类似明确表述给出一个确定结论，哪怕证据不完美也要大胆判定，绝不使用"各有千秋""取决于情况""无法判断"这类逃避性话术"""

                what_if_rules = """当用户提出假设性问题（"假如""如果"等），你就是符文之地的世界线推演师。此时必须：
- 先明确被改变的"原事件"是什么，检索相关角色的背景故事作为推演锚点
- 从改变点出发，按因果关系逐级推演：直接影响 → 角色反应 → 他人连锁反应 → 长期格局变化
- 每一步标注依据等级：【基于原著铁证】【基于角色性格合理推演】【纯猜测】
- 区分短期影响（几周到几个月）和长期影响（数年），可以有多个可能分支但必须说明哪种最可能及原因
- 最后对比「原世界线 vs 假如世界线」，给出鲜明结论，不要含糊"""

                messages = [{
                    "role": "system",
                    "content": f"你是符文之地的博学贤者，通晓英雄联盟宇宙的一切传奇。{kb_overview}请保持对话自然流畅，记住之前聊过的内容，大胆给出判断和见解。{debate_rules}{what_if_rules}回答精炼，正文控制在 600 字以内，避免重复啰嗦。"
                }]

                if context.strip():
                    messages.append({
                        "role": "system",
                        "content": f"以下是与用户问题相关的参考文档：\n<参考文档>\n{context}\n</参考文档>\n请基于这些文档和你的知识回答。如果文档信息不完整，大胆用你的知识补充。"
                    })

                # 历史对话（最近 10 轮 = 20 条）
                for h in history[-20:]:
                    role = h.get('role', 'user')
                    if role in ('user', 'assistant'):
                        messages.append({"role": role, "content": h.get('content', '')})

                messages.append({"role": "user", "content": message})

                # 流式生成
                full_response = ""
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        max_tokens=1500,
                        timeout=30,
                        stream=True
                    )
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield f"data: {json.dumps({'type': 'ai_chunk', 'content': content})}\n\n"
                    yield f"data: {json.dumps({'type': 'ai_complete', 'content': full_response})}\n\n"
                except Exception as e:
                    log_message(f"❌ 对话AI生成失败: {e}", "ERROR")
                    yield f"data: {json.dumps({'type': 'ai_error', 'error': f'AI回答失败：{str(e)}'})}\n\n"

                # 保存历史
                try:
                    init_history_manager()
                    history_manager.add_history(
                        query=message, results=results, ai_response=full_response,
                        strategy='chat', top_k=top_k,
                        performance={
                            'retrieval_time': float(retrieval_time),
                            'total_time': float(time.time() - start_time)
                        }
                    )
                except Exception as e:
                    log_message(f"⚠️ 保存对话历史失败: {e}", "WARN")

                total_time = time.time() - start_time
                log_message(f"💬 对话完成，总耗时: {total_time:.3f}秒")
                yield f"data: {json.dumps({'type': 'complete', 'total_time': total_time})}\n\n"

            except Exception as e:
                log_message(f"❌ 对话请求失败: {e}", "ERROR")
                yield f"data: {json.dumps({'type': 'error', 'error': f'对话出错：{str(e)}'})}\n\n"

        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache', 'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type'
        })

    except Exception as e:
        log_message(f"❌ 对话请求处理失败: {e}", "ERROR")
        return jsonify({'error': f'请求处理失败：{str(e)}'})

@app.route('/api/update', methods=['POST'])
def api_update():
    """更新数据库"""
    log_message("🔄 开始更新向量数据库...")
    
    try:
        # 获取请求参数
        data = request.get_json() or {}
        questions_per_chunk = data.get('questions_per_chunk', 10)  # 默认10个问题/块
        
        log_message(f"📊 问题数量设置: {questions_per_chunk} 个/块")
        
        # 初始化文档管理器
        init_document_manager()
        
        # 检查原始文档是否存在
        original_doc_path = os.path.join(os.path.dirname(BASE_DIR), "documents", "document.txt")
        if not os.path.exists(original_doc_path):
            log_message(f"❌ 原始文档不存在: {original_doc_path}", "ERROR")
            return jsonify({'success': False, 'error': '原始文档 document.txt 不存在'})
        
        # 读取原始文档
        with open(original_doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        log_message(f"📄 读取原始文档: {len(content)} 字符")
        
        # 重新创建数据库
        log_message("🔄 使用 DocumentManager 重新创建数据库...")
        success = document_manager.rebuild_database_from_original(content, questions_per_chunk)
        
        if success:
            log_message("✅ 数据库重建成功")
            log_message("🔄 重新初始化检索器...")
            
            # 重新初始化检索器
            global retriever
            retriever = None
            init_retriever()
            
            log_message("✅ 数据库更新完成")
            return jsonify({'success': True, 'message': f'数据库更新成功，使用了{questions_per_chunk}个问题/块'})
        else:
            log_message("❌ 数据库重建失败", "ERROR")
            return jsonify({'success': False, 'error': '数据库重建失败'})
            
    except Exception as e:
        log_message(f"❌ 数据库更新过程出错: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """处理文件上传"""
    log_message("📤 收到文件上传请求")
    
    try:
        if 'file' not in request.files:
            log_message("⚠️ 上传请求中没有文件", "WARN")
            return jsonify({'success': False, 'error': '没有文件被上传'})
        
        file = request.files['file']
        if file.filename == '':
            log_message("⚠️ 未选择文件", "WARN")
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        log_message(f"📁 上传文件: {file.filename}")
        
        if file and file.filename.endswith('.txt'):
            # 保存文件
            save_path = os.path.join(os.path.dirname(BASE_DIR), "documents", "document.txt")
            file.save(save_path)
            log_message(f"✅ 文件保存成功: {save_path}")
            log_message("💡 提示: 请点击'更新数据库'按钮重新处理文档")
            return jsonify({'success': True, 'message': '文件上传成功，请点击更新数据库'})
        else:
            log_message(f"❌ 不支持的文件格式: {file.filename}", "ERROR")
            return jsonify({'success': False, 'error': '只支持.txt文件'})
            
    except Exception as e:
        log_message(f"❌ 文件上传失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/stats')
def api_document_stats():
    """获取文档数据库统计信息"""
    try:
        init_document_manager()
        
        log_message("📊 获取文档数据库统计信息...")
        stats = document_manager.get_database_stats()
        
        log_message(f"✅ 统计信息获取成功: {stats.get('total_documents', 0)}个文档")
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        log_message(f"❌ 获取统计信息失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/sources')
def api_document_sources():
    """获取文档来源列表"""
    try:
        init_document_manager()
        
        log_message("📂 获取文档来源列表...")
        sources = document_manager.list_document_sources()
        
        log_message(f"✅ 文档来源获取成功: {len(sources)}个来源")
        
        return jsonify({
            'success': True,
            'sources': sources
        })
        
    except Exception as e:
        log_message(f"❌ 获取文档来源失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/add', methods=['POST'])
def api_document_add():
    """添加文档"""
    try:
        init_document_manager()
        
        data = request.get_json()
        content = data.get('content', '').strip()
        name = data.get('name', '').strip()
        skip_duplicates = data.get('skip_duplicates', True)
        questions_per_chunk = data.get('questions_per_chunk', 10)
        
        log_message(f"📝 收到文档添加请求: '{name}' ({len(content)}字符, {questions_per_chunk}问题/块)")
        
        if not content:
            log_message("⚠️ 文档内容为空", "WARN")
            return jsonify({'success': False, 'error': '文档内容不能为空'})
        
        if not name:
            name = f"Web添加文档_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        log_message(f"🔄 开始添加文档: {name}")
        success = document_manager.add_document_from_text(content, name, skip_duplicates, questions_per_chunk)
        
        if success:
            log_message(f"✅ 文档添加成功: {name}")
            
            # 重新初始化检索器以加载新数据
            global retriever
            retriever = None
            init_retriever()
            
            return jsonify({
                'success': True,
                'message': f'文档 "{name}" 添加成功'
            })
        else:
            log_message(f"❌ 文档添加失败: {name}", "ERROR")
            return jsonify({'success': False, 'error': '文档添加失败'})
            
    except Exception as e:
        log_message(f"❌ 文档添加过程中出错: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/upload', methods=['POST'])
def api_document_upload():
    """上传文档文件"""
    try:
        init_document_manager()
        
        if 'file' not in request.files:
            log_message("⚠️ 没有选择文件", "WARN")
            return jsonify({'success': False, 'error': '请选择文件'})
        
        file = request.files['file']
        if file.filename == '':
            log_message("⚠️ 文件名为空", "WARN")
            return jsonify({'success': False, 'error': '文件名不能为空'})
        
        # 检查文件扩展名
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in document_manager.supported_formats:
            log_message(f"⚠️ 不支持的文件格式: {file_ext}", "WARN")
            return jsonify({
                'success': False, 
                'error': f'不支持的文件格式: {file_ext}\n支持的格式: {", ".join(document_manager.supported_formats)}'
            })
        
        # 读取文件内容
        content = file.read().decode('utf-8')
        
        log_message(f"📄 收到文件上传: {filename} ({len(content)}字符)")
        
        # 获取其他参数
        skip_duplicates = request.form.get('skip_duplicates', 'true').lower() == 'true'
        questions_per_chunk = int(request.form.get('questions_per_chunk', '10'))
        
        log_message(f"🔄 开始处理上传文件: {filename} ({questions_per_chunk}问题/块)")
        success = document_manager.add_document_from_text(content, filename, skip_duplicates, questions_per_chunk)
        
        if success:
            log_message(f"✅ 文件上传成功: {filename}")
            
            # 重新初始化检索器以加载新数据
            global retriever
            retriever = None
            init_retriever()
            
            return jsonify({
                'success': True,
                'message': f'文件 "{filename}" 上传成功'
            })
        else:
            log_message(f"❌ 文件上传失败: {filename}", "ERROR")
            return jsonify({'success': False, 'error': '文件上传失败'})
            
    except Exception as e:
        log_message(f"❌ 文件上传过程中出错: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/content/<path:source_name>')
def api_document_content(source_name):
    """获取文档内容"""
    try:
        init_document_manager()
        
        log_message(f"📖 获取文档内容: {source_name}")
        result = document_manager.get_document_content(source_name)
        
        if result['success']:
            log_message(f"✅ 文档内容获取成功: {len(result['chunks'])}个块")
        else:
            log_message(f"❌ 文档内容获取失败: {result['error']}", "ERROR")
        
        return jsonify(result)
        
    except Exception as e:
        log_message(f"❌ 获取文档内容失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/update', methods=['POST'])
def api_document_update():
    """更新文档内容"""
    try:
        init_document_manager()
        
        data = request.get_json()
        source_name = data.get('source_name', '').strip()
        new_content = data.get('content', '').strip()
        
        log_message(f"✏️ 收到文档更新请求: {source_name} ({len(new_content)}字符)")
        
        if not source_name:
            log_message("⚠️ 文档名称为空", "WARN")
            return jsonify({'success': False, 'error': '文档名称不能为空'})
        
        if not new_content:
            log_message("⚠️ 文档内容为空", "WARN")
            return jsonify({'success': False, 'error': '文档内容不能为空'})
        
        log_message(f"🔄 开始更新文档: {source_name}")
        success = document_manager.update_document_content(source_name, new_content)
        
        if success:
            log_message(f"✅ 文档更新成功: {source_name}")
            
            # 重新初始化检索器以加载新数据
            global retriever
            retriever = None
            init_retriever()
            
            return jsonify({
                'success': True,
                'message': f'文档 "{source_name}" 更新成功'
            })
        else:
            log_message(f"❌ 文档更新失败: {source_name}", "ERROR")
            return jsonify({'success': False, 'error': '文档更新失败'})
            
    except Exception as e:
        log_message(f"❌ 文档更新过程中出错: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/delete', methods=['POST'])
def api_document_delete():
    """删除文档"""
    try:
        init_document_manager()
        
        data = request.get_json()
        source_name = data.get('source_name', '').strip()
        
        log_message(f"🗑️ 收到文档删除请求: {source_name}")
        
        if not source_name:
            log_message("⚠️ 文档名称为空", "WARN")
            return jsonify({'success': False, 'error': '文档名称不能为空'})
        
        # 不允许删除原始文档
        if source_name == "document.txt (原始文档)":
            log_message("⚠️ 不允许删除原始文档", "WARN")
            return jsonify({'success': False, 'error': '不允许删除原始文档'})
        
        log_message(f"🗑️ 开始删除文档: {source_name}")
        success = document_manager.delete_document(source_name)
        
        if success:
            log_message(f"✅ 文档删除成功: {source_name}")
            
            # 重新初始化检索器以加载新数据
            global retriever
            retriever = None
            init_retriever()
            
            return jsonify({
                'success': True,
                'message': f'文档 "{source_name}" 删除成功'
            })
        else:
            log_message(f"❌ 文档删除失败: {source_name}", "ERROR")
            return jsonify({'success': False, 'error': '文档删除失败'})
            
    except Exception as e:
        log_message(f"❌ 文档删除过程中出错: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/document/search', methods=['POST'])
def api_document_search():
    """在文档中搜索关键词"""
    try:
        init_document_manager()
        
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        max_results = data.get('max_results', 10)
        
        log_message(f"🔍 收到文档搜索请求: '{keyword}'")
        
        if not keyword:
            log_message("⚠️ 搜索关键词为空", "WARN")
            return jsonify({'success': False, 'error': '搜索关键词不能为空'})
        
        log_message(f"🔎 开始在文档中搜索: {keyword}")
        results = document_manager.search_in_documents(keyword, max_results)
        
        log_message(f"✅ 搜索完成，找到 {len(results)} 个结果")
        
        return jsonify({
            'success': True,
            'keyword': keyword,
            'results': results,
            'total_results': len(results)
        })
        
    except Exception as e:
        log_message(f"❌ 文档搜索失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/backup/list')
def api_backup_list():
    """获取备份文件列表"""
    try:
        init_document_manager()
        
        log_message("📋 获取备份文件列表")
        backups = document_manager.list_backups()
        
        backup_info = []
        for backup_file in backups:
            file_size = os.path.getsize(backup_file)
            file_time = datetime.fromtimestamp(os.path.getctime(backup_file))
            
            backup_info.append({
                'name': os.path.basename(backup_file),
                'path': backup_file,
                'size': file_size,
                'size_mb': file_size / (1024 * 1024),
                'created_time': file_time.isoformat(),
                'created_time_str': file_time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        log_message(f"✅ 找到 {len(backup_info)} 个备份文件")
        
        return jsonify({
            'success': True,
            'backups': backup_info,
            'total_count': len(backup_info),
            'total_size_mb': sum(b['size_mb'] for b in backup_info)
        })
        
    except Exception as e:
        log_message(f"❌ 获取备份列表失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/backup/cleanup', methods=['POST'])
def api_backup_cleanup():
    """清理备份文件"""
    try:
        init_document_manager()
        
        data = request.get_json()
        cleanup_type = data.get('type', 'old')  # 'old' 或 'all'
        
        if cleanup_type == 'all':
            log_message("🧹 清理所有备份文件")
            document_manager.cleanup_all_backups()
            message = "所有备份文件已清理"
        else:
            log_message("🧹 清理旧备份文件")
            document_manager._cleanup_old_backups(3)
            message = "旧备份文件已清理，保留最新3个"
        
        return jsonify({
            'success': True,
            'message': message
        })
        
    except Exception as e:
        log_message(f"❌ 备份清理失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

# 历史记录相关API端点
@app.route('/api/history/list')
def api_history_list():
    """获取历史记录列表"""
    try:
        init_history_manager()
        
        # 获取查询参数
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        log_message(f"📚 获取历史记录列表 (limit: {limit}, offset: {offset})")
        
        history_list = history_manager.get_history(limit=limit, offset=offset)
        
        # 格式化时间显示
        for record in history_list:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'])
                record['time_str'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                record['date_str'] = timestamp.strftime('%Y-%m-%d')
            except:
                record['time_str'] = record['timestamp']
                record['date_str'] = record['timestamp'][:10]
        
        log_message(f"✅ 获取历史记录成功: {len(history_list)} 条记录")
        
        return jsonify({
            'success': True,
            'history': history_list,
            'total_count': len(history_list)
        })
        
    except Exception as e:
        log_message(f"❌ 获取历史记录失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/detail/<record_id>')
def api_history_detail(record_id):
    """获取历史记录详情"""
    try:
        init_history_manager()
        
        log_message(f"📖 获取历史记录详情: {record_id}")
        
        record = history_manager.get_history_by_id(record_id)
        
        if record:
            # 格式化时间显示
            try:
                timestamp = datetime.fromisoformat(record['timestamp'])
                record['time_str'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                record['date_str'] = timestamp.strftime('%Y-%m-%d')
            except:
                record['time_str'] = record['timestamp']
                record['date_str'] = record['timestamp'][:10]
            
            log_message(f"✅ 获取历史记录详情成功")
            return jsonify({
                'success': True,
                'record': record
            })
        else:
            log_message(f"❌ 未找到历史记录: {record_id}", "WARN")
            return jsonify({'success': False, 'error': '未找到指定的历史记录'})
        
    except Exception as e:
        log_message(f"❌ 获取历史记录详情失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/search', methods=['POST'])
def api_history_search():
    """搜索历史记录"""
    try:
        init_history_manager()
        
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        limit = data.get('limit', 20, type=int)
        
        log_message(f"🔍 搜索历史记录: '{keyword}'")
        
        if not keyword:
            log_message("⚠️ 搜索关键词为空", "WARN")
            return jsonify({'success': False, 'error': '搜索关键词不能为空'})
        
        matched_records = history_manager.search_history(keyword, limit=limit)
        
        # 格式化时间显示
        for record in matched_records:
            try:
                timestamp = datetime.fromisoformat(record['timestamp'])
                record['time_str'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                record['date_str'] = timestamp.strftime('%Y-%m-%d')
            except:
                record['time_str'] = record['timestamp']
                record['date_str'] = record['timestamp'][:10]
        
        log_message(f"✅ 搜索历史记录完成: 找到 {len(matched_records)} 条记录")
        
        return jsonify({
            'success': True,
            'keyword': keyword,
            'results': matched_records,
            'total_count': len(matched_records)
        })
        
    except Exception as e:
        log_message(f"❌ 搜索历史记录失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/delete', methods=['POST'])
def api_history_delete():
    """删除历史记录"""
    try:
        init_history_manager()
        
        data = request.get_json()
        record_id = data.get('record_id', '').strip()
        
        log_message(f"🗑️ 删除历史记录: {record_id}")
        
        if not record_id:
            log_message("⚠️ 记录ID为空", "WARN")
            return jsonify({'success': False, 'error': '记录ID不能为空'})
        
        success = history_manager.delete_history(record_id)
        
        if success:
            log_message(f"✅ 历史记录删除成功: {record_id}")
            return jsonify({
                'success': True,
                'message': f'历史记录删除成功'
            })
        else:
            log_message(f"❌ 历史记录删除失败: {record_id}", "ERROR")
            return jsonify({'success': False, 'error': '历史记录删除失败'})
        
    except Exception as e:
        log_message(f"❌ 删除历史记录失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    """清空所有历史记录"""
    try:
        init_history_manager()
        
        log_message("🗑️ 清空所有历史记录")
        
        success = history_manager.clear_history()
        
        if success:
            log_message("✅ 历史记录清空成功")
            return jsonify({
                'success': True,
                'message': '所有历史记录已清空'
            })
        else:
            log_message("❌ 历史记录清空失败", "ERROR")
            return jsonify({'success': False, 'error': '历史记录清空失败'})
        
    except Exception as e:
        log_message(f"❌ 清空历史记录失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/stats')
def api_history_stats():
    """获取历史记录统计信息"""
    try:
        init_history_manager()
        
        log_message("📊 获取历史记录统计信息")
        
        stats = history_manager.get_history_stats()
        
        log_message(f"✅ 历史记录统计获取成功: {stats.get('total_count', 0)} 条记录")
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        log_message(f"❌ 获取历史记录统计失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/export')
def api_history_export():
    """导出历史记录"""
    try:
        init_history_manager()
        
        log_message("📤 导出历史记录")
        
        success = history_manager.export_history()
        
        if success:
            log_message("✅ 历史记录导出成功")
            return jsonify({
                'success': True,
                'message': '历史记录导出成功'
            })
        else:
            log_message("❌ 历史记录导出失败", "ERROR")
            return jsonify({'success': False, 'error': '历史记录导出失败'})
        
    except Exception as e:
        log_message(f"❌ 导出历史记录失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history/cleanup', methods=['POST'])
def api_history_cleanup():
    """清理旧历史记录"""
    try:
        init_history_manager()
        
        data = request.get_json()
        days = data.get('days', 30, type=int)
        
        log_message(f"🧹 清理 {days} 天前的历史记录")
        
        cleaned_count = history_manager.cleanup_old_history(days=days)
        
        log_message(f"✅ 历史记录清理完成: 删除了 {cleaned_count} 条记录")
        
        return jsonify({
            'success': True,
            'message': f'清理完成，删除了 {cleaned_count} 条过期记录',
            'cleaned_count': cleaned_count
        })
        
    except Exception as e:
        log_message(f"❌ 清理历史记录失败: {e}", "ERROR")
        return jsonify({'success': False, 'error': str(e)})

def run_web_server(host='localhost', port=5000):
    """运行Web服务器"""
    print("="*60)
    print("🌐 RAG智能问答系统 - Web服务器")
    print("="*60)
    log_message(f"🚀 启动Web服务器...")
    log_message(f"📍 服务地址: http://{host}:{port}")
    log_message(f"🏠 主页地址: http://{host}:{port}")
    log_message(f"📡 API地址: http://{host}:{port}/api/")
    print("="*60)
    
    # 初始化检索器
    try:
        init_retriever()
        log_message("🎯 Web服务器初始化完成")
    except Exception as e:
        log_message(f"❌ 初始化失败: {e}", "ERROR")
        return
    
    def open_browser():
        """延迟打开浏览器"""
        time.sleep(1.5)
        log_message(f"🌐 自动打开浏览器: http://{host}:{port}")
        webbrowser.open(f'http://{host}:{port}')
    
    # 在后台线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    log_message("✅ Web服务器启动中...")
    log_message("💡 提示: 按 Ctrl+C 停止服务器")
    print("="*60)
    
    # 启动Flask服务器
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        log_message("👋 Web服务器已停止", "INFO")
    except Exception as e:
        log_message(f"❌ Web服务器启动失败: {e}", "ERROR")

if __name__ == "__main__":
    run_web_server() 