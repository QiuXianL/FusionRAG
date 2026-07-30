#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - 命令行界面模块
提供命令行交互功能
"""

import os
import sys
import subprocess
from smart_retrieval import SmartRetriever
from document_manager import DocumentManager
from query_optimizer import QueryOptimizer
from openai import OpenAI

def display_banner():
    """显示程序启动横幅"""
    print("="*60)
    print("🚀 RAG智能问答系统")
    print("="*60)
    print("✨ 特性：")
    print("  🧠 智能检索：多策略自适应检索")
    print("  🚀 GPU加速：自动检测并使用GPU")
    print("  🌐 双模式：支持命令行和Web界面")
    print("  📝 中文优化：专为中文文档优化")
    print("  💾 本地模型：优先使用本地模型")
    print("  📁 文档管理：支持追加和批量添加文档")
    print("="*60)

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查关键文件（相对于 src/ 目录）
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_dir = os.path.dirname(_src_dir)
    required_files = [
        os.path.join(_proj_dir, 'documents', 'document.txt'),
        os.path.join(_src_dir, 'template.html'),
        os.path.join(_src_dir, 'smart_retrieval.py'),
        os.path.join(_src_dir, 'web_app.py'),
        os.path.join(_src_dir, 'document_manager.py')
    ]
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    if not os.path.exists(os.path.join(_proj_dir, 'documents', 'document_vectors_enhanced.json')):
        print("⚠️ 未找到增强向量数据库文件")
        print("💡 请先运行: python web_app.py (在Web界面中点击'更新数据库')")
        return False
    
    # 检查DeepSeek API
    api_key = os.getenv("DEEPSEEK_API")
    if not api_key:
        print("⚠️ 未设置DEEPSEEK_API环境变量")
        print("💡 将无法使用AI增强回答功能")
    else:
        print("✅ DeepSeek API已配置")
    
    print("✅ 环境检查完成")
    return True

def ask_update_database():
    """询问用户是否更新数据库"""
    while True:
        user_input = input("\n🔄 是否需要更新向量数据库？(y/n): ").strip().lower()
        if user_input in ['y', 'yes', '是']:
            return True
        elif user_input in ['n', 'no', '否']:
            return False
        else:
            print("请输入 y/yes/是 或 n/no/否")

def update_database():
    """更新向量数据库"""
    print("\n🔄 开始更新向量数据库...")
    print("=" * 60)
    
    try:
        # 使用 DocumentManager 重新构建数据库
        from document_manager import DocumentManager
        
        # 检查原始文档
        if not os.path.exists('documents/document.txt'):
            print("❌ 原始文档不存在: documents/document.txt")
            return False
        
        # 读取原始文档
        with open('documents/document.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📄 读取原始文档: {len(content)} 字符")
        
        # 初始化文档管理器
        manager = DocumentManager()
        
        # 重建数据库（使用默认的问题数量）
        success = manager.rebuild_database_from_original(content, questions_per_chunk=10)
        
        if success:
            print("✅ 数据库更新成功!")
            return True
        else:
            print("❌ 数据库更新失败")
            return False
            
    except Exception as e:
        print(f"❌ 更新过程中出错: {e}")
        return False

def ask_mode_selection():
    """询问用户选择运行模式"""
    print("\n🎯 请选择运行模式：")
    print("1. 🌐 Web界面模式（推荐）")
    print("2. 💻 命令行模式")
    print("3. 📁 文档管理模式")
    print("4. ❌ 退出")
    
    while True:
        choice = input("\n请输入选择 (1/2/3/4): ").strip()
        if choice == '1':
            return 'web'
        elif choice == '2':
            return 'cli'
        elif choice == '3':
            return 'document'
        elif choice == '4':
            return 'exit'
        else:
            print("请输入有效选择：1、2、3 或 4")

def run_cli_mode():
    """运行命令行模式"""
    print("\n💻 启动命令行模式...")
    
    # 初始化检索器
    try:
        retriever = SmartRetriever()
        print("✅ 智能检索器初始化成功")
    except Exception as e:
        print(f"❌ 检索器初始化失败: {e}")
        return
    
    # 当前使用的策略
    current_strategy = 'fusion'

    print("\n🎯 命令行RAG问答系统已启动")
    print("💡 提示：")
    print("  - 输入问题开始查询")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'help' 查看帮助")
    print("  - 输入 'strategy <策略名>' 切换检索策略")
    
    while True:
        print("\n" + "="*50)
        print(f"📊 当前策略: {current_strategy}")
        query = input("🤔 请输入您的问题: ").strip()
        
        if query.lower() in ['quit', 'exit', '退出', 'q']:
            print("👋 感谢使用RAG系统！")
            break
        
        if query.lower() in ['help', '帮助']:
            print_help()
            continue
        
        if query.lower().startswith('strategy '):
            new_strategy = query.split(' ', 1)[1].strip()
            if new_strategy in ['auto', 'enhanced', 'fusion']:
                current_strategy = new_strategy
                print(f"✅ 已切换到 {new_strategy} 策略")
            else:
                print("❌ 无效的策略名称，支持的策略: auto, enhanced, fusion")
            continue
        
        if not query:
            print("请输入有效的问题")
            continue
        
        # 执行查询
        perform_query(retriever, query, current_strategy)

def print_help():
    """打印帮助信息"""
    print("\n📚 帮助信息：")
    print("  查询命令:")
    print("    - 直接输入问题进行查询")
    print("    - 支持中文问题，如：'南航什么时候成立的？'")
    print("  ")
    print("  特殊命令:")
    print("    - help/帮助    显示帮助信息")
    print("    - quit/exit/退出  退出程序")
    print("    - strategy <策略名> 切换检索策略")
    print("  ")
    print("  检索策略:")
    print("    - auto: 自动选择最佳策略（默认使用enhanced）")
    print("    - enhanced: 分别从文档向量和问题向量中检索文本块，然后合并")
    print("  ")
    print("  说明:")
    print("    - enhanced策略会分别从文档向量和问题向量中各检索top_k个文本块")
    print("    - 然后将两份结果合并发送给AI，提供更全面的上下文信息")

def perform_query(retriever, query, strategy='auto'):
    """执行查询"""
    try:
        print(f"\n🔍 正在搜索: '{query}'")
        print(f"📊 使用策略: {strategy}")
        
        results = []
        
        # Super Brain Logic for Enhanced/Fusion Strategy
        if strategy in ('enhanced', 'auto', 'fusion'):
            # Note: auto currently defaults to enhanced in SmartRetriever
            try:
                optimizer = QueryOptimizer()
                if optimizer.client: # Only if API is available
                    print(f"🧠 Super Brain: 正在多角度思考问题...")
                    queries = optimizer.expand_query(query)
                    
                    if len(queries) > 1:
                        print(f"  ↳ 扩展查询: {queries[1:]}")
                        all_results = {}
                        for q in queries:
                            # Retrieve top 3 for each variation
                            res = retriever.retrieve_with_strategy(q, strategy='enhanced', top_k=3)
                            all_results[q] = res

                        # Fuse results with RRF (Weighted)
                        results = optimizer.fuse_results(all_results, original_query=query)
                        # Keep top 5
                        results = results[:5]
                        print(f"✅ 多路召回融合完成")
                    else:
                        results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=3)
                else:
                    results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=3)
            except Exception as e:
                print(f"⚠️ Super Brain 优化失败，回退到普通检索: {e}")
                results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=3)
        else:
            # 执行普通检索
            results = retriever.retrieve_with_strategy(query, strategy=strategy, top_k=3)
        
        if not results:
            print("❌ 未找到相关文档")
            return
        
        print(f"✅ 找到 {len(results)} 个相关结果：")
        
        # 显示检索结果
        for i, result in enumerate(results, 1):
            print(f"\n📄 结果 {i}:")
            print(f"  📊 相似度: {result['similarity_score']:.4f}")
            print(f"  📝 匹配类型: {result['match_type']}")
            
            # 显示检索来源（如果存在）
            if 'search_source' in result:
                source_desc = "文档向量库" if result['search_source'] == 'document_vector' else "问题向量库"
                print(f"  🔍 检索来源: {source_desc}")
            
            # 显示对应的问题（如果是问题向量）
            if result['match_type'] == 'question' and 'question_text' in result:
                print(f"  ❓ 生成问题: {result['question_text']}")
            
            print(f"  📑 文档块索引: {result['chunk_index']}")
            print(f"  📃 内容预览: {result['original_text'][:200]}...")
        
        # 统计检索来源
        if strategy == 'enhanced' and results:
            doc_count = sum(1 for r in results if r.get('search_source') == 'document_vector')
            question_count = sum(1 for r in results if r.get('search_source') == 'question_vector')
            print(f"\n📊 检索来源统计:")
            print(f"  📝 文档向量库: {doc_count} 个结果")
            print(f"  ❓ 问题向量库: {question_count} 个结果")
        
        # 生成AI增强回答
        generate_ai_response(results, query)
        
    except Exception as e:
        print(f"❌ 查询过程中出错: {e}")

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
        print(f"⚠️ 提取回答内容时出错: {e}")
        return response_text.strip()

def generate_ai_response(results, query):
    """生成AI增强回答"""
    api_key = os.getenv("DEEPSEEK_API")
    
    if not api_key:
        print(f"\n{'='*50}")
        print("⚠️ 未设置DEEPSEEK_API环境变量，跳过AI增强回答")
        print("💡 设置环境变量后可获得AI增强回答")
        print(f"{'='*50}")
        return
    
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
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
        2. 如果文档信息不完整，大胆用你的知识补充，并在文中自然说明哪些推断来自文档、哪些来自背景知识。不要让"文档中没有"成为你不回答的借口。
        3. 请使用Markdown格式回答，包括适当的标题、列表、引用、表格等格式，让回答既有深度又易读。
        4. 像一位真正的学者那样写作——有洞察力，有判断力，敢于下结论。

        请在<回答>标签内写下你的Markdown格式答案。
        <回答>
        [在此给出你作为符文之地贤者的回答]
        </回答>
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        ai_response = response.choices[0].message.content
        
        # 提取<回答>标签之间的内容
        ai_response = extract_answer_content(ai_response)
        
        print(f"\n{'='*50}")
        print("🤖 AI增强回答:")
        print(f"{'='*50}")
        print(ai_response)
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\n❌ AI回答生成失败: {e}")

def run_document_mode():
    """运行文档管理模式"""
    print("\n📁 启动文档管理模式...")
    
    try:
        manager = DocumentManager()
        print("✅ 文档管理器初始化成功")
    except Exception as e:
        print(f"❌ 文档管理器初始化失败: {e}")
        return
    
    while True:
        print("\n📋 文档管理功能菜单：")
        print("1. 📄 添加单个文档")
        print("2. 📚 批量添加文档")
        print("3. 📊 查看数据库统计")
        print("4. 📂 查看文档来源")
        print("5. 📝 从文本添加文档")
        print("6. 🍵 查看文档内容")
        print("7. ✏️ 编辑文档")
        print("8. 🗑️ 删除文档")
        print("9. 🔍 搜索文档")
        print("10. 📋 备份管理")
        print("11. 🔙 返回主菜单")
        
        choice = input("\n请选择功能 (1-11): ").strip()
        
        if choice == '1':
            add_single_document(manager)
        elif choice == '2':
            add_batch_documents(manager)
        elif choice == '3':
            show_database_stats(manager)
        elif choice == '4':
            show_document_sources(manager)
        elif choice == '5':
            add_text_document(manager)
        elif choice == '6':
            view_document_content(manager)
        elif choice == '7':
            edit_document_content(manager)
        elif choice == '8':
            delete_document_content(manager)
        elif choice == '9':
            search_document_content(manager)
        elif choice == '10':
            backup_management(manager)
        elif choice == '11':
            print("🔙 返回主菜单")
            break
        else:
            print("请输入有效选择：1-11")

def add_single_document(manager):
    """添加单个文档"""
    print("\n📄 添加单个文档")
    
    while True:
        file_path = input("请输入文档文件路径（或输入 'back' 返回）: ").strip()
        
        if file_path.lower() == 'back':
            return
        
        if not file_path:
            print("请输入有效的文件路径")
            continue
        
        if not os.path.exists(file_path):
            print("❌ 文件不存在，请检查路径")
            continue
        
        # 询问文档名称
        doc_name = input("请输入文档名称（直接回车使用文件名）: ").strip()
        if not doc_name:
            doc_name = os.path.basename(file_path)
        
        # 询问是否跳过重复
        skip_dup = input("是否跳过重复内容？(y/n，默认y): ").strip().lower()
        skip_duplicates = skip_dup not in ['n', 'no', '否']
        
        try:
            success = manager.add_document_from_file(file_path, doc_name, skip_duplicates)
            if success:
                print("✅ 文档添加成功！")
            else:
                print("❌ 文档添加失败")
        except Exception as e:
            print(f"❌ 添加过程中出错: {e}")
        
        # 询问是否继续添加
        continue_add = input("\n是否继续添加其他文档？(y/n): ").strip().lower()
        if continue_add not in ['y', 'yes', '是']:
            break

def add_batch_documents(manager):
    """批量添加文档"""
    print("\n📚 批量添加文档")
    
    while True:
        folder_path = input("请输入文档文件夹路径（或输入 'back' 返回）: ").strip()
        
        if folder_path.lower() == 'back':
            return
        
        if not folder_path:
            print("请输入有效的文件夹路径")
            continue
        
        if not os.path.exists(folder_path):
            print("❌ 文件夹不存在，请检查路径")
            continue
        
        if not os.path.isdir(folder_path):
            print("❌ 路径不是文件夹")
            continue
        
        # 获取支持的文件
        supported_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in manager.supported_formats):
                    supported_files.append(os.path.join(root, file))
        
        if not supported_files:
            print("❌ 文件夹中没有支持的文档格式")
            print(f"支持的格式: {', '.join(manager.supported_formats)}")
            continue
        
        print(f"📋 找到 {len(supported_files)} 个支持的文档文件:")
        for i, file in enumerate(supported_files[:10], 1):  # 只显示前10个
            print(f"  {i}. {os.path.basename(file)}")
        
        if len(supported_files) > 10:
            print(f"  ... 还有 {len(supported_files) - 10} 个文件")
        
        # 确认处理
        confirm = input(f"\n确认处理这 {len(supported_files)} 个文件？(y/n): ").strip().lower()
        if confirm not in ['y', 'yes', '是']:
            continue
        
        # 询问是否跳过重复
        skip_dup = input("是否跳过重复内容？(y/n，默认y): ").strip().lower()
        skip_duplicates = skip_dup not in ['n', 'no', '否']
        
        try:
            results = manager.add_documents_batch(supported_files, skip_duplicates)
            
            successful = sum(results.values())
            failed = len(results) - successful
            
            print(f"\n📊 批量处理结果:")
            print(f"✅ 成功: {successful} 个文件")
            print(f"❌ 失败: {failed} 个文件")
            
            if failed > 0:
                print("\n❌ 失败文件:")
                for file_path, success in results.items():
                    if not success:
                        print(f"  - {os.path.basename(file_path)}")
                        
        except Exception as e:
            print(f"❌ 批量处理过程中出错: {e}")
        
        break

def add_text_document(manager):
    """从文本添加文档"""
    print("\n📝 从文本添加文档")
    print("请输入文档内容（输入 'END' 结束输入）:")
    
    lines = []
    while True:
        line = input()
        if line.strip() == 'END':
            break
        lines.append(line)
    
    content = '\n'.join(lines).strip()
    
    if not content:
        print("❌ 文档内容为空")
        return
    
    # 询问文档名称
    doc_name = input("请输入文档名称: ").strip()
    if not doc_name:
        doc_name = "手动输入文档"
    
    # 询问是否跳过重复
    skip_dup = input("是否跳过重复内容？(y/n，默认y): ").strip().lower()
    skip_duplicates = skip_dup not in ['n', 'no', '否']
    
    try:
        success = manager.add_document_from_text(content, doc_name, skip_duplicates)
        if success:
            print("✅ 文档添加成功！")
        else:
            print("❌ 文档添加失败")
    except Exception as e:
        print(f"❌ 添加过程中出错: {e}")

def show_database_stats(manager):
    """显示数据库统计信息"""
    print("\n📊 数据库统计信息")
    
    try:
        stats = manager.get_database_stats()
        
        if not stats:
            print("❌ 数据库为空或不存在")
            return
        
        print(f"{'='*40}")
        print(f"📚 总文档块数: {stats['total_documents']}")
        print(f"❓ 总问题数: {stats['total_questions']}")
        print(f"🔍 总向量数: {stats['total_vectors']}")
        print(f"📄 文档向量数: {stats['document_vectors']}")
        print(f"❓ 问题向量数: {stats['question_vectors']}")
        print(f"📂 文档来源数: {stats['sources_count']}")
        print(f"🔢 版本: {stats['version']}")
        print(f"🤖 模型: {stats['model']}")
        print(f"{'='*40}")
        
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")

def show_document_sources(manager):
    """显示文档来源"""
    print("\n📂 文档来源信息")
    
    try:
        sources = manager.list_document_sources()
        
        if not sources:
            print("❌ 没有文档来源信息")
            return
        
        print(f"{'='*60}")
        for i, source in enumerate(sources, 1):
            print(f"{i}. 📄 {source['name']}")
            print(f"   ⏰ 添加时间: {source['added_time']}")
            print(f"   📊 文档块: {source['chunk_count']} 个")
            print(f"   🔍 向量数: {source['vector_count']} 个")
            print(f"   📍 块范围: {source['chunk_range'][0]}-{source['chunk_range'][1]}")
            print()
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ 获取文档来源失败: {e}")

def view_document_content(manager):
    """查看文档内容"""
    print("\n🍵 查看文档内容")
    
    # 先显示所有文档
    try:
        sources = manager.list_document_sources()
        
        if not sources:
            print("❌ 没有文档来源信息")
            return
            
        print(f"\n📂 可用文档 ({len(sources)} 个):")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source['name']} (块数: {source['chunk_count']}, 添加时间: {source['added_time'][:19]})")
        
        # 选择文档
        while True:
            choice = input("\n请选择要查看的文档编号（或输入 'back' 返回）: ").strip()
            
            if choice.lower() == 'back':
                return
            
            try:
                doc_index = int(choice) - 1
                if 0 <= doc_index < len(sources):
                    source_name = sources[doc_index]['name']
                    break
                else:
                    print("请输入有效的编号")
                    continue
            except ValueError:
                print("请输入有效的数字")
                continue
        
        # 获取文档内容
        result = manager.get_document_content(source_name)
        
        if result['success']:
            print(f"\n{'='*60}")
            print(f"📖 文档名称: {result['source']['name']}")
            print(f"⏰ 添加时间: {result['source']['added_time'][:19]}")
            print(f"📄 文档块数: {result['source']['chunk_count']}")
            print(f"🔍 向量数: {result['source']['vector_count']}")
            print(f"📍 块范围: {result['source']['chunk_range'][0]}-{result['source']['chunk_range'][1]}")
            print(f"{'='*60}")
            
            # 显示内容
            print("\n📃 文档内容:")
            print("-" * 60)
            
            # 分段显示，每次显示500字符
            content = result['full_content']
            if len(content) <= 2000:
                print(content)
            else:
                print(f"文档较长（{len(content)}字符），是否完整显示？")
                show_all = input("输入 'y' 显示全部，其他键显示前2000字符: ").strip().lower()
                
                if show_all == 'y':
                    print(content)
                else:
                    print(content[:2000])
                    print(f"\n... （还有{len(content)-2000}字符，输入 'y' 查看完整内容）")
                    
            print("-" * 60)
        else:
            print(f"❌ 获取文档内容失败: {result['error']}")
            
    except Exception as e:
        print(f"❌ 查看文档内容失败: {e}")

def edit_document_content(manager):
    """编辑文档内容"""
    print("\n✏️ 编辑文档内容")
    
    # 先显示所有文档
    try:
        sources = manager.list_document_sources()
        
        if not sources:
            print("❌ 没有文档来源信息")
            return
            
        print(f"\n📂 可用文档 ({len(sources)} 个):")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source['name']} (块数: {source['chunk_count']}, 添加时间: {source['added_time'][:19]})")
        
        # 选择文档
        while True:
            choice = input("\n请选择要编辑的文档编号（或输入 'back' 返回）: ").strip()
            
            if choice.lower() == 'back':
                return
            
            try:
                doc_index = int(choice) - 1
                if 0 <= doc_index < len(sources):
                    source_name = sources[doc_index]['name']
                    break
                else:
                    print("请输入有效的编号")
                    continue
            except ValueError:
                print("请输入有效的数字")
                continue
        
        # 获取文档内容
        result = manager.get_document_content(source_name)
        
        if not result['success']:
            print(f"❌ 获取文档内容失败: {result['error']}")
            return
        
        print(f"\n📖 正在编辑文档: {source_name}")
        print(f"📄 当前内容长度: {len(result['full_content'])} 字符")
        print(f"📍 当前块数: {result['source']['chunk_count']}")
        
        # 显示编辑选项
        print("\n编辑选项:")
        print("1. 📝 重新输入全部内容")
        print("2. 🍵 查看当前内容后再编辑")
        
        edit_choice = input("请选择编辑方式 (1/2): ").strip()
        
        if edit_choice == '2':
            print("\n📃 当前内容:")
            print("-" * 60)
            print(result['full_content'])
            print("-" * 60)
        
        print("\n请输入新的文档内容（输入 'END' 结束输入）:")
        
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        
        new_content = '\n'.join(lines).strip()
        
        if not new_content:
            print("❌ 新内容为空，取消编辑")
            return
        
        if new_content == result['full_content']:
            print("❌ 内容没有变化，取消编辑")
            return
        
        # 确认编辑
        print(f"\n📊 新内容长度: {len(new_content)} 字符")
        print(f"📊 原内容长度: {len(result['full_content'])} 字符")
        
        confirm = input("\n确定要保存编辑吗？这将重新生成向量，可能需要几分钟时间。(y/n): ").strip().lower()
        
        if confirm not in ['y', 'yes', '是']:
            print("❌ 取消编辑")
            return
        
        # 执行更新
        success = manager.update_document_content(source_name, new_content)
        
        if success:
            print("✅ 文档编辑成功！")
        else:
            print("❌ 文档编辑失败")
            
    except Exception as e:
        print(f"❌ 编辑文档失败: {e}")

def delete_document_content(manager):
    """删除文档"""
    print("\n🗑️ 删除文档")
    
    # 先显示所有文档
    try:
        sources = manager.list_document_sources()
        
        if not sources:
            print("❌ 没有文档来源信息")
            return
            
        print(f"\n📂 可用文档 ({len(sources)} 个):")
        for i, source in enumerate(sources, 1):
            delete_hint = " (不可删除)" if source['name'] == "document.txt (原始文档)" else ""
            print(f"  {i}. {source['name']} (块数: {source['chunk_count']}, 添加时间: {source['added_time'][:19]}){delete_hint}")
        
        # 选择文档
        while True:
            choice = input("\n请选择要删除的文档编号（或输入 'back' 返回）: ").strip()
            
            if choice.lower() == 'back':
                return
            
            try:
                doc_index = int(choice) - 1
                if 0 <= doc_index < len(sources):
                    source_name = sources[doc_index]['name']
                    break
                else:
                    print("请输入有效的编号")
                    continue
            except ValueError:
                print("请输入有效的数字")
                continue
        
        # 检查是否为原始文档
        if source_name == "document.txt (原始文档)":
            print("❌ 不允许删除原始文档")
            return
        
        # 显示文档信息
        result = manager.get_document_content(source_name)
        
        if result['success']:
            print(f"\n📖 即将删除文档: {source_name}")
            print(f"📄 文档块数: {result['source']['chunk_count']}")
            print(f"🔍 向量数: {result['source']['vector_count']}")
            print(f"📍 块范围: {result['source']['chunk_range'][0]}-{result['source']['chunk_range'][1]}")
            print(f"⏰ 添加时间: {result['source']['added_time'][:19]}")
        
        # 确认删除
        print(f"\n⚠️ 警告：删除操作不可撤销！")
        confirm = input(f"确定要删除文档 '{source_name}' 吗？(y/n): ").strip().lower()
        
        if confirm not in ['y', 'yes', '是']:
            print("❌ 取消删除")
            return
        
        # 执行删除
        success = manager.delete_document(source_name)
        
        if success:
            print("✅ 文档删除成功！")
        else:
            print("❌ 文档删除失败")
            
    except Exception as e:
        print(f"❌ 删除文档失败: {e}")

def search_document_content(manager):
    """搜索文档内容"""
    print("\n🔍 搜索文档内容")
    
    while True:
        keyword = input("\n请输入搜索关键词（或输入 'back' 返回）: ").strip()
        
        if keyword.lower() == 'back':
            return
        
        if not keyword:
            print("请输入有效的关键词")
            continue
        
        # 询问最大结果数
        max_results = input("请输入最大结果数（默认10）: ").strip()
        if not max_results:
            max_results = 10
        else:
            try:
                max_results = int(max_results)
                if max_results <= 0:
                    max_results = 10
            except ValueError:
                max_results = 10
        
        try:
            print(f"\n🔎 正在搜索关键词: '{keyword}'")
            results = manager.search_in_documents(keyword, max_results)
            
            if not results:
                print("❌ 未找到包含关键词的文档")
                continue
            
            print(f"\n✅ 找到 {len(results)} 个结果：")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n📄 结果 {i}:")
                print(f"  📂 来源: {result['source_name']}")
                print(f"  📍 块索引: {result['chunk_index']}")
                print(f"  🔍 匹配次数: {result['keyword_count']}")
                print(f"  📃 内容预览:")
                print("  " + "-" * 60)
                
                # 显示高亮内容的前300个字符
                content = result['highlighted_content']
                if len(content) > 300:
                    content = content[:300] + "..."
                
                # 简单高亮显示（命令行中用[]标记）
                content = content.replace("**", "[")
                print(f"  {content}")
                print("  " + "-" * 60)
            
            print("=" * 80)
            
            # 询问是否继续搜索
            continue_search = input("\n是否继续搜索其他关键词？(y/n): ").strip().lower()
            if continue_search not in ['y', 'yes', '是']:
                break
                
        except Exception as e:
            print(f"❌ 搜索失败: {e}")

def backup_management(manager):
    """备份管理"""
    print("\n📋 备份管理")
    
    while True:
        print("\n备份管理选项:")
        print("1. 📋 查看备份文件")
        print("2. 🧹 清理所有备份")
        print("3. 🔙 返回上级菜单")
        
        choice = input("\n请选择操作 (1-3): ").strip()
        
        if choice == '1':
            # 查看备份文件
            print("\n📋 正在查看备份文件...")
            backups = manager.list_backups()
            
            if backups:
                print("\n操作选项:")
                print("1. 🧹 清理旧备份（保留最新3个）")
                print("2. 🗑️ 清理所有备份")
                print("3. 🔙 返回")
                
                sub_choice = input("\n请选择操作: ").strip()
                
                if sub_choice == '1':
                    confirm = input("\n确定要清理旧备份吗？(y/n): ").strip().lower()
                    if confirm in ['y', 'yes', '是']:
                        manager._cleanup_old_backups(3)
                elif sub_choice == '2':
                    confirm = input("\n⚠️ 确定要清理所有备份吗？此操作不可撤销！(y/n): ").strip().lower()
                    if confirm in ['y', 'yes', '是']:
                        manager.cleanup_all_backups()
        
        elif choice == '2':
            # 清理所有备份
            print("\n🧹 清理所有备份文件")
            backups = manager.list_backups()
            
            if not backups:
                continue
            
            print("\n⚠️ 警告：此操作将删除所有备份文件，不可撤销！")
            confirm = input("确定要继续吗？(y/n): ").strip().lower()
            
            if confirm in ['y', 'yes', '是']:
                manager.cleanup_all_backups()
            else:
                print("❌ 取消操作")
        
        elif choice == '3':
            print("🔙 返回上级菜单")
            break
        
        else:
            print("请输入有效选择：1-3")

def main():
    """CLI主函数"""
    try:
        # 显示启动横幅
        display_banner()
        
        # 检查运行环境
        if not check_environment():
            print("\n❌ 环境检查失败，程序退出")
            return
        
        # 询问是否更新数据库
        if ask_update_database():
            if not update_database():
                print("❌ 数据库更新失败，程序退出")
                return
        
        # 选择运行模式
        mode = ask_mode_selection()
        
        if mode == 'cli':
            run_cli_mode()
        elif mode == 'document':
            run_document_mode()
        elif mode == 'exit':
            print("👋 再见！")
        else:
            print(f"❌ 不支持的模式: {mode}")
        
    except KeyboardInterrupt:
        print("\n\n👋 程序已被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 