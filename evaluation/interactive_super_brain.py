import os
import sys
import time
import numpy as np
from dotenv import load_dotenv

# Add src to path to allow importing modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from document_manager import DocumentManager
from query_optimizer import QueryOptimizer

def main():
    # Load environment variables
    load_dotenv()
    if not os.getenv("DEEPSEEK_API"):
        print("❌ Error: DEEPSEEK_API key not found in .env file. Please check your .env file.")
        return

    print("\n🧠 Initializing Super Brain Core Components...")
    print("   (Loading embedding model, this may take a few seconds...)")
    
    # Initialize DocumentManager with a temporary DB path to avoid affecting the main DB
    # We disable backup to avoid unnecessary file operations
    doc_manager = DocumentManager(db_path="evaluation/temp_interactive_test.json", backup_enabled=False)
    
    # Initialize QueryOptimizer
    query_optimizer = QueryOptimizer()
    
    print("✅ Initialization Complete.\n")

    while True:
        print("\n" + "="*60)
        print("🤖 Super Brain Interactive Logic Test")
        print("="*60)
        print("此测试包含两个核心部分：")
        print("1. 索引端：测试【文档 -> 多角度问题 -> 聚类去重】的生成逻辑")
        print("2. 检索端：测试【用户问题 -> 思维链扩展】的多路召回逻辑")
        print("-" * 60)
        
        # --- Part 1: Document Processing Test ---
        print("\n📝 [Part 1] 请输入一段示例文档内容 (输入完成后按两次回车):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        document = "\n".join(lines).strip()
        
        if not document:
            print("⚠️ 未输入文档，跳过文档生成测试。")
        else:
            print(f"\n⚙️  正在运行最新生成逻辑 (CoT Generation + Semantic Clustering)...")
            print(f"   文档长度: {len(document)} 字符")
            
            # Step 1: Generate Questions
            print("\n   [Step 1.1] 正在使用思维链 (CoT) 生成候选问题...")
            print("   (AI正在分析关键信息并从不同角度提问...)")
            start_t = time.time()
            
            # Force generate 10 questions to demonstrate clustering capability
            # Accessing protected method for demonstration purpose
            raw_questions = doc_manager._generate_enhanced_questions(document, num_questions=10)
            print(f"   ⏱️  耗时: {time.time()-start_t:.2f}s")
            
            if not raw_questions:
                print("   ❌ 生成失败。")
            else:
                print(f"\n   👉 初始生成的候选问题 ({len(raw_questions)}个):")
                for i, q in enumerate(raw_questions):
                    print(f"      {i+1}. {q}")
                
                # Step 2: Calculate Embeddings
                print("\n   [Step 1.2] 计算语义向量 (Embeddings)...")
                embeddings = []
                for q in raw_questions:
                    # E5 model requires "passage: " prefix for asymmetric tasks
                    embeddings.append(doc_manager.model.encode(f"passage: {q}", normalize_embeddings=True))
                
                # Step 3: Semantic Clustering
                print("\n   [Step 1.3] 执行语义聚类去重 (Semantic Clustering)...")
                # Accessing protected method for demonstration purpose
                opt_questions, _ = doc_manager._optimize_generated_questions(raw_questions, embeddings)
                
                print(f"\n   ✅ 最终入库的高质量问题 ({len(opt_questions)}个):")
                for i, q in enumerate(opt_questions):
                    print(f"      {i+1}. {q}")
                
                removed = len(raw_questions) - len(opt_questions)
                if removed > 0:
                    print(f"\n   ✨ 优化效果: 识别并移除了 {removed} 个语义重复的问题，精简了索引库。")
                else:
                    print(f"\n   ✨ 优化效果: 所有问题均具有独特语义，无需移除。")

        # --- Part 2: Query Expansion Test ---
        print("\n" + "-"*60)
        print("📝 [Part 2] 请输入一个用户查询问题 (用于测试思维链扩展):")
        print("(如果不输入直接回车，将跳过此步骤)")
        question = input("> ").strip()
        
        if question:
            print(f"\n⚙️  正在运行最强大脑检索逻辑 (Super Brain Retrieval)...")
            
            print("\n   [Step 2.1] 激活思维链 (CoT) 进行多视角扩展...")
            start_t = time.time()
            expanded_queries = query_optimizer.expand_query(question)
            print(f"   ⏱️  耗时: {time.time()-start_t:.2f}s")
            
            print(f"\n   👉 原始问题: {question}")
            print(f"   🧠 AI 扩展出的多维搜索视角:")
            for i, q in enumerate(expanded_queries):
                if q == question: 
                    continue # Skip original in list
                print(f"      🔍 {q}")
            
            print(f"\n   💡 原理: 系统将并行检索这些查询，并使用 RRF 算法融合结果，从而大幅提升召回率。")

        # Exit prompt
        print("\n" + "="*60)
        choice = input("🔄 是否再测一次? (y/n): ").lower()
        if choice != 'y':
            break
    
    # Cleanup temp file if exists
    if os.path.exists("evaluation/temp_interactive_test.json"):
        try:
            os.remove("evaluation/temp_interactive_test.json")
        except:
            pass

if __name__ == "__main__":
    main()
