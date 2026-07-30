#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档管理器 - 支持追加文档库功能
支持多种文档格式和批量处理
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import shutil
import hashlib

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class DocumentManager:
    """文档管理器，负责文档的追加、更新和管理"""
    
    def __init__(self,
                 model_path: str = None,
                 db_path: str = None,
                 backup_enabled: bool = True):
        """
        初始化文档管理器

        Args:
            model_path: 模型路径
            db_path: 向量数据库路径
            backup_enabled: 是否启用备份
        """
        # 使用脚本所在目录的上级目录（项目根目录）来解析相对路径
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(_base, "models", "e5-base-v2")
        if db_path is None:
            db_path = os.path.join(_base, "documents", "document_vectors_enhanced.json")
        self.model_path = model_path
        self.db_path = db_path
        self.backup_enabled = backup_enabled
        self.device = self._get_device()
        self.model = self._load_model()
        
        # 文本分块器配置
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # chunk大小
            chunk_overlap=30,  # 重叠大小
            separators=["\n\n", "\n", "。", "？", "！", "……", "．", ".", "?", "!", "\r"],
            length_function=len,
            is_separator_regex=False
        )
        
        # 支持的文档格式
        self.supported_formats = ['.txt', '.md', '.py', '.json', '.csv']
        
        print(f"📁 文档管理器初始化完成")
        print(f"  💾 模型路径: {model_path}")
        print(f"  🗄️ 数据库路径: {db_path}")
        print(f"  🔄 备份功能: {'开启' if backup_enabled else '关闭'}")
    
    def _get_device(self) -> str:
        """检测并返回最佳设备"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 检测到GPU: {gpu_name}")
            return device
        else:
            print("💻 使用CPU处理")
            return 'cpu'
    
    def _load_model(self) -> SentenceTransformer:
        """加载模型"""
        try:
            if os.path.exists(self.model_path):
                print(f"💾 加载本地模型: {self.model_path}")
                return SentenceTransformer(self.model_path, device=self.device)
            else:
                print("⚠️ 本地模型不存在，从网络下载...")
                return SentenceTransformer("intfloat/e5-base-v2", device=self.device)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _create_backup(self) -> bool:
        """创建数据库备份（智能备份策略）"""
        if not self.backup_enabled or not os.path.exists(self.db_path):
            return True
        
        try:
            # 智能备份策略：检查是否需要创建新备份
            if not self._should_create_backup():
                return True
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"
            shutil.copy2(self.db_path, backup_path)
            print(f"✅ 备份创建成功: {backup_path}")
            
            # 清理旧备份文件
            self._cleanup_old_backups()
            
            return True
        except Exception as e:
            print(f"❌ 备份创建失败: {e}")
            return False
    
    def _should_create_backup(self) -> bool:
        """判断是否需要创建新备份"""
        try:
            # 查找现有备份文件
            backup_pattern = f"{self.db_path}.backup_*"
            import glob
            backup_files = glob.glob(backup_pattern)
            
            if not backup_files:
                return True  # 没有备份文件，需要创建
            
            # 获取最新备份文件
            latest_backup = max(backup_files, key=os.path.getctime)
            
            # 检查最新备份的创建时间
            backup_time = os.path.getctime(latest_backup)
            current_time = datetime.now().timestamp()
            
            # 如果最新备份是5分钟内创建的，不创建新备份
            if current_time - backup_time < 300:  # 5分钟 = 300秒
                print(f"📋 使用现有备份: {os.path.basename(latest_backup)}")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ 检查备份状态失败: {e}")
            return True  # 出错时选择创建备份
    
    def _cleanup_old_backups(self, keep_count: int = 3):
        """清理旧备份文件，保留最新的几个"""
        try:
            # 查找所有备份文件
            backup_pattern = f"{self.db_path}.backup_*"
            import glob
            backup_files = glob.glob(backup_pattern)
            
            if len(backup_files) <= keep_count:
                return  # 备份文件数量不超过限制
            
            # 按创建时间排序
            backup_files.sort(key=os.path.getctime, reverse=True)
            
            # 删除多余的备份文件
            files_to_delete = backup_files[keep_count:]
            deleted_count = 0
            
            for backup_file in files_to_delete:
                try:
                    os.remove(backup_file)
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ 删除备份文件失败: {os.path.basename(backup_file)}")
            
            if deleted_count > 0:
                print(f"🧹 清理了 {deleted_count} 个旧备份文件，保留最新 {keep_count} 个")
                
        except Exception as e:
            print(f"❌ 清理备份文件失败: {e}")
    
    def cleanup_all_backups(self):
        """清理所有备份文件"""
        try:
            backup_pattern = f"{self.db_path}.backup_*"
            import glob
            backup_files = glob.glob(backup_pattern)
            
            deleted_count = 0
            total_size = 0
            
            for backup_file in backup_files:
                try:
                    file_size = os.path.getsize(backup_file)
                    os.remove(backup_file)
                    deleted_count += 1
                    total_size += file_size
                except Exception as e:
                    print(f"⚠️ 删除备份文件失败: {os.path.basename(backup_file)}")
            
            if deleted_count > 0:
                size_mb = total_size / (1024 * 1024)
                print(f"🧹 清理了 {deleted_count} 个备份文件，释放空间 {size_mb:.2f} MB")
            else:
                print("📋 没有备份文件需要清理")
                
        except Exception as e:
            print(f"❌ 清理备份文件失败: {e}")
    
    def list_backups(self):
        """列出所有备份文件"""
        try:
            backup_pattern = f"{self.db_path}.backup_*"
            import glob
            backup_files = glob.glob(backup_pattern)
            
            if not backup_files:
                print("📋 没有找到备份文件")
                return []
            
            # 按创建时间排序
            backup_files.sort(key=os.path.getctime, reverse=True)
            
            print(f"📋 找到 {len(backup_files)} 个备份文件:")
            total_size = 0
            
            for i, backup_file in enumerate(backup_files, 1):
                file_size = os.path.getsize(backup_file)
                file_time = datetime.fromtimestamp(os.path.getctime(backup_file))
                size_mb = file_size / (1024 * 1024)
                total_size += file_size
                
                print(f"  {i}. {os.path.basename(backup_file)}")
                print(f"     时间: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     大小: {size_mb:.2f} MB")
                print()
            
            total_mb = total_size / (1024 * 1024)
            print(f"📊 总计: {len(backup_files)} 个文件，{total_mb:.2f} MB")
            
            return backup_files
            
        except Exception as e:
            print(f"❌ 列出备份文件失败: {e}")
            return []
    
    def _load_database(self) -> Optional[Dict[str, Any]]:
        """加载现有数据库"""
        if not os.path.exists(self.db_path):
            print("⚠️ 数据库文件不存在，将创建新数据库")
            return None
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 加载数据库成功，包含 {len(data.get('documents', []))} 个文档块")
            return data
        except Exception as e:
            print(f"❌ 数据库加载失败: {e}")
            return None
    
    def _save_database(self, data: Dict[str, Any]) -> bool:
        """保存数据库"""
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ 数据库保存成功: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ 数据库保存失败: {e}")
            return False
    
    def _extract_keywords(self, client: OpenAI, content: str) -> str:
        """Extract keywords from content"""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", 
                     "content": """
                    You are a keyword extraction expert. Please extract key information from the provided document content.
                    Keywords should cover: important entities, time, locations, core concepts, etc.
                    Please output the keywords directly, separated by commas, without any other content.
                     """},
                    {"role": "user", "content": f"Document content:\n{content}"}
                ],
                stream=False,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Keyword extraction failed: {e}")
            return ""

    def _generate_enhanced_questions(self, content: str, num_questions: int = 3) -> List[str]:
        """
        使用DeepSeek API生成增强问题
        """
        # Ensure env vars are loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("DEEPSEEK_API")
        
        if not api_key:
            print("⚠️ DEEPSEEK_API not set, skipping question generation")
            return []
        
        try:
            print(f"DEBUG: Calling DeepSeek API with key ending in ...{api_key[-4:]}")
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            # Extract keywords
            keywords = self._extract_keywords(client, content)
            # print(f"  Keywords: {keywords}") # Optional logging
            
            # 2. Generate questions based on keywords and content using CoT (Chain of Thought)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", 
                     "content": f"""
                    You are an expert content analyst and question generator.
                    
                    Step 1: Analyze the text and keywords to identify the {num_questions} most critical pieces of information (facts, causal relationships, or definitions).
                    Step 2: For each piece of information, formulate a specific question that requires understanding that information to answer.
                    
                    Extracted Keywords: {keywords}
                    
                    Requirements:
                    - Questions must be self-contained (avoid "he", "it", "they" without context).
                    - Questions must be diverse (Who/When/Where vs Why/How).
                    - Generate exactly {num_questions} questions.
                    - Output ONLY the questions inside the tags.
                    
                    Output format:
                    <question_list>
                    [Question 1]
                    [Question 2]
                    ...
                    </question_list>
                     """},
                    {"role": "user", "content": f"Document content:\n{content}"}
                ],
                stream=False,
                temperature=0.3
            )
            
            generated_content = response.choices[0].message.content
            print(f"DEBUG: Raw DeepSeek Response:\n{generated_content}")
            
            if "<question_list>" in generated_content:
                generated_content = generated_content.split("<question_list>")[1].split("</question_list>")[0]
                
            questions = [q.strip() for q in generated_content.split('\n') if q.strip() and ('？' in q or '?' in q)]
            print(f"DEBUG: Parsed questions (before filter): {questions}")
            
            # Quality filtering
            quality_questions = []
            for q in questions:
                # Check question length and question mark count (support both Chinese and English question marks)
                question_mark_count = q.count('？') + q.count('?')
                # Relaxed length limit to 200 to accommodate English questions
                if 5 <= len(q) <= 200 and question_mark_count >= 1:
                    quality_questions.append(q)
                else:
                    print(f"DEBUG: Filtered out: {q} (Len: {len(q)}, Marks: {question_mark_count})")
            
            return quality_questions[:num_questions]
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            return []
    
    def _optimize_generated_questions(self, questions: List[str], embeddings: List[np.ndarray]) -> Tuple[List[str], List[np.ndarray]]:
        """
        使用语义相似度去重优化生成的问题 (Smart Clustering)
        通过计算余弦相似度，去除语义重复的问题，优先保留信息量大（较长）的问题
        """
        if not questions or len(questions) < 2:
            return questions, embeddings
            
        try:
            # 转换 list 为 numpy array 以便计算
            emb_matrix = np.array(embeddings)
            
            # 计算余弦相似度矩阵 (假设 embeddings 已经是 normalized 的)
            # 如果不确定是否 normalized，可以重新 normalize:
            # norm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            # normalized_embs = emb_matrix / (norm + 1e-10)
            
            # 由于 sentence-transformers encode(normalize_embeddings=True) 已经归一化
            similarity_matrix = np.dot(emb_matrix, emb_matrix.T)
            
            # 贪心去重策略
            # 1. 按问题长度排序（优先保留信息量大的长问题）
            # 创建 (index, length) 列表
            indexed_questions = [(i, len(q)) for i, q in enumerate(questions)]
            # 按长度降序排序
            sorted_indices = [x[0] for x in sorted(indexed_questions, key=lambda x: x[1], reverse=True)]
            
            keep_indices = []
            removed_indices = set()
            
            for i in sorted_indices:
                if i in removed_indices:
                    continue
                
                keep_indices.append(i)
                
                # 找到与当前问题相似度高的其他问题，标记为移除
                for j in sorted_indices:
                    if j != i and j not in removed_indices:
                        # 阈值 0.90: 稍微放松，仅去除极度相似的问题
                        if similarity_matrix[i][j] > 0.90:
                            removed_indices.add(j)
            
            # 重新构建结果
            optimized_questions = [questions[i] for i in keep_indices]
            optimized_embeddings = [embeddings[i] for i in keep_indices]
            
            if len(questions) != len(optimized_questions):
                print(f"    ✨ 智能聚类优化: {len(questions)} -> {len(optimized_questions)} 个问题 (去除 {len(questions)-len(optimized_questions)} 个冗余)")
                
            return optimized_questions, optimized_embeddings
            
        except Exception as e:
            print(f"    ⚠️ 优化问题失败: {e}")
            return questions, embeddings

    def _create_document_vectors(self, chunks: List[str], questions_per_chunk: int = 10) -> Dict[str, Any]:
        """为文档块创建向量"""
        all_embeddings = []
        all_questions = []
        vector_types = []
        vector_to_chunk_map = []
        
        print(f"🔄 开始处理 {len(chunks)} 个文档块...")
        
        for i, chunk in enumerate(chunks):
            print(f"  处理第 {i+1}/{len(chunks)} 个文档块...")
            
            try:
                # 1. 创建文档向量 - 添加e5-base-v2要求的"passage: "前缀
                chunk_with_prefix = f"passage: {chunk}"
                doc_embedding = self.model.encode(chunk_with_prefix, normalize_embeddings=True)
                all_embeddings.append(doc_embedding)
                vector_types.append('document')
                vector_to_chunk_map.append(i)
                
                # 2. 生成问题向量
                questions = self._generate_enhanced_questions(chunk, num_questions=questions_per_chunk)
                
                # 先生成所有问题的向量
                temp_embeddings = []
                temp_questions = []
                
                for question in questions:
                    try:
                        # 为生成问题添加"passage: "前缀
                        question_with_prefix = f"passage: {question}"
                        question_embedding = self.model.encode(question_with_prefix, normalize_embeddings=True)
                        temp_embeddings.append(question_embedding)
                        temp_questions.append(question)
                    except Exception as e:
                        print(f"    问题向量生成失败: {e}")
                        continue
                
                # 3. 应用智能聚类优化
                final_questions, final_embeddings = self._optimize_generated_questions(temp_questions, temp_embeddings)
                
                # 4. 添加到最终列表
                all_questions.extend(final_questions)
                all_embeddings.extend(final_embeddings)
                
                # 批量更新类型和映射
                vector_types.extend(['question'] * len(final_questions))
                vector_to_chunk_map.extend([i] * len(final_questions))
                
                print(f"    生成 {1 + len(final_questions)} 个向量 (1 文档 + {len(final_questions)} 问题)")
                
            except Exception as e:
                print(f"    文档块处理失败: {e}")
                continue
        
        return {
            'chunks': chunks,
            'embeddings': all_embeddings,
            'questions': all_questions,
            'vector_types': vector_types,
            'vector_to_chunk_map': vector_to_chunk_map
        }
    
    def _calculate_document_hash(self, content: str) -> str:
        """计算文档内容哈希值，用于检测重复"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _check_duplicate_content(self, new_chunks: List[str], existing_data: Dict[str, Any]) -> List[int]:
        """检查重复内容，返回重复的块索引"""
        if not existing_data or 'documents' not in existing_data:
            return []
        
        existing_docs = existing_data['documents']
        existing_hashes = {self._calculate_document_hash(doc): i for i, doc in enumerate(existing_docs)}
        
        duplicate_indices = []
        for i, chunk in enumerate(new_chunks):
            chunk_hash = self._calculate_document_hash(chunk)
            if chunk_hash in existing_hashes:
                duplicate_indices.append(i)
        
        return duplicate_indices
    
    def add_document_from_file(self, file_path: str, 
                              document_name: Optional[str] = None,
                              skip_duplicates: bool = True) -> bool:
        """
        从文件添加文档到向量数据库
        
        Args:
            file_path: 文档文件路径
            document_name: 文档名称（可选）
            skip_duplicates: 是否跳过重复内容
            
        Returns:
            bool: 是否成功添加
        """
        
        # 检查文件格式
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            print(f"❌ 不支持的文件格式: {file_ext}")
            print(f"支持的格式: {', '.join(self.supported_formats)}")
            return False
        
        # 读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print("❌ 文件内容为空")
                return False
                
            print(f"📄 读取文件成功: {file_path}")
            print(f"📊 文件大小: {len(content)} 字符")
            
        except Exception as e:
            print(f"❌ 文件读取失败: {e}")
            return False
        
        return self.add_document_from_text(content, document_name or os.path.basename(file_path), skip_duplicates)
    
    def add_document_from_text(self, content: str, 
                              document_name: str = "unnamed_document",
                              skip_duplicates: bool = True,
                              questions_per_chunk: int = 10) -> bool:
        """
        从文本内容添加文档到向量数据库
        
        Args:
            content: 文档内容
            document_name: 文档名称
            skip_duplicates: 是否跳过重复内容
            questions_per_chunk: 每个文档块生成的问题数量
            
        Returns:
            bool: 是否成功添加
        """
        
        print(f"\n🔄 开始添加文档: {document_name}")
        print(f"{'='*60}")
        
        # 创建备份
        if not self._create_backup():
            print("❌ 备份创建失败，为安全起见，停止操作")
            return False
        
        # 加载现有数据库
        existing_data = self._load_database()
        
        # 文本分块
        chunks = self.text_splitter.split_text(content)
        print(f"📄 文本分块完成，共 {len(chunks)} 个块")
        
        # 检查重复内容
        if skip_duplicates and existing_data:
            duplicate_indices = self._check_duplicate_content(chunks, existing_data)
            if duplicate_indices:
                print(f"⚠️ 检测到 {len(duplicate_indices)} 个重复块，将跳过")
                chunks = [chunk for i, chunk in enumerate(chunks) if i not in duplicate_indices]
                
                if not chunks:
                    print("❌ 所有内容都重复，无需添加")
                    return False
        
        # 创建向量
        vector_data = self._create_document_vectors(chunks, questions_per_chunk)
        
        if not vector_data['embeddings']:
            print("❌ 未能生成任何向量")
            return False
        
        # 合并数据
        if existing_data:
            # 更新现有数据
            chunk_offset = len(existing_data['documents'])
            
            # 更新文档
            existing_data['documents'].extend(vector_data['chunks'])
            existing_data['questions'].extend(vector_data['questions'])
            
            # 更新向量
            existing_embeddings = np.array(existing_data['embeddings'])
            new_embeddings = np.array([emb.tolist() for emb in vector_data['embeddings']])
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
            existing_data['embeddings'] = combined_embeddings.tolist()
            
            # 更新向量类型和映射
            existing_data['vector_types'].extend(vector_data['vector_types'])
            adjusted_map = [idx + chunk_offset for idx in vector_data['vector_to_chunk_map']]
            existing_data['vector_to_chunk_map'].extend(adjusted_map)
            
            # 更新元数据
            existing_data['version'] = f"2.1.0_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 添加文档来源信息
            if 'document_sources' not in existing_data:
                existing_data['document_sources'] = []
            
            existing_data['document_sources'].append({
                'name': document_name,
                'added_time': datetime.now().isoformat(),
                'chunk_range': [chunk_offset, chunk_offset + len(vector_data['chunks']) - 1],
                'chunk_count': len(vector_data['chunks']),
                'vector_count': len(vector_data['embeddings'])
            })
            
            final_data = existing_data
            
        else:
            # 创建新数据库
            final_data = {
                "documents": vector_data['chunks'],
                "questions": vector_data['questions'],
                "embeddings": [emb.tolist() for emb in vector_data['embeddings']],
                "vector_types": vector_data['vector_types'],
                "vector_to_chunk_map": vector_data['vector_to_chunk_map'],
                "model": "intfloat/e5-base-v2",
                "version": f"2.1.0_created_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "config": {
                    "chunk_size": 300,
                    "chunk_overlap": 30,
                    "questions_per_chunk": 10,
                    "hybrid_approach": True
                },
                "document_sources": [{
                    'name': document_name,
                    'added_time': datetime.now().isoformat(),
                    'chunk_range': [0, len(vector_data['chunks']) - 1],
                    'chunk_count': len(vector_data['chunks']),
                    'vector_count': len(vector_data['embeddings'])
                }]
            }
        
        # 保存数据库
        if self._save_database(final_data):
            print(f"\n✅ 文档添加成功!")
            print(f"📊 统计信息:")
            print(f"  📄 新增文档块: {len(vector_data['chunks'])}")
            print(f"  🔍 新增向量: {len(vector_data['embeddings'])}")
            print(f"  ❓ 新增问题: {len(vector_data['questions'])}")
            print(f"  📚 总文档块: {len(final_data['documents'])}")
            print(f"  🔢 总向量数: {len(final_data['embeddings'])}")
            print(f"{'='*60}")
            return True
        else:
            print("❌ 数据库保存失败")
            return False

    def add_documents_from_texts_batch(self, contents_dict: Dict[str, str], 
                                      questions_per_chunk: int = 10) -> bool:
        """
        批量从文本字典添加文档
        contents_dict: {document_name: content}
        """
        print(f"\n🔄 开始批量添加 {len(contents_dict)} 个文档...")
        print(f"{'='*60}")
        
        # 1. 创建备份
        if not self._create_backup():
            print("❌ 备份创建失败")
            return False
        
        # 2. 加载现有数据库
        existing_data = self._load_database()
        
        # 初始化数据库结构如果为空
        if not existing_data:
            existing_data = {
                "documents": [],
                "questions": [],
                "embeddings": [],
                "vector_types": [],
                "vector_to_chunk_map": [],
                "model": "intfloat/e5-base-v2",
                "version": f"2.1.0_created_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "config": {
                    "chunk_size": 300,
                    "chunk_overlap": 30,
                    "questions_per_chunk": questions_per_chunk,
                    "hybrid_approach": True
                },
                "document_sources": []
            }
        
        # 3. 处理每个文档
        processed_count = 0
        total_docs = len(contents_dict)
        
        for doc_name, content in contents_dict.items():
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processing {processed_count}/{total_docs}: {doc_name}...")
             
            # 分块
            chunks = self.text_splitter.split_text(content)
            if not chunks: continue
             
            # 创建向量 (复用现有的方法)
            vector_data = self._create_document_vectors(chunks, questions_per_chunk)
            if not vector_data['embeddings']: continue
             
            # 追加数据
            chunk_offset = len(existing_data['documents'])
             
            existing_data['documents'].extend(vector_data['chunks'])
            existing_data['questions'].extend(vector_data['questions'])
             
            # 直接追加列表格式的embedding
            existing_data['embeddings'].extend([emb.tolist() for emb in vector_data['embeddings']])
             
            existing_data['vector_types'].extend(vector_data['vector_types'])
            adjusted_map = [idx + chunk_offset for idx in vector_data['vector_to_chunk_map']]
            existing_data['vector_to_chunk_map'].extend(adjusted_map)
             
            # 更新元数据
            if 'document_sources' not in existing_data: existing_data['document_sources'] = []
            existing_data['document_sources'].append({
                'name': doc_name,
                'added_time': datetime.now().isoformat(),
                'chunk_range': [chunk_offset, chunk_offset + len(vector_data['chunks']) - 1],
                'chunk_count': len(vector_data['chunks']),
                'vector_count': len(vector_data['embeddings'])
            })
            
        # 4. 一次性保存
        print(f"💾 正在保存数据库 ({len(existing_data['documents'])} chunks)...")
        return self._save_database(existing_data)
    
    def add_documents_batch(self, file_paths: List[str], 
                           skip_duplicates: bool = True) -> Dict[str, bool]:
        """
        批量添加文档
        
        Args:
            file_paths: 文档文件路径列表
            skip_duplicates: 是否跳过重复内容
            
        Returns:
            Dict[str, bool]: 每个文件的处理结果
        """
        
        print(f"\n🔄 开始批量添加文档")
        print(f"📝 文件数量: {len(file_paths)}")
        print(f"{'='*60}")
        
        results = {}
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n📄 处理文件 {i}/{len(file_paths)}: {os.path.basename(file_path)}")
            
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                results[file_path] = False
                continue
            
            try:
                success = self.add_document_from_file(file_path, skip_duplicates=skip_duplicates)
                results[file_path] = success
                
                if success:
                    print(f"✅ 处理成功: {os.path.basename(file_path)}")
                else:
                    print(f"❌ 处理失败: {os.path.basename(file_path)}")
                    
            except Exception as e:
                print(f"❌ 处理异常: {e}")
                results[file_path] = False
        
        # 显示汇总结果
        successful = sum(results.values())
        failed = len(results) - successful
        
        print(f"\n📊 批量处理完成")
        print(f"✅ 成功: {successful} 个文件")
        print(f"❌ 失败: {failed} 个文件")
        print(f"{'='*60}")
        
        return results
    
    def list_document_sources(self) -> List[Dict[str, Any]]:
        """列出所有文档来源"""
        data = self._load_database()
        if not data:
            return []
        
        sources = data.get('document_sources', [])
        documents = data.get('documents', [])
        vector_to_chunk_map = data.get('vector_to_chunk_map', [])
        
        # 检查是否有原始文档没有被记录在document_sources中
        if sources and documents:
            # 获取所有已记录的chunk范围
            recorded_chunks = set()
            for source in sources:
                chunk_range = source.get('chunk_range', [])
                if len(chunk_range) >= 2:
                    for i in range(chunk_range[0], chunk_range[1] + 1):
                        recorded_chunks.add(i)
            
            # 检查是否有未记录的chunk
            max_chunk = max(vector_to_chunk_map) if vector_to_chunk_map else 0
            unrecorded_chunks = []
            for i in range(max_chunk + 1):
                if i not in recorded_chunks:
                    unrecorded_chunks.append(i)
            
            # 如果有未记录的chunk，为原始文档创建虚拟条目
            if unrecorded_chunks:
                # 计算原始文档的向量数量
                original_vector_count = sum(1 for chunk in vector_to_chunk_map if chunk in unrecorded_chunks)
                
                original_source = {
                    "name": "document.txt (原始文档)",
                    "added_time": "2025-01-01T00:00:00",  # 使用一个较早的时间表示原始文档
                    "chunk_range": [min(unrecorded_chunks), max(unrecorded_chunks)],
                    "chunk_count": len(unrecorded_chunks),
                    "vector_count": original_vector_count
                }
                
                # 将原始文档插入到列表开头
                sources.insert(0, original_source)
        
        elif documents and not sources:
            # 如果没有任何document_sources记录，但有documents，说明全部都是原始文档
            chunk_count = len(documents)
            vector_count = len(vector_to_chunk_map)
            
            original_source = {
                "name": "document.txt (原始文档)",
                "added_time": "2025-01-01T00:00:00",
                "chunk_range": [0, chunk_count - 1],
                "chunk_count": chunk_count,
                "vector_count": vector_count
            }
            
            sources = [original_source]
        
        return sources
    
    def get_document_content(self, source_name: str) -> Dict[str, Any]:
        """获取指定文档的内容"""
        data = self._load_database()
        if not data:
            return {'success': False, 'error': '数据库为空'}
        
        sources = self.list_document_sources()
        target_source = None
        
        for source in sources:
            if source['name'] == source_name:
                target_source = source
                break
        
        if not target_source:
            return {'success': False, 'error': f'未找到文档: {source_name}'}
        
        # 获取文档块内容
        documents = data.get('documents', [])
        chunk_range = target_source['chunk_range']
        
        content_chunks = []
        for i in range(chunk_range[0], chunk_range[1] + 1):
            if i < len(documents):
                content_chunks.append({
                    'chunk_index': i,
                    'content': documents[i]
                })
        
        return {
            'success': True,
            'source': target_source,
            'chunks': content_chunks,
            'full_content': '\n\n'.join([chunk['content'] for chunk in content_chunks])
        }
    
    def update_document_content(self, source_name: str, new_content: str) -> bool:
        """更新指定文档的内容"""
        try:
            # 创建备份
            if not self._create_backup():
                print("⚠️ 备份创建失败，但继续执行更新")
            
            data = self._load_database()
            if not data:
                print("❌ 数据库为空")
                return False
            
            sources = self.list_document_sources()
            target_source = None
            
            for source in sources:
                if source['name'] == source_name:
                    target_source = source
                    break
            
            if not target_source:
                print(f"❌ 未找到文档: {source_name}")
                return False
            
            print(f"🔄 开始更新文档: {source_name}")
            
            # 删除原有内容
            if not self.delete_document(source_name, update_database=False):
                print("❌ 删除原文档失败")
                return False
            
            # 重新加载数据（因为删除操作已经修改了数据库）
            data = self._load_database()
            
            # 添加新内容
            chunks = self.text_splitter.split_text(new_content)
            print(f"📄 新内容分块完成，共 {len(chunks)} 个块")
            
            # 创建向量
            vector_data = self._create_document_vectors(chunks)
            
            if not vector_data['embeddings']:
                print("❌ 未能生成任何向量")
                return False
            
            # 添加到数据库
            chunk_offset = len(data['documents'])
            
            # 更新文档
            data['documents'].extend(vector_data['chunks'])
            data['questions'].extend(vector_data['questions'])
            
            # 更新向量
            if data['embeddings']:
                existing_embeddings = np.array(data['embeddings'])
                new_embeddings = np.array([emb.tolist() for emb in vector_data['embeddings']])
                combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
                data['embeddings'] = combined_embeddings.tolist()
            else:
                data['embeddings'] = [emb.tolist() for emb in vector_data['embeddings']]
            
            # 更新向量类型和映射
            data['vector_types'].extend(vector_data['vector_types'])
            adjusted_map = [idx + chunk_offset for idx in vector_data['vector_to_chunk_map']]
            data['vector_to_chunk_map'].extend(adjusted_map)
            
            # 更新元数据
            data['version'] = f"2.1.0_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 添加文档来源信息
            if 'document_sources' not in data:
                data['document_sources'] = []
            
            data['document_sources'].append({
                'name': source_name,
                'added_time': datetime.now().isoformat(),
                'chunk_range': [chunk_offset, chunk_offset + len(vector_data['chunks']) - 1],
                'chunk_count': len(vector_data['chunks']),
                'vector_count': len(vector_data['embeddings'])
            })
            
            # 保存数据库
            if self._save_database(data):
                print(f"✅ 文档更新成功: {source_name}")
                print(f"📊 新文档块: {len(vector_data['chunks'])}")
                print(f"🔍 新向量数: {len(vector_data['embeddings'])}")
                return True
            else:
                print("❌ 数据库保存失败")
                return False
                
        except Exception as e:
            print(f"❌ 更新文档失败: {e}")
            return False
    
    def delete_document(self, source_name: str, update_database: bool = True) -> bool:
        """删除指定文档"""
        try:
            if update_database:
                # 创建备份
                if not self._create_backup():
                    print("⚠️ 备份创建失败，但继续执行删除")
            
            data = self._load_database()
            if not data:
                print("❌ 数据库为空")
                return False
            
            sources = self.list_document_sources()
            target_source = None
            source_index = -1
            
            for i, source in enumerate(sources):
                if source['name'] == source_name:
                    target_source = source
                    source_index = i
                    break
            
            if not target_source:
                print(f"❌ 未找到文档: {source_name}")
                return False
            
            print(f"🗑️ 开始删除文档: {source_name}")
            
            # 获取要删除的chunk范围
            chunk_range = target_source['chunk_range']
            chunks_to_delete = set(range(chunk_range[0], chunk_range[1] + 1))
            
            print(f"📄 删除chunk范围: {chunk_range[0]}-{chunk_range[1]} (共{len(chunks_to_delete)}个)")
            
            # 删除documents
            new_documents = []
            chunk_mapping = {}  # 旧索引 -> 新索引
            new_chunk_index = 0
            
            for old_index, doc in enumerate(data['documents']):
                if old_index not in chunks_to_delete:
                    new_documents.append(doc)
                    chunk_mapping[old_index] = new_chunk_index
                    new_chunk_index += 1
            
            data['documents'] = new_documents
            
            # 删除questions和相关vectors
            vector_to_chunk_map = data.get('vector_to_chunk_map', [])
            new_questions = []
            new_embeddings = []
            new_vector_types = []
            new_vector_to_chunk_map = []
            
            for i, chunk_idx in enumerate(vector_to_chunk_map):
                if chunk_idx not in chunks_to_delete:
                    # 保留这个向量
                    if i < len(data.get('questions', [])):
                        new_questions.append(data['questions'][i])
                    if i < len(data.get('embeddings', [])):
                        new_embeddings.append(data['embeddings'][i])
                    if i < len(data.get('vector_types', [])):
                        new_vector_types.append(data['vector_types'][i])
                    
                    # 更新chunk映射
                    new_chunk_idx = chunk_mapping.get(chunk_idx, chunk_idx)
                    new_vector_to_chunk_map.append(new_chunk_idx)
            
            data['questions'] = new_questions
            data['embeddings'] = new_embeddings
            data['vector_types'] = new_vector_types
            data['vector_to_chunk_map'] = new_vector_to_chunk_map
            
            # 更新document_sources
            if 'document_sources' in data and source_name != "document.txt (原始文档)":
                data['document_sources'] = [s for s in data['document_sources'] if s['name'] != source_name]
                
                # 更新其他source的chunk_range
                for source in data['document_sources']:
                    old_start, old_end = source['chunk_range']
                    new_start = chunk_mapping.get(old_start, old_start)
                    new_end = chunk_mapping.get(old_end, old_end)
                    source['chunk_range'] = [new_start, new_end]
            
            # 更新元数据
            data['version'] = f"2.1.0_deleted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if update_database:
                # 保存数据库
                if self._save_database(data):
                    print(f"✅ 文档删除成功: {source_name}")
                    print(f"📊 剩余文档块: {len(data['documents'])}")
                    print(f"🔍 剩余向量数: {len(data['embeddings'])}")
                    return True
                else:
                    print("❌ 数据库保存失败")
                    return False
            else:
                # 不保存数据库，只更新内存中的数据
                return True
                
        except Exception as e:
            print(f"❌ 删除文档失败: {e}")
            return False
    
    def search_in_documents(self, keyword: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """在文档中搜索关键词"""
        data = self._load_database()
        if not data:
            return []
        
        documents = data.get('documents', [])
        sources = self.list_document_sources()
        
        results = []
        
        for doc_idx, doc_content in enumerate(documents):
            if keyword.lower() in doc_content.lower():
                # 找到对应的source
                source_name = "未知来源"
                for source in sources:
                    chunk_range = source['chunk_range']
                    if chunk_range[0] <= doc_idx <= chunk_range[1]:
                        source_name = source['name']
                        break
                
                # 高亮关键词
                highlighted_content = doc_content.replace(
                    keyword, f"**{keyword}**"
                ).replace(
                    keyword.lower(), f"**{keyword.lower()}**"
                ).replace(
                    keyword.upper(), f"**{keyword.upper()}**"
                )
                
                results.append({
                    'chunk_index': doc_idx,
                    'source_name': source_name,
                    'content': doc_content,
                    'highlighted_content': highlighted_content,
                    'keyword_count': doc_content.lower().count(keyword.lower())
                })
                
                if len(results) >= max_results:
                    break
        
        # 按关键词出现次数排序
        results.sort(key=lambda x: x['keyword_count'], reverse=True)
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        data = self._load_database()
        if not data:
            return {}
        
        # 使用list_document_sources来获取准确的来源数量（包括原始文档）
        sources = self.list_document_sources()
        
        return {
            'total_documents': len(data.get('documents', [])),
            'total_questions': len(data.get('questions', [])),
            'total_vectors': len(data.get('embeddings', [])),
            'document_vectors': data.get('vector_types', []).count('document'),
            'question_vectors': data.get('vector_types', []).count('question'),
            'version': data.get('version', 'unknown'),
            'sources_count': len(sources),
            'model': data.get('model', 'unknown')
        }
    
    def rebuild_database_from_original(self, content: str, questions_per_chunk: int = 10) -> bool:
        """
        从原始文档重新构建数据库
        
        Args:
            content: 原始文档内容
            questions_per_chunk: 每个文档块生成的问题数量
            
        Returns:
            bool: 是否成功重建
        """
        print(f"\n🔄 开始重建数据库...")
        print(f"{'='*60}")
        
        # 创建备份
        if not self._create_backup():
            print("❌ 备份创建失败，为安全起见，停止操作")
            return False
        
        # 文本分块
        chunks = self.text_splitter.split_text(content)
        print(f"📄 文本分块完成，共 {len(chunks)} 个块")
        
        # 创建向量
        vector_data = self._create_document_vectors(chunks, questions_per_chunk)
        
        if not vector_data['embeddings']:
            print("❌ 未能生成任何向量")
            return False
        
        # 创建新数据库
        new_data = {
            "documents": vector_data['chunks'],
            "questions": vector_data['questions'],
            "embeddings": [emb.tolist() for emb in vector_data['embeddings']],
            "vector_types": vector_data['vector_types'],
            "vector_to_chunk_map": vector_data['vector_to_chunk_map'],
            "model": "intfloat/e5-base-v2",
            "version": f"2.1.0_rebuilt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": {
                "chunk_size": 300,
                "chunk_overlap": 30,
                "questions_per_chunk": questions_per_chunk,
                "hybrid_approach": True
            },
            "document_sources": [{
                'name': "document.txt (原始文档)",
                'added_time': datetime.now().isoformat(),
                'chunk_range': [0, len(vector_data['chunks']) - 1],
                'chunk_count': len(vector_data['chunks']),
                'vector_count': len(vector_data['embeddings'])
            }]
        }
        
        # 保存数据库
        if self._save_database(new_data):
            print(f"\n✅ 数据库重建成功!")
            print(f"📊 统计信息:")
            print(f"  📄 文档块: {len(vector_data['chunks'])}")
            print(f"  🔍 向量数: {len(vector_data['embeddings'])}")
            print(f"  ❓ 问题数: {len(vector_data['questions'])}")
            print(f"{'='*60}")
            return True
        else:
            print("❌ 数据库保存失败")
            return False

def main():
    """测试文档管理器"""
    manager = DocumentManager()
    
    # 显示当前统计信息
    stats = manager.get_database_stats()
    if stats:
        print("\n📊 当前数据库统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 列出文档来源
    sources = manager.list_document_sources()
    if sources:
        print("\n📚 文档来源:")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source['name']} ({source['added_time']})")
            print(f"     文档块: {source['chunk_count']}, 向量: {source['vector_count']}")

if __name__ == "__main__":
    main() 