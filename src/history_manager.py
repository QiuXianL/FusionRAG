#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史记录管理器 - 管理用户的搜索历史记录
支持历史记录的增删改查和持久化存储
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

class HistoryManager:
    """历史记录管理器，负责管理用户的搜索历史记录"""
    
    def __init__(self, history_file: str = None):
        """
        初始化历史记录管理器

        Args:
            history_file: 历史记录文件路径
        """
        # 使用脚本所在目录的上级目录（项目根目录）来解析路径
        if history_file is None:
            _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            history_file = os.path.join(_base, "documents", "search_history.json")
        self.history_file = history_file
        self.max_history_count = 1000  # 最大历史记录数量
        self.history_data = self._load_history()
        
        print(f"📚 历史记录管理器初始化完成")
        print(f"  💾 历史文件: {history_file}")
        print(f"  📊 当前记录数: {len(self.history_data.get('history', []))}")
    
    def _load_history(self) -> Dict[str, Any]:
        """加载历史记录文件"""
        if not os.path.exists(self.history_file):
            # 创建目录（如果路径包含目录）
            history_dir = os.path.dirname(self.history_file)
            if history_dir:
                os.makedirs(history_dir, exist_ok=True)
            
            # 创建初始历史记录结构
            initial_data = {
                "version": "1.0.0",
                "created_time": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_count": 0,
                "history": []
            }
            
            self._save_history(initial_data)
            print(f"✅ 创建新的历史记录文件: {self.history_file}")
            return initial_data
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 加载历史记录成功: {len(data.get('history', []))} 条记录")
            return data
        except Exception as e:
            print(f"❌ 加载历史记录失败: {e}")
            # 返回空的历史记录结构
            return {
                "version": "1.0.0",
                "created_time": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_count": 0,
                "history": []
            }
    
    def _save_history(self, data: Dict[str, Any]) -> bool:
        """保存历史记录到文件"""
        try:
            # 确保目录存在（如果路径包含目录）
            history_dir = os.path.dirname(self.history_file)
            if history_dir:
                os.makedirs(history_dir, exist_ok=True)
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ 保存历史记录失败: {e}")
            return False
    
    def add_history(self, query: str, results: List[Dict], ai_response: str = "", 
                   strategy: str = "auto", top_k: int = 5, performance: Dict = None) -> bool:
        """
        添加搜索历史记录
        
        Args:
            query: 用户查询
            results: 检索结果
            ai_response: AI回答
            strategy: 使用的策略
            top_k: 检索数量
            performance: 性能指标
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 生成记录ID
            record_id = self._generate_record_id(query, datetime.now())
            
            # 创建历史记录
            history_record = {
                "id": record_id,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy,
                "top_k": top_k,
                "result_count": len(results),
                "ai_response": ai_response,
                "performance": performance or {},
                "results_summary": self._create_results_summary(results)
            }
            
            # 添加到历史记录列表
            self.history_data["history"].insert(0, history_record)  # 插入到开头
            self.history_data["total_count"] += 1
            self.history_data["last_updated"] = datetime.now().isoformat()
            
            # 限制历史记录数量
            if len(self.history_data["history"]) > self.max_history_count:
                self.history_data["history"] = self.history_data["history"][:self.max_history_count]
                print(f"📊 历史记录已达到最大数量限制，保留最新的 {self.max_history_count} 条")
            
            # 保存到文件
            if self._save_history(self.history_data):
                print(f"✅ 添加历史记录成功: '{query[:50]}...'")
                return True
            else:
                print(f"❌ 保存历史记录失败")
                return False
                
        except Exception as e:
            print(f"❌ 添加历史记录失败: {e}")
            return False
    
    def _generate_record_id(self, query: str, timestamp: datetime) -> str:
        """生成历史记录ID"""
        # 使用查询内容和时间戳生成唯一ID
        content = f"{query}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    
    def _create_results_summary(self, results: List[Dict]) -> List[Dict]:
        """创建结果摘要，避免存储过大的数据"""
        summary = []
        for result in results[:5]:  # 只保存前5个结果的摘要
            summary.append({
                "similarity_score": result.get('similarity_score', 0),
                "match_type": result.get('match_type', ''),
                "chunk_index": result.get('chunk_index', 0),
                "text_preview": result.get('original_text', '')[:100] + "..." if len(result.get('original_text', '')) > 100 else result.get('original_text', '')
            })
        return summary
    
    def get_history(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        获取历史记录列表
        
        Args:
            limit: 返回记录数量限制
            offset: 偏移量
            
        Returns:
            List[Dict]: 历史记录列表
        """
        try:
            history_list = self.history_data.get("history", [])
            start_index = offset
            end_index = min(start_index + limit, len(history_list))
            
            return history_list[start_index:end_index]
        except Exception as e:
            print(f"❌ 获取历史记录失败: {e}")
            return []
    
    def get_history_by_id(self, record_id: str) -> Optional[Dict]:
        """
        根据ID获取历史记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            Optional[Dict]: 历史记录，如果不存在返回None
        """
        try:
            for record in self.history_data.get("history", []):
                if record.get("id") == record_id:
                    return record
            return None
        except Exception as e:
            print(f"❌ 根据ID获取历史记录失败: {e}")
            return None
    
    def search_history(self, keyword: str, limit: int = 20) -> List[Dict]:
        """
        搜索历史记录
        
        Args:
            keyword: 搜索关键词
            limit: 返回记录数量限制
            
        Returns:
            List[Dict]: 匹配的历史记录列表
        """
        try:
            keyword_lower = keyword.lower()
            matched_records = []
            
            for record in self.history_data.get("history", []):
                query = record.get("query", "").lower()
                if keyword_lower in query:
                    matched_records.append(record)
                    if len(matched_records) >= limit:
                        break
            
            return matched_records
        except Exception as e:
            print(f"❌ 搜索历史记录失败: {e}")
            return []
    
    def delete_history(self, record_id: str) -> bool:
        """
        删除指定的历史记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            history_list = self.history_data.get("history", [])
            original_length = len(history_list)
            
            # 过滤掉要删除的记录
            self.history_data["history"] = [record for record in history_list if record.get("id") != record_id]
            
            if len(self.history_data["history"]) < original_length:
                self.history_data["total_count"] = len(self.history_data["history"])
                self.history_data["last_updated"] = datetime.now().isoformat()
                
                if self._save_history(self.history_data):
                    print(f"✅ 删除历史记录成功: {record_id}")
                    return True
                else:
                    print(f"❌ 保存历史记录失败")
                    return False
            else:
                print(f"⚠️ 未找到要删除的历史记录: {record_id}")
                return False
                
        except Exception as e:
            print(f"❌ 删除历史记录失败: {e}")
            return False
    
    def clear_history(self) -> bool:
        """
        清空所有历史记录
        
        Returns:
            bool: 是否清空成功
        """
        try:
            if not self.history_data.get("history"):
                print("📊 历史记录已经是空的")
                return True
            
            # 清空历史记录
            self.history_data["history"] = []
            self.history_data["total_count"] = 0
            self.history_data["last_updated"] = datetime.now().isoformat()
            
            if self._save_history(self.history_data):
                print(f"✅ 清空历史记录成功")
                return True
            else:
                print(f"❌ 保存历史记录失败")
                return False
                
        except Exception as e:
            print(f"❌ 清空历史记录失败: {e}")
            return False
    
    def get_history_stats(self) -> Dict[str, Any]:
        """
        获取历史记录统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            history_list = self.history_data.get("history", [])
            
            # 计算统计信息
            total_count = len(history_list)
            
            # 按日期统计
            date_stats = {}
            for record in history_list:
                timestamp = record.get("timestamp", "")
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
                        date_stats[date] = date_stats.get(date, 0) + 1
                    except:
                        pass
            
            # 按策略统计
            strategy_stats = {}
            for record in history_list:
                strategy = record.get("strategy", "unknown")
                strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1
            
            # 平均性能指标
            total_retrieval_time = 0
            total_ai_time = 0
            total_time = 0
            performance_count = 0
            
            for record in history_list:
                performance = record.get("performance", {})
                if performance:
                    total_retrieval_time += performance.get("retrieval_time", 0)
                    total_ai_time += performance.get("ai_time", 0)
                    total_time += performance.get("total_time", 0)
                    performance_count += 1
            
            avg_performance = {}
            if performance_count > 0:
                avg_performance = {
                    "avg_retrieval_time": total_retrieval_time / performance_count,
                    "avg_ai_time": total_ai_time / performance_count,
                    "avg_total_time": total_time / performance_count
                }
            
            return {
                "total_count": total_count,
                "date_stats": date_stats,
                "strategy_stats": strategy_stats,
                "avg_performance": avg_performance,
                "last_updated": self.history_data.get("last_updated", ""),
                "file_size_mb": os.path.getsize(self.history_file) / (1024 * 1024) if os.path.exists(self.history_file) else 0
            }
            
        except Exception as e:
            print(f"❌ 获取历史记录统计失败: {e}")
            return {
                "total_count": 0,
                "date_stats": {},
                "strategy_stats": {},
                "avg_performance": {},
                "last_updated": "",
                "file_size_mb": 0
            }
    
    def export_history(self, export_file: str = None) -> bool:
        """
        导出历史记录
        
        Args:
            export_file: 导出文件路径，如果为None则使用默认路径
            
        Returns:
            bool: 是否导出成功
        """
        try:
            if export_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                export_file = os.path.join(_base, "documents", f"history_export_{timestamp}.json")
            
            # 确保目录存在（如果路径包含目录）
            export_dir = os.path.dirname(export_file)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            
            # 导出数据
            export_data = {
                "export_info": {
                    "export_time": datetime.now().isoformat(),
                    "total_records": len(self.history_data.get("history", [])),
                    "version": "1.0.0"
                },
                "history_data": self.history_data
            }
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 历史记录导出成功: {export_file}")
            return True
            
        except Exception as e:
            print(f"❌ 导出历史记录失败: {e}")
            return False
    
    def cleanup_old_history(self, days: int = 30) -> int:
        """
        清理指定天数前的历史记录
        
        Args:
            days: 保留天数
            
        Returns:
            int: 清理的记录数量
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
            history_list = self.history_data.get("history", [])
            original_count = len(history_list)
            
            # 过滤掉过期的记录
            filtered_history = []
            for record in history_list:
                timestamp = record.get("timestamp", "")
                if timestamp:
                    try:
                        record_time = datetime.fromisoformat(timestamp).timestamp()
                        if record_time >= cutoff_time:
                            filtered_history.append(record)
                    except:
                        # 如果时间戳格式有问题，保留记录
                        filtered_history.append(record)
            
            # 更新历史记录
            self.history_data["history"] = filtered_history
            self.history_data["total_count"] = len(filtered_history)
            self.history_data["last_updated"] = datetime.now().isoformat()
            
            # 保存到文件
            if self._save_history(self.history_data):
                cleaned_count = original_count - len(filtered_history)
                print(f"✅ 清理历史记录成功: 删除了 {cleaned_count} 条过期记录")
                return cleaned_count
            else:
                print(f"❌ 保存历史记录失败")
                return 0
                
        except Exception as e:
            print(f"❌ 清理历史记录失败: {e}")
            return 0 