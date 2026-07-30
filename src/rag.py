#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能问答系统 - 主程序
支持命令行和Web界面两种运行模式
"""

import sys
from web_app import run_web_server
from rag_cli import (
    display_banner,
    check_environment,
    ask_update_database,
    update_database,
    ask_mode_selection,
    run_cli_mode,
    run_document_mode
)

def main():
    """主函数"""
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
        
        if mode == 'web':
            print("\n🌐 启动Web界面模式...")
            run_web_server()
        elif mode == 'cli':
            run_cli_mode()
        elif mode == 'document':
            run_document_mode()
        elif mode == 'exit':
            print("👋 再见！")
        
    except KeyboardInterrupt:
        print("\n\n👋 程序已被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()