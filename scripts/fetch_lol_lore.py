#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Riot Data Dragon 官方 CDN 获取所有英雄联盟英雄背景故事
输出格式化的中文文本文件，可直接导入 RAG 系统
"""

import json
import urllib.request
import urllib.error
import os
import sys
import time

# Data Dragon 配置
LATEST_VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CHAMPION_DATA_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/zh_CN/champion.json"

# 英雄定位中文映射
TAG_CN_MAP = {
    "Assassin": "刺客",
    "Fighter": "战士",
    "Mage": "法师",
    "Marksman": "射手",
    "Support": "辅助",
    "Tank": "坦克",
}

# 输出文件路径
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lol_champions.txt")


def http_get(url: str, retries: int = 3) -> dict:
    """带重试的 HTTP GET 请求"""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RAG-LoL-Collector/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                print(f"  ⚠️ 请求失败 (第{attempt+1}次重试): {e}")
                time.sleep(2)
            else:
                raise


def get_latest_version() -> str:
    """获取最新的 Data Dragon 版本号"""
    print("📡 正在获取最新版本号...")
    versions = http_get(LATEST_VERSION_URL)
    version = versions[0]
    print(f"✅ 最新版本: {version}")
    return version


def fetch_all_champions(version: str) -> list:
    """获取所有英雄数据"""
    url = CHAMPION_DATA_URL.format(version=version)
    print(f"📡 正在获取英雄数据...")
    print(f"   URL: {url}")
    data = http_get(url)

    champions = list(data["data"].values())
    print(f"✅ 获取到 {len(champions)} 个英雄")
    return champions


def format_champion(champion: dict, index: int) -> str:
    """将单个英雄数据格式化为文本块"""
    name = champion.get("name", "未知")
    en_name = champion.get("id", "unknown")
    title = champion.get("title", "")
    lore = champion.get("lore", "") or champion.get("blurb", "")
    tags = champion.get("tags", [])
    tags_cn = [TAG_CN_MAP.get(t, t) for t in tags]

    lines = []
    lines.append(f"【英雄 #{index}】")
    lines.append(f"英雄：{name} {en_name}")
    lines.append(f"称号：{title}")
    lines.append(f"定位：{' / '.join(tags_cn) if tags_cn else '未知'}")
    lines.append(f"背景故事：{lore}" if lore else f"背景故事：（暂无）")
    lines.append("")

    return "\n".join(lines)


def main():
    """主函数"""
    print("=" * 60)
    print("🏆 英雄联盟宇宙背景故事采集器")
    print("   数据源: Riot Data Dragon (官方CDN)")
    print("=" * 60)
    print()

    try:
        # 1. 获取版本号
        version = get_latest_version()

        # 2. 获取所有英雄
        champions = fetch_all_champions(version)

        # 3. 格式化并写入文件
        print(f"\n📝 正在格式化并写入文件...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        total_chars = 0
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for i, champ in enumerate(champions, 1):
                text = format_champion(champ, i)
                f.write(text + "\n")
                total_chars += len(text)

        # 4. 统计
        print(f"\n{'=' * 60}")
        print(f"✅ 采集完成！")
        print(f"   英雄数量: {len(champions)}")
        print(f"   总字符数: {total_chars:,}")
        print(f"   输出文件: {OUTPUT_FILE}")
        print(f"{'=' * 60}")
        print()
        print("💡 下一步：")
        print("   1. 启动 RAG 系统: python src/rag.py")
        print("   2. 选择「Web界面模式」")
        print("   3. 在「文档管理」中导入该文件")
        print("   4. 尝试搜索: '亚托克斯的背景故事是什么？'")

    except urllib.error.URLError as e:
        print(f"❌ 网络连接失败: {e}")
        print("💡 请检查网络连接，Data Dragon 可能需要科学上网")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
