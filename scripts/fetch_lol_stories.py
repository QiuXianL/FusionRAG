#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 League of Legends Universe 官方 API 获取所有短篇故事和英雄传记
输出格式化的中文文本文件，可直接导入 RAG 系统
"""

import json
import urllib.request
import urllib.error
import os
import sys
import time
import re

# Universe API 端点
EXPLORE_API = "https://universe-meeps.leagueoflegends.com/v1/zh_cn/explore2/index.json"

# 输出路径
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "documents", "lol_stories.txt")

# 地区映射
REGION_MAP = {
    "freljord": "弗雷尔卓德", "demacia": "德玛西亚", "noxus": "诺克萨斯",
    "ionia": "艾欧尼亚", "shurima": "恕瑞玛", "bilgewater": "比尔吉沃特",
    "piltover": "皮尔特沃夫", "zaun": "祖安", "shadow-isles": "暗影岛",
    "targon": "巨神峰", "ixtal": "以绪塔尔", "bandle-city": "班德尔城",
    "void": "虚空", "mount-targon": "巨神峰", "ixtal": "以绪塔尔",
}

def http_get(url: str, retries: int = 3) -> dict:
    """带重试的 GET 请求"""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FusionRAG-LoreBot/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5)
            else:
                raise

def strip_html(text: str) -> str:
    """去除 HTML 标签"""
    if not text:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</p>', '\n\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&apos;', "'").replace('&quot;', '"')
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&#39;', "'")
    return text.strip()

def get_champion_region(champ: dict) -> str:
    """从英雄数据中提取地区"""
    faction = champ.get("associated-faction-slug", "")
    if faction and faction in REGION_MAP:
        return REGION_MAP[faction]
    # 尝试从 faction 名称推断
    faction_name = champ.get("associated-faction", "")
    return faction_name or "未知地区"

def format_story(story: dict, index: int) -> str:
    """将一篇故事格式化为文本块"""
    title = story.get("title", "未知标题")
    subtitle = story.get("subtitle", "") or ""
    description = strip_html(story.get("description", ""))
    slug = story.get("story-slug", story.get("slug", ""))
    read_time = story.get("minutes-to-read", "?")
    release_date = story.get("release-date", "")[:10]
    champions = story.get("featured-champions", [])

    # 收集英雄信息
    champ_names = []
    champ_regions = set()
    for c in champions:
        name = c.get("name", "")
        if name:
            champ_names.append(name)
        region = get_champion_region(c)
        if region and region != "未知地区":
            champ_regions.add(region)

    # 获取正文
    full_text = ""
    # 策略1: 从英雄传记中提取
    for c in champions:
        bio = c.get("biography", {})
        if isinstance(bio, dict):
            bio_text = bio.get("full", "")
            if bio_text:
                full_text = strip_html(bio_text)
                break

    # 策略2: 从 description 提取（对于非传记类故事）
    if not full_text and description:
        full_text = description

    # 策略3: 尝试从 story 的 content 字段提取
    if not full_text:
        story_content = story.get("content", "") or story.get("body", "")
        if story_content:
            full_text = strip_html(str(story_content))

    if not full_text or len(full_text) < 50:
        return None  # 跳过内容太短的故事

    # 构建格式化文本
    lines = []
    lines.append(f"【官方小说 #{index}】")
    lines.append(f"标题：{title}")
    if subtitle and not subtitle.startswith("by "):
        lines.append(f"作者：{subtitle}")
    elif subtitle:
        lines.append(f"作者：{subtitle[3:]}")
    lines.append(f"发布日期：{release_date}")
    lines.append(f"阅读时长：约 {read_time} 分钟")
    if champ_names:
        lines.append(f"相关英雄：{'、'.join(champ_names)}")
    if champ_regions:
        lines.append(f"涉及地区：{'、'.join(sorted(champ_regions))}")
    lines.append(f"正文：")
    lines.append(full_text)
    lines.append("")

    return "\n".join(lines)

def main():
    print("=" * 60)
    print("📖 英雄联盟宇宙官方小说采集器")
    print("   数据源: Universe API (universe-meeps.leagueoflegends.com)")
    print("=" * 60)
    print()

    try:
        # 1. 获取故事索引
        print("📡 正在获取故事索引...")
        data = http_get(EXPLORE_API)
        modules = data.get("modules", [])
        stories = [m for m in modules if m["type"] == "story-preview"]
        print(f"✅ 找到 {len(stories)} 篇故事")

        # 2. 逐个处理
        print(f"\n📝 正在处理故事...")
        output_lines = []
        success_count = 0
        skip_count = 0

        for i, story in enumerate(stories, 1):
            try:
                formatted = format_story(story, i)
                if formatted:
                    output_lines.append(formatted)
                    success_count += 1
                    title = story.get("title", "")[:40]
                    print(f"  [{i}/{len(stories)}] ✅ {title}")
                else:
                    skip_count += 1
                    title = story.get("title", "")[:40]
                    print(f"  [{i}/{len(stories)}] ⏭️  {title} (内容太短)")
            except Exception as e:
                skip_count += 1
                title = story.get("title", "")[:40]
                print(f"  [{i}/{len(stories)}] ⚠️  {title} - {e}")

            # 避免请求过快
            if i % 20 == 0:
                time.sleep(0.5)

        # 3. 写入文件
        if output_lines:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            full_content = "\n".join(output_lines)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(full_content)

        # 4. 统计
        total_chars = sum(len(l) for l in output_lines)
        print(f"\n{'=' * 60}")
        print(f"✅ 采集完成！")
        print(f"   成功: {success_count} 篇")
        print(f"   跳过: {skip_count} 篇")
        print(f"   总字符数: {total_chars:,}")
        print(f"   输出文件: {OUTPUT_FILE}")
        print(f"{'=' * 60}")
        print()
        print("💡 下一步：")
        print("   1. 在 Web 界面「文档管理」中导入该文件")
        print("   2. 试搜: '布隆的盾牌是怎么来的？'")

    except urllib.error.URLError as e:
        print(f"❌ 网络连接失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
