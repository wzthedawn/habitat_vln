#!/usr/bin/env python3
"""修复项目中的硬编码路径"""

import os
import re
from pathlib import Path

# 基础目录
BASE_DIR = Path(__file__).parent.parent
OLD_BASE = "/root/habitat_vln"
NEW_BASE = "/home/WZ/MA_VLN"

# 需要替换的路径映射
PATH_REPLACEMENTS = {
    "/root/habitat_vln": "/home/WZ/MA_VLN",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B": "/data/WZ/Model/Qwen/Qwen3___5-2B",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct": "/data/WZ/Model/Qwen/Qwen2-VL-2B-Instruct",
    "/root/habitat-lab/data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json": "/data/WZ/Dataset/R2R/val_seen.json",
}

# 需要跳过的文件/目录
SKIP_PATTERNS = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".claude",
    "*.egg-info",
]

def should_skip(path: Path) -> bool:
    """检查是否应该跳过该路径"""
    path_str = str(path)
    for pattern in SKIP_PATTERNS:
        if pattern in path_str:
            return True
    return False

def fix_file(path: Path, dry_run: bool = True) -> int:
    """修复单个文件中的路径"""
    if should_skip(path):
        return 0

    try:
        content = path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return 0

    original_content = content
    replacements_count = 0

    # 替换所有配置的路径
    for old_path, new_path in PATH_REPLACEMENTS.items():
        if old_path in content:
            count = content.count(old_path)
            content = content.replace(old_path, new_path)
            replacements_count += count
            if not dry_run:
                print(f"  替换：{old_path} -> {new_path} ({count} 处)")

    # 保存修改
    if replacements_count > 0 and not dry_run:
        path.write_text(content, encoding='utf-8')
        print(f"✓ 已修复：{path.relative_to(BASE_DIR)}")

    return replacements_count

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="修复项目中的硬编码路径")
    parser.add_argument("--apply", action="store_true", help="应用修改（默认只检查）")
    parser.add_argument("--add-custom", type=str, nargs=2, metavar=("OLD", "NEW"),
                       help="添加自定义路径替换")
    args = parser.parse_args()

    if args.add_custom:
        old, new = args.add_custom
        PATH_REPLACEMENTS[old] = new
        print(f"添加自定义替换：{old} -> {new}")

    print("=" * 60)
    print("路径修复工具")
    print("=" * 60)
    print(f"项目目录：{BASE_DIR}")
    print(f"模式：{'应用修改' if args.apply else '仅检查'}")
    print()

    # 显示要替换的路径
    print("路径替换映射:")
    for old, new in PATH_REPLACEMENTS.items():
        print(f"  {old}")
        print(f"    -> {new}")
    print()

    # 查找所有 Python 文件和 YAML 文件
    files_to_check = []
    for pattern in ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json"]:
        files_to_check.extend(BASE_DIR.glob(pattern))

    total_replacements = 0
    files_with_issues = []

    print(f"检查 {len(files_to_check)} 个文件...")
    print()

    for file_path in files_to_check:
        count = fix_file(file_path, dry_run=not args.apply)
        if count > 0:
            files_with_issues.append((file_path, count))
            total_replacements += count

    # 打印总结
    print()
    print("=" * 60)
    print("总结")
    print("=" * 60)

    if total_replacements == 0:
        print("✓ 没有发现需要修复的路径")
    else:
        print(f"发现 {total_replacements} 处需要修复")
        print(f"涉及 {len(files_with_issues)} 个文件:")
        for path, count in files_with_issues:
            rel_path = path.relative_to(BASE_DIR)
            print(f"  {rel_path}: {count} 处")

        if not args.apply:
            print()
            print("使用 --apply 参数来应用这些修改")

    return 0

if __name__ == "__main__":
    exit(main())
