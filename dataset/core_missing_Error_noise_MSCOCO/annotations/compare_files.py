#!/usr/bin/env python3
"""
文件差异比较工具
将两个文件的差异及上下文输出到指定文件夹中
"""

import os
import sys
import datetime
from typing import List

def ensure_directory(directory: str) -> bool:
    """确保目录存在，如果不存在则创建"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ 创建目录失败: {e}")
        return False

def compare_files_with_context(file1_path: str, file2_path: str, output_dir: str, context_lines: int = 3):
    """
    比较两个文件的差异，并输出差异行及其上下文到指定文件夹
    
    Args:
        file1_path: 第一个文件路径
        file2_path: 第二个文件路径
        output_dir: 输出文件夹路径
        context_lines: 上下文的行数(默认3行)
    """
    
    # 确保输出目录存在
    if not ensure_directory(output_dir):
        return
    
    # 生成输出文件名（基于输入文件名和时间戳）
    file1_name = os.path.basename(file1_path)
    file2_name = os.path.basename(file2_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"diff_report_{file1_name}_vs_{file2_name}_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # 读取文件内容
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        with open(file2_path, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return
    
    # 找出差异行
    diff_lines = []
    min_len = min(len(lines1), len(lines2))
    max_len = max(len(lines1), len(lines2))
    
    for i in range(max_len):
        line1 = lines1[i] if i < len(lines1) else None
        line2 = lines2[i] if i < len(lines2) else None
        
        if line1 != line2:
            diff_lines.append(i)
    
    # 生成输出内容
    output_content = []
    output_content.append("=" * 80)
    output_content.append("文件差异比较报告")
    output_content.append("=" * 80)
    output_content.append(f"比较时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_content.append(f"文件1: {file1_path}")
    output_content.append(f"文件2: {file2_path}")
    output_content.append(f"输出文件: {output_path}")
    output_content.append(f"总行数 - 文件1: {len(lines1)}, 文件2: {len(lines2)}")
    output_content.append(f"发现差异行数: {len(diff_lines)}")
    output_content.append("=" * 80)
    output_content.append("")
    
    # 处理每个差异区域
    processed_lines = set()
    
    for diff_line in diff_lines:
        if diff_line in processed_lines:
            continue
            
        # 计算上下文范围
        start_line = max(0, diff_line - context_lines)
        end_line = min(max_len, diff_line + context_lines + 1)
        
        output_content.append(f"🎯 差异区域 {len(processed_lines) + 1}: 行 {start_line + 1}-{end_line}")
        output_content.append("-" * 60)
        
        # 输出上下文内容
        for i in range(start_line, end_line):
            if i >= len(lines1) and i >= len(lines2):
                break
                
            line_num = i + 1
            marker = "  "
            
            if i == diff_line:
                marker = ">>>"
            elif i in diff_lines:
                marker = " * "
            
            line1_content = lines1[i].rstrip() if i < len(lines1) else "<文件结束>"
            line2_content = lines2[i].rstrip() if i < len(lines2) else "<文件结束>"
            
            # 如果是差异行，分别显示两行内容
            if i == diff_line:
                output_content.append(f"{marker} 行{line_num:4d} | 文件1: {line1_content}")
                output_content.append(f"{' ':>7} | 文件2: {line2_content}")
            else:
                # 对于相同行，只显示一次
                if line1_content == line2_content:
                    output_content.append(f"{marker} 行{line_num:4d} | {line1_content}")
                else:
                    output_content.append(f"{marker} 行{line_num:4d} | 文件1: {line1_content}")
                    output_content.append(f"{' ':>7} | 文件2: {line2_content}")
            
            processed_lines.add(i)
        
        output_content.append("")
    
    # 处理文件长度不同的情况
    if len(lines1) != len(lines2):
        output_content.append("📏 文件长度差异")
        output_content.append("-" * 40)
        if len(lines1) > len(lines2):
            output_content.append(f"文件1 多出 {len(lines1) - len(lines2)} 行:")
            for i in range(len(lines2), len(lines1)):
                output_content.append(f"  行{i+1:4d} | {lines1[i].rstrip()}")
        else:
            output_content.append(f"文件2 多出 {len(lines2) - len(lines1)} 行:")
            for i in range(len(lines1), len(lines2)):
                output_content.append(f"  行{i+1:4d} | {lines2[i].rstrip()}")
        output_content.append("")
    
    # 写入输出文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_content))
        print(f"✅ 差异报告已保存到: {output_path}")
        print("📊 统计信息:")
        print(f"   - 总差异行数: {len(diff_lines)}")
        print(f"   - 文件1行数: {len(lines1)}")
        print(f"   - 文件2行数: {len(lines2)}")
        if diff_lines:
            print(f"   - 相似度: {((max_len - len(diff_lines)) / max_len * 100):.2f}%")
    except Exception as e:
        print(f"❌ 写入输出文件时出错: {e}")

def main():
    """主函数"""
    # 文件路径配置 - 你可以修改这些路径
    base_dir = "/home/zbm/xjd/NPC-master/dataset/Entity_Referential_Error_noise_MSCOCO/annotations/scan_split"
    
    # 输入文件
    file1 = "0_noise_train_caps.txt"
    file2 = "1.0_noise_train_caps.txt"
    
    # 输出文件夹 - 你可以指定任何文件夹
    output_dir = "/home/zbm/xjd/NPC-master/dataset/Entity_Referential_Error_noise_MSCOCO/annotations/dfii_reports"  # 修改为你想要的输出文件夹
    
    # 构建完整路径
    file1_path = os.path.join(base_dir, file1)
    file2_path = os.path.join(base_dir, file2)
    
    print("🔍 开始比较文件差异...")
    print(f"📄 文件1: {file1_path}")
    print(f"📄 文件2: {file2_path}")
    print(f"📁 输出文件夹: {output_dir}")
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        print(f"❌ 文件1不存在: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"❌ 文件2不存在: {file2_path}")
        return
    
    # 执行比较
    compare_files_with_context(file1_path, file2_path, output_dir, context_lines=2)

if __name__ == "__main__":
    main()