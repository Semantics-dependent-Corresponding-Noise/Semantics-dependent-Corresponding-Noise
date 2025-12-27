import os
import glob
import asyncio
import aiofiles
from collections import defaultdict

async def merge_description_files(input_dir, output_file):
    """
    将所有分割的描述文件按序号顺序合并成一个文件
    
    参数:
        input_dir: 包含分割文件的目录
        output_file: 合并后的输出文件路径
    """
    
    # 获取所有匹配的文件
    pattern = os.path.join(input_dir, "train_caps_5_per_image_part*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ 错误: 在 {input_dir} 中没有找到匹配的文件")
        return False
    
    # 按文件名中的数字排序（确保顺序正确）
    def extract_number(filename):
        try:
            basename = os.path.basename(filename)
            # 从 "part001.txt" 中提取数字 1
            number_part = ''.join(filter(str.isdigit, basename))
            return int(number_part) if number_part else 0
        except:
            return 0
    
    files.sort(key=extract_number)
    
    total_files = len(files)
    total_lines = 0
    total_images = 0
    
    print("="*60)
    print(f"📁 找到 {total_files} 个文件:")
    for i, f in enumerate(files[:10], 1):  # 显示前10个
        print(f"   {i:3d}. {os.path.basename(f)}")
    if total_files > 10:
        print(f"   ... 还有 {total_files - 10} 个文件")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 开始合并
    print("\n" + "="*60)
    print("🔄 开始合并文件...")
    
    try:
        # 使用异步IO进行高效合并
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as outfile:
            for i, file_path in enumerate(files, 1):
                filename = os.path.basename(file_path)
                print(f"\r📄 正在处理 ({i}/{total_files}): {filename}", end="", flush=True)
                
                # 读取文件内容
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as infile:
                    content = await infile.read()
                    lines = content.splitlines()
                    
                    # 写入到目标文件（保持原有格式）
                    if lines:
                        await outfile.write(content)
                        # 如果不是以换行符结尾，添加换行
                        if content and not content.endswith('\n'):
                            await outfile.write('\n')
                        
                        total_lines += len(lines)
                        total_images += len(lines) // 5  # 每5行是一张图片
        
        # 输出统计信息
        print("\n" + "="*60)
        print("✅ 合并完成!")
        print(f"📊 统计信息:")
        print(f"  合并文件数: {total_files}")
        print(f"  总图片数: {total_images:,}")
        print(f"  总行数: {total_lines:,}")
        print(f"  输出文件: {output_file}")
        
        # 验证输出文件
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"  文件大小: {file_size:,} 字节 ({file_size/1024/1024:.2f} MB)")
            
            # 显示文件前几个描述示例
            print(f"\n📝 文件内容验证:")
            async with aiofiles.open(output_file, 'r', encoding='utf-8') as f:
                first_lines = []
                async for line in f:
                    first_lines.append(line.strip())
                    if len(first_lines) >= 15:  # 读取前15行（3张图片）
                        break
                
                for img_idx in range(min(3, len(first_lines) // 5)):
                    start = img_idx * 5
                    print(f"\n  图片 {img_idx+1} 的5条描述:")
                    for line_idx in range(5):
                        desc = first_lines[start + line_idx]
                        print(f"    行{line_idx+1}: {desc}")
                print(f"\n  ... 剩余 {total_lines - 15:,} 行")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def sync_merge_description_files(input_dir, output_file):
    """
    同步版本（不使用异步）
    """
    pattern = os.path.join(input_dir, "train_caps_5_per_image_part*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ 错误: 在 {input_dir} 中没有找到匹配的文件")
        return False
    
    # 排序
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
    
    total_files = len(files)
    
    print("="*60)
    print(f"📁 找到 {total_files} 个文件")
    print("="*60)
    print("🔄 开始合并...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, file_path in enumerate(files, 1):
                filename = os.path.basename(file_path)
                print(f"\r📄 正在处理 ({i}/{total_files}): {filename}", end="", flush=True)
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    if content and not content.endswith('\n'):
                        outfile.write('\n')
        
        print("\n" + "="*60)
        print("✅ 合并完成!")
        print(f"输出文件: {output_file}")
        
        # 验证
        if os.path.exists(output_file):
            total_lines = 0
            with open(output_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
            
            print(f"总行数: {total_lines:,}")
            print(f"图片数: {total_lines // 5:,}")
            print(f"文件大小: {os.path.getsize(output_file):,} 字节")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        return False

# ==================== 主程序 ====================
if __name__ == "__main__":
    # ==================== 配置参数 ====================
    # 输入目录（包含分割文件的目录）
    INPUT_DIR = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/test'
    
    # 输出文件（合并后的完整文件）
    OUTPUT_FILE = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/1.0_noise_train_caps.txt'
    
    # 是否使用异步模式（推荐True，速度更快）
    USE_ASYNC = True
    
    # ==================== 执行合并 ====================
    print("🚀 开始合并描述文件...")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"模式: {'异步' if USE_ASYNC else '同步'}")
    print("="*60)
    
    if USE_ASYNC:
        asyncio.run(merge_description_files(INPUT_DIR, OUTPUT_FILE))
    else:
        sync_merge_description_files(INPUT_DIR, OUTPUT_FILE)
    
    print("\n程序结束")