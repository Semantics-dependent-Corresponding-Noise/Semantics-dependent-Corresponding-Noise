import pandas as pd
import ast
import random
from openai import OpenAI
import time
from tqdm import tqdm
import os
import glob
import logging
import sys
from datetime import datetime

# 设置日志配置
def setup_logging(log_dir="logs"):
    """设置日志配置"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"noise_generation_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return log_file

# 设置 KimiChat API 密钥和 API 端点
client = OpenAI(
    api_key="yours api",
    base_url="https://api.moonshot.cn/v1"
)

class SimpleProgress:
    """简单的文本进度显示，用于后台运行"""
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_log_percent = -1
        self.start_time = time.time()
        
    def update(self, n=1):
        self.current += n
        current_percent = int(self.current / self.total * 100)
        
        # 每5%或者完成时记录一次
        if current_percent != self.last_log_percent and (current_percent % 5 == 0 or current_percent == 100):
            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: calculating..."
                
            logging.info(f"{self.desc}: {self.current}/{self.total} ({current_percent}%) {eta_str}")
            self.last_log_percent = current_percent
    
    def close(self):
        elapsed = time.time() - self.start_time
        logging.info(f"{self.desc}: 完成 {self.current}/{self.total} (100%) 耗时: {elapsed:.2f}s")

def process_single_file(input_file_path, output_file_path, replace_ratio=1.0):
    """
    处理单个文本文件，生成噪声文本并保存
    
    参数:
        input_file_path: 输入文件路径
        output_file_path: 输出文件路径
        replace_ratio: 替换比例
    """
    
    logging.info(f"开始处理文件: {input_file_path}")
    
    # 读取文本文件
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            raw_texts = f.readlines()
    except Exception as e:
        logging.error(f"读取文件失败: {e}")
        return

    # 去除文本中的换行符
    raw_texts = [text.strip() for text in raw_texts if text.strip()]
    
    if not raw_texts:
        logging.warning(f"文件为空，跳过处理: {input_file_path}")
        return

    logging.info(f"文件信息 - 文件名: {os.path.basename(input_file_path)}, 文本数量: {len(raw_texts)}")

    original_texts = raw_texts.copy()
    
    num_texts = len(raw_texts)
    num_to_replace = int(num_texts * replace_ratio)  # 需要替换的文本数量

    # 批量处理函数
    def generate_noisy_text_batch(text_list):
        prompt1 = """You are a professional Sentence revision Assistant., and your only task is to condense sentences without altering their essential meaning. Please strictly follow the following rules and output format:
Core Rules for Condense sentence:
1.Extraction of the sentence subject: First, accurately identify every subject component in the input sentence (for example, object category, color, scene, action，numerical expressions).
2.Partial removal or simplification of sentence components: Remove some sentence components to make the sentence more concise. Retain some descriptive words of the components, but ensure the sentence no longer fully describes the original scene. Avoid altering the verb structure (e.g., do not remove or change the form of the verb).
3.Ensure the Modified Sentence Omits at Least One Key Action or Detail: The modified sentence should omit at least one key action or detail, making it less descriptive than the original. The action or detail omitted should result in a change in the meaning or completeness of the sentence.
4.Avoid simply removing adjectives, adverbs, or other modifying elements: the revised sentence should not be entirely identical to the original in meaning. For instance, When multiple subjects appear in parallel, such as concurrent actions, parallel subjects or objects, one or more may be omitted, but at least one must be retained, resulting in the sentence describing a default state.
5.Where none of the above rules can be applied to modify a sentence, simply return the subject of the original sentence.
- Input Sentence: A man in a pink shirt climbs a rock face.
- Output Sentence: A man in a pink shirt.
- Input Sentence: A boys jumps into the water upside down.
- Output Sentence: A boys jumps into the water.
- Input Sentence: This is a young boy playing with a dollhouse.
- Output Sentence: A young boy.
- Input Sentence: A man wearing a cap and glasses is fixing the seat of a bicycle.
- Output Sentence: A man wearing a cap is fixing the seat of a bicycle.
- Input Sentence:A young boy is frantically staring and shaking his hands.
- Output Sentence: A young boy is frantically staring.
Strict Output Format:
Only output the modified sentence directly. Do NOT add any extra content (such as explanations, notes, or greetings)."""
        prompt2 = (
            "Please process the following sentences in batches according to the rules, outputting one modified sentence per line:\n"
            + "\n".join([f"{i+1}. {text}" for i, text in enumerate(text_list)])
        )
        try:
            logging.debug(f"发送API请求，批次大小: {len(text_list)}")
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "system", "content": prompt1},
                    {"role": "user", "content": prompt2}
                ],
                temperature=0.3,
            )
            # 处理API返回的结果，去除开头的数字和点
            processed_results = []
            for line in completion.choices[0].message.content.strip().split('\n'):
                line = line.strip()
                if line:
                    # 去除开头的 "1. ", "2. " 等编号
                    if '. ' in line and line.split('. ')[0].isdigit():
                        line = line.split('. ', 1)[1]
                    processed_results.append(line)
            logging.debug(f"API响应处理完成，返回 {len(processed_results)} 个结果")
            return processed_results
        except Exception as e:
            logging.error(f"API调用错误: {e}")
            return text_list

    requests_per_minute = 50
    delay_between_requests = 60 / requests_per_minute

    # 批次处理
    batch_size = 50
    max_retries = 5
    modified_count = 0
    target_modified = num_to_replace
    used_indices = set()

    # 检查是否在终端中运行
    is_tty = sys.stdout.isatty()
    
    if is_tty:
        # 在终端中运行，使用 tqdm 进度条
        logging.info("使用交互式进度条显示")
        main_pbar = tqdm(total=target_modified, 
                        desc=f"处理 {os.path.basename(input_file_path)}", 
                        unit="text",
                        ncols=100,
                        bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')
    else:
        # 在后台运行，使用简单文本进度
        logging.info(f"开始生成噪声文本，目标数量: {target_modified}")
        main_pbar = SimpleProgress(target_modified, desc=f"处理 {os.path.basename(input_file_path)}")

    batch_count = 0
    total_batches = (target_modified + batch_size - 1) // batch_size
    logging.info(f"批次信息 - 总批次数: {total_batches}, 批次大小: {batch_size}")
    
    while modified_count < target_modified:
        # 采样未被修改过的索引
        remaining_indices = list(set(range(num_texts)) - used_indices)
        if not remaining_indices:
            logging.warning("没有更多可用的文本进行修改")
            break
            
        batch_indices = random.sample(remaining_indices, min(batch_size, len(remaining_indices)))
        logging.debug(f"采样批次索引: {len(batch_indices)} 个文本")
        
        retry_count = 0
        failed_indices = batch_indices.copy()  # 初始所有都视为失败（待处理）
        
        batch_count += 1
        logging.info(f"开始处理第 {batch_count}/{total_batches} 批次")
        
        while retry_count < max_retries and failed_indices:
            try:
                # 准备当前需要处理的文本
                current_batch_texts = [raw_texts[idx] for idx in failed_indices]
                
                logging.info(f"重试 {retry_count + 1}: 发送 {len(failed_indices)} 个文本到API")
                batch_modified = generate_noisy_text_batch(current_batch_texts)
                
                if len(batch_modified) == len(failed_indices):
                    new_failed_indices = []
                    success_in_batch = 0
                    
                    for j, idx in enumerate(failed_indices):
                        original_text = raw_texts[idx]
                        modified_text = batch_modified[j]
                        
                        if modified_text and modified_text != original_text:
                            # ✅ 成功修改：存储修改后的文本
                            raw_texts[idx] = modified_text
                            used_indices.add(idx)
                            modified_count += 1
                            success_in_batch += 1
                            
                            # 更新进度条
                            if is_tty:
                                main_pbar.update(1)
                                main_pbar.set_description(f"处理 {os.path.basename(input_file_path)} ({modified_count}/{target_modified})")
                            else:
                                main_pbar.update(1)
                        else:
                            # ❌ 修改失败：保持原始文本，加入重试列表
                            new_failed_indices.append(idx)
                    
                    failed_indices = new_failed_indices
                    logging.info(f"批次处理结果 - 成功: {success_in_batch}, 失败: {len(failed_indices)}")
                    
                    if not failed_indices:
                        logging.info("当前批次所有文本处理成功")
                        break  # 全部成功，退出重试循环
                        
                else:
                    logging.warning(f"API返回数量不匹配: 期望 {len(failed_indices)}, 实际 {len(batch_modified)}")
                    
            except Exception as e:
                logging.error(f"API调用异常: {e}")
            
            retry_count += 1
            if retry_count < max_retries and failed_indices:
                logging.info(f"准备第 {retry_count} 次重试，剩余 {len(failed_indices)} 个文本")
                time.sleep(5)
        
        # 最终统计
        success_count = len(batch_indices) - len(failed_indices)
        if success_count > 0:
            logging.info(f"批次完成 - 成功修改: {success_count} 个文本")
        if failed_indices:
            logging.warning(f"批次完成 - 保持原文本: {len(failed_indices)} 个文本")
            # 对于最终失败的文本，标记为已使用，避免重复处理
            for idx in failed_indices:
                used_indices.add(idx)
        
        time.sleep(delay_between_requests)
    
    # 关闭进度条
    main_pbar.close()

    logging.info(f"文件处理完成 - 实际生成噪声文本: {modified_count}/{num_texts}")

    # 保存处理后的文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for text in raw_texts:
                f.write(text + "\n")
        logging.info(f"文件保存成功: {output_file_path}")
    except Exception as e:
        logging.error(f"文件保存失败: {e}")
        return

    logging.info(f"单个文件处理完成: {os.path.basename(input_file_path)}")
    logging.info("=" * 70)

def process_split_files_sequential(input_dir, output_dir, start_num=1, end_num=None, replace_ratio=1.0):
    """
    按顺序处理分割文件 (split_file_001.txt, split_file_002.txt, ...)
    
    参数:
        input_dir: 输入文件目录
        output_dir: 输出文件目录
        start_num: 起始文件编号
        end_num: 结束文件编号
        replace_ratio: 替换比例
    """
    
    logging.info(f"开始批量处理文件 - 输入目录: {input_dir}, 输出目录: {output_dir}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"创建输出目录: {output_dir}")
    
    # 确定要处理的文件范围
    if end_num is None:
        # 自动检测文件数量
        existing_files = glob.glob(os.path.join(input_dir, "split_file_*.txt"))
        if existing_files:
            file_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            end_num = max(file_numbers)
            logging.info(f"自动检测到文件范围: 001 到 {end_num:03d}")
        else:
            logging.error("没有找到 split_file_*.txt 文件")
            return
    
    logging.info(f"处理文件范围: {start_num:03d} 到 {end_num:03d}, 总共 {end_num - start_num + 1} 个文件")
    
    # 检查是否在终端中运行
    is_tty = sys.stdout.isatty()
    
    if is_tty:
        # 在终端中运行，使用 tqdm 进度条
        total_files = end_num - start_num + 1
        overall_pbar = tqdm(total=total_files, 
                           desc="总体进度", 
                           unit="file",
                           position=0,
                           ncols=100)
    else:
        # 在后台运行，使用简单进度
        total_files = end_num - start_num + 1
        overall_pbar = SimpleProgress(total_files, desc="总体文件进度")
    
    # 按顺序处理文件
    processed_count = 0
    for file_num in range(start_num, end_num + 1):
        input_filename = f"split_file_{file_num:03d}.txt"
        input_file_path = os.path.join(input_dir, input_filename)
        
        if not os.path.exists(input_file_path):
            logging.warning(f"文件不存在，跳过: {input_file_path}")
            overall_pbar.update(1)
            continue
            
        output_filename = f"noisy_split_file_{file_num:03d}.txt"
        output_file_path = os.path.join(output_dir, output_filename)
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file_path):
            logging.info(f"输出文件已存在，跳过: {output_file_path}")
            overall_pbar.update(1)
            continue
        
        logging.info(f"开始处理文件: {input_filename}")
        file_start_time = time.time()
        
        # 处理单个文件
        process_single_file(input_file_path, output_file_path, replace_ratio)
        
        # 更新进度
        processed_count += 1
        overall_pbar.update(1)
        
        file_elapsed_time = time.time() - file_start_time
        logging.info(f"文件处理完成: {input_filename}, 耗时: {file_elapsed_time:.2f}秒")
        
        # 处理完一个文件后休息一下
        time.sleep(1)
    
    # 关闭总体进度条
    overall_pbar.close()
    
    logging.info(f"批量处理完成 - 共处理 {processed_count} 个文件")

# 使用示例
if __name__ == "__main__":
    # 设置日志
    log_file = setup_logging()
    
    # 配置参数
    input_directory = "/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/incomplete_description/original"
    output_directory = "/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/incomplete_description/noise"
    replace_ratio = 1.0
    
    logging.info("=" * 80)
    logging.info("开始批量处理分割文本文件")
    logging.info(f"输入目录: {input_directory}")
    logging.info(f"输出目录: {output_directory}")
    logging.info(f"替换比例: {replace_ratio}")
    logging.info(f"日志文件: {log_file}")
    logging.info("=" * 80)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 方法1: 按顺序处理所有文件
        process_split_files_sequential(input_directory, output_directory, 
                                     start_num=1, end_num=None,
                                     replace_ratio=replace_ratio)
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logging.info(f"程序执行完成 - 总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info("所有文件处理完成！")
        
    except KeyboardInterrupt:
        logging.warning("程序被用户中断")
    except Exception as e:
        logging.error(f"程序执行出错: {e}", exc_info=True)
        raise
