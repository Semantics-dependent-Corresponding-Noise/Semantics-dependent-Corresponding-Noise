import base64
import os
import sys
import time
import signal
import json
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
from asyncio import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# ==================== 配置参数 ====================
# API 配置
client = AsyncOpenAI(
    api_key="yours api",
    base_url="https://ark.cn-beijing.volces.com/api/v3/",
)

# 测试模式配置
TEST_MODE = False  # 设置为True启用测试模式
if TEST_MODE:
    MAX_CONCURRENT_REQUESTS = 2  # 测试模式并发数
    SAVE_INTERVAL = 10  # 每10张图片保存一个文件
    MAX_TEST_IMAGES = 10  # 只测试前10张图片
else:
    MAX_CONCURRENT_REQUESTS = 20  # 生产模式并发数
    SAVE_INTERVAL = 1000  # 每1000张图片保存一个文件
    MAX_TEST_IMAGES = None  # 不限制图片数量

# 路径配置
TEST_IDS_PATH = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/test_ids.txt'
IMAGE_NAMES_PATH = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/annotations/scan_split/image_name.txt'
IMAGE_DIR = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_f30k/images'
OUTPUT_DIR = '/home/zbm/xjd/NPC-master/f30k_noise_construct/core_missing/test_flickr'

# 文件配置
LOG_FILE = os.path.join(OUTPUT_DIR, 'processing_flickr_test.log')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint_flickr_test.json')
PID_FILE = os.path.join(OUTPUT_DIR, 'processing_flickr_test.pid')

# 处理参数
MAX_API_RETRIES = 3
MAX_REGENERATION_ATTEMPTS = 5
API_TIMEOUT = 120.0

# 全局线程池用于文件I/O
io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="FileIO")

# ==================== 异步日志系统 ====================
class AsyncLogger:
    """异步双重输出：控制台+日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.lock = Lock()
        
    async def write(self, message):
        async with self.lock:
            self.terminal.write(message)
            self.terminal.flush()
            async with aiofiles.open(self.log_file, 'a', encoding='utf-8') as f:
                await f.write(message)
            
    def flush(self):
        self.terminal.flush()

    async def close(self):
        pass

# ==================== 全局锁定义 ====================
checkpoint_lock = None
failed_images_lock = None
logger = None

# ==================== 信号处理 ====================
def signal_handler(signum, frame):
    print(f"\n收到信号 {signum}，正在退出...")
    sys.exit(0)

# ==================== 数据加载 ====================
async def load_data():
    """加载Flickr30k测试集图片数据 - 去重处理"""
    print(f"正在加载测试集ID文件: {TEST_IDS_PATH}")
    print(f"正在加载图片名文件: {IMAGE_NAMES_PATH}")
    
    # 读取test_ids文件（包含重复ID）
    async with aiofiles.open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
        content = await f.read()
        lines = content.split('\n')
        test_indices = [int(line.strip()) for line in lines if line.strip().isdigit()]
    
    print(f"原始测试ID数量（包含重复）: {len(test_indices)}")
    
    # 读取image_names文件
    async with aiofiles.open(IMAGE_NAMES_PATH, 'r', encoding='utf-8') as f:
        content = await f.read()
        all_image_names = [line.strip() for line in content.split('\n') if line.strip()]
    
    print(f"总图片名数量: {len(all_image_names)}")
    
    # ✅ 关键修改：去重处理
    # 提取所有唯一索引
    unique_indices = list(set(test_indices))
    unique_indices.sort()  # 保持顺序
    
    # 统计重复情况
    id_counts = Counter(test_indices)
    total_repeats = len(test_indices) - len(unique_indices)
    
    print(f"✅ 唯一ID数量（去重后）: {len(unique_indices)}")
    print(f"📊 重复统计:")
    print(f"  重复ID总数: {total_repeats}")
    print(f"  平均每个ID重复次数: {len(test_indices)/len(unique_indices):.2f}")
    
    # 显示重复最多的几个ID
    if total_repeats > 0:
        print(f"  重复最多的前5个ID:")
        for i, (id, count) in enumerate(id_counts.most_common(5), 1):
            print(f"    {i}. ID {id}: 出现 {count} 次")
    
    # 根据唯一索引提取对应的图像文件名
    image_names_to_process = []
    for idx in unique_indices:
        if 0 <= idx < len(all_image_names):
            image_names_to_process.append(all_image_names[idx])
        else:
            print(f"⚠ 警告: 索引 {idx} 超出范围 (0-{len(all_image_names)-1})")
    
    print(f"生成 {len(image_names_to_process)} 个唯一图片名")
    
    # 验证前几个文件名
    if len(image_names_to_process) > 0:
        print(f"\n🔍 前10个唯一图片名:")
        for i in range(min(10, len(image_names_to_process))):
            print(f"  {i+1}. {image_names_to_process[i]}")
        
        # 检查文件是否存在
        print(f"\n📁 检查前5个文件是否存在:")
        for i in range(min(5, len(image_names_to_process))):
            img_path = os.path.join(IMAGE_DIR, image_names_to_process[i])
            exists = os.path.exists(img_path)
            print(f"  {image_names_to_process[i]}: {'✅ 存在' if exists else '❌ 不存在'}")
    
    # 测试模式：只处理前N张图片
    if TEST_MODE and MAX_TEST_IMAGES:
        if len(image_names_to_process) > MAX_TEST_IMAGES:
            print(f"\n🔬 测试模式: 只处理前 {MAX_TEST_IMAGES} 张图片")
            image_names_to_process = image_names_to_process[:MAX_TEST_IMAGES]
    
    await logger.write(f"📊 Flickr30k数据统计:\n")
    await logger.write(f"  原始ID数量（包含重复）: {len(test_indices)}\n")
    await logger.write(f"  唯一ID数量（去重后）: {len(unique_indices)}\n")
    await logger.write(f"  平均重复次数: {len(test_indices)/len(unique_indices):.2f}\n")
    await logger.write(f"  生成 {len(image_names_to_process)} 个唯一文件名\n")
    
    return image_names_to_process

# ==================== 检查点系统 ====================
async def load_checkpoint():
    """加载检查点"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            file_size = os.path.getsize(CHECKPOINT_FILE)
            if file_size == 0:
                await logger.write("检查点文件为空，使用默认配置\n")
                return create_default_checkpoint()
            
            async with aiofiles.open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                await logger.write("检查点文件内容为空，使用默认配置\n")
                return create_default_checkpoint()
                
            checkpoint = json.loads(content)
            await logger.write(f"成功加载检查点，已处理 {checkpoint.get('processed_count', 0)} 张图片\n")
            return checkpoint
            
        except json.JSONDecodeError as e:
            await logger.write(f"检查点文件JSON格式错误: {e}，使用默认配置\n")
            backup_file = f"{CHECKPOINT_FILE}.backup_{int(time.time())}"
            if os.path.exists(CHECKPOINT_FILE):
                os.rename(CHECKPOINT_FILE, backup_file)
            await logger.write(f"已备份损坏文件到: {backup_file}\n")
            return create_default_checkpoint()
        except Exception as e:
            await logger.write(f"加载检查点时出错: {e}，使用默认配置\n")
            return create_default_checkpoint()
    
    return create_default_checkpoint()

def create_default_checkpoint():
    """创建默认检查点"""
    return {
        "processed_count": 0,
        "file_count": 0,
        "failed_images": [],
        "all_descriptions": {},
        "current_file_number": 1,
        "current_file_start_idx": 0,
        "timestamp": time.time()
    }

def save_checkpoint_sync(checkpoint_data, checkpoint_path):
    """同步保存检查点文件（在线程池中执行）"""
    try:
        temp_file = f"{checkpoint_path}.tmp"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # 原子性地替换文件
        os.replace(temp_file, checkpoint_path)
        return True
    except Exception as e:
        print(f"保存检查点失败: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

async def save_checkpoint(checkpoint_data):
    """异步保存检查点"""
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(
        io_executor,
        save_checkpoint_sync,
        checkpoint_data,
        CHECKPOINT_FILE
    )
    return success

# ==================== 图像编码 ====================
def encode_image_to_base64(image_path):
    """同步编码图片为base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败 {image_path}: {e}")
        return None

# ==================== 核心生成逻辑 ====================
async def generate_single_description(image_path, img_name, temperature=0.8, retry_count=0, semaphore=None):
    """单张图片生成描述，使用信号量控制并发"""
    prompt1 = """You are a professional image description assistant. Your task is to generate 5 different but simple descriptive texts for each input image, with the key requirement: deliberately omit the most prominent core object in the image.Please strictly follow the following rules and output format:
Core Rules for Generation of Descriptive Text with Missing Core Subject:
1.Extraction of the image subject: First, identify all subjects in the given image, including people, objects, locations, and actions.
2.Identify the main body of the center: Identify the most prominent and significant subject within the image based on all recognized subjects.  
3.Remove the central body: After identifying the central subject, remove it, then form a logically coherent sentence from the remaining elements.
4.Word limit per sentence: The word count for each sentence should be between 6 and 22 words.
Example :
- Input Image: A guy stitching up another man's coat.
- Output Sentence: A man's coat.
- Input Image: A boys jumps into the water upside down.
- Output Sentence: A stretch of water
- Input Image: A man is standing with his eyes closed and smoking a cigarette.
- Output Sentence: A room.
Strict Output Format:
Only output the modified sentence directly. Do NOT add any extra content (such as explanations, notes, or greetings)."""
    
    if retry_count > 0:
        prompt1 += f"\n\nIMPORTANT: You previously generated {retry_count} descriptions, but we need exactly 5 unique descriptions. Please generate {5 - retry_count} more unique descriptions that are different from the previous ones."
    
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return ""
    
    user_content = [
        {"type": "text", "text": f"Image {img_name}:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    
    async with semaphore:
        for attempt in range(MAX_API_RETRIES):
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="doubao-seed-1-6-vision-250815",
                        messages=[
                            {"role": "system", "content": prompt1},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=temperature,
                        max_tokens=500,
                    ),
                    timeout=API_TIMEOUT
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                if attempt == MAX_API_RETRIES - 1:
                    await logger.write(f"❌ {img_name}: API调用失败: {str(e)[:50]}\n")
                    return ""
                await asyncio.sleep(2)
    
    return ""

def parse_single_response(response_text):
    """解析单张图片的描述 - 只提取纯描述文本"""
    if not response_text:
        return []
    
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    filtered_lines = []
    
    for line in lines:
        # 移除图片文件名
        if any(ext in line for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
            continue
        # 移除数字编号
        elif re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
            filtered_lines.append(line)
        # 移除其他可能的前缀
        elif re.match(r'^[•\-*]\s*', line):
            line = re.sub(r'^[•\-*]\s*', '', line)
            filtered_lines.append(line)
        # 移除包含"Image:"等前缀的行
        elif re.match(r'^(Image|图片|IMG|img)\s*[:：]', line, re.IGNORECASE):
            continue
        else:
            filtered_lines.append(line)
    
    # 清洗描述文本
    cleaned_lines = []
    for line in filtered_lines:
        # 移除可能包含的图片文件名
        line = re.sub(r'\b\d+\.jpg\b', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\bimage\s*\d+\b', '', line, flags=re.IGNORECASE)
        line = line.strip()
        
        # 移除开头结尾的标点
        line = re.sub(r'^[":\-\s]+', '', line)
        line = re.sub(r'[":\-\s]+$', '', line)
        
        if line:
            cleaned_lines.append(line)
    
    return cleaned_lines

async def process_single_image_with_retry(img_name, img_path, semaphore):
    """处理单张图片，如果不足5条描述则重新生成"""
    all_descriptions = []
    seen_descriptions = set()
    temperature = 0.8
    regeneration_attempts = 0
    
    while len(all_descriptions) < 5 and regeneration_attempts < MAX_REGENERATION_ATTEMPTS:
        needed = 5 - len(all_descriptions)
        
        if regeneration_attempts > 0:
            await logger.write(f"  ↳ {img_name}: 第{regeneration_attempts+1}次重试，还需要{needed}条描述\n")
            temperature = min(1.2, temperature + 0.1)
        
        response_text = await generate_single_description(img_path, img_name, temperature, len(all_descriptions), semaphore)
        new_descriptions = parse_single_response(response_text)
        
        if not new_descriptions:
            await logger.write(f"  ↳ {img_name}: 第{regeneration_attempts+1}次调用返回空结果\n")
            regeneration_attempts += 1
            await asyncio.sleep(2)
            continue
        
        unique_new_descriptions = []
        for desc in new_descriptions:
            cleaned_desc = desc.strip()
            # 移除过短的描述
            if len(cleaned_desc.split()) < 3:
                continue
            # 检查是否重复
            if cleaned_desc not in seen_descriptions and cleaned_desc not in all_descriptions:
                unique_new_descriptions.append(cleaned_desc)
                seen_descriptions.add(cleaned_desc)
        
        if unique_new_descriptions:
            all_descriptions.extend(unique_new_descriptions[:needed])
        
        regeneration_attempts += 1
        
        if len(all_descriptions) < 5:
            await asyncio.sleep(1)
    
    if len(all_descriptions) >= 5:
        result = all_descriptions[:5]
        await logger.write(f"✓ {img_name}: 成功生成5条描述（尝试{regeneration_attempts}次）\n")
        return result, False
    
    await logger.write(f"❌ {img_name}: 无法生成5条描述，只有{len(all_descriptions)}条\n")
    return [f"描述生成失败_{i+1}" for i in range(5)], True

async def process_single_image(img_name, img_path, semaphore):
    """处理单张图片的主函数"""
    try:
        return await process_single_image_with_retry(img_name, img_path, semaphore)
    except Exception as e:
        await logger.write(f"❌ {img_name}: 处理过程中发生异常: {str(e)[:50]}\n")
        return [f"处理异常_{i+1}" for i in range(5)], True

# ==================== 批量保存功能 ====================
async def save_images_file(file_number, all_descriptions, image_names_to_process, start_idx, end_idx):
    """保存每批图片的描述到一个文件 - 只保存纯描述文本"""
    output_file_path = os.path.join(OUTPUT_DIR, f'test_caps_5_per_image_part{file_number:03d}.txt')
    
    print(f"\n💾 正在保存第 {file_number} 个文件: {output_file_path}")
    print(f"  图片索引范围: {start_idx} 到 {end_idx-1}")
    print(f"  应包含图片数: {end_idx - start_idx}")
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        lines_to_write = []
        saved_count = 0
        missing_count = 0
        failed_count = 0
        
        print(f"  正在保存图片描述（纯文本）:")
        
        for i in range(start_idx, end_idx):
            img_name = image_names_to_process[i]
            
            # 检查是否有描述
            if img_name not in all_descriptions:
                descriptions = [f"描述未生成_{j+1}" for j in range(5)]
                missing_count += 1
            else:
                descriptions = all_descriptions[img_name]
                
                if not descriptions:
                    descriptions = [f"描述为空_{j+1}" for j in range(5)]
                    missing_count += 1
            
            # 检查是否是失败描述
            is_failed = any("失败" in desc or "未生成" in desc or "为空" in desc or "异常" in desc for desc in descriptions)
            if is_failed:
                failed_count += 1
            
            # 确保有5个描述
            if len(descriptions) < 5:
                missing = 5 - len(descriptions)
                placeholders = [f"补充描述_{j+1}" for j in range(missing)]
                descriptions = descriptions + placeholders
                missing_count += 1
            elif len(descriptions) > 5:
                descriptions = descriptions[:5]
            
            # 添加纯描述文本
            for desc in descriptions:
                clean_desc = desc.strip()
                # 清理可能的图片名残留
                clean_desc = re.sub(r'\b\d+\.jpg\b', '', clean_desc, flags=re.IGNORECASE)
                clean_desc = re.sub(r'\bimage\s*\d+\b', '', clean_desc, flags=re.IGNORECASE)
                clean_desc = clean_desc.strip()
                
                if not clean_desc:
                    clean_desc = "描述内容为空"
                
                lines_to_write.append(clean_desc)
            
            saved_count += 1
            
            if (i - start_idx + 1) % 10 == 0:
                print(f"    已处理 {i-start_idx+1}/{end_idx-start_idx} 张图片")
        
        print(f"\n  准备写入 {len(lines_to_write)} 行纯描述文本")
        print(f"  按顺序处理了 {saved_count} 张图片")
        if missing_count > 0:
            print(f"  ⚠ 有 {missing_count} 张图片没有完整描述数据")
        if failed_count > 0:
            print(f"  ❌ 有 {failed_count} 张图片生成失败")
        
        # 写入文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in lines_to_write:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        
        # 验证文件
        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path)
            print(f"\n  ✅ 文件保存成功!")
            print(f"    大小: {file_size} 字节")
            print(f"    图片: {saved_count} 张")
            print(f"    描述行数: {len(lines_to_write)} 行")
            
            # 验证文件内容
            with open(output_file_path, 'r', encoding='utf-8') as f:
                first_few_lines = [f.readline().strip() for _ in range(5)]
                print(f"\n    文件前5行内容:")
                for idx, line in enumerate(first_few_lines, 1):
                    print(f"      行{idx}: {line[:60]}...")
            
            await logger.write(f"\n✅ 已保存第 {file_number} 个文件: {output_file_path}\n")
            await logger.write(f"   包含图片索引 {start_idx} 到 {end_idx-1} (共{saved_count}张图片)\n")
            await logger.write(f"   文件大小: {file_size} 字节, {len(lines_to_write)} 行描述\n")
            if missing_count > 0:
                await logger.write(f"   ⚠ {missing_count}张图片没有完整描述数据\n")
            
            return saved_count, failed_count
        else:
            print(f"  ❌ 文件未创建!")
            return 0, 0
            
    except Exception as e:
        print(f"  ❌ 保存文件失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

# ==================== 进度跟踪器 ====================
class ProgressTracker:
    """跟踪处理进度"""
    def __init__(self, total_images):
        self.total = total_images
        self.processed = 0
        self.lock = Lock()
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_progress_log = 0
    
    async def update(self, count=1):
        async with self.lock:
            self.processed += count
            
            if self.processed > self.total:
                self.processed = self.total
            
            progress = self.processed / self.total * 100 if self.total > 0 else 0
            elapsed = time.time() - self.start_time
            
            current_time = time.time()
            if self.processed - self.last_progress_log >= 10 or current_time - self.last_log_time > 5:
                if elapsed > 0:
                    speed = self.processed / elapsed
                    eta = (self.total - self.processed) / speed if speed > 0 and self.processed < self.total else 0
                    
                    await logger.write(
                        f"📊 进度: {self.processed}/{self.total} ({progress:.1f}%) | "
                        f"速度: {speed:.2f} 张/秒 | ETA: {eta/60:.1f} 分钟\n"
                    )
                self.last_log_time = current_time
                self.last_progress_log = self.processed

# ==================== 处理单个文件的图片 ====================
async def process_single_file_images(file_number, start_idx, end_idx, image_names_to_process, 
                                    all_descriptions, failed_images, semaphore, progress_tracker):
    """处理单个文件的所有图片"""
    print(f"\n📁 开始处理第 {file_number} 个文件")
    print(f"  图片索引: {start_idx} 到 {end_idx-1}")
    print(f"  共 {end_idx - start_idx} 张图片")
    
    file_successful_count = 0
    file_failed_count = 0
    
    # 创建并发任务
    tasks = []
    for idx in range(start_idx, end_idx):
        img_name = image_names_to_process[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            await logger.write(f"❌ {img_name}: 文件不存在\n")
            async with failed_images_lock:
                all_descriptions[img_name] = [f"文件缺失_{j+1}" for j in range(5)]
                failed_images.append(img_name)
                file_failed_count += 1
            await progress_tracker.update(1)
            continue
        
        # 检查是否已经处理过
        if img_name in all_descriptions and all_descriptions[img_name]:
            await logger.write(f"⏩ {img_name}: 已处理，跳过\n")
            await progress_tracker.update(1)
            continue
        
        # 创建并发任务
        task = asyncio.create_task(process_single_image(img_name, img_path, semaphore))
        tasks.append((task, idx, img_name))
    
    if not tasks:
        print(f"  ⏩ 当前文件没有需要处理的任务")
        return file_successful_count, file_failed_count
    
    print(f"  🚀 开始并发处理 {len(tasks)} 个任务 (并发数: {MAX_CONCURRENT_REQUESTS})...")
    
    # 并发执行所有任务
    results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)
    
    # 处理结果
    for (task, idx, img_name), result in zip(tasks, results):
        if isinstance(result, Exception):
            await logger.write(f"❌ {img_name}: 任务执行失败 {str(result)[:50]}\n")
            async with checkpoint_lock:
                all_descriptions[img_name] = [f"任务失败_{i+1}" for i in range(5)]
            async with failed_images_lock:
                failed_images.append(img_name)
            file_failed_count += 1
            await progress_tracker.update(1)
        elif isinstance(result, tuple) and len(result) == 2:
            descriptions, is_failed = result
            async with checkpoint_lock:
                all_descriptions[img_name] = descriptions
            if is_failed:
                async with failed_images_lock:
                    failed_images.append(img_name)
                file_failed_count += 1
            else:
                file_successful_count += 1
            await progress_tracker.update(1)
        else:
            await logger.write(f"⚠ {img_name}: 返回结果格式异常\n")
            async with checkpoint_lock:
                all_descriptions[img_name] = [f"格式错误_{i+1}" for i in range(5)]
            async with failed_images_lock:
                failed_images.append(img_name)
            file_failed_count += 1
            await progress_tracker.update(1)
    
    return file_successful_count, file_failed_count

# ==================== 主程序 ====================
async def main_async():
    """主异步函数"""
    global logger, checkpoint_lock, failed_images_lock
    
    # 初始化全局锁
    checkpoint_lock = Lock()
    failed_images_lock = Lock()
    
    logger = AsyncLogger(LOG_FILE)
    sys.stdout = sys.stderr = logger.terminal
    
    try:
        print("=" * 60)
        print("🚀 Flickr30k测试集图片描述生成程序启动（去重模式）")
        if TEST_MODE:
            print("🔬 测试模式: 2并发，每10张图片保存一个文件")
        print(f"输出目录: {OUTPUT_DIR}")
        print(f"并发数: {MAX_CONCURRENT_REQUESTS}")
        print(f"保存间隔: 每{SAVE_INTERVAL}张图片保存一个文件")
        print("=" * 60)
        
        # 加载数据（去重处理）
        print("\n📋 正在加载Flickr30k测试集数据（去重处理）...")
        image_names_to_process = await load_data()
        total_images = len(image_names_to_process)
        
        if total_images == 0:
            print("❌ 错误: 没有找到要处理的图片!")
            return
        
        print(f"\n✅ 总共需要处理 {total_images} 张唯一图片")
        print(f"   预期结果: {total_images} 张图片 × 5条描述 = {total_images * 5} 行描述")
        
        total_files = (total_images + SAVE_INTERVAL - 1) // SAVE_INTERVAL
        print(f"   将保存 {total_files} 个文件")
        
        # 加载检查点
        print("\n🔍 正在加载检查点...")
        checkpoint = await load_checkpoint()
        processed_count = checkpoint.get("processed_count", 0)
        file_count = checkpoint.get("file_count", 0)
        failed_images = checkpoint.get("failed_images", [])
        all_descriptions = checkpoint.get("all_descriptions", {})
        current_file_number = checkpoint.get("current_file_number", 1)
        current_file_start_idx = checkpoint.get("current_file_start_idx", 0)
        
        # 初始化描述字典
        for img_name in image_names_to_process:
            if img_name not in all_descriptions:
                all_descriptions[img_name] = []
        
        print(f"📊 检查点恢复:")
        print(f"  已处理图片: {processed_count}/{total_images}")
        print(f"  已保存文件: {file_count}/{total_files}")
        print(f"  当前处理文件: 第{current_file_number}个")
        print(f"  失败图片数: {len(failed_images)}")
        print(f"  并发数: {MAX_CONCURRENT_REQUESTS}")
        
        await logger.write(f"🔍 检查点恢复: {processed_count}/{total_images} 张图片已处理\n")
        await logger.write(f"❌ 失败图片数: {len(failed_images)}\n")
        await logger.write(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 创建信号量控制并发
        semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
        progress_tracker = ProgressTracker(total_images)
        
        # 设置进度从已处理数量开始
        progress_tracker.processed = processed_count
        
        # 按文件顺序处理
        total_successful = 0
        total_failed = 0
        
        for file_number in range(current_file_number, total_files + 1):
            start_idx = (file_number - 1) * SAVE_INTERVAL
            end_idx = min(start_idx + SAVE_INTERVAL, total_images)
            
            print(f"\n{'='*60}")
            print(f"📁 处理第 {file_number}/{total_files} 个文件")
            print(f"  图片索引: {start_idx} 到 {end_idx-1}")
            print(f"  共 {end_idx - start_idx} 张图片")
            print(f"{'='*60}")
            
            await logger.write(f"\n📁 开始处理第 {file_number} 个文件 (图片 {start_idx}-{end_idx-1})\n")
            
            # 处理当前文件的所有图片
            file_successful, file_failed = await process_single_file_images(
                file_number, start_idx, end_idx, image_names_to_process,
                all_descriptions, failed_images, semaphore, progress_tracker
            )
            
            total_successful += file_successful
            total_failed += file_failed
            
            print(f"\n📊 第 {file_number} 个文件处理完成:")
            print(f"  成功: {file_successful} 张")
            print(f"  失败: {file_failed} 张")
            
            # 立即保存当前文件
            print(f"\n💾 立即保存第 {file_number} 个文件（纯描述文本）...")
            saved_count, failed_in_file = await save_images_file(
                file_number, all_descriptions, image_names_to_process, start_idx, end_idx
            )
            
            if saved_count > 0:
                print(f"  ✅ 第 {file_number} 个文件保存成功!")
                print(f"     包含 {saved_count} 张图片的描述")
                
                # 更新检查点
                checkpoint_data = {
                    "processed_count": end_idx,
                    "file_count": file_number,
                    "failed_images": failed_images,
                    "all_descriptions": all_descriptions,
                    "current_file_number": file_number + 1,
                    "current_file_start_idx": end_idx,
                    "timestamp": time.time()
                }
                
                async with checkpoint_lock:
                    await save_checkpoint(checkpoint_data)
                
                print(f"  💾 检查点已更新: 文件{file_number}, 图片{end_idx}")
            else:
                print(f"  ❌ 第 {file_number} 个文件保存失败!")
            
            # 输出统计信息
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > 0:
                current_processed = min(file_number * SAVE_INTERVAL, total_images)
                speed = current_processed / elapsed
                remaining_images = total_images - current_processed
                eta = remaining_images / speed if speed > 0 else 0
                
                print(f"\n📈 总体进度:")
                print(f"  已处理文件: {file_number}/{total_files}")
                print(f"  已处理图片: {current_processed}/{total_images}")
                print(f"  平均速度: {speed:.2f} 张/秒")
                if remaining_images > 0:
                    print(f"  预计剩余时间: {eta/60:.1f} 分钟")
            
            # 每个文件处理后等待一下
            if file_number < total_files:
                print(f"\n⏳ 准备处理下一个文件，等待3秒...")
                await asyncio.sleep(3)
        
        # 所有文件处理完成
        print(f"\n{'='*60}")
        print("🎉 所有文件处理完成!")
        
        # 保存失败图片列表
        if failed_images:
            failed_file = os.path.join(OUTPUT_DIR, 'failed_test_images.txt')
            print(f"\n📋 保存失败图片列表到: {failed_file}")
            
            try:
                with open(failed_file, 'w', encoding='utf-8') as f:
                    for img_name in failed_images:
                        f.write(f"{img_name}\n")
                print(f"失败图片列表已保存 ({len(failed_images)}张)")
            except Exception as e:
                print(f"保存失败图片列表时出错: {e}")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("✅ 处理完成!")
        print(f"⏱️  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"🚀 平均速度: {total_images/total_time:.2f} 张/秒")
        print(f"📊 成功: {total_successful} 张 | 失败: {total_failed} 张")
        print(f"📁 保存文件数: {total_files} 个")
        print(f"📝 总描述行数: {total_images * 5} 行")
        print(f"{'='*60}")
        
        # 列出生成的文件
        print("\n📋 生成的文件列表:")
        import glob
        files = glob.glob(os.path.join(OUTPUT_DIR, "test_caps_5_per_image_part*.txt"))
        for file in sorted(files):
            size = os.path.getsize(file)
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"  {os.path.basename(file)} - {size} 字节, {len(lines)} 行, {len(lines)//5} 张图片")
            
            if file == files[0]:
                print(f"    前3行内容:")
                for i, line in enumerate(lines[:3], 1):
                    print(f"      行{i}: {line.strip()[:60]}...")
        
        # 删除检查点文件
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("\n🧹 清理检查点文件")
        
        await logger.write(f"\n{'='*60}\n")
        await logger.write("✅ 处理完成!\n")
        await logger.write(f"⏱️  总耗时: {total_time:.2f} 秒\n")
        await logger.write(f"🚀 平均速度: {total_images/total_time:.2f} 张/秒\n")
        await logger.write(f"📊 成功: {total_successful} 张 | 失败: {total_failed} 张\n")
        await logger.write(f"📁 保存文件数: {total_files} 个\n")
        await logger.write(f"📝 总描述行数: {total_images * 5} 行\n")
        await logger.write(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        import traceback
        traceback.print_exc(file=logger.terminal)
        
        # 发生异常时保存当前状态
        try:
            checkpoint_data = {
                "processed_count": progress_tracker.processed if 'progress_tracker' in locals() else 0,
                "file_count": file_count if 'file_count' in locals() else 0,
                "failed_images": failed_images if 'failed_images' in locals() else [],
                "all_descriptions": all_descriptions if 'all_descriptions' in locals() else {},
                "current_file_number": current_file_number if 'current_file_number' in locals() else 1,
                "current_file_start_idx": current_file_start_idx if 'current_file_start_idx' in locals() else 0,
                "timestamp": time.time()
            }
            
            async with checkpoint_lock:
                await save_checkpoint(checkpoint_data)
            print("💾 已保存异常时的检查点")
        except:
            pass
    finally:
        await asyncio.sleep(1)
        await logger.close()

# ==================== 入口 ====================
if __name__ == "__main__":
    print("Flickr30k测试集处理程序启动（去重模式）...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 检查是否重复运行
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            old_pid = f.read().strip()
        try:
            os.kill(int(old_pid), 0)
            print(f"警告: 进程 {old_pid} 已在运行!")
            print(f"如需重启，请删除: {PID_FILE}")
            sys.exit(1)
        except OSError:
            os.remove(PID_FILE)
    
    # 保存PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    print(f"PID文件: {PID_FILE}")
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("开始异步主程序...")
        asyncio.run(main_async())
    except Exception as e:
        print(f"主程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        print("程序结束")
