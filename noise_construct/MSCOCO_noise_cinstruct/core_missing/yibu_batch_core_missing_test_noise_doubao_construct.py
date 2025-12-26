import base64
import os
import sys
import time
import signal
import json
import re
import asyncio
import aiofiles
from openai import AsyncOpenAI
from asyncio import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor

# ==================== 配置参数 ====================
client = AsyncOpenAI(
    api_key="3d866616-54c8-4222-bb96-d5b6e208fbb7",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)

# 测试模式配置
TEST_MODE = False  # 设置为True启用测试模式
if TEST_MODE:
    # 测试模式参数
    MAX_CONCURRENT_REQUESTS = 2  # 并发数改为2
    SAVE_INTERVAL = 10  # 每10张图片保存一个文件（测试用）
    MAX_TEST_IMAGES = 10  # 只测试前10张图片
else:
    # 生产模式参数
    MAX_CONCURRENT_REQUESTS = 20  # 真正的并发数
    SAVE_INTERVAL = 1000  # 每1000张图片保存一个文件
    MAX_TEST_IMAGES = None  # 不限制图片数量

# 其他配置
CHECKPOINT_INTERVAL = 50

# 数据集配置
DATASET_TYPE = 'coco'
IMAGE_DIR = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/images'
TEST_IDS_PATH = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/annotations/scan_split/test_ids.txt'
OUTPUT_DIR = '/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/core_missing/test_testid'

# 文件配置
LOG_FILE = os.path.join(OUTPUT_DIR, 'processing_test.log')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint_test.json')
PID_FILE = os.path.join(OUTPUT_DIR, 'processing_test.pid')

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
    """加载测试集图片数据 - 处理重复ID"""
    print(f"正在加载测试集ID文件: {TEST_IDS_PATH}")
    
    # 首先读取原始文件内容
    async with aiofiles.open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
        content = await f.read()
        
        # 调试：显示文件内容
        print(f"原始文件内容前100字符: {content[:100]}")
        print(f"文件总字符数: {len(content)}")
        
        lines = content.strip().split('\n')
        print(f"原始行数: {len(lines)}")
        print(f"前10行: {lines[:10]}")
    
    # 提取所有ID（包含重复）
    all_ids = []
    for line in lines:
        line = line.strip()
        if line and line.isdigit():
            all_ids.append(int(line))
    
    print(f"\n📊 原始ID数量（包含重复）: {len(all_ids)}")
    
    # 验证文件格式
    print(f"\n🔍 验证test_ids.txt格式:")
    print(f"  总行数: {len(all_ids)}")
    
    if len(all_ids) % 5 == 0:
        print(f"  ✅ 格式正确: 行数是5的倍数")
    else:
        print(f"  ⚠ 警告: 行数 {len(all_ids)} 不是5的倍数")
    
    # 检查前几组
    print(f"\n  前3组ID验证:")
    for i in range(0, min(15, len(all_ids)), 5):
        group = all_ids[i:i+5]
        if len(group) == 5 and len(set(group)) == 1:
            print(f"    组 {i//5 + 1}: ID {group[0]} 重复5次 - ✅ 正确")
        else:
            print(f"    组 {i//5 + 1}: {group} - ❌ 错误")
    
    # ✅ 关键修改：按照test_ids.txt的格式，每5个重复ID对应一张图片
    unique_ids = []
    processed_ids_info = []
    
    # 每5个ID为一组，提取第一个（因为是重复的）
    for i in range(0, len(all_ids), 5):
        if i < len(all_ids):
            current_id = all_ids[i]
            unique_ids.append(current_id)
            processed_ids_info.append({
                'original_index': i,
                'id': current_id,
                'group_size': 5
            })
    
    print(f"\n✅ 处理后唯一ID数量（按5个一组）: {len(unique_ids)}")
    print(f"📊 重复统计:")
    print(f"  总共 {len(all_ids)} 行，每5行对应一个ID的5个描述")
    print(f"  提取了 {len(unique_ids)} 个唯一ID")
    
    # 验证ID分布
    print(f"\n🔍 ID分布验证:")
    from collections import Counter
    id_counts = Counter(all_ids)
    for i in range(min(5, len(unique_ids))):
        id = unique_ids[i]
        count = id_counts[id]
        expected_count = 5  # 期望每个ID重复5次
        if count == expected_count:
            print(f"  ID {id}: 出现 {count} 次 - ✅ 符合期望")
        else:
            print(f"  ID {id}: 出现 {count} 次 - ⚠ 期望是{expected_count}次")
    
    # ✅ 生成图片名：直接使用提取的ID
    image_names_to_process = []
    valid_ids = []
    
    print(f"\n📁 检查图片文件是否存在:")
    for idx, id in enumerate(unique_ids):
        # COCO图片格式：COCO_2014_000000130524.jpg
        image_name = f'COCO_2014_{str(id).rjust(12, "0")}.jpg'
        image_path = os.path.join(IMAGE_DIR, image_name)
        
        # 检查图片是否存在
        if os.path.exists(image_path):
            image_names_to_process.append(image_name)
            valid_ids.append(id)
            if idx < 5:  # 只显示前5个
                print(f"  {idx+1}. {image_name} - ✅ 存在")
        else:
            print(f"  ❌ {image_name} - 文件不存在，跳过ID {id}")
            # 使用备用检查方式
            # 尝试查找其他可能的文件名格式
            alt_names = [
                f'COCO_train2014_{str(id).rjust(12, "0")}.jpg',
                f'{id}.jpg',
                f'COCO_2014_{id}.jpg'
            ]
            found_alt = False
            for alt_name in alt_names:
                alt_path = os.path.join(IMAGE_DIR, alt_name)
                if os.path.exists(alt_path):
                    image_names_to_process.append(alt_name)
                    valid_ids.append(id)
                    print(f"  ➡ 找到备用文件名: {alt_name}")
                    found_alt = True
                    break
    
    if not image_names_to_process:
        print(f"\n❌ 错误: 没有找到任何图片文件!")
        print(f"请检查图片目录: {IMAGE_DIR}")
        print(f"图片应该类似: COCO_2014_000000130524.jpg")
        return []
    
    print(f"\n📊 图片文件验证结果:")
    print(f"  总共找到 {len(image_names_to_process)} 个有效图片文件")
    print(f"  前5个图片:")
    for i in range(min(5, len(image_names_to_process))):
        img_path = os.path.join(IMAGE_DIR, image_names_to_process[i])
        print(f"  {i+1}. {image_names_to_process[i]}")
        print(f"     完整路径: {img_path}")
        print(f"     是否存在: {'✅ 是' if os.path.exists(img_path) else '❌ 否'}")
    
    # 测试模式：只处理前N张图片
    if TEST_MODE and MAX_TEST_IMAGES:
        if len(image_names_to_process) > MAX_TEST_IMAGES:
            print(f"\n🔬 测试模式: 只处理前 {MAX_TEST_IMAGES} 张图片")
            image_names_to_process = image_names_to_process[:MAX_TEST_IMAGES]
    
    await logger.write(f"📊 数据统计:\n")
    await logger.write(f"  原始ID行数: {len(all_ids)}\n")
    await logger.write(f"  处理后的唯一ID数量: {len(unique_ids)}\n")
    await logger.write(f"  有效图片文件数量: {len(image_names_to_process)}\n")
    
    # ✅ 关键：记录图片名和原始ID的对应关系
    image_id_mapping = {}
    for img_name, orig_id in zip(image_names_to_process, valid_ids):
        image_id_mapping[img_name] = orig_id
        await logger.write(f"  图片 {img_name} -> 原始ID {orig_id}\n")
    
    # 保存映射关系到临时文件用于调试
    mapping_file = os.path.join(OUTPUT_DIR, 'image_id_mapping.json')
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(image_id_mapping, f, indent=2, ensure_ascii=False)
        print(f"\n📝 图片ID映射已保存到: {mapping_file}")
    except Exception as e:
        print(f"⚠ 保存映射文件失败: {e}")
    
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
        "current_file_number": 1,  # 当前正在处理的文件编号
        "current_file_start_idx": 0,  # 当前文件开始的索引
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
        # 清理临时文件
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
    """同步编码（本地IO，无需异步）"""
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
    
    # ✅ 关键修改：发送提示时只包含纯文本图片名，不发送数字ID
    user_content = [
        {"type": "text", "text": f"Image: {img_name}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    
    async with semaphore:
        for attempt in range(MAX_API_RETRIES):
            try:
                # 调试：记录发送的内容
                print(f"  发送图片: {img_name}")
                
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
                
                response = completion.choices[0].message.content.strip()
                print(f"  收到响应长度: {len(response)} 字符")
                if len(response) < 100:
                    print(f"  响应内容: {response}")
                
                return response
            except Exception as e:
                error_msg = str(e)
                print(f"  API调用失败 ({attempt+1}/{MAX_API_RETRIES}): {error_msg[:50]}")
                if attempt == MAX_API_RETRIES - 1:
                    await logger.write(f"❌ {img_name}: API调用失败: {error_msg[:50]}\n")
                    return ""
                await asyncio.sleep(2)
    
    return ""

def parse_single_response(response_text):
    """解析单张图片的描述 - 只提取纯描述文本"""
    if not response_text:
        print("  响应文本为空")
        return []
    
    print(f"  解析响应文本: {len(response_text)} 字符")
    
    # 如果响应文本很短，直接返回
    if len(response_text.strip()) < 20:
        print(f"  响应文本过短，可能有问题: {response_text}")
        return []
    
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    print(f"  分割成 {len(lines)} 行")
    
    filtered_lines = []
    
    for idx, line in enumerate(lines):
        # 调试：显示原始行
        print(f"    行 {idx}: '{line}'")
        
        # 移除图片文件名
        if any(ext in line for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
            print(f"      移除: 包含图片扩展名")
            continue
        # 移除数字编号（如 "1. "、"2. "等）
        elif re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
            filtered_lines.append(line)
            print(f"      保留: 去除编号后: '{line}'")
        # 移除其他可能的前缀
        elif re.match(r'^[•\-*]\s*', line):
            line = re.sub(r'^[•\-*]\s*', '', line)
            filtered_lines.append(line)
            print(f"      保留: 去除项目符号后: '{line}'")
        # 移除包含"Image:"、"图片:"等前缀的行
        elif re.match(r'^(Image|图片|IMG|img)\s*[:：]', line, re.IGNORECASE):
            print(f"      移除: 包含Image前缀")
            continue
        else:
            filtered_lines.append(line)
            print(f"      保留: 原始行")
    
    # ✅ 关键：清洗描述文本，移除图片名等无关内容
    cleaned_lines = []
    for idx, line in enumerate(filtered_lines):
        # 移除可能包含的图片文件名
        original_line = line
        line = re.sub(r'\bCOCO_2014_\d{12}\.jpg\b', '', line)
        line = re.sub(r'\b\d{12}\b', '', line)  # 移除12位数字ID
        line = re.sub(r'\bimage\s*\d+\b', '', line, flags=re.IGNORECASE)
        line = line.strip()
        
        # 移除开头结尾的标点
        line = re.sub(r'^[":\-\s]+', '', line)
        line = re.sub(r'[":\-\s]+$', '', line)
        
        if line:
            cleaned_lines.append(line)
            if line != original_line:
                print(f"      清洗后行 {idx}: '{line}' (原: '{original_line}')")
    
    print(f"  最终得到 {len(cleaned_lines)} 个描述")
    if cleaned_lines:
        print(f"  前3个描述: {cleaned_lines[:3]}")
    
    return cleaned_lines

async def process_single_image_with_retry(img_name, img_path, semaphore):
    """处理单张图片，如果不足5条描述则重新生成"""
    all_descriptions = []
    seen_descriptions = set()
    temperature = 0.8
    regeneration_attempts = 0
    
    print(f"\n🔄 开始处理图片: {img_name}")
    print(f"  图片路径: {img_path}")
    print(f"  是否存在: {'✅ 是' if os.path.exists(img_path) else '❌ 否'}")
    
    while len(all_descriptions) < 5 and regeneration_attempts < MAX_REGENERATION_ATTEMPTS:
        needed = 5 - len(all_descriptions)
        
        if regeneration_attempts > 0:
            print(f"  ↳ 第{regeneration_attempts+1}次重试，还需要{needed}条描述")
            temperature = min(1.2, temperature + 0.1)
        
        response_text = await generate_single_description(img_path, img_name, temperature, len(all_descriptions), semaphore)
        new_descriptions = parse_single_response(response_text)
        
        if not new_descriptions:
            print(f"  ↳ 第{regeneration_attempts+1}次调用返回空结果")
            regeneration_attempts += 1
            await asyncio.sleep(2)
            continue
        
        unique_new_descriptions = []
        for desc in new_descriptions:
            # 清洗描述文本
            cleaned_desc = desc.strip()
            # 移除过短的描述（小于3个单词）
            if len(cleaned_desc.split()) < 3:
                print(f"    跳过过短描述: '{cleaned_desc}'")
                continue
            # 检查是否重复
            if cleaned_desc not in seen_descriptions and cleaned_desc not in all_descriptions:
                unique_new_descriptions.append(cleaned_desc)
                seen_descriptions.add(cleaned_desc)
                print(f"    添加新描述: '{cleaned_desc[:50]}...'")
            else:
                print(f"    跳过重复描述: '{cleaned_desc[:50]}...'")
        
        if unique_new_descriptions:
            all_descriptions.extend(unique_new_descriptions[:needed])
            print(f"    当前总共 {len(all_descriptions)} 条描述")
        
        regeneration_attempts += 1
        
        if len(all_descriptions) < 5:
            await asyncio.sleep(1)
    
    if len(all_descriptions) >= 5:
        result = all_descriptions[:5]
        print(f"✅ {img_name}: 成功生成5条描述（尝试{regeneration_attempts}次）")
        for i, desc in enumerate(result, 1):
            print(f"    {i}. {desc[:60]}...")
        return result, False  # 第二个返回值表示是否失败
    
    print(f"❌ {img_name}: 无法生成5条描述，只有{len(all_descriptions)}条")
    # ✅ 修改：失败时返回纯占位符，不包含图片名
    placeholder = [f"描述生成失败_{i+1}" for i in range(5)]
    for i, desc in enumerate(placeholder, 1):
        print(f"    {i}. {desc}")
    return placeholder, True

async def process_single_image(img_name, img_path, semaphore):
    """处理单张图片的主函数"""
    try:
        print(f"\n🎯 开始处理图片: {img_name}")
        print(f"  完整路径: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"❌ 错误: 图片文件不存在!")
            placeholder = [f"文件缺失_{i+1}" for i in range(5)]
            return placeholder, True
        
        result, is_failed = await process_single_image_with_retry(img_name, img_path, semaphore)
        return result, is_failed
    except Exception as e:
        print(f"❌ {img_name}: 处理过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        # ✅ 修改：异常时返回纯占位符
        placeholder = [f"处理异常_{i+1}" for i in range(5)]
        for i, desc in enumerate(placeholder, 1):
            print(f"    {i}. {desc}")
        return placeholder, True

# ==================== 每N张图片保存一个文件的功能 ====================
async def save_images_file(file_number, all_descriptions, image_names_to_process, start_idx, end_idx):
    """保存每批图片的描述到一个文件 - 需要与test_ids.txt格式对应"""
    output_file_path = os.path.join(OUTPUT_DIR, f'test_caps_5_per_image_part{file_number:03d}.txt')
    
    print(f"\n💾 正在保存第 {file_number} 个文件: {output_file_path}")
    print(f"  图片索引范围: {start_idx} 到 {end_idx-1}")
    print(f"  应包含图片数: {end_idx - start_idx}")
    
    # ✅ 关键修改：按照test_ids.txt的格式保存（每个ID重复5次）
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        lines_to_write = []
        saved_count = 0
        missing_count = 0
        failed_count = 0
        
        print(f"  正在保存图片描述（按test_ids.txt格式）:")
        
        for i in range(start_idx, end_idx):
            img_name = image_names_to_process[i]
            img_num = i + 1
            
            if (i - start_idx) % 10 == 0:
                print(f"    处理第 {img_num} 张图片: {img_name}")
            
            # 获取描述
            if img_name in all_descriptions:
                descriptions = all_descriptions[img_name]
                print(f"      找到 {len(descriptions)} 条描述")
            else:
                # 使用占位符
                print(f"      ⚠ 没有描述数据，使用占位符")
                descriptions = [f"描述未生成_{j+1}" for j in range(5)]
                missing_count += 1
            
            # 确保有5个描述
            if len(descriptions) < 5:
                missing = 5 - len(descriptions)
                print(f"      ⚠ 只有 {len(descriptions)} 条描述，补充 {missing} 条占位符")
                placeholders = [f"补充描述_{j+1}" for j in range(missing)]
                descriptions = descriptions + placeholders
                missing_count += 1
            elif len(descriptions) > 5:
                print(f"      ⚠ 有 {len(descriptions)} 条描述，只取前5条")
                descriptions = descriptions[:5]
            
            # 检查是否是失败描述
            is_failed = any("失败" in desc or "未生成" in desc or "为空" in desc or "异常" in desc or "缺失" in desc for desc in descriptions)
            if is_failed:
                failed_count += 1
                print(f"      ❌ 包含失败描述")
            
            # ✅ 关键：每个ID重复5次，对应5个描述
            for j, desc in enumerate(descriptions, 1):
                clean_desc = desc.strip()
                # 移除可能包含的图片名信息
                clean_desc = re.sub(r'\bCOCO_2014_\d{12}\b', '', clean_desc)
                clean_desc = re.sub(r'\b\d{12}\b', '', clean_desc)
                clean_desc = clean_desc.strip()
                
                if not clean_desc:
                    clean_desc = "描述内容为空"
                
                lines_to_write.append(clean_desc)
            
            saved_count += 1
        
        print(f"\n  统计:")
        print(f"    准备写入 {len(lines_to_write)} 行描述")
        print(f"    对应 {saved_count} 张图片，每张图片5个描述")
        print(f"    缺失数据: {missing_count} 张图片")
        print(f"    生成失败: {failed_count} 张图片")
        
        # 验证格式：每5行对应一个图片ID
        print(f"\n  格式验证（前15行，每5行一组）:")
        sample_lines = lines_to_write[:15] if len(lines_to_write) >= 15 else lines_to_write
        for group_idx in range(0, len(sample_lines), 5):
            group = sample_lines[group_idx:group_idx+5]
            if group:
                print(f"    图片 {group_idx//5 + 1} 的5个描述:")
                for j, line in enumerate(group, 1):
                    print(f"      {j}. {line[:50]}...")
        
        # 写入文件（纯描述文本，每行一条描述）
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in lines_to_write:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        
        # 验证文件
        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path)
            print(f"\n  ✅ 文件保存成功!")
            print(f"    文件路径: {output_file_path}")
            print(f"    文件大小: {file_size} 字节")
            print(f"    描述行数: {len(lines_to_write)} 行")
            print(f"    对应图片: {saved_count} 张")
            
            # 读取并显示文件前几行内容
            print(f"\n    文件前10行内容:")
            with open(output_file_path, 'r', encoding='utf-8') as f:
                first_lines = []
                for j in range(10):
                    line = f.readline()
                    if line:
                        first_lines.append(line.strip())
                
                for j, line in enumerate(first_lines, 1):
                    if (j-1) % 5 == 0:
                        print(f"      图片 {(j-1)//5 + 1}:")
                    print(f"        行{j}: {line[:50]}...")
            
            await logger.write(f"\n✅ 已保存第 {file_number} 个文件: {output_file_path}\n")
            await logger.write(f"   包含图片索引 {start_idx} 到 {end_idx-1} (共{saved_count}张图片)\n")
            await logger.write(f"   文件大小: {file_size} 字节, {len(lines_to_write)} 行描述\n")
            if missing_count > 0:
                await logger.write(f"   ⚠ {missing_count}张图片没有完整描述数据\n")
            if failed_count > 0:
                await logger.write(f"   ❌ {failed_count}张图片生成失败\n")
            
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
            
            # 确保不会超过总数
            if self.processed > self.total:
                self.processed = self.total
            
            progress = self.processed / self.total * 100 if self.total > 0 else 0
            elapsed = time.time() - self.start_time
            
            # 每10张图片或每5秒记录一次进度
            current_time = time.time()
            if self.processed - self.last_progress_log >= 10 or current_time - self.last_log_time > 5:
                if elapsed > 0:
                    speed = self.processed / elapsed
                    # 修复ETA计算，避免负数
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
    
    # 获取当前文件的所有图片
    file_image_names = image_names_to_process[start_idx:end_idx]
    print(f"  前5张图片: {file_image_names[:5]}")
    
    # 检查是否有重复图片（理论上不应该有，因为已经去重了）
    from collections import Counter
    duplicates = Counter(file_image_names)
    duplicate_count = sum(1 for count in duplicates.values() if count > 1)
    if duplicate_count > 0:
        print(f"  ⚠ 警告：发现 {duplicate_count} 张重复图片（不应该发生）")
    
    # 创建并发任务
    tasks = []
    for idx in range(start_idx, end_idx):
        img_name = image_names_to_process[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"❌ {img_name}: 文件不存在，路径: {img_path}")
            async with failed_images_lock:
                # ✅ 修改：失败描述不包含图片名
                all_descriptions[img_name] = [f"文件缺失_{j+1}" for j in range(5)]
                failed_images.append(img_name)
                file_failed_count += 1
            await progress_tracker.update(1)
            continue
        
        # 检查是否已经处理过
        if img_name in all_descriptions and all_descriptions[img_name]:
            print(f"⏩ {img_name}: 已处理，跳过")
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
            print(f"❌ {img_name}: 任务执行失败 {str(result)[:50]}")
            async with checkpoint_lock:
                # ✅ 修改：异常描述不包含图片名
                all_descriptions[img_name] = [f"任务失败_{i+1}" for i in range(5)]
            async with failed_images_lock:
                failed_images.append(img_name)
            file_failed_count += 1
            await progress_tracker.update(1)
        elif isinstance(result, tuple) and len(result) == 2:
            # 正常返回 (descriptions, is_failed)
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
            
            # 显示生成的结果
            print(f"📝 {img_name}: 生成 {len(descriptions)} 条描述")
            for i, desc in enumerate(descriptions[:3], 1):
                print(f"    {i}. {desc[:50]}...")
        else:
            print(f"⚠ {img_name}: 返回结果格式异常")
            async with checkpoint_lock:
                # ✅ 修改：异常描述不包含图片名
                all_descriptions[img_name] = [f"格式错误_{i+1}" for i in range(5)]
            async with failed_images_lock:
                failed_images.append(img_name)
            file_failed_count += 1
            await progress_tracker.update(1)
    
    return file_successful_count, file_failed_count

# ==================== 主程序 - 按文件顺序处理 ====================
async def main_async():
    """主异步函数 - 按文件顺序处理"""
    global logger, checkpoint_lock, failed_images_lock
    
    # 初始化全局锁
    checkpoint_lock = Lock()
    failed_images_lock = Lock()
    
    logger = AsyncLogger(LOG_FILE)
    sys.stdout = sys.stderr = logger.terminal
    
    try:
        print("=" * 80)
        print("🚀 测试集图片描述生成程序启动")
        print(f"📊 注意: test_ids.txt格式检测")
        print(f"📋 预期格式: 每个ID重复5次，对应5个描述")
        print(f"📁 图片目录: {IMAGE_DIR}")
        print(f"📄 ID文件: {TEST_IDS_PATH}")
        print(f"💾 输出目录: {OUTPUT_DIR}")
        print(f"⚡ 并发数: {MAX_CONCURRENT_REQUESTS}")
        print(f"💾 保存间隔: 每{SAVE_INTERVAL}张图片保存一个文件")
        
        if TEST_MODE:
            print(f"🔬 测试模式: 开启，只处理前{MAX_TEST_IMAGES}张图片")
        print("=" * 80)
        
        # 首先验证test_ids.txt文件格式
        print("\n📋 验证test_ids.txt文件格式...")
        try:
            with open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"  文件总行数: {len(lines)}")
                print(f"  前5行内容:")
                for i, line in enumerate(lines[:5], 1):
                    print(f"    行{i}: '{line.strip()}'")
                
                # 验证是否都是数字
                valid_ids = [int(line.strip()) for line in lines if line.strip().isdigit()]
                print(f"  有效ID数量: {len(valid_ids)}")
                
                if len(valid_ids) % 5 != 0:
                    print(f"  ⚠ 警告: 有效ID数量 {len(valid_ids)} 不是5的倍数!")
                else:
                    print(f"  ✅ 格式正确: 有效ID数量是5的倍数")
                    
                    # 检查前几组
                    print(f"  前3组ID验证:")
                    for i in range(0, min(15, len(valid_ids)), 5):
                        group = valid_ids[i:i+5]
                        if len(group) == 5 and len(set(group)) == 1:
                            print(f"    组 {i//5 + 1}: ID {group[0]} 重复5次 - ✅ 正确")
                        else:
                            print(f"    组 {i//5 + 1}: {group} - ❌ 错误")
        except Exception as e:
            print(f"  ❌ 读取test_ids.txt失败: {e}")
            return
        
        # 加载数据（保证顺序，已去重）
        print("\n📋 正在加载测试集图片数据...")
        image_names_to_process = await load_data()
        total_images = len(image_names_to_process)
        
        if total_images == 0:
            print("❌ 错误: 没有找到要处理的图片!")
            print(f"请检查图片目录: {IMAGE_DIR}")
            print(f"图片应该类似: COCO_2014_000000130524.jpg")
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
        
        # ✅ 关键修改：按文件顺序处理，处理完一个文件就保存
        total_successful = 0
        total_failed = 0
        
        for file_number in range(current_file_number, total_files + 1):
            # 计算当前文件的图片范围
            start_idx = (file_number - 1) * SAVE_INTERVAL
            end_idx = min(start_idx + SAVE_INTERVAL, total_images)
            
            print(f"\n{'='*80}")
            print(f"📁 处理第 {file_number}/{total_files} 个文件")
            print(f"  图片索引: {start_idx} 到 {end_idx-1}")
            print(f"  共 {end_idx - start_idx} 张图片")
            print(f"  前5张图片: {image_names_to_process[start_idx:start_idx+5]}")
            print(f"{'='*80}")
            
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
            
            # ✅ 关键：立即保存当前文件（只保存纯描述文本）
            print(f"\n💾 立即保存第 {file_number} 个文件（纯描述文本）...")
            saved_count, failed_in_file = await save_images_file(
                file_number, all_descriptions, image_names_to_process, start_idx, end_idx
            )
            
            if saved_count > 0:
                print(f"  ✅ 第 {file_number} 个文件保存成功!")
                print(f"     包含 {saved_count} 张图片的描述")
                
                # 更新检查点
                checkpoint_data = {
                    "processed_count": end_idx,  # 已处理到哪个索引
                    "file_count": file_number,    # 已保存的文件数
                    "failed_images": failed_images,
                    "all_descriptions": all_descriptions,
                    "current_file_number": file_number + 1,  # 下一个要处理的文件
                    "current_file_start_idx": end_idx,      # 下一个文件的开始索引
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
            
            # 每个文件处理后等待一下，避免API限制
            if file_number < total_files:
                print(f"\n⏳ 准备处理下一个文件，等待3秒...")
                await asyncio.sleep(3)
        
        # 所有文件处理完成
        print(f"\n{'='*80}")
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
        
        print(f"\n{'='*80}")
        print("✅ 处理完成!")
        print(f"⏱️  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"🚀 平均速度: {total_images/total_time:.2f} 张/秒")
        print(f"📊 成功: {total_successful} 张 | 失败: {total_failed} 张")
        print(f"📁 保存文件数: {total_files} 个")
        print(f"📝 总描述行数: {total_images * 5} 行")
        print(f"{'='*80}")
        
        # 列出生成的文件
        print("\n📋 生成的文件列表（纯描述文本）:")
        import glob
        files = glob.glob(os.path.join(OUTPUT_DIR, "test_caps_5_per_image_part*.txt"))
        for file in sorted(files):
            size = os.path.getsize(file)
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"  {os.path.basename(file)} - {size} 字节, {len(lines)} 行, {len(lines)//5} 张图片")
            
            # 显示文件前几行内容
            if file == files[0]:  # 只显示第一个文件的内容
                print(f"    前5行内容:")
                for i, line in enumerate(lines[:5], 1):
                    print(f"      行{i}: {line.strip()[:60]}...")
        
        # 删除检查点文件（处理完成）
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("\n🧹 清理检查点文件")
        
        await logger.write(f"\n{'='*80}\n")
        await logger.write("✅ 处理完成!\n")
        await logger.write(f"⏱️  总耗时: {total_time:.2f} 秒\n")
        await logger.write(f"🚀 平均速度: {total_images/total_time:.2f} 张/秒\n")
        await logger.write(f"📊 成功: {total_successful} 张 | 失败: {total_failed} 张\n")
        await logger.write(f"📁 保存文件数: {total_files} 个\n")
        await logger.write(f"📝 总描述行数: {total_images * 5} 行\n")
        await logger.write(f"{'='*80}\n")
        
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
        # 等待所有I/O操作完成
        await asyncio.sleep(1)
        await logger.close()

# ==================== 入口 ====================
if __name__ == "__main__":
    print("测试集处理程序启动...")
    
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