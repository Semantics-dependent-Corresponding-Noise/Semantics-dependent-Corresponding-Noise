import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# ==================== Configuration ====================
IMAGE_DIR = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/images'
CAPTIONS_FILE = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/annotations/scan_split/test_caps_core.txt'
TRAIN_IDS_PATH = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/annotations/scan_split/test_ids.txt'
NUM_IMAGES_TO_SHOW = 5
OUTPUT_DIR = '/home/zbm/xjd/NPC-master/MSCOCO_noise_cinstruct/core_missing/display_samples'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 函数定义 ====================
def load_train_ids_and_captions(train_ids_path, captions_file, mode='train'):
    """
    加载训练ID和对应的描述
    
    Args:
        mode: 'train' - 每个ID对应5个描述
              'test' - 每个ID重复5次，每个描述对应一个ID
    """
    with open(train_ids_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if mode == 'train':
        # train模式：每个ID出现一次，每个ID对应5个描述
        train_ids = [int(line) for line in lines if line.isdigit()]
        print(f"[Train Mode] Loaded {len(train_ids)} unique train IDs")
    else:  # test模式
        # test模式：每个ID重复5次，需要提取不重复的ID
        train_ids_raw = [int(line) for line in lines if line.isdigit()]
        # 每5个一组提取第一个作为ID
        train_ids = train_ids_raw[::5]
        print(f"[Test Mode] Loaded {len(train_ids)} unique train IDs from {len(train_ids_raw)} lines")
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        caption_lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(caption_lines)} caption lines")
    return train_ids, caption_lines

def get_image_path_by_train_id(train_id, image_dir=IMAGE_DIR):
    """根据train_id获取图像路径"""
    filename = f"COCO_2014_{train_id:012d}.jpg"
    return os.path.join(image_dir, filename), filename

def get_captions_by_train_id_index(caption_lines, index, mode='train'):
    """根据索引获取对应的5个描述"""
    if mode == 'train':
        # train模式：每个ID对应连续的5个描述
        start_idx = index * 5
        return caption_lines[start_idx:start_idx + 5]
    else:  # test模式
        # test模式：每个ID的5个描述是连续的，不需要乘以5
        start_idx = index * 5
        return caption_lines[start_idx:start_idx + 5]

def save_image_with_captions(image_path, captions, train_id, output_path, mode='train'):
    """保存图像和描述到文件"""
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return False
    
    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(11, 6), 
                                          gridspec_kw={'width_ratios': [1, 1.1]})
    
    # Show image
    img = Image.open(image_path)
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title(f"Train ID: {train_id} ({mode} mode)", fontsize=11, pad=10)
    
    # Show captions
    caption_text = f"Train ID: {train_id}\n"
    caption_text += f"Mode: {mode}\n\n"
    caption_text += "Generated Captions:\n"
    caption_text += "="*40 + "\n\n"
    
    for i, caption in enumerate(captions, 1):
        caption_text += f"{i}. {caption}\n\n"
    
    ax_text.text(0.05, 0.95, caption_text, 
                 transform=ax_text.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 family='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax_text.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形，避免内存泄漏
    return True

def find_train_id_by_filename(train_ids, filename):
    """根据文件名查找对应的train_id"""
    # 尝试从文件名中提取train_id
    try:
        # 格式: COCO_2014_{train_id:012d}.jpg
        if filename.startswith('COCO_2014_'):
            id_str = filename.replace('COCO_2014_', '').replace('.jpg', '')
            train_id = int(id_str)
            return train_id
        # 如果文件名就是数字
        elif filename.replace('.jpg', '').isdigit():
            train_id = int(filename.replace('.jpg', ''))
            return train_id
    except:
        pass
    
    # 在train_ids中查找
    for train_id in train_ids:
        expected_filename = f"COCO_2014_{train_id:012d}.jpg"
        if expected_filename == filename:
            return train_id
    
    return None

def process_specific_image(train_ids, caption_lines, filename, mode='train'):
    """处理指定的图像文件"""
    print(f"\n{'='*60}")
    print(f"Processing specific image: {filename}")
    print(f"Mode: {mode}")
    print('='*60)
    
    # 查找对应的train_id
    train_id = find_train_id_by_filename(train_ids, filename)
    
    if train_id is None:
        print(f"ERROR: Could not find train_id for filename: {filename}")
        print("Please check if the filename format is correct.")
        print("Expected format: COCO_2014_000000130524.jpg or 130524.jpg")
        return
    
    print(f"Found train_id: {train_id}")
    
    # 查找train_id在列表中的索引
    try:
        index = train_ids.index(train_id)
        print(f"Found at index: {index}")
    except ValueError:
        print(f"ERROR: Train_id {train_id} not found in the loaded train_ids list")
        return
    
    # 获取图像路径和描述
    image_path, image_name = get_image_path_by_train_id(train_id, IMAGE_DIR)
    captions = get_captions_by_train_id_index(caption_lines, index, mode)
    
    # 保存到文件
    output_path = os.path.join(OUTPUT_DIR, f"specific_{train_id}_{mode}.png")
    success = save_image_with_captions(image_path, captions, train_id, output_path, mode)
    
    if success:
        print(f"  ✓ Saved to: {output_path}")
        print("\nCaptions:")
        for i, caption in enumerate(captions, 1):
            print(f"  {i}. {caption}")
    else:
        print(f"  ✗ Failed to save")

def main():
    print("="*60)
    print("Image and Caption Saver")
    print("="*60)
    
    # 选择模式
    print("\nSelect mode:")
    print("1. Train mode (each ID appears once)")
    print("2. Test mode (each ID repeats 5 times)")
    mode_choice = input("Enter 1 or 2 (default: 1): ").strip()
    
    mode = 'train' if mode_choice != '2' else 'test'
    
    print(f"\nSelected mode: {mode}")
    
    # 选择处理方式
    print("\nSelect processing method:")
    print("1. Randomly select images")
    print("2. Process specific image by filename")
    process_choice = input("Enter 1 or 2 (default: 1): ").strip()
    
    # 加载train IDs和描述
    train_ids, caption_lines = load_train_ids_and_captions(TRAIN_IDS_PATH, CAPTIONS_FILE, mode)
    num_images = len(train_ids)
    
    if process_choice == '2':
        # 处理指定的图像
        filename = input("\nEnter image filename (e.g., COCO_2014_000000130524.jpg or 130524.jpg): ").strip()
        process_specific_image(train_ids, caption_lines, filename, mode)
    else:
        # 随机选择图像
        selected_indices = random.sample(range(num_images), min(NUM_IMAGES_TO_SHOW, num_images))
        selected_train_ids = [train_ids[i] for i in selected_indices]
        
        print(f"\nRandomly selected {len(selected_train_ids)} images in {mode} mode:")
        for i, train_id in enumerate(selected_train_ids, 1):
            print(f"  {i}. Train ID: {train_id}")
        
        # 保存每个图像
        for display_idx, (train_id_idx, train_id) in enumerate(zip(selected_indices, selected_train_ids), 1):
            print(f"\nProcessing {display_idx}/{len(selected_train_ids)}: Train ID {train_id}")
            
            # 获取图像路径和描述
            image_path, image_name = get_image_path_by_train_id(train_id, IMAGE_DIR)
            captions = get_captions_by_train_id_index(caption_lines, train_id_idx, mode)
            
            # 保存到文件
            output_path = os.path.join(OUTPUT_DIR, f"sample_{train_id}_{mode}.png")
            success = save_image_with_captions(image_path, captions, train_id, output_path, mode)
            
            if success:
                print(f"  ✓ Saved to: {output_path}")
            else:
                print(f"  ✗ Failed to save")
        
        print("\n" + "="*60)
        print(f"✅ All samples saved to: {OUTPUT_DIR}")
        print("="*60)

if __name__ == "__main__":
    main()