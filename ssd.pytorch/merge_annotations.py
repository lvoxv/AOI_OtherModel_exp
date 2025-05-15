import json
import os
import argparse
from tqdm import tqdm

def merge_coco_annotations(train_json, val_json, output_json):
    """
    合并train和val的COCO格式注释文件
    
    参数:
        train_json: 训练集注释JSON文件路径
        val_json: 验证集注释JSON文件路径
        output_json: 输出合并后的JSON文件路径
    """
    print(f"加载训练集注释: {train_json}")
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    
    print(f"加载验证集注释: {val_json}")
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    
    # 合并categories (通常两个文件中的categories应该相同)
    # 使用train_data中的categories
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": train_data["categories"]
    }
    
    # 添加训练集图像和注释
    print("合并训练集数据...")
    image_id_mapping = {}  # 用于存储原始image_id到新image_id的映射
    next_image_id = 0
    next_ann_id = 0
    
    # 处理训练集图像
    for img in tqdm(train_data["images"], desc="处理训练集图像"):
        original_id = img["id"]
        img["id"] = next_image_id
        image_id_mapping[original_id] = next_image_id
        merged_data["images"].append(img)
        next_image_id += 1
    
    # 处理训练集注释
    for ann in tqdm(train_data["annotations"], desc="处理训练集注释"):
        ann["id"] = next_ann_id
        ann["image_id"] = image_id_mapping[ann["image_id"]]
        merged_data["annotations"].append(ann)
        next_ann_id += 1
    
    # 添加验证集图像和注释
    print("合并验证集数据...")
    image_id_mapping = {}  # 重置映射
    
    # 处理验证集图像
    for img in tqdm(val_data["images"], desc="处理验证集图像"):
        original_id = img["id"]
        img["id"] = next_image_id
        image_id_mapping[original_id] = next_image_id
        merged_data["images"].append(img)
        next_image_id += 1
    
    # 处理验证集注释
    for ann in tqdm(val_data["annotations"], desc="处理验证集注释"):
        ann["id"] = next_ann_id
        ann["image_id"] = image_id_mapping[ann["image_id"]]
        merged_data["annotations"].append(ann)
        next_ann_id += 1
    
    # 保存合并后的数据
    print(f"正在保存合并后的注释到: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(merged_data, f)
    
    print(f"合并完成! 共 {len(merged_data['images'])} 张图像和 {len(merged_data['annotations'])} 个标注。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并COCO格式的训练集和验证集注释")
    parser.add_argument("--train", required=True, help="训练集注释JSON文件路径")
    parser.add_argument("--val", required=True, help="验证集注释JSON文件路径")
    parser.add_argument("--output", required=True, help="输出合并后的JSON文件路径")
    
    args = parser.parse_args()
    merge_coco_annotations(args.train, args.val, args.output)