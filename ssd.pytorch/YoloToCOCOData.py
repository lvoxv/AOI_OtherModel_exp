import os
import json
import glob
from PIL import Image
import shutil
from tqdm import tqdm
import argparse

def yolo_to_coco(yolo_images_path, yolo_labels_path, class_names_file, output_json_path):
    """
    将YOLO格式的数据集转换为COCO格式
    
    参数:
        yolo_images_path: YOLO图像目录路径
        yolo_labels_path: YOLO标签目录路径
        class_names_file: 包含类别名称的文件路径
        output_json_path: 输出COCO JSON文件路径
    """
    # 读取类别名称
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 创建COCO数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别信息
    for class_id, class_name in enumerate(class_names):
        category = {
            "id": class_id,
            "name": class_name,
            "supercategory": "none"
        }
        coco_data["categories"].append(category)
    
    # 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(yolo_images_path, "*.*")))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    annotation_id = 0
    
    # 遍历所有图像
    for image_id, image_file in enumerate(tqdm(image_files, desc="处理图像")):
        # 获取图像文件名（不带扩展名）
        image_basename = os.path.basename(image_file)
        image_name = os.path.splitext(image_basename)[0]
        
        # 读取图像尺寸
        with Image.open(image_file) as img:
            width, height = img.size
        
        # 添加图像信息
        image_info = {
            "id": image_id,
            "file_name": image_basename,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_info)
        
        # 构建对应的标签文件路径
        label_file = os.path.join(yolo_labels_path, f"{image_name}.txt")
        
        # 如果标签文件存在，则处理标签
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # YOLO格式: <class_id> <x_center> <y_center> <width> <height> (归一化)
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        
                        # 转换为COCO格式 (像素坐标)
                        x = int((x_center - bbox_width / 2) * width)
                        y = int((y_center - bbox_height / 2) * height)
                        w = int(bbox_width * width)
                        h = int(bbox_height * height)
                        
                        # 创建标注
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        }
                        
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
    
    # 保存COCO JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"转换完成! 共处理 {len(image_files)} 张图像，生成 {annotation_id} 个标注。")
    print(f"COCO格式数据已保存到: {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="将YOLO数据集转换为COCO格式")
    parser.add_argument("--yolo_path", required=True, help="YOLO数据集根目录")
    parser.add_argument("--split", default="train", help="数据集划分(train, val, test)")
    parser.add_argument("--class_file", required=True, help="包含类别名称的文件路径")
    parser.add_argument("--output", default="", help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    # 构建路径
    yolo_images_path = os.path.join(args.yolo_path, "images", args.split)
    yolo_labels_path = os.path.join(args.yolo_path, "labels", args.split)
    
    # 如果未指定输出路径，则默认保存在数据集根目录
    if not args.output:
        output_json_path = os.path.join(args.yolo_path, f"annotations_{args.split}.json")
    else:
        output_json_path = args.output
    
    # 检查路径是否存在
    if not os.path.exists(yolo_images_path):
        print(f"错误: 图像目录不存在: {yolo_images_path}")
        return
    
    if not os.path.exists(yolo_labels_path):
        print(f"错误: 标签目录不存在: {yolo_labels_path}")
        return
    
    if not os.path.exists(args.class_file):
        print(f"错误: 类别文件不存在: {args.class_file}")
        return
    
    yolo_to_coco(yolo_images_path, yolo_labels_path, args.class_file, output_json_path)

if __name__ == "__main__":
    main()

# YOLO数据集转换为COCO格式
# 该脚本将YOLO格式的数据集转换为COCO格式，支持训练、验证和测试集的转换
# 使用方法:
# python YoloToCOCOData.py --yolo_path /home/xj/xu/data/250423_the_thrid_optimization_datasets --split train --class_file /home/xj/xu/data/cls.txt --output /home/xj/xu/data/COCO_250423_the_thrid_optimization_datasets/annotations/train.json
# 注意: YOLO格式的标签文件应与图像文件同名，且位于同一目录下
# 例如: images/train/image1.jpg 和 labels/train/image1.txt
# 其中 image1.txt 的内容格式为:
# <class_id> <x_center> <y_center> <width> <height>
# 其中 x_center, y_center, width, height 为归一化坐标
# 该脚本会生成一个COCO格式的JSON文件，包含图像信息、标注信息和类别信息
# 该脚本使用了PIL库来读取图像尺寸，使用了tqdm库来显示进度条
# 需要安装的库:
# pip install pillow tqdm
# 该脚本支持命令行参数，可以指定YOLO数据集的根目录、划分类型、类别文件路径和输出JSON文件路径
# 该脚本会检查输入路径是否存在，并在转换完成后输出转换结果
# 该脚本可以用于YOLO格式数据集的转换，方便与COCO格式的数据集进行兼容