#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from pathlib import Path
import shutil

def create_voc_xml(image_path, image_shape, boxes, class_names, output_path):
    """
    创建VOC格式的XML文件
    
    参数:
        image_path: 图片路径
        image_shape: 图片尺寸 (高, 宽, 通道数)
        boxes: 边界框列表，每个边界框是 [class_id, x_min, y_min, x_max, y_max]
        class_names: 类别名称列表
        output_path: 输出XML文件路径
    """
    height, width, depth = image_shape
    
    # 创建根元素
    annotation = ET.Element('annotation')
    
    # 添加基本信息
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.basename(os.path.dirname(image_path))
    
    filename_elem = ET.SubElement(annotation, 'filename')
    filename_elem.text = os.path.basename(image_path)
    
    path = ET.SubElement(annotation, 'path')
    path.text = image_path
    
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    
    # 添加图像尺寸信息
    size = ET.SubElement(annotation, 'size')
    width_elem = ET.SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, 'height')
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, 'depth')
    depth_elem.text = str(depth)
    
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # 添加每个目标的边界框信息
    for box in boxes:
        class_id, x_min, y_min, x_max, y_max = box
        
        if class_id >= len(class_names):
            print(f"警告：类别ID {class_id} 超出了类别名称列表范围，使用 'unknown' 代替")
            class_name = 'unknown'
        else:
            class_name = class_names[class_id]
        
        object_elem = ET.SubElement(annotation, 'object')
        
        name = ET.SubElement(object_elem, 'name')
        name.text = class_name
        
        pose = ET.SubElement(object_elem, 'pose')
        pose.text = 'Unspecified'
        
        truncated = ET.SubElement(object_elem, 'truncated')
        truncated.text = '0'
        
        difficult = ET.SubElement(object_elem, 'difficult')
        difficult.text = '0'
        
        bndbox = ET.SubElement(object_elem, 'bndbox')
        
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(x_min) + 1)  # VOC格式是1-based索引
        
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(y_min) + 1)
        
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(x_max) + 1)
        
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(y_max) + 1)
    
    # 创建格式良好的XML字符串
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent='    ')
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)


def find_label_file(image_file, label_dir, image_dir):
    """
    智能查找对应的标签文件
    
    参数:
        image_file: 图片文件路径 (Path对象)
        label_dir: 标签目录
        image_dir: 图片目录
    
    返回:
        标签文件路径或None
    """
    image_stem = image_file.stem
    
    # 直接在标签目录下查找
    direct_path = os.path.join(label_dir, f"{image_stem}.txt")
    if os.path.exists(direct_path):
        return direct_path
    
    # 尝试获取图片相对于图片目录的路径
    try:
        rel_path = image_file.relative_to(Path(image_dir))
        subdir = rel_path.parts[0] if len(rel_path.parts) > 0 else ""
        
        # 如果图片在子文件夹中，尝试在标签目录的相应子文件夹中查找
        if subdir:
            subdir_path = os.path.join(label_dir, subdir, f"{image_stem}.txt")
            if os.path.exists(subdir_path):
                return subdir_path
            
            # 如果在子文件夹中找不到，尝试移除子文件夹后缀中的数字
            # 例如，train_1234.jpg 的标签可能是 train/1234.txt
            prefix = subdir.rstrip('0123456789_')
            if prefix and prefix != subdir:
                rest = image_stem
                subdir_path = os.path.join(label_dir, prefix, f"{rest}.txt")
                if os.path.exists(subdir_path):
                    return subdir_path
    except Exception as e:
        print(f"获取图片 {image_file} 的相对路径时出错: {e}")
    
    # 在标签目录的子文件夹中递归查找
    for root, _, files in os.walk(label_dir):
        label_path = os.path.join(root, f"{image_stem}.txt")
        if os.path.exists(label_path):
            return label_path
    
    return None


def get_dataset_split(image_file, image_dir):
    """
    根据图片文件路径确定其所属的数据集划分（train/val/test）
    
    参数:
        image_file: 图片文件路径 (Path对象)
        image_dir: 图片目录
    
    返回:
        数据集划分名称: 'train', 'val', 'test' 或 'unknown'
    """
    try:
        rel_path = image_file.relative_to(Path(image_dir))
        parts = rel_path.parts
        
        if len(parts) > 0:
            if parts[0] in ['train', 'val', 'test']:
                return parts[0]
    except Exception as e:
        print(f"获取图片 {image_file} 的数据集划分时出错: {e}")
    
    return 'unknown'


def convert_yolo_to_voc_with_subfolders(image_dir, label_dir, output_dir, class_names_file):
    """
    将YOLO格式的标注转换为VOC格式，支持子文件夹结构
    
    参数:
        image_dir: 图片目录 (包含train/val/test子文件夹)
        label_dir: YOLO标注文件目录 (可能也包含子文件夹)
        output_dir: VOC XML输出目录
        class_names_file: 类别名称文件，每行一个类别
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载类别名称
    with open(class_names_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print(f"加载了 {len(class_names)} 个类别: {', '.join(class_names)}")
    
    # 获取所有图片文件，包括子文件夹中的
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(image_dir).glob(f'**/*{ext}')))
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 检查标签目录是否有子文件夹
    label_subfolders = [f for f in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, f))]
    if label_subfolders:
        print(f"标签目录含有子文件夹: {', '.join(label_subfolders)}")
    
    # 跟踪数据集划分
    split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
    dataset_splits = {}  # 用于记录每个图像属于哪个集合
    
    # 处理每个图片及其标注
    converted_count = 0
    missing_label_count = 0
    
    for image_file in tqdm(image_files, desc="转换标注"):
        image_path = str(image_file)
        image_stem = image_file.stem
        
        # 确定数据集划分
        split = get_dataset_split(image_file, image_dir)
        split_counts[split] += 1
        dataset_splits[image_stem] = split
        
        # 智能查找对应的标注文件
        label_path = find_label_file(image_file, label_dir, image_dir)
        
        # 检查对应的标注文件是否存在
        if not label_path:
            print(f"警告：图片 {image_path} 没有对应的标注文件")
            missing_label_count += 1
            continue
        
        # 读取图片获取尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图片 {image_path}")
            continue
        
        height, width, channels = img.shape
        
        # 读取YOLO格式标注
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # YOLO格式: <class_id> <x_center> <y_center> <width> <height>
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        box_width = float(parts[3]) * width
                        box_height = float(parts[4]) * height
                        
                        # 转换为 VOC 格式: (xmin, ymin, xmax, ymax)
                        x_min = max(0, int(x_center - box_width / 2))
                        y_min = max(0, int(y_center - box_height / 2))
                        x_max = min(width - 1, int(x_center + box_width / 2))
                        y_max = min(height - 1, int(y_center + box_height / 2))
                        
                        boxes.append([class_id, x_min, y_min, x_max, y_max])
        
        # 创建VOC XML文件
        xml_output_path = os.path.join(output_dir, f"{image_stem}.xml")
        create_voc_xml(image_path, (height, width, channels), boxes, class_names, xml_output_path)
        converted_count += 1
    
    print(f"转换完成! 已将 {converted_count} 个VOC格式标注保存到: {output_dir}")
    print(f"缺失标签文件: {missing_label_count} 个")
    print(f"数据集划分统计: {split_counts}")
    
    return dataset_splits


def create_voc_structure_with_subfolders(base_dir, image_dir, label_dir, class_names_file, dataset_name='VOC2007'):
    """
    创建完整的VOC数据集结构，支持子文件夹分割
    
    参数:
        base_dir: 基础目录
        image_dir: 原始图片目录 (包含train/val/test子文件夹)
        label_dir: 原始YOLO标注目录 (可能也包含子文件夹)
        class_names_file: 类别名称文件
        dataset_name: 数据集名称，默认为VOC2007
    """
    # 创建VOC目录结构
    voc_dir = os.path.join(base_dir, dataset_name)
    annotations_dir = os.path.join(voc_dir, 'Annotations')
    imagesets_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
    jpegimages_dir = os.path.join(voc_dir, 'JPEGImages')
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(imagesets_dir, exist_ok=True)
    os.makedirs(jpegimages_dir, exist_ok=True)
    
    # 转换标注并获取数据集划分信息
    dataset_splits = convert_yolo_to_voc_with_subfolders(image_dir, label_dir, annotations_dir, class_names_file)
    
    # 获取所有图片文件，包括子文件夹中的
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(image_dir).glob(f'**/*{ext}')))
    
    print(f"正在复制 {len(image_files)} 个图片文件到 JPEGImages 目录...")
    
    for image_file in tqdm(image_files, desc="复制图片"):
        # 如果不是jpg格式，需要转换
        if image_file.suffix.lower() not in ['.jpg', '.jpeg']:
            img = cv2.imread(str(image_file))
            if img is not None:
                output_path = os.path.join(jpegimages_dir, f"{image_file.stem}.jpg")
                cv2.imwrite(output_path, img)
        else:
            output_path = os.path.join(jpegimages_dir, f"{image_file.stem}.jpg")
            # 使用二进制模式复制
            with open(image_file, 'rb') as src_file, open(output_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
    
    # 创建imagesets文件
    print("创建ImageSets文件...")
    file_stems = [file.stem for file in image_files]
    
    # 根据子文件夹结构创建train.txt、val.txt和test.txt
    train_set = [stem for stem in file_stems if dataset_splits.get(stem, 'unknown') == 'train']
    val_set = [stem for stem in file_stems if dataset_splits.get(stem, 'unknown') == 'val']
    test_set = [stem for stem in file_stems if dataset_splits.get(stem, 'unknown') == 'test']
    
    # 如果没有验证集，则从训练集中分配20%作为验证集
    if not val_set and train_set:
        import random
        random.shuffle(train_set)
        val_size = int(len(train_set) * 0.2)
        val_set = train_set[:val_size]
        train_set = train_set[val_size:]
    
    # 如果没有明确的测试集，使用验证集作为测试集
    if not test_set and val_set:
        test_set = val_set
    
    # 如果既没有验证集也没有测试集，则随机分配
    if not val_set and not test_set and file_stems:
        import random
        all_files = file_stems.copy()
        random.shuffle(all_files)
        train_size = int(len(all_files) * 0.7)
        val_size = int(len(all_files) * 0.15)
        train_set = all_files[:train_size]
        val_set = all_files[train_size:train_size+val_size]
        test_set = all_files[train_size+val_size:]
    
    trainval_set = train_set + val_set
    
    # 写入数据集划分文件
    with open(os.path.join(imagesets_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_set))
    
    with open(os.path.join(imagesets_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_set))
    
    with open(os.path.join(imagesets_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_set))
    
    with open(os.path.join(imagesets_dir, 'trainval.txt'), 'w') as f:
        f.write('\n'.join(trainval_set))
    
    print(f"数据集划分统计: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")
    
    # 创建每个类别的文件
    with open(class_names_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print(f"创建 {len(class_names)} 个类别的ImageSets文件...")
    
    # 复制类别文件到VOC目录
    with open(os.path.join(voc_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    
    print(f"VOC格式数据集已创建完成: {voc_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将YOLO格式标注转换为VOC格式 (支持子文件夹)')
    parser.add_argument('--image_dir', required=True, help='图片目录 (可包含train/val/test子文件夹)')
    parser.add_argument('--label_dir', required=True, help='YOLO标注文件目录 (可包含子文件夹)')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--class_file', required=True, help='类别名称文件，每行一个类别')
    parser.add_argument('--create_voc_structure', action='store_true', help='创建完整的VOC数据集结构')
    parser.add_argument('--dataset_name', default='VOC2007', help='数据集名称，默认为VOC2007')
    
    args = parser.parse_args()
    
    if args.create_voc_structure:
        create_voc_structure_with_subfolders(args.output_dir, args.image_dir, args.label_dir, args.class_file, args.dataset_name)
    else:
        convert_yolo_to_voc_with_subfolders(args.image_dir, args.label_dir, args.output_dir, args.class_file)

## 运行示例:
# python YoloToVOC.py --image_dir /path/to/images --label_dir /path/to/labels --output_dir /path/to/output --class_file /path/to/classes.txt
# python YoloToVOC.py --image_dir /path/to/images --label_dir /path/to/labels --output_dir /path/to/output --class_file /path/to/classes.txt --create_voc_structure
# python YoloToVOC.py --image_dir /home/xj/xu/data/250423_the_thrid_optimization_datasets/images --label_dir /home/xj/xu/data/250423_the_thrid_optimization_datasets/labels --output_dir /home/xj/xu/data/VOC_250423 --class_file /home/xj/xu/data/cls.txt --create_voc_structure