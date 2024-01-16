import os
import json
import shutil

import numpy as np
import cv2

# 定义COCO到DOTA的类映射
class_mapping = {
    1: 'ship',
    # 添加其他类别映射
}


def coco_to_dota(coco_json_file, dota_output_dir):
    # 加载COCO格式的标注数据
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    # 创建DOTA格式的标注数据结构
    for image_info in coco_data['images']:
        img_id = image_info['id']
        img_filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']

        # 获取与图像关联的标注信息
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        txt_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.txt'
        txt_file_path = os.path.join(dota_output_dir, 'labels', txt_filename)

        with open(txt_file_path, 'w') as txt_file:
            for ann in img_annotations:
                category_id = ann['category_id']
                class_name = class_mapping.get(category_id, 'unknown')
                bbox = ann['bbox']

                # 将COCO格式的坐标(x, y, width, height)转换为DOTA格式的坐标(x1, y1, x2, y2, x3, y3, x4, y4)
                x, y, w, h = bbox
                x1, y1 = x, y
                x2, y2 = x + w, y
                x3, y3 = x + w, y + h
                x4, y4 = x, y + h

                line = f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {class_name} {category_id-1}\n"
                txt_file.write(line)

        shutil.copy(os.path.join('./JPEGImages',img_filename),os.path.join(dota_output_dir,'images'))

if __name__ == "__main__":
    coco_json_file = "./annotations/train2017.json"  # 替换为你的COCO格式标注文件的路径
    dota_output_dir = "./dota/train"  # 替换为你想要保存DOTA格式数据的目录

    if os.path.exists(os.path.join(dota_output_dir,'images')) is not True:
        os.makedirs(os.path.join(dota_output_dir,'images'))
    if os.path.exists(os.path.join(dota_output_dir,'labels')) is not True:
        os.makedirs(os.path.join(dota_output_dir,'labels'))
    
    coco_to_dota(coco_json_file, dota_output_dir)