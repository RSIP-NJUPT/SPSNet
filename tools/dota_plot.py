import os
import json
import cv2
import numpy as np


def draw_dota_annotations(image_dir, dota_annotation, output_dir):
    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        dota_txt_annotations = os.path.join(dota_annotation, os.path.splitext(image_file)[0] + '.txt')
        output_path = os.path.join(output_dir, image_file)

        # 读取图像
        image = cv2.imread(image_path)

        if os.path.exists(dota_txt_annotations) is not True:
            continue
        with open(dota_txt_annotations, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(' ')
            if parts.__len__() == 10:
                x1, y1, x2, y2, x3, y3, x4, y4, class_name, _ = map(str, parts)

                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, (x1, y1, x2, y2, x3, y3, x4, y4))
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, (x1, y1, x2, y2, x3, y3, x4, y4))

                # 在图像上绘制多边形和类别
                # pts = np.array([[100, 100], [200, 200], [300, 100], [200, 50]], dtype=np.int32)
                points = [np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int32)]
                # points = [map(int, point) for point in points]
                cv2.polylines(image, points, isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存标注后的图像
        cv2.imwrite(output_path, image)


if __name__ == "__main__":
    image_path = "./JPEGImages"  # 替换为你的图像文件路径
    dota_annotations = "./dota"  # 替换为你的DOTA格式标注文件的路径
    output_path = "./plot"  # 替换为要保存标注后的图像的路径

    draw_dota_annotations(image_path, dota_annotations, output_path)