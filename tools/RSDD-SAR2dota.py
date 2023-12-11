import os
import xml.etree.ElementTree as ET
import argparse
import math


def convert_to_dota(xml_path):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    dota_output = []
    # 获取目标信息
    for obj in root.findall('object'):
        obj_type = obj.find('type').text
        obj_name = obj.find('name').text
        difficult = int(obj.find('difficult').text)

        # 获取robndbox信息
        robndbox = obj.find('robndbox')
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        h = float(robndbox.find('h').text)
        w = float(robndbox.find('w').text)
        angle = float(robndbox.find('angle').text)

        # RSDD数据集标注框初始是竖着的
        angle += 1 / 2 * math.pi

        # 将robndbox信息格式化为8个点表示的旋转目标框
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        Ax = cx - 0.5 * w * cos_a - 0.5 * h * sin_a
        Ay = cy - 0.5 * w * sin_a + 0.5 * h * cos_a

        Bx = cx + 0.5 * w * cos_a - 0.5 * h * sin_a
        By = cy + 0.5 * w * sin_a + 0.5 * h * cos_a

        Cx = cx + 0.5 * w * cos_a + 0.5 * h * sin_a
        Cy = cy + 0.5 * w * sin_a - 0.5 * h * cos_a

        Dx = cx - 0.5 * w * cos_a + 0.5 * h * sin_a
        Dy = cy - 0.5 * w * sin_a - 0.5 * h * cos_a

        # 将8个点表示添加到输出字符串
        dota_output.append(f"{Ax} {Ay} {Bx} {By} {Cx} {Cy} {Dx} {Dy} {obj_name} {difficult}\n")

    return dota_output


def batch_convert_to_dota(xml_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历XML文件夹中的所有文件
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)

            # 生成输出文件路径
            output_file = os.path.splitext(xml_file)[0] + ".txt"
            output_path = os.path.join(output_folder, output_file)

            # 进行转换并写入文件
            dota_output = convert_to_dota(xml_path)
            with open(output_path, 'w') as f:
                f.writelines(dota_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XML annotations to DOTA format.")
    parser.add_argument("--xml_folder", required=True, help="Path to the folder containing XML annotations.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder for DOTA format files.")
    args = parser.parse_args()

    batch_convert_to_dota(args.xml_folder, args.output_folder)
