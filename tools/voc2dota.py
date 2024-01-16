# @Time    : 2023/2/24 10:04
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : voc2dota.py
# @Software: PyCharm
import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='VOC TO DOTA')
    parser.add_argument('xml_path')
    parser.add_argument('txt_path')
    args = parser.parse_args()
    return args


def edit_xml(xml_file, txt):
    print('Processing ' + xml_file)
    if ".xml" not in xml_file:
        return

    tree = ET.parse(xml_file)
    objs = tree.findall('object')

    with open(txt, 'w') as wf:
        for ix, obj in enumerate(objs):

            x0text = ""
            y0text = ""
            x1text = ""
            y1text = ""
            x2text = ""
            y2text = ""
            x3text = ""
            y3text = ""
            difficulttext = ""
            className = ""

            obj_type = obj.find('type')
            type = 'bndbox'

            obj_name = obj.find('name')
            className = obj_name.text

            obj_difficult = obj.find('difficult')
            difficulttext = obj_difficult.text

            if type == 'bndbox':
                obj_bnd = obj.find('bndbox')
                obj_xmin = obj_bnd.find('xmin')
                obj_ymin = obj_bnd.find('ymin')
                obj_xmax = obj_bnd.find('xmax')
                obj_ymax = obj_bnd.find('ymax')
                xmin = float(obj_xmin.text)
                ymin = float(obj_ymin.text)
                xmax = float(obj_xmax.text)
                ymax = float(obj_ymax.text)

                x0text = str(xmin)
                y0text = str(ymin)
                x1text = str(xmax)
                y1text = str(ymin)
                x2text = str(xmax)
                y2text = str(ymax)
                x3text = str(xmin)
                y3text = str(ymax)

                points = np.array([[int(float(x0text)), int(float(y0text))], [int(float(x1text)), int(float(y1text))], [
                    int(float(x2text)), int(float(y2text))], [int(float(x3text)), int(float(y3text))]], np.int32)
            wf.write(
                "{} {} {} {} {} {} {} {} {} {}\n".format(x0text, y0text, x1text, y1text, x2text, y2text, x3text, y3text,
                                                         className, difficulttext))


if __name__ == '__main__':
    args = parse_args()
    xml = args.xml_path
    txt = args.txt_path
    if os.path.exists(txt):
        shutil.rmtree(txt)
    os.mkdir(txt)
    filelist = os.listdir(xml)
    for fil in filelist:
        edit_xml(os.path.join(xml, fil), os.path.join(txt, fil[:-3] + 'txt'))
