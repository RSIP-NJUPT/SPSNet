import os
import shutil

if os.path.exists('datasets/LS-SSDD-v1.0/dota/train/images/') is not True:
    os.makedirs('datasets/LS-SSDD-v1.0/dota/train/images/')
if os.path.exists('datasets/LS-SSDD-v1.0/dota/val/images/') is not True:
    os.makedirs('datasets/LS-SSDD-v1.0/dota/val/images/')

if os.path.exists('datasets/LS-SSDD-v1.0/dota/train/labels/') is not True:
    os.makedirs('datasets/LS-SSDD-v1.0/dota/train/labels/')
if os.path.exists('datasets/LS-SSDD-v1.0/dota/val/labels/') is not True:
    os.makedirs('datasets/LS-SSDD-v1.0/dota/val/labels/')

train = open('./datasets/LS-SSDD-v1.0/ImageSet/train.txt','r')
for line in train.readlines():
    image_name = line.strip('\n').split('_')[0] + '.jpg'
    if os.path.exists('datasets/LS-SSDD-v1.0/dota/train/images/' + image_name) is not True:
        shutil.copy('datasets/LS-SSDD-v1.0/images/' + image_name,
                    'datasets/LS-SSDD-v1.0/dota/train/images/' + image_name)
        txt_name = line.strip('\n').split('_')[0] + '.txt'
        shutil.copy('datasets/LS-SSDD-v1.0/dota/' + txt_name, 'datasets/LS-SSDD-v1.0/dota/train/labels/' + txt_name)
train.close()

test = open('./datasets/LS-SSDD-v1.0/ImageSet/test.txt','r')
for line in test.readlines():
    image_name = line.strip('\n').split('_')[0] + '.jpg'
    if os.path.exists('datasets/LS-SSDD-v1.0/dota/val/images/' + image_name) is not True:
        shutil.copy('datasets/LS-SSDD-v1.0/images/' + image_name, 'datasets/LS-SSDD-v1.0/dota/val/images/' + image_name)
        txt_name = line.strip('\n').split('_')[0] + '.txt'
        shutil.copy('datasets/LS-SSDD-v1.0/dota/' + txt_name, 'datasets/LS-SSDD-v1.0/dota/val/labels/' + txt_name)
test.close()