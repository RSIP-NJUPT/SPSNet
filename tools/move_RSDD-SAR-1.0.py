import os
import shutil

if os.path.exists('datasets/RSDD-SAR/dota/train/images/') is not True:
    os.makedirs('datasets/RSDD-SAR/dota/train/images/')
if os.path.exists('datasets/RSDD-SAR/dota/test/images/') is not True:
    os.makedirs('datasets/RSDD-SAR/dota/test/images/')

if os.path.exists('datasets/RSDD-SAR/dota/train/labels/') is not True:
    os.makedirs('datasets/RSDD-SAR/dota/train/labels/')
if os.path.exists('datasets/RSDD-SAR/dota/test/labels/') is not True:
    os.makedirs('datasets/RSDD-SAR/dota/test/labels/')

train = open('./datasets/RSDD-SAR/ImageSets/train.txt','r')
for line in train.readlines():
    image_name = line.strip('\n') + '.jpg'
    shutil.copy('datasets/RSDD-SAR/JPEGImages/' + image_name, 'datasets/RSDD-SAR/dota/train/images/' + image_name)
    txt_name = line.strip('\n') + '.txt'
    shutil.copy('datasets/RSDD-SAR/dota/' + txt_name, 'datasets/RSDD-SAR/dota/train/labels/'+txt_name)
train.close()

test = open('./datasets/RSDD-SAR/ImageSets/test.txt','r')
for line in test.readlines():
    image_name = line.strip('\n') + '.jpg'
    shutil.copy('datasets/RSDD-SAR/JPEGImages/' + image_name, 'datasets/RSDD-SAR/dota/test/images/' + image_name)
    txt_name = line.strip('\n') + '.txt'
    shutil.copy('datasets/RSDD-SAR/dota/' + txt_name, 'datasets/RSDD-SAR/dota/test/labels/'+txt_name)
test.close()