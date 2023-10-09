# GCFNet
This an official Pytorch implementation of our paper "**GCFNet: Global Attention And Class-balanced Focal Network for SAR Target Detection**".

## Installation

Step 1: Create a conda environment

```shell
conda create --name GCFNet python=3.9
conda activate GCFNet
```

Step 2: Install PyTorch 2.0.0+CU118

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Step 3: Install OpenMMLab codebases

```shell
# openmmlab codebases
pip install -U openmim dadaptation --no-input
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmsegmentation>=1.0.0" "mmrotate>=1.0.0rc1"
# other dependencies
pip install ninja --no-input
```

Step 4: Install `GCFNet`

**Note**: make sure you have `cd` to the root directory of `GCFNet`

```shell
python setup.py develop
```
