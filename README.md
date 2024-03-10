# SPSNet
This an official Pytorch implementation of our paper "**SPSNet: A Selected Pyramidal Shape-constrained Network for SAR Small Target Detection**".

## Installation

Step 1: Create a conda environment

```shell
conda create --name SPSNet python=3.9
conda activate SPSNet
```

Step 2: Install PyTorch 2.0.1+CU118

```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Step 3: Install OpenMMLab codebases

```shell
# openmmlab codebases
pip install -U openmim dadaptation --no-input
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmsegmentation>=1.0.0" "mmrotate>=1.0.0rc1" mmyolo
# heatmap generation dependencies
pip install grad-cam
# other dependencies
pip install ninja --no-input
pip install scikit-learn
pip install psutil
```

Step 4: Install `SPSNet`

**Note**: make sure you have `cd` to the root directory of `SPSNet`

```shell
python setup.py develop
```
