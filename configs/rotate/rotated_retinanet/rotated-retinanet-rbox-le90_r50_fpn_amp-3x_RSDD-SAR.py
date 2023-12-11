_base_ = ['./rotated-retinanet-rbox-le90_r50_fpn_3x_RSDD-SAR.py']

optim_wrapper = dict(type='AmpOptimWrapper')