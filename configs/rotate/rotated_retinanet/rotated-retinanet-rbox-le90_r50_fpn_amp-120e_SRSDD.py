_base_ = ['./rotated-retinanet-rbox-le90_r50_fpn_120e_SRSDD.py']

optim_wrapper = dict(type='AmpOptimWrapper')