_base_ = '../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_120e_SRSDD.py'

model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        loss_bbox_type='kfiou',
        loss_bbox=dict(type='KFLoss', loss_weight=5.0)),
    train_cfg=dict(
        assigner=dict(iou_calculator=dict(type='FakeRBboxOverlaps2D'))))
