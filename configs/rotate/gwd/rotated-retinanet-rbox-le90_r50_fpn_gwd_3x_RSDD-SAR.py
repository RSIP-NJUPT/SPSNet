_base_ = [
    '../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_3x_RSDD-SAR.py'
]

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))
