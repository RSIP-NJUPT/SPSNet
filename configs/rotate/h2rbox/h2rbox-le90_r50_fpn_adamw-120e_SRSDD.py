_base_ = [
    '../_base_/datasets/SRSDD_rbox.py',
    '../_base_/schedules/schedule_120e.py',
    'mmrotate::_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='H2RBoxDetector',
    crop_size=(1024, 1024),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[7.7832, 7.7832, 7.7832],
        std=[15.77581, 15.77581, 15.77581],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='H2RBoxHead',
        num_classes=6,
        in_channels=256,
        angle_version='le90',
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        square_classes=[9, 11],
        crop_size=(1024, 1024),
        loss_bbox_ss=dict(
            type='H2RBoxConsistencyLoss',
            loss_weight=0.4,
            center_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=0.0),
            shape_loss_cfg=dict(type='mmdet.IoULoss', loss_weight=1.0),
            angle_loss_cfg=dict(type='mmdet.L1Loss', loss_weight=1.0))),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# load hbox annotations
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    # Horizontal GTBox, (x1,y1,x2,y2)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    # Horizontal GTBox, (x,y,w,h,theta)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    # Horizontal GTBox, (x1,y1,x2,y2)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    # Horizontal GTBox, (x,y,w,h,theta)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    # Horizontal GTBox, (x1,y1,x2,y2)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    # Horizontal GTBox, (x,y,w,h,theta)
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(batch_size=4,num_workers=4,dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=4,num_workers=4,dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(batch_size=4,num_workers=4,dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=-1,
        _scope_='mmdet',
        save_best='auto'))
