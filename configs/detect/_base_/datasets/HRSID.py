# dataset settings
dataset_type = 'SPSNet.HRSIDDataset'
backend_args = None

METAINFO = {
        'classes':
        ('ship',),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228),]
    }

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=True, clip_object_border=False),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='mmrotate.ConvertBoxType', box_type_mapping=dict(gt_bboxes='hbox')),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_prefix=dict(img_path='datasets/HRSID/dota/train/images'),
        img_suffix='jpg',
        ann_file='datasets/HRSID/dota/train/labels',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=METAINFO,
        data_prefix=dict(img_path='datasets/HRSID/dota/test/images'),
        img_suffix='jpg',
        ann_file='datasets/HRSID/dota/test/labels/',
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(type='SPSNet.HorizontalDOTAMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator