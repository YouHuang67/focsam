test_cfg = dict(_delete_=True, type='ClickTestLoop')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCOD10KAnnotations'),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    dict(type='ObjectSampler',
         max_num_merged_objects=1,
         min_area_ratio=0.0),
    dict(type='PackSegInputs')
]
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='COD10KDataset',
        data_root='data/COD10K-v3',
        pipeline=test_pipeline))
test_evaluator = dict()
# set num_clicks to 20 for NoC
model = dict(test_cfg=dict(num_clicks=20))
