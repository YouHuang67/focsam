test_cfg = dict(_delete_=True, type='ClickTestLoop')
# COCO-Val settings
dataset_type = 'COCOPanopticValDataset'
data_root = 'data/coco2017'
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCOCOPanopticAnnotations'),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    dict(type='ObjectSampler',
         max_num_merged_objects=1,
         min_area_ratio=0.0),
    dict(type='InterSegPackSegInputs')
]
coco_dataset = dict(type=dataset_type,
                    data_root=data_root,
                    pipeline=test_pipeline)

# COCO-Val dataloader settings
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=coco_dataset
)
test_evaluator = dict()
# set num_clicks to 20 for NoC
model = dict(test_cfg=dict(num_clicks=20))
