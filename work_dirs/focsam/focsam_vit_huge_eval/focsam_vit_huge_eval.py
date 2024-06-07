custom_imports = dict(
    allow_failed_imports=False, imports=[
        'focsam',
        'mmdet.models',
    ])
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=0,
    size=(
        1024,
        1024,
    ),
    size_divisor=None,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, type='CheckpointHook'),
    logger=dict(interval=200, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        depth=32,
        embed_dim=1280,
        global_attn_indexes=(
            7,
            15,
            23,
            31,
        ),
        img_size=1024,
        in_dim=3,
        mlp_ratio=4.0,
        num_heads=16,
        out_dim=256,
        patch_size=16,
        qkv_bias=True,
        type='SAMWindowViT',
        use_abs_pos_embed=True,
        use_rel_pos_embed=True,
        window_size=14),
    decode_head=dict(
        align_corners=False,
        attn_cfg=dict(depth=2, mlp_dim=2048, num_heads=8),
        in_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        loss_decode=[
            dict(loss_weight=1.0, type='NormalizedFocalLoss'),
            dict(type='BinaryIoU'),
        ],
        num_multimask_outputs=3,
        type='SAMDecoderForRefiner'),
    image_embed_loader=None,
    init_cfg=dict(
        checkpoint='pretrain/sam_pretrain_vit_huge_mmcls.pth',
        type='Pretrained'),
    neck=dict(
        embed_dim=256,
        image_embed_size=(
            64,
            64,
        ),
        input_image_size=(
            1024,
            1024,
        ),
        mask_in_dim=16,
        type='SAMPromptEncoder'),
    refine_extra_params=dict(mode='single mask with token'),
    refine_head=dict(
        depth=12,
        embed_dim=256,
        mlp_ratio=4.0,
        num_heads=8,
        type='FocusRefiner',
        window_size=16),
    test_cfg=dict(target_size=1024),
    train_cfg=dict(
        gamma=0.6, max_num_clicks=20, sfc_inner_k=1.7, target_size=1024),
    type='ClickMixSegmentorRefine')
randomness = dict(seed=42)
resume = False
test_cfg = None
train_cfg = None
val_cfg = None
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = None
work_dir = './work_dirs/focsam/focsam_vit_huge_eval'
