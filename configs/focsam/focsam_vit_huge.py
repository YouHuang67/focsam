_base_ = '../sam/sam_vit_huge.py'
model = dict(
    type='ClickMixSegmentorRefine',
    decode_head=dict(type='SAMDecoderForRefiner'),
    refine_head=dict(type='FocusRefiner',
                     embed_dim=256,
                     depth=12,
                     num_heads=8,
                     mlp_ratio=4.0,
                     window_size=16),
    refine_extra_params=dict(mode='single mask with token')
)
