_base_ = ['../../_base_/optimizer_adamw_320k.py',
          '../../_base_/train_colaug_coco_lvis_1024x1024.py',
          '../sam_vit_huge.py']
batch_size = 4
train_dataloader = dict(batch_size=batch_size, num_workers=batch_size)
model = dict(
    type='ClickMixSegmentorDecode',
    remove_backbone=False,
    init_cfg=dict(type='Pretrained',
                  checkpoint='pretrain/sam_pretrain_vit_huge.pth'),  # SAM pretrained
    image_embed_loader=dict(
        _delete_=True,
        type='BaseEmbedLoader',
        embed_dir='data/embeds/colaug_coco_1024x1024_sam_vit_huge',
        list_format=True,
        update_prefixes_each_step=False),
    train_cfg=dict(
        interact_params={'coco': dict(gamma=0.6), 'lvis': dict(gamma=0.9)})
)
