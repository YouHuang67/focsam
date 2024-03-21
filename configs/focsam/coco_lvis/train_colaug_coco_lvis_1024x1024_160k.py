_base_ = ['../../_base_/optimizer_adamw_160k.py',
          '../../_base_/train_colaug_coco_lvis_1024x1024.py',
          '../focsam_vit_huge.py']
batch_size = 4
train_dataloader = dict(batch_size=batch_size, num_workers=batch_size)
model = dict(
    type='ClickMixSegmentorRefine',
    remove_backbone=False,
    init_cfg=dict(type='Pretrained',
                  checkpoint='work_dirs/sam/coco_lvis/'
                             'train_colaug_coco_lvis_1024x1024_320k/'
                             'iter_320000.pth'),  # COCO-LVIS fine-tuned
    image_embed_loader=dict(
        _delete_=True,
        type='BaseEmbedLoader',
        embed_dir='data/embeds/colaug_coco_1024x1024_sam_vit_huge',
        list_format=True,
        update_prefixes_each_step=True),
    train_cfg=dict(
        interact_params={'coco': dict(gamma=0.6, refine_gamma=0.6),
                         'lvis': dict(gamma=0.9, refine_gamma=0.35)},
        expand_ratio_range=(1.0, 1.4))
)
find_unused_parameters = False
