# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch


def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('image_encoder'):
            new_k = k.replace('image_encoder', 'backbone')
        elif k.startswith('prompt_encoder'):
            new_k = k.replace('prompt_encoder', 'neck')
            if 'pe_layer' in k:
                new_k = new_k.replace('pe_layer', 'pos_embed_layer')
            if 'point_embeddings' in k:
                new_k = new_k.replace('point_embeddings', 'point_embeds')
        elif k.startswith('mask_decoder'):
            new_k = k.replace('mask_decoder', 'decode_head')
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
