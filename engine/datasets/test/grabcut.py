from pathlib import Path
from PIL import Image

import numpy as np
import mmengine
from mmengine.dist import get_dist_info
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS

from ..base import BaseInterSegDataset


@TRANSFORMS.register_module()
class LoadGrabCutAnnotations(LoadAnnotations):

    def __init__(self):
        super(LoadGrabCutAnnotations, self).__init__()

    def _load_seg_map(self, results):
        with Image.open(results['seg_map_path'], 'r') as gt_semantic_seg:
            gt_semantic_seg = np.array(gt_semantic_seg)
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')


@DATASETS.register_module()
class GrabCutDataset(BaseInterSegDataset):

    default_meta_root = 'data/meta-info/grabcut.json'

    def __init__(self,
                 data_root,
                 pipeline,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        super(GrabCutDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            img_suffix='.png',
            ann_suffix='.png',
            filter_cfg=None,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None)

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            img_files = {f'{p.stem}#001': p for p in
                         data_root.rglob(f'*{self.img_suffix}')
                         if not p.stem.endswith('#001')}
            ann_files = {p.stem: p for p in
                         data_root.rglob(f'*{self.ann_suffix}')
                         if p.stem.endswith('#001')}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            for prefix in sorted(list(prefixes)):
                img_file = str(img_files[prefix])
                ann_file = str(ann_files[prefix])
                data_list.append(
                    dict(img_path=img_file, seg_map_path=ann_file,
                         seg_fields=[], segments_info=[dict(id=1)],
                         reduce_zero_label=False)
                )
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root, indent=4)
        return data_list
