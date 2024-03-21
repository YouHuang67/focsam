from pathlib import Path
from scipy.io import loadmat

import numpy as np
import mmengine
from mmengine.dist import get_dist_info
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS

from ..base import BaseInterSegDataset


@TRANSFORMS.register_module()
class LoadSBDAnnotations(LoadAnnotations):

    def __init__(self):
        super(LoadSBDAnnotations, self).__init__()

    def _load_seg_map(self, results):
        gt_semantic_seg = loadmat(results['seg_map_path'])['GTinst'][0][0][0]
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')


@DATASETS.register_module()
class SBDDataset(BaseInterSegDataset):

    default_meta_root = 'data/meta-info/sbd.json'

    def __init__(self,
                 data_root,
                 pipeline,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        super(SBDDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            img_suffix='.jpg',
            ann_suffix='.mat',
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
            with open(data_root / 'val.txt', 'r') as file:
                val_set = set(line.strip() for line in file.readlines())
            img_files = {p.stem: p for p in
                         data_root.rglob(f'*{self.img_suffix}')
                         if p.stem in val_set}
            ann_files = {p.stem: p for p in
                         data_root.rglob(f'*inst/*{self.ann_suffix}')
                         if p.stem in val_set}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            for prefix in sorted(list(prefixes)):
                img_file = str(img_files[prefix])
                ann_file = str(ann_files[prefix])
                mask = loadmat(ann_file)['GTinst'][0][0][0]
                for idx in np.unique(mask):
                    if idx == 0:
                        continue
                    data_list.append(
                        dict(img_path=img_file, seg_map_path=ann_file,
                             seg_fields=[], segments_info=[dict(id=idx)],
                             reduce_zero_label=False)
                    )
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root, indent=4)
        return data_list


@DATASETS.register_module()
class SBDSubDataset(SBDDataset):

    def load_data_list(self):
        data_list = super(SBDSubDataset, self).load_data_list()
        return data_list[::10]
