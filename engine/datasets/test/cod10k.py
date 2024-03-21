from pathlib import Path
from PIL import Image

import numpy as np
import mmengine
from mmengine.dist import get_dist_info
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS

from ..base import BaseInterSegDataset


@TRANSFORMS.register_module()
class LoadCOD10KAnnotations(LoadAnnotations):

    def _load_seg_map(self, results):
        with Image.open(results['seg_map_path'], 'r') as gt_seg_map:
            gt_seg_map = np.array(gt_seg_map.convert('L'))
        gt_seg_map[gt_seg_map == 255] = 1
        results['gt_seg_map'] = gt_seg_map


@DATASETS.register_module()
class COD10KDataset(BaseInterSegDataset):

    default_meta_root = 'data/meta-info/cod10k.json'

    def __init__(self,
                 data_root,
                 pipeline,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        super(COD10KDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            img_suffix='.jpg',
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
        rank, _ = get_dist_info()
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            img_suffix, ann_suffix = self.img_suffix, self.ann_suffix
            img_files = {
                p.stem: p for p in
                (data_root / 'Test' / 'Image').glob(f'*{img_suffix}')}
            ann_files = {
                p.stem: p for p in
                (data_root / 'Test' / 'GT_Object').glob(f'*{ann_suffix}')}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            cam_names = [
                Path(line.strip().split()[0]).stem for line in
                open(data_root / 'Info' / 'CAM_test.txt', 'r').readlines()]
            prefixes = prefixes & set(cam_names)
            for prefix in sorted(list(prefixes)):
                data_list.append(
                    dict(img_path=str(img_files[prefix]),
                         seg_map_path=str(ann_files[prefix]),
                         seg_fields=[], segments_info=[dict(id=1)],
                         reduce_zero_label=False)
                )
            if rank == 0:
                mmengine.dump(dict(data_list=data_list), meta_root, indent=4)
        return data_list
