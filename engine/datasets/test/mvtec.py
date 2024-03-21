import warnings
from pathlib import Path

import mmengine
from mmengine.dist import get_dist_info
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS

from ..base import BaseInterSegDataset


@TRANSFORMS.register_module()
class LoadMVTecAnnotations(LoadAnnotations):

    def _load_seg_map(self, results):
        super(LoadMVTecAnnotations, self)._load_seg_map(results)
        gt_seg_map = results.pop('gt_seg_map')
        gt_seg_map[gt_seg_map == 255] = 1
        results['gt_seg_map'] = gt_seg_map


@DATASETS.register_module()
class MVTecDataset(BaseInterSegDataset):

    default_meta_root = 'data/meta-info/mvtec.json'
    default_link_root = 'data/mvtec-soft-link'
    IGNORE_DEFECT = [('transistor', 'cut_lead'), ('transistor', 'misplaced')]

    def __init__(self,
                 data_root,
                 pipeline,
                 link_root=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        self.link_root = link_root or self.default_link_root
        super(MVTecDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            img_suffix='.png',
            ann_suffix='_mask.png',
            filter_cfg=None,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None)

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)
        link_root = Path(self.link_root)
        rank, _ = get_dist_info()
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            img_suffix, ann_suffix = self.img_suffix, self.ann_suffix
            img_files = {
                f'{p.parents[2].name}_{p.parent.name}_'
                f'{p.name.replace(img_suffix, "")}': p
                for p in data_root.rglob(f'*{img_suffix}')
                if p.parents[1].name == 'test' and p.parent.name != 'good'}
            ann_files = {
                f'{p.parents[2].name}_{p.parent.name}_'
                f'{p.name.replace(ann_suffix, "")}': p
                for p in data_root.rglob(f'*{ann_suffix}')
                if p.parents[1].name == 'ground_truth'}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            if rank == 0:
                link_root.mkdir(parents=True, exist_ok=True)
            for prefix in sorted(list(prefixes)):
                ignore = False
                for cls_type, def_type in self.IGNORE_DEFECT:
                    if cls_type.lower() in prefix.lower() and \
                       def_type.lower() in prefix.lower():
                        ignore = True
                        break
                if ignore:
                    continue
                src_img_path = img_files[prefix]
                tar_img_path = link_root / f'{prefix}{img_suffix}'
                src_ann_path = ann_files[prefix]
                tar_ann_path = link_root / f'{prefix}{ann_suffix}'
                if rank == 0:
                    if not tar_img_path.is_file():
                        tar_img_path.symlink_to(src_img_path.resolve())
                    else:
                        warnings.warn(f'{str(tar_img_path)} already exists')
                    if not tar_ann_path.is_file():
                        tar_ann_path.symlink_to(src_ann_path.resolve())
                    else:
                        warnings.warn(f'{str(tar_ann_path)} already exists')
                data_list.append(
                    dict(img_path=str(tar_img_path),
                         seg_map_path=str(tar_ann_path),
                         seg_fields=[], segments_info=[dict(id=1)],
                         reduce_zero_label=False)
                )
            if rank == 0:
                mmengine.dump(dict(data_list=data_list), meta_root, indent=4)
        return data_list
