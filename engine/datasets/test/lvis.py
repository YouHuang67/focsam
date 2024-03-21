from functools import partial
from pathlib import Path

import mmengine
from mmengine.dist import get_dist_info
from mmengine.logging import print_log as _print_log
from mmseg.registry import DATASETS

from ..base import BaseInterSegDataset


print_log = partial(_print_log, logger='current')


@DATASETS.register_module()
class LVISValDataset(BaseInterSegDataset):

    default_meta_file = 'data/meta-info/lvis-val.json'
    default_meta_root = 'data/meta-info/lvis-val-segments-infos'

    def __init__(self,
                 data_root,
                 pipeline,
                 meta_file=None,
                 meta_root=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        self.meta_file = meta_file or self.default_meta_file
        super(LVISValDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            meta_root=meta_root,
            img_suffix='.jpg',
            ann_suffix=None,
            filter_cfg=None,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None)

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_file = Path(self.meta_file)
        if meta_file.is_file():
            data_list = mmengine.load(meta_file)['data_list']
        else:
            meta_root = Path(self.meta_root)
            data_list = []
            img_files = {
                int(p.stem): p for p in data_root.rglob(f'*{self.img_suffix}')}
            prefixes = set(img_files.keys())

            annotations = mmengine.load(
                next(data_root.rglob(f'lvis_v1_val.json')))['annotations']
            not_found = set()
            for info in sorted(annotations, key=lambda x: x['id']):
                prefix = info['image_id']
                if prefix not in prefixes:
                    if prefix not in not_found:
                        not_found.add(prefix)
                        print_log(f'Not found image {prefix} , skip it.')
                    continue
                segments_info_file = str(
                    meta_root / f'{prefix}-{info["id"]}.json')
                if get_dist_info()[0] == 0:
                    segments_info_dict = dict(segments_info=[info])
                    meta_root.mkdir(parents=True, exist_ok=True)
                    mmengine.dump(segments_info_dict, segments_info_file)
                data_list.append(
                    dict(img_path=str(img_files[prefix]),
                         seg_map_path=None,
                         segments_info_file=segments_info_file,
                         seg_fields=[], reduce_zero_label=False)
                )
            if len(not_found) > 0:
                print_log(f'Not found {len(not_found)} images.')
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_file)
        return data_list
