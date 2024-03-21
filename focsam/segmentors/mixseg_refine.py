import random
import warnings
from copy import deepcopy
from functools import partial, reduce

import torch

from mmengine.utils.misc import to_2tuple
from mmseg.registry import MODELS
from mmseg.models.builder import HEADS
from mmseg.utils import add_prefix
from engine.utils import rearrange
from engine.timers import Timer
from .base import BaseClickSegmentor


@MODELS.register_module()
class ClickMixSegmentorRefine(BaseClickSegmentor):

    def __init__(self,
                 image_embed_loader,
                 backbone,
                 neck,
                 decode_head,
                 refine_head,
                 refine_extra_params=dict(),
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None,
                 remove_backbone=False):
        super(ClickMixSegmentorRefine, self).__init__(
            image_embed_loader=image_embed_loader,
            backbone=backbone,
            neck=neck,
            decode_head=decode_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            remove_backbone=remove_backbone)
        self.refine_head = HEADS.build(refine_head)
        self.refine_extra_params = refine_extra_params
        for param in self.decode_head.parameters():
            param.requires_grad = False

    def reset_refine_head_expand_ratio(self, cfg):
        if not hasattr(self.refine_head, 'expand_ratio'):
            raise AttributeError(
                f'Not found `expand_ratio` in `self.refine_head`, '
                f'please check whether {self.refine_head.__class__.__name__} '
                f'needs `expand_ratio`'
            )
        if hasattr(cfg, 'expand_ratio_range'):
            expand_ratio_range = cfg.expand_ratio_range
            h_ratio = random.uniform(*to_2tuple(expand_ratio_range))
            w_ratio = random.uniform(*to_2tuple(expand_ratio_range))
            self.refine_head.expand_ratio = (h_ratio, w_ratio)
        else:
            warnings.warn(
                f'Not found `expand_ratio_range` in `cfg`, '
                f'will use `self.refine_head.expand_ratio '
                f'{self.refine_head.expand_ratio} as default`')

    def parse_train_cfg(self, dataset):
        cfg = self.train_cfg
        if hasattr(cfg, 'interact_params'):
            interact_params = cfg.interact_params
        else:
            warnings.warn(f'Not found interact_params in train_cfg')
            interact_params = dict()
        if dataset in interact_params:
            params = interact_params[dataset]
            max_num_clicks = params.get('max_num_clicks', cfg.max_num_clicks)
            gamma = params.get('gamma', cfg.gamma)
            if 'refine_gamma' in params:
                refine_gamma = params['refine_gamma']
            else:
                if hasattr(cfg, 'refine_gamma'):
                    warnings.warn(f'Not found refine_gamma of {dataset}, '
                                  f'use default refine_gamma={gamma} instead')
                    refine_gamma = cfg.refine_gamma
                else:
                    warnings.warn(f'Not found refine_gamma of {dataset} and '
                                  f'default refine_gamma, '
                                  f'use default gamma={gamma} instead')
                    refine_gamma = gamma
        else:
            warnings.warn(f'Not found interact_params of {dataset}')
            max_num_clicks = cfg.max_num_clicks
            gamma = cfg.gamma
            if hasattr(cfg, 'refine_gamma'):
                refine_gamma = cfg.refine_gamma
            else:
                warnings.warn(f'Not found refine_gamma in train_cfg, '
                              f'use default gamma={gamma} instead')
                refine_gamma = gamma
        return max_num_clicks, gamma, refine_gamma

    @torch.no_grad()
    @Timer('Simulate')
    def interact_simulate_on_single_dataset(self, inputs,
                                            data_samples, dataset):
        cfg = self.train_cfg
        gt_sem_seg = self.check_gt_validity(data_samples, train=True)
        if hasattr(cfg, 'downsample_factor'):
            downsample_factor = cfg.downsample_factor
        else:
            downsample_factor = 1.0

        if hasattr(cfg, 'extra_click_cfg'):
            extra_click_cfg = cfg.extra_click_cfg
        else:
            extra_click_cfg = dict()

        max_num_clicks, gamma, refine_gamma = self.parse_train_cfg(dataset)

        # initialize
        pre_labels = torch.zeros_like(gt_sem_seg)
        seg_labels = gt_sem_seg.detach().clone()
        prev_logits = None
        device = inputs.device
        num_clicks = self.sample_num_clicks(max_num_clicks, gamma)
        refine_step = self.sample_num_clicks(num_clicks, refine_gamma)
        points_list = self.update_clicks(
            pre_labels, seg_labels, None, 1.0,
            downsample_factor, extra_click_cfg)
        if hasattr(cfg, 'target_image_size'):
            h, w = inputs.shape[-2:]
            if h != w:
                scale = float(cfg.target_image_size) / max(h, w)
                target_shape = (int(scale * h + 0.5), int(scale * w + 0.5))
                warnings.warn(
                    f'The image has different height and width: '
                    f'{h} and {w}, resize the image to {target_shape} '
                    f'as large as possible')
            else:
                scale = float(cfg.target_image_size) / h
                target_shape = (cfg.target_image_size,) * 2
            if tuple(inputs.shape[-2:]) != target_shape:
                inputs = self.interpolate(inputs, target_shape)
        else:
            scale = 1.0

        # before refinement
        ori_image_embeds = self.preprocess_inputs(inputs, data_samples)
        for _ in range(refine_step):
            logits = self.encode_decode(
                inputs, data_samples,
                image_embeds=ori_image_embeds,
                prev_logits=prev_logits,
                points=self.point_lists_to_coords(
                    points_list, device, scale=scale)
            )
            prev_logits = logits
            logits = self.interpolate(logits, pre_labels.shape[-2:])
            pre_labels = (logits > 0.0).to(pre_labels)
            points_list = self.update_clicks(
                pre_labels, seg_labels, points_list,
                cfg.sfc_inner_k, downsample_factor, extra_click_cfg)

        # refine step
        if prev_logits is None:
            refine_prev_logits = None
        else:
            refine_prev_logits = prev_logits.detach().clone()
        refine_points_list = deepcopy(points_list)

        coarse_logits, *refine_extra_inputs = self.decode_head(
            self.neck(ori_image_embeds,
                      prev_logits=prev_logits,
                      points=self.point_lists_to_coords(
                          points_list, device, scale=scale)),
            **self.refine_extra_params)
        self.reset_refine_head_expand_ratio(cfg)
        image_embeds = self.refine_head(
            self.copy(ori_image_embeds), inputs,
            self.copy(coarse_logits),
            *self.copy(refine_extra_inputs))

        # after refinement
        for _ in range(num_clicks - refine_step):
            logits = self.encode_decode(
                inputs, data_samples,
                image_embeds=image_embeds,
                prev_logits=prev_logits,
                points=self.point_lists_to_coords(
                    points_list, device, scale=scale)
            )
            prev_logits = logits
            logits = self.interpolate(logits, pre_labels.shape[-2:])
            pre_labels = (logits > 0.0).to(pre_labels)
            points_list = self.update_clicks(
                pre_labels, seg_labels, points_list,
                cfg.sfc_inner_k, downsample_factor, extra_click_cfg)

        return dict(gt_sem_seg=gt_sem_seg,
                    image_embeds=ori_image_embeds,
                    refine_prev_logits=refine_prev_logits,
                    refine_points_list=refine_points_list,
                    prev_logits=prev_logits,
                    points_list=points_list,
                    scale=scale)

    def interact_train(self, inputs, data_samples):
        device = inputs.device
        inputs_dict, data_samples_dict, _ = \
            self.redistribute_tensor(inputs, data_samples)

        self.eval()

        inputs = []
        data_samples = []
        gt_sem_seg = []
        image_embeds = []
        scales = []

        refine_prev_logits = []
        refine_points_list = []
        prev_logits = []
        points_list = []
        max_refine_num_points = 0
        max_num_points = 0

        for dataset in inputs_dict.keys():

            results = self.interact_simulate_on_single_dataset(
                inputs_dict[dataset], data_samples_dict[dataset], dataset)

            inputs.append(inputs_dict[dataset])
            data_samples.extend(data_samples_dict[dataset])
            gt_sem_seg.append(results['gt_sem_seg'])
            image_embeds.append(results['image_embeds'])
            scales.append(results['scale'])

            refine_prev_logits.append(results['refine_prev_logits'])
            refine_points_list.append(results['refine_points_list'])
            prev_logits.append(results['prev_logits'])
            points_list.append(results['points_list'])
            max_refine_num_points = max(
                max_refine_num_points,
                *[len(_) for _ in refine_points_list[-1]])
            max_num_points = max(
                max_num_points, *[len(_) for _ in points_list[-1]])

        if len(set(scales)) != 1:
            raise ValueError(f'Found different scales: {scales}')
        scale = scales[0]
        point_map = partial(
            self.point_lists_to_coords, device=device, scale=scale)

        refine_prompt_embeds = []
        prompt_embeds = []
        for dataset_idx in range(len(inputs)):
            B, *_ = inputs[dataset_idx].shape
            refine_prompt_embeds.append(
                {k: (v.repeat(B, *[1] * (v.ndim - 1)) if v.size(0) == 1 else v)
                 for k, v in self.neck(
                    image_embeds[dataset_idx],
                    points=point_map(refine_points_list[dataset_idx],
                                     max_num_points=max_refine_num_points),
                    prev_logits=refine_prev_logits[dataset_idx]
                 ).items()
                 if 'image_embeds' not in k}
            )
            prompt_embeds.append(
                {k: (v.repeat(B, *[1] * (v.ndim - 1)) if v.size(0) == 1 else v)
                 for k, v in self.neck(
                    image_embeds[dataset_idx],
                    points=point_map(points_list[dataset_idx],
                                     max_num_points=max_num_points),
                    prev_logits=prev_logits[dataset_idx]
                 ).items()
                 if 'image_embeds' not in k}
            )

        inputs = self.merge_tensors(inputs)
        gt_sem_seg = self.merge_tensors(gt_sem_seg)
        image_embeds = self.merge_tensors(image_embeds)
        refine_prompt_embeds = self.merge_tensors(refine_prompt_embeds)
        prompt_embeds = self.merge_tensors(prompt_embeds)

        self.train()

        def convert_image_embeds(x):  # convert list to tensor
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, (list, tuple)):
                if len(x) > 1:
                    warnings.warn(f'Found more than one image_embeds, '
                                  f'use the last one instead')
                return x[-1]
            else:
                raise NotImplementedError(f'Unknown type {type(x)}')

        losses = dict()
        with torch.no_grad():
            seg_logits = self.decode_head(
                dict(image_embeds=convert_image_embeds(image_embeds),
                     **prompt_embeds))
            seg_logits = self.interpolate(
                seg_logits, gt_sem_seg.shape[-2:])
        loss = self.metric_by_decode(seg_logits, gt_sem_seg)
        losses.update(add_prefix(loss, 'dec'))

        # refine head forward-loss
        with torch.no_grad():
            coarse_logits, *refine_extra_inputs = self.decode_head(
                dict(image_embeds=convert_image_embeds(image_embeds),
                     **refine_prompt_embeds),
                **self.refine_extra_params)
        self.reset_refine_head_expand_ratio(self.train_cfg)
        image_embeds = self.refine_head(
            image_embeds, inputs, coarse_logits, *refine_extra_inputs)
        seg_logits = self.decode_head(
            dict(image_embeds=convert_image_embeds(image_embeds),
                 **prompt_embeds))
        seg_logits = self.interpolate(
            seg_logits, gt_sem_seg.shape[-2:])
        loss = self.loss_by_decode(seg_logits, gt_sem_seg)
        if self.train_cfg.get('refine_with_click_loss', True):
            points_list = reduce(lambda x, y: x + y, points_list, [])
            loss.update(dict(clk_loss=self.click_loss(
                seg_logits,
                *self.point_lists_to_coords(points_list, inputs.device))
            ))
        if self.with_metric:
            loss.update(self.metric_by_decode(seg_logits, gt_sem_seg))
        losses.update(add_prefix(loss, 'ref'))
        return losses

    @torch.no_grad()
    def interact_test(self, inputs, data_samples):
        assert self.with_losses, 'Not found loss function in test mode.'
        cfg = self.test_cfg
        gt_sem_seg = self.check_gt_validity(data_samples, train=False)

        if hasattr(cfg, 'sfc_inner_k'):
            sfc_inner_k = cfg.sfc_inner_k
        else:
            sfc_inner_k = 1.0

        self.eval()
        resized_padded_inputs = \
            self.resize_and_pad_to_target_size(inputs, cfg.target_size)
        image_embeds = \
            self.preprocess_inputs(resized_padded_inputs, data_samples)
        ori_image_embeds = image_embeds
        pre_labels = torch.zeros_like(gt_sem_seg)
        seg_labels = gt_sem_seg
        prev_logits = None

        self.intertest_init()
        results, points = [], None
        for step in range(cfg.num_clicks):
            points, *_ = self.update_clicks(
                pre_labels, seg_labels, [points], sfc_inner_k)
            image_embeds, logits = self.intertest_predict(
                inputs=inputs,
                data_samples=data_samples,
                resized_padded_inputs=resized_padded_inputs,
                image_embeds=image_embeds,
                ori_image_embeds=ori_image_embeds,
                step=step,
                prev_logits=prev_logits,
                points=points)
            prev_logits = logits
            logits = self.interpolate(logits, resized_padded_inputs.shape[-2:])
            logits = self.crop_and_resize_to_original_size(
                logits, inputs.shape[-2:], cfg.target_size)
            pre_labels = (logits > 0.0).to(pre_labels)
            h, w = seg_labels.shape[-2:]
            pre_labels = pre_labels[..., :h, :w]
            results.append(pre_labels.squeeze().detach().cpu().numpy())
        gt_sem_seg = gt_sem_seg.squeeze().detach().cpu().numpy()
        return points, results, gt_sem_seg

    def intertest_predict(self,
                          inputs,
                          data_samples,
                          resized_padded_inputs,
                          image_embeds,
                          ori_image_embeds,
                          step,
                          prev_logits,
                          points):
        cfg = self.test_cfg
        if hasattr(cfg, 'refine_start_step'):
            refine_start_step = cfg.refine_start_step
        else:
            refine_start_step = 1

        points = self.resize_coord_to_target_size(
            points, inputs.shape[-2:], cfg.target_size, inputs.device)
        if step != refine_start_step:
            logits = self.encode_decode(
                inputs, data_samples,
                image_embeds=image_embeds,
                prev_logits=prev_logits,
                points=points)
            return image_embeds, logits

        coarse_logits, *refine_extra_inputs = self.decode_head(
            self.neck(ori_image_embeds,
                      prev_logits=prev_logits, points=points),
            **self.refine_extra_params
        )
        image_embeds = self.refine_head(
            ori_image_embeds, resized_padded_inputs,
            coarse_logits, *refine_extra_inputs
        )

        logits = self.encode_decode(
            inputs, data_samples,
            image_embeds=image_embeds, prev_logits=prev_logits, points=points)
        return image_embeds, logits

    @staticmethod
    def click_loss(logits, coords, labels):
        H, _ = logits.shape[-2:]
        logits = rearrange(logits,
                           'b () h w -> b (w h)')  # since x is placed before y in coords
        coords = coords[..., 0] * H + coords[..., 1]
        logits = torch.gather(logits, 1, coords.long())
        losses = (torch.sigmoid(logits) - labels.float()).square()[labels >= 0]
        return losses.mean()
