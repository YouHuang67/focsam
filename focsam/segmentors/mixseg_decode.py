import warnings
from functools import partial

import torch

from mmseg.registry import MODELS
from mmseg.utils import add_prefix
from engine.timers import Timer
from .base import BaseClickSegmentor


@MODELS.register_module()
class ClickMixSegmentorDecode(BaseClickSegmentor):

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
        else:
            warnings.warn(f'Not found interact_params of {dataset}')
            max_num_clicks = cfg.max_num_clicks
            gamma = cfg.gamma
        return max_num_clicks, gamma

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

        max_num_clicks, gamma = self.parse_train_cfg(dataset)

        # initialize
        pre_labels = torch.zeros_like(gt_sem_seg)
        seg_labels = gt_sem_seg.detach().clone()
        prev_logits = None
        device = inputs.device
        num_clicks = self.sample_num_clicks(max_num_clicks, gamma)
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

        image_embeds = self.preprocess_inputs(inputs, data_samples)
        for _ in range(num_clicks):
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
                    image_embeds=image_embeds,
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

        prev_logits = []
        points_list = []
        max_num_points = 0

        for dataset in inputs_dict.keys():

            results = self.interact_simulate_on_single_dataset(
                inputs_dict[dataset], data_samples_dict[dataset], dataset)

            inputs.append(inputs_dict[dataset])
            data_samples.extend(data_samples_dict[dataset])
            gt_sem_seg.append(results['gt_sem_seg'])
            image_embeds.append(results['image_embeds'])
            scales.append(results['scale'])

            prev_logits.append(results['prev_logits'])
            points_list.append(results['points_list'])
            max_num_points = max(
                max_num_points, *[len(_) for _ in points_list[-1]])

        if len(set(scales)) != 1:
            raise ValueError(f'Found different scales: {scales}')
        scale = scales[0]
        point_map = partial(
            self.point_lists_to_coords, device=device, scale=scale)

        prompt_embeds = []
        for dataset_idx in range(len(inputs)):
            B, *_ = inputs[dataset_idx].shape
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

        gt_sem_seg = self.merge_tensors(gt_sem_seg)
        image_embeds = self.merge_tensors(image_embeds)
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

        # decode head forward-loss
        seg_logits = self.decode_head(
            dict(image_embeds=convert_image_embeds(image_embeds),
                 **prompt_embeds))
        seg_logits = self.interpolate(
            seg_logits, gt_sem_seg.shape[-2:])
        loss = self.loss_by_decode(seg_logits, gt_sem_seg)
        if self.with_metric:
            loss.update(self.metric_by_decode(seg_logits, gt_sem_seg))
        losses.update(add_prefix(loss, 'dec'))
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
        points = self.resize_coord_to_target_size(
            points, inputs.shape[-2:], cfg.target_size, inputs.device)
        logits = self.encode_decode(
            inputs, data_samples,
            image_embeds=image_embeds,
            prev_logits=prev_logits,
            points=points)
        return image_embeds, logits
