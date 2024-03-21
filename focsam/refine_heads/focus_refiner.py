import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
from mmengine.utils.misc import to_2tuple
from mmseg.models.builder import HEADS

from engine.utils import rearrange, repeat, reduce, memory_efficient_attention
from engine.utils import get_bbox_from_mask, expand_bbox, convert_bbox_to_mask
from engine.timers import Timer


class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads=8):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    @Timer('Attention')
    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = rearrange(q, 'b n (hn d) -> b hn n d', hn=self.num_heads)
        k = rearrange(k, 'b n (hn d) -> b hn n d', hn=self.num_heads)
        v = rearrange(v, 'b n (hn d) -> b hn n d', hn=self.num_heads)

        k = torch.cat([k, torch.zeros_like(k[..., :1, :])], dim=-2)
        v = torch.cat([v, torch.zeros_like(v[..., :1, :])], dim=-2)
        out = memory_efficient_attention(q, k, v, scale=self.scale)

        out = rearrange(out, 'b hn n d -> b n (hn d)', hn=self.num_heads)
        out = self.proj(out)
        return out


class CrossAttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(CrossAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.gamma1 = nn.Parameter(torch.ones(embed_dim).float())
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio * embed_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(mlp_ratio * embed_dim), embed_dim))
        self.gamma2 = nn.Parameter(torch.ones(embed_dim).float())

    @Timer('CrossAttentionBlock')
    def forward(self, q, kv):
        x = q
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        x = x + self.gamma1 * self.attn(q, kv, kv)
        x = x + self.gamma2 * self.mlp(self.norm(x))
        return x


class PDyReLU(nn.Module):

    def __init__(self, channels):
        super(PDyReLU, self).__init__()
        self.channels = channels
        self.pixel_scale = nn.Parameter(torch.zeros(2 * channels).float())
        self.pixel_bias = nn.Parameter(torch.zeros(2 * channels).float())
        self.pixel_mlp = nn.Sequential(
            nn.Linear(1, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2 * channels, bias=True),
            nn.Tanh())
        self.channel_scale = nn.Parameter(torch.zeros(2 * channels).float())
        self.channel_bias = nn.Parameter(torch.zeros(2 * channels).float())
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2 * channels, bias=True),
            nn.Tanh())

    @Timer('PDyReLU')
    def forward(self, x, cls):
        """
        :param x: shape (B, N, C)
        :param cls: shape (B, 1, C)
        :return: shape (B, N, C)
        """
        # pixel-wise
        pixel_relu = self.pixel_mlp(x @ cls.transpose(-1, -2))
        pixel_relu = pixel_relu * \
                     rearrange(self.pixel_scale.exp(), "c -> () () c") + \
                     rearrange(self.pixel_bias, "c -> () () c")
        pixel_act = pixel_relu[..., :self.channels] * x + \
                    pixel_relu[..., self.channels:]

        # channel-wise
        channel_relu = self.channel_mlp(x.mean(1, keepdim=True))
        channel_relu = channel_relu * \
                       rearrange(self.channel_scale.exp(), "c -> () () c") + \
                       rearrange(self.channel_bias, "c -> () () c")
        channel_act = channel_relu[..., :self.channels] * x + \
                      channel_relu[..., self.channels:]

        # combine
        out = torch.max(pixel_act, channel_act)
        return out


class DeformLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 modulate_deform=True,
                 num_groups=1,
                 deform_num_groups=1,
                 dilation=1,
                 bias=False):
        super(DeformLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modulate_deform = modulate_deform
        self.num_groups = num_groups
        self.deform_num_groups = deform_num_groups
        self.dilation = dilation

        if modulate_deform:
            deform_conv_op = ModulatedDeformConv2d
            offset_channels = 27
        else:
            deform_conv_op = DeformConv2d
            offset_channels = 18
        self.dcn_offset = nn.Conv2d(
            in_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation)
        self.dcn = deform_conv_op(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=bias,
            groups=num_groups,
            dilation=dilation,
            deform_groups=deform_num_groups)
        self.c2_msra_fill(self.dcn)

    @Timer('DeformConv')
    def forward(self, x):
        if self.modulate_deform:
            offset_mask = self.dcn_offset(x)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.dcn(x, offset, mask)
        else:
            offset = self.dcn_offset(x)
            out = self.dcn(x, offset)
        return out

    @staticmethod
    def c2_msra_fill(module, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            fan_out = module.weight.size(0)
            fan_in = np.prod(module.weight.shape[1:])
            std = np.sqrt(2.0 / (fan_in + fan_out))
            module.weight.data.normal_(0, std)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(bias)
        return module


class MSAModule(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(MSAModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.cls_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.LayerNorm(embed_dim))
        self.attn = Attention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.relu1 = PDyReLU(embed_dim)

        self.conv2 = DeformLayer(embed_dim, embed_dim)
        self.norm2 = nn.GroupNorm(1, embed_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.gamma1 = nn.Parameter(torch.ones(1, embed_dim, 1, 1).float())
        self.gamma2 = nn.Parameter(torch.ones(1, embed_dim, 1, 1).float())

    @Timer('MSAModule')
    def forward(self, x, cls):
        """
        :param x: shape (B, C, H, W)
        :param cls: shape (B, 1, C)
        :return: shape (B, C, H, W)
        """
        out = rearrange(x, "b c h w -> b (h w) c")
        out = self.attn(out, out, out)
        out = self.relu1(self.norm1(out), self.cls_proj(cls))
        out = rearrange(out, "b (h w) c -> b c h w", h=x.size(2), w=x.size(3))
        out = x + self.gamma1 * out

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = x + self.gamma2 * out
        return out


class RefineBlock(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio,
                 window_size,
                 cls_scale,
                 img2cls_type=CrossAttentionBlock,
                 cls2img_type=MSAModule):
        super(RefineBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.img2cls = img2cls_type(embed_dim, num_heads, mlp_ratio)
        self.cls2img = cls2img_type(embed_dim, num_heads=num_heads)
        if cls_scale:
            self.cls_scale_proj = nn.Linear(1, 1, bias=True)
            self.cls_scale_proj.weight.data.fill_(1)
            self.cls_scale_proj.bias.data.fill_(0)

    def forward(self, x, cls, mask):
        """
        :param x: shape (B, C, H, W)
        :param cls: shape (B, 1, C)
        :param mask: shape (B, 1, H, W)
        :return: shape (B, C, H, W)
        """
        pre_cls = cls
        ori_hw = x.shape[-2:]
        x, pad_hw = self.window_partition(x)
        cls = repeat(cls, 'b () c -> (b hw) () c', hw=x.size(0) // cls.size(0))

        mask, _ = self.window_partition(mask)
        mask = reduce(mask, 'bhw () Wh Ww -> bhw', 'max').bool()
        indices = rearrange(torch.nonzero(mask), 'bhw () -> bhw')
        ori_x, ori_cls = x, cls
        x, cls = x[indices], cls[indices]

        cls = self.img2cls(cls, rearrange(x, "b c h w -> b (h w) c"))
        x = self.cls2img(x, cls)

        ori_x[indices], ori_cls[indices] = x, cls
        x, cls = ori_x, ori_cls

        x = self.window_unpartition(x, pad_hw, ori_hw)
        cls = reduce(cls, '(b hw) () c -> b () c', 'mean', b=x.size(0))
        if hasattr(self, 'cls_scale_proj'):
            cls_scale = reduce(
                mask.float(), '(b hw) -> b () ()', 'mean', b=x.size(0))
            cls_scale = \
                self.cls_scale_proj(-cls_scale.clip(1e-2).log()).exp()
            cls = cls_scale * (cls - pre_cls) + pre_cls
        return x, cls

    def window_partition(self, x):
        B, C, H, W = x.shape
        window_size = self.window_size
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0))
        Hp, Wp = H + pad_h, W + pad_w
        x = rearrange(x, 'b c (h Wh) (w Ww) -> (b h w) c Wh Ww',
                      Wh=window_size, Ww=window_size)
        return x, (Hp, Wp)

    def window_unpartition(self, x, pad_hw, ori_hw):
        Hp, Wp = pad_hw
        H, W = ori_hw
        window_size = self.window_size
        x = rearrange(x, '(b h w) c Wh Ww -> b c (h Wh) (w Ww)',
                      h=Hp // window_size, w=Wp // window_size)
        if Hp > H or Wp > W:
            x = x[..., :H, :W].contiguous()
        return x


class ShiftRefineBlock(RefineBlock):

    def forward(self, x, cls, mask):
        """
        :param x: shape (B, C, H, W)
        :param cls: shape (B, 1, C)
        :param mask: shape (B, 1, H, W)
        :return: shape (B, C, H, W)
        """
        pre_cls = cls
        H, W = x.shape[-2:]
        window_size = self.window_size
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b, 0, 0))
        Hp, Wp = x.shape[-2:]
        shift_size = window_size // 2
        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(-2, -1))
        x, pad_hw = self.window_partition(x)
        cls = repeat(cls, 'b () c -> (b hw) () c', hw=x.size(0) // cls.size(0))

        if pad_r > 0 or pad_b > 0:
            mask = F.pad(mask, (0, pad_r, 0, pad_b, 0, 0))
        mask = torch.roll(
            mask, shifts=(-shift_size, -shift_size), dims=(-2, -1))
        mask, _ = self.window_partition(mask)
        mask = reduce(mask, 'bhw () Wh Ww -> bhw', 'max').bool()
        indices = rearrange(torch.nonzero(mask), 'bhw () -> bhw')
        ori_x, ori_cls = x, cls
        x, cls = x[indices], cls[indices]

        cls = self.img2cls(cls, rearrange(x, "b c h w -> b (h w) c"))
        x = self.cls2img(x, cls)

        ori_x[indices], ori_cls[indices] = x, cls
        x, cls = ori_x, ori_cls

        x = self.window_unpartition(x, pad_hw, (Hp, Wp))
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(-2, -1))
        if pad_r > 0 or pad_b > 0:
            x = x[..., :H, :W].contiguous()
        cls = reduce(cls, '(b hw) () c -> b () c', 'mean', b=x.size(0))
        if hasattr(self, 'cls_scale_proj'):
            cls_scale = reduce(
                mask.float(), '(b hw) -> b () ()', 'mean', b=x.size(0)
            )
            cls_scale = \
                self.cls_scale_proj(-cls_scale.clip(1e-2).log()).exp()
            cls = cls_scale * (cls - pre_cls) + pre_cls
        return x, cls


@HEADS.register_module()
class FocusRefiner(nn.Module):

    def __init__(self,
                 embed_dim=256,
                 depth=12,
                 num_heads=8,
                 mlp_ratio=4.0,
                 window_size=14,
                 img2cls_type=CrossAttentionBlock,
                 cls2img_type=MSAModule,
                 expand_ratio=1.4):
        super(FocusRefiner, self).__init__()
        self.embed_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.token_proj = nn.Linear(embed_dim, embed_dim)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = [RefineBlock, ShiftRefineBlock][i % 2]
            self.blocks.append(
                block(embed_dim=embed_dim,
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      window_size=window_size,
                      cls_scale=(i < depth - 1),
                      img2cls_type=img2cls_type,
                      cls2img_type=cls2img_type)
            )
        self.expand_ratio = to_2tuple(expand_ratio)

    def forward(self,
                image_embeds,
                inputs,
                coarse_logits,
                mask_token,
                mask=None):
        if isinstance(image_embeds, (list, tuple)):
            if len(image_embeds) != 1:
                raise ValueError(f'`image_embeds` is expected to have single '
                                 f'embed, but got {len(image_embeds)} embeds')
            image_embeds = image_embeds[0]
        if not isinstance(image_embeds, torch.Tensor):
            raise TypeError(f'`image_embeds` is expected to be a Tensor or '
                            f'a list of Tensors, but got {type(image_embeds)}')
        if mask is None:
            coarse_logits = F.interpolate(coarse_logits, inputs.shape[-2:],
                                          mode='bilinear', align_corners=False)
            bbox = get_bbox_from_mask((coarse_logits > 0.0).long())
            bbox = expand_bbox(bbox, *self.expand_ratio, *inputs.shape[-2:])
            mask = convert_bbox_to_mask(bbox, inputs.shape[-2:], inputs.device)
        if mask_token.ndim == 2:
            mask_token = rearrange(mask_token, 'b c -> b () c')
        if mask_token.ndim != 3 or mask_token.size(1) != 1:
            raise ValueError(f'`mask_token` is expected to have the shape: '
                             f'(B, 1, C), but got `mask_token` of shape: '
                             f'{tuple(mask_token.shape)}')
        return self.stem(image_embeds, mask_token, mask)

    @Timer('FocusRefiner')
    def stem(self, image_embeds, mask_token, mask):
        """
        :param image_embeds: shape (B, C, H, W)
        :param mask_token: shape (B, 1, C)
        :param mask: shape (B, 1, oriH, oriW)
        :return: shape (B, C, H, W)
        """
        x = self.embed_proj(image_embeds)
        cls = self.token_proj(mask_token)
        mask = F.adaptive_max_pool2d(mask.float(), x.shape[-2:])
        for block in self.blocks:
            x, cls = block(x, cls, mask)
        return x
