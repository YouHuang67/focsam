import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmseg.models.builder import BACKBONES
from engine.utils import rearrange, memory_efficient_attention
from engine.timers import Timer


class TwoLayerMLP(nn.Module):

    def __init__(self, embed_dim, mlp_dim, act_layer=nn.GELU):
        super(TwoLayerMLP, self).__init__()
        self.lin1 = nn.Linear(embed_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embed_dim)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in
            zip([input_dim] + dims, dims + [output_dim])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x) if i < self.num_layers - 1 else x
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).square().mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = rearrange(self.weight, 'c -> c () ()') * x + \
            rearrange(self.bias, 'c -> c () ()')
        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 kernel_size=(16, 16),
                 stride=(16, 16),
                 padding=(0, 0),
                 in_dim=3,
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(
            in_dim, embed_dim,
            kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x


class Attention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 qkv_bias=False,
                 use_rel_pos_embed=False,
                 input_size=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.use_rel_pos_embed = use_rel_pos_embed
        if self.use_rel_pos_embed:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if "
                    "using relative positional encoding.")
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        q, k, v = rearrange(
            self.qkv(x), 'b h w (n3 hn c) -> n3 (b hn) (h w) c',
            n3=3, hn=self.num_heads)

        if self.use_rel_pos_embed:
            rel_pos_bias = self.decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        else:
            rel_pos_bias = None

        x = memory_efficient_attention(
            q, k, v, attn_bias=rel_pos_bias, scale=self.scale)
        x = rearrange(x,
                      '(b hn) (h w) c -> b h w (hn c)',
                      b=B, hn=self.num_heads, h=H, w=W)
        x = self.proj(x)
        return x

    @staticmethod
    def get_rel_pos(q_size: int,
                    k_size: int,
                    rel_pos: torch.Tensor) -> torch.Tensor:
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos = rearrange(rel_pos, 'l c -> () c l')
            rel_pos_resized = F.interpolate(
                rel_pos, max_rel_dist, mode='linear'
            )
            rel_pos_resized = rearrange(rel_pos_resized, '() c l -> l c')
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if q & k shapes are different.
        q_coords = rearrange(torch.arange(q_size), 'q -> q ()')
        q_coords = q_coords * max(k_size / q_size, 1.0)
        k_coords = rearrange(torch.arange(k_size), 'k -> () k')
        k_coords = k_coords * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + \
                          (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos_resized[relative_coords.long()]

    @classmethod
    def decomposed_rel_pos(cls, q, rel_pos_h, rel_pos_w, q_size, k_size):
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = cls.get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = cls.get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = rearrange(q, 'b (qh qw) c -> b qh qw c', qh=q_h, qw=q_w)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        rel_h = rearrange(rel_h, 'b qh qw kh -> b qh qw kh ()')
        rel_w = rearrange(rel_w, 'b qh qw kw -> b qh qw () kw')
        rel_pos_bias = rearrange(rel_h + rel_w,
                                 'b qh qw kh kw -> b (qh qw) (kh kw)')
        return rel_pos_bias


class TransformerBlock(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 use_rel_pos_embed=False,
                 window_size=0,
                 input_size=None,
                 attn_cls=Attention):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = attn_cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos_embed=use_rel_pos_embed,
            input_size=(input_size if window_size == 0
                        else (window_size, window_size))
        )
        self.norm2 = norm_layer(embed_dim)
        self.mlp = TwoLayerMLP(embed_dim=embed_dim,
                               mlp_dim=int(embed_dim * mlp_ratio),
                               act_layer=act_layer)
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            ori_hw = x.size()[1:3]
            x, pad_hw = self.window_partition(x, self.window_size)
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = self.window_unpartition(x, self.window_size, pad_hw, ori_hw)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = rearrange(x, 'b (h Wh) (w Ww) c -> (b h w) Wh Ww c',
                      Wh=window_size, Ww=window_size)
        return x, (Hp, Wp)

    @staticmethod
    def window_unpartition(x, window_size, pad_hw, ori_hw):
        Hp, Wp = pad_hw
        H, W = ori_hw
        x = rearrange(x, '(b h w) Wh Ww c -> b (h Wh) (w Ww) c',
                      h=Hp // window_size, w=Wp // window_size)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


@BACKBONES.register_module()
class SAMWindowViT(BaseModule):

    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_dim=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 out_dim=256,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 use_abs_pos_embed=True,
                 use_rel_pos_embed=False,
                 window_size=0,
                 global_attn_indexes=(),
                 output_indices=(),
                 patch_embed_cls=PatchEmbed,
                 attn_cls=Attention,
                 block_cls=TransformerBlock,
                 freeze=False,
                 init_cfg=None,
                 pretrained=None):
        super(SAMWindowViT, self).__init__(init_cfg=init_cfg)
        self.img_size = img_size
        self.use_abs_pos_embed = use_abs_pos_embed
        self.global_attn_indexes = global_attn_indexes
        self.output_indices = output_indices
        self.patch_embed = patch_embed_cls(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_dim=in_dim, embed_dim=embed_dim)
        if use_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1,
                            img_size // patch_size,
                            img_size // patch_size,
                            embed_dim)
            )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(block_cls(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos_embed=use_rel_pos_embed,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                attn_cls=attn_cls))
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, bias=False),
            LayerNorm2d(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_dim))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

    @Timer('SAMEncoder')
    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed
            if tuple(pos_embed.size()[1:3]) != tuple(x.size()[1:3]):
                pos_embed = rearrange(pos_embed, 'b h w c -> b c h w')
                pos_embed = F.interpolate(
                    pos_embed, size=tuple(x.shape[1:3]),
                    mode='bilinear', align_corners=False)
                pos_embed = rearrange(pos_embed, 'b c h w -> b h w c')
            x = x + pos_embed
        stage_embeds = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.global_attn_indexes:
                stage_embeds.append(rearrange(x, 'b h w c -> b c h w'))
        image_embeds = self.neck(rearrange(x, 'b h w c -> b c h w'))
        return [stage_embeds[i] for i in self.output_indices] + [image_embeds]
