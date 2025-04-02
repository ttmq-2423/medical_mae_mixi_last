import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from model.swin_transformer_v2 import SwinTransformerBlock

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class SwinStage(nn.Module):
    """Adapter class để sử dụng SwinTransformerBlock trong ConvNeXt"""
    def __init__(self, blocks, resolution):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.resolution = resolution

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        for block in self.blocks:
            x = block(x)

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class ConvSwin(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 window_size=7, swin_stage3_heads=12, swin_stage4_heads=24,
                 input_size=224, mlp_ratio=4.0):
        super().__init__()

        self.window_size = window_size
        self.input_size = input_size

        patch_size = 4
        # Calculate resolutions for stages 3 and 4
        self.stage3_resolution = input_size // patch_size // 2 // 2
        self.stage4_resolution = self.stage3_resolution // 2

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Stages 1-2: ConvNeXt Blocks
        for i in range(2):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Stage 3: Swin Transformer Blocks
        # Ensure resolution is compatible with window size
        assert self.stage3_resolution % window_size == 0, f"Stage3 resolution {self.stage3_resolution} must be divisible by window_size {window_size}"
        assert dims[2] % swin_stage3_heads == 0, f"Dimension {dims[2]} must be divisible by num_heads {swin_stage3_heads}"

        swin_stage3_blocks = []
        for j in range(depths[2]):
            block = SwinTransformerBlock(
                dim=dims[2],
                input_resolution=(self.stage3_resolution, self.stage3_resolution),
                num_heads=swin_stage3_heads,
                window_size=window_size,
                shift_size=0 if (j % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=dp_rates[cur + j],
                norm_layer=nn.LayerNorm,
                pretrained_window_size=0
            )
            swin_stage3_blocks.append(block)
        
        self.stages.append(SwinStage(swin_stage3_blocks, self.stage3_resolution))
        cur += depths[2]

        # Stage 4: Swin Transformer Blocks
        assert self.stage4_resolution % window_size == 0, f"Stage4 resolution {self.stage4_resolution} must be divisible by window_size {window_size}"
        assert dims[3] % swin_stage4_heads == 0, f"Dimension {dims[3]} must be divisible by num_heads {swin_stage4_heads}"

        swin_stage4_blocks = []
        for j in range(depths[3]):
            block = SwinTransformerBlock(
                dim=dims[3],
                input_resolution=(self.stage4_resolution, self.stage4_resolution),
                num_heads=swin_stage4_heads,
                window_size=window_size,
                shift_size=0 if (j % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=dp_rates[cur + j],
                norm_layer=nn.LayerNorm,
                pretrained_window_size=0
            )
            swin_stage4_blocks.append(block)
        
        self.stages.append(SwinStage(swin_stage4_blocks, self.stage4_resolution))
        cur += depths[3]

        # Final norm layer, pooling và classifier head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # Khởi tạo trọng số
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Global pooling và normalization
        x_pooled = x.mean([-2, -1])  # [B, C]
        return self.norm(x_pooled)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def convnext_swin_tiny_34(**kwargs):
    model = ConvSwin(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                         window_size=7, swin_stage3_heads=12, swin_stage4_heads=24, **kwargs)
    return model

@register_model
def convnext_swin_small_34(**kwargs):
    model = ConvSwin(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],
                         window_size=7, swin_stage3_heads=12, swin_stage4_heads=24, **kwargs)
    return model

@register_model
def convnext_swin_base_34(**kwargs):
    model = ConvSwin(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                         window_size=7, swin_stage3_heads=16, swin_stage4_heads=32, **kwargs)
    return model

@register_model
def convnext_swin_large_34(**kwargs):
    model = ConvSwin(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536],
                         window_size=7, swin_stage3_heads=24, swin_stage4_heads=48, **kwargs)
    return model

@register_model
def convnext_swin_xlarge_34(**kwargs):
    model = ConvSwin(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048],
                         window_size=7, swin_stage3_heads=32, swin_stage4_heads=64, **kwargs)
    return model