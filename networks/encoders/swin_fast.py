""" MatteFormer
Copyright (c) 2022-present NAVER Webtoon
Apache-2.0
"""

import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from networks.ops import SpectralNorm
import os
from einops import rearrange, repeat, reduce
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class PatchAggregation(nn.Module):
    """down sample the feature resolution, build with conv 2x2 stride 2
    """

    def __init__(self, in_channel, out_channel, kernel_size=2, stride_size=2):
        super(PatchAggregation, self).__init__()
        self.patch_size=2
        self.patch_aggregation = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=make_pairs(kernel_size),
            stride=make_pairs(stride_size)
        )

    def forward(self, x):
        _, _, H, W = x.size()

        # padding
        if W % self.patch_size != 0:
            x = F.pad(x, (0, 2 - W % 2))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, 2 - H % 2))

        x = self.patch_aggregation(x)
        return x
class ConvGeluBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride_size, padding=1):
        """build the conv3x3 + gelu + bn module
        """
        super(ConvGeluBN, self).__init__()
        self.kernel_size = make_pairs(kernel_size)
        self.stride_size = make_pairs(stride_size)
        self.padding_size = make_pairs(padding)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv3x3_gelu_bn = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.kernel_size,
                      stride=self.stride_size,
                      padding=self.padding_size),
            nn.GELU(),
            nn.BatchNorm2d(self.out_channel)
        )

    def forward(self, x):

        x = self.conv3x3_gelu_bn(x)
        return x

def make_pairs(x):
    """make the int -> tuple
    """
    return x if isinstance(x, tuple) else (x, x)
class CMTStem(nn.Module):
    """make the model conv stem module
    """
    def __init__(self, kernel_size, in_channel, out_channel, layers_num):
        super(CMTStem, self).__init__()
        self.layers_num = layers_num
        self.patch_size=2
        self.conv3x3_gelu_bn_downsample = ConvGeluBN(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            stride_size=make_pairs(2)
            )
        self.conv3x3_gelu_bn_list = nn.ModuleList(
            [ConvGeluBN(kernel_size=kernel_size, in_channel=out_channel, out_channel=out_channel, stride_size=1) for _ in range(self.layers_num)]
        )

    def forward(self, x):
        _, _, H, W = x.size()

        # padding
        if W % self.patch_size != 0:
            x = F.pad(x, (0, 2 - W % 2))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, 2 - H % 2))
        x = self.conv3x3_gelu_bn_downsample(x)
        for i in range(self.layers_num):
            x = self.conv3x3_gelu_bn_list[i](x)
        return x


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        forward='split_cat'
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x) :
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x)  :
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """


    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    """Forward function."""
    def forward(self, x):

        _, _, H, W = x.size()

        # padding
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class PAWSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) , num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        # params
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, _inp_shape=None):
        # get shape
        B_, N, C = x.shape
        B, H, W, C = _inp_shape

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1] , -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # qkv projection for x
        qkv = self.qkv(x).reshape(B_, N , 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q[:, :, :self.window_size[0] * self.window_size[1], :] * self.scale

        # get self-attention features
        attn = (q @ k.transpose(-2, -1))
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N ) .contiguous().unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N,  N )
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PASTBlock(nn.Module):
    def __init__(self, layer_idx, block_idx, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio


        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = PAWSA(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    # swin transformer block
    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        # shortcut features
        shortcut = x


        # forward
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b)) # [1, 259, 518, 96]
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # foward attention layer
        attn_windows = self.attn(x_windows, mask=attn_mask, _inp_shape=shifted_x.shape)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN : second module
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):

    def __init__(self, layer_idx,
                 dim, depth, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.layer_idx=layer_idx
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.PyConv=Partial_conv3(dim,4)

        # build blocks
        self.blocks = nn.ModuleList([
            PASTBlock(
                layer_idx=layer_idx,
                block_idx=i,
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def _make_shortcut(self, inplane, planes, norm_layer=nn.BatchNorm2d):
        return nn.Sequential(
            nn.Linear(inplane, planes, bias=False),
            nn.ReLU(inplace=True),
        )

    # basic layer
    def forward(self, x, H, W):

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        # img_mask -> index map (0~8) # [1, 259, 518, 1]
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # get attn_mask
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # forward blocks

        for b, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                shortcut = x
                x = blk(x, attn_mask)
                x = rearrange(x, 'b ( h w ) c -> b c h w', h=H, w=W)
                x = self.PyConv(x)
                x = rearrange(x, 'b c h w -> b ( h w ) c ')
                x = shortcut + x

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class MatteFormer(nn.Module):

    def __init__(self,
                 patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 patch_norm=True, out_indices=(0, 1, 2, 3), use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        # self.patch_embed = PatchEmbed(
        #     patch_size=patch_size, in_chans=in_chans + 3, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        self.stem = CMTStem(
            kernel_size=3,
            in_channel=6,
            out_channel=32,
            layers_num=2
        )
        self.pool1 = PatchAggregation(in_channel=32, out_channel=96)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                layer_idx = i_layer,
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,

            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output [from Swin-Transformer-Semantic-Segmentation]
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # add shortcut layer [from MG-Matting]
        self.shortcut = nn.ModuleList()
        shortcut_inplanes = [[6, 32], [32, 32], [96, 64], [192, 128], [384, 256], [768, 512]]
        for shortcut in shortcut_inplanes:
            inplane, planes = shortcut
            self.shortcut.append(self._make_shortcut(inplane=inplane, planes=planes, norm_layer=nn.BatchNorm2d))

    def _make_shortcut(self, inplane, planes, norm_layer=nn.BatchNorm2d):
        '''
        came from MGMatting
        '''
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            norm_layer(planes)
        )

    def forward(self, x, trimapmask, sampleidx=None):
        # set outputs
        outs = []

        # get outs[0]
        outs.append(self.shortcut[0](x))
        x=self.stem(x)
        # x = self.patch_embed(x)

        # outs.append(self.shortcut[1](F.upsample_bilinear(x, scale_factor=2.0)))
        outs.append(self.shortcut[1](x))
        x=self.pool1(x)
        _, _, Wh, Ww = x.shape
        # dropout
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        # get outs[2~5]
        for i in range(self.num_layers):
            layer = self.layers[i]

            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                out = self.shortcut[i+2](out)
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super(MatteFormer, self).train(mode)


if __name__ == '__main__':
    model = MatteFormer().cuda()
    # print(model)

    out = model(torch.ones(2, 6, 451, 411).cuda(), torch.ones(2,3,512,512).cuda())
    print(len(out))
    for outs in out:
        print(outs.shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    # 打印模型的参数量
    print(f"Model parameters: {num_params}")
    # for key, value in model.items():
    #     print(f"{key}: {value.shape}")

    print('MODEL DEBUG')


