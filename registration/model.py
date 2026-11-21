"""Registration model: SwinUNETR + velocity head, diffeomorphic integration and warping."""
import itertools
import types

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from monai.networks.blocks.dynunet_block import UnetResBlock
from monai.networks.nets import SwinUNETR
from monai.networks.nets import swin_unetr
from monai.networks.nets.swin_unetr import BasicLayer, WindowAttention, filter_swinunetr, window_partition
from monai.networks.utils import copy_model_state

SSL_WEIGHTS_URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
                   "download/0.8.1/ssl_pretrained_weights.pth")
TESTED_MONAI_VERSION = "1.6"


class SpatialTransformer3D(nn.Module):
    """Warps a volume by sampling it at x + u(x), with u expressed in voxels."""

    def __init__(self, size):
        super().__init__()
        axes = [torch.arange(s, dtype=torch.float32) for s in size]
        self.register_buffer("grid", torch.stack(torch.meshgrid(axes, indexing="ij")).unsqueeze(0),
                             persistent=False)
        scale = torch.tensor([s - 1 for s in size], dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, src, dvf, mode="bilinear"):
        pos = 2.0 * (self.grid + dvf) / self.scale - 1.0
        # grid_sample expects coordinates in (z, y, x) order with respect to the tensor axes
        pos = pos.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src.to(pos.dtype), pos, mode=mode, padding_mode="border", align_corners=True)


class ScalingAndSquaring(nn.Module):
    """Integrates a stationary velocity field into a diffeomorphic DVF."""

    def __init__(self, steps=7, use_checkpoint=False):
        super().__init__()
        self.steps = steps
        self.use_checkpoint = use_checkpoint

    def forward(self, v, warp_layer):
        v = v / (2 ** self.steps)
        for _ in range(self.steps):
            if self.use_checkpoint and torch.is_grad_enabled():
                v = v + checkpoint(warp_layer, v, v, use_reentrant=False)
            else:
                v = v + warp_layer(v, v)
        return v


class PretrainedSwinUNETR_Reg(nn.Module):
    """SwinUNETR backbone with concatenated (moving, fixed) input and a head predicting the velocity."""

    def __init__(self, in_channels=2, feature_size=48):
        super().__init__()
        self.backbone = SwinUNETR(spatial_dims=3, in_channels=in_channels, out_channels=16,
                                  feature_size=feature_size, use_checkpoint=True)
        self.reg_head = nn.Conv3d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.reg_head(self.backbone(x))


def _efficient_window_attention(self, x, mask):
    b, n, c = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    bias = self.relative_position_bias_table[self.relative_position_index[:n, :n].reshape(-1)]
    bias = bias.reshape(n, n, -1).permute(2, 0, 1).unsqueeze(0).to(q.dtype)
    dropout = self.attn_drop.p if self.training else 0.0
    out = []
    for start in range(0, b, self.window_chunk):
        stop = min(start + self.window_chunk, b)
        attn_bias = bias
        if mask is not None:
            windows = torch.arange(start, stop, device=x.device) % mask.shape[0]
            window_mask = mask[windows]
            if window_mask.dim() == 2:
                window_mask = window_mask.unsqueeze(1) - window_mask.unsqueeze(2)
                window_mask = window_mask.masked_fill(window_mask != 0, -100.0).masked_fill(window_mask == 0, 0.0)
            attn_bias = bias + window_mask.unsqueeze(1).to(q.dtype)
        out.append(F.scaled_dot_product_attention(q[start:stop], k[start:stop], v[start:stop], attn_mask=attn_bias,
                                                  dropout_p=dropout, scale=self.scale))
    x = torch.cat(out).transpose(1, 2).reshape(b, n, c)
    return self.proj_drop(self.proj(x))


def _window_region_ids(dims, window_size, shift_size, device):
    img_mask = torch.zeros((1, *dims, 1), device=device)
    cnt = 0
    regions = [(slice(-ws), slice(-ws, -ss), slice(-ss, None)) for ws, ss in zip(window_size, shift_size)]
    for index in itertools.product(*regions):
        img_mask[(slice(None), *index, slice(None))] = cnt
        cnt += 1
    return window_partition(img_mask, window_size).squeeze(-1)


def _compact_mask_forward(self, x):
    compute_mask = swin_unetr.compute_mask
    swin_unetr.compute_mask = _window_region_ids
    try:
        return type(self).forward(self, x)
    finally:
        swin_unetr.compute_mask = compute_mask


def set_window_attention(model, efficient=True, window_chunk=256):
    """
    efficient=True: window attention with scaled_dot_product_attention on groups of windows and a
    compact shifted-window mask, without the fp32 attention matrices; checkpoints unchanged.
    """
    if efficient and ".".join(monai.__version__.split(".")[:2]) != TESTED_MONAI_VERSION:
        print(f"[WARNING] Efficient attention verified with MONAI {TESTED_MONAI_VERSION}, found {monai.__version__}")
    for module in model.modules():
        if isinstance(module, (WindowAttention, BasicLayer)):
            if efficient:
                forward = _efficient_window_attention if isinstance(module, WindowAttention) else _compact_mask_forward
                module.window_chunk = window_chunk
                module.forward = types.MethodType(forward, module)
            else:
                module.__dict__.pop("forward", None)
    return model


def _checkpointed_forward(self, *args):
    if torch.is_grad_enabled():
        return checkpoint(type(self).forward, self, *args, use_reentrant=False)
    return type(self).forward(self, *args)


def _res_block_first_half(self, inp):
    return self.lrelu(self.norm1(self.conv1(inp)))


def _res_block_second_half(self, out, inp):
    out = self.norm2(self.conv2(out))
    residual = inp
    if hasattr(self, "conv3"):
        residual = self.conv3(residual)
    if hasattr(self, "norm3"):
        residual = self.norm3(residual)
    out += residual
    return self.lrelu(out)


def _segmented_res_block_forward(self, inp):
    if not torch.is_grad_enabled():
        return type(self).forward(self, inp)
    out = checkpoint(_res_block_first_half, self, inp, use_reentrant=False)
    return checkpoint(_res_block_second_half, self, out, inp, use_reentrant=False)


def set_activation_checkpointing(model, integrator=None, enabled=True, blocks=("encoder1", "decoder2", "decoder1"),
                                 segmented_blocks=()):
    """
    enabled=True: the given blocks and each squaring step are recomputed in the backward;
    segmented_blocks also split their residual block in two. Same result.
    """
    base = model.module if hasattr(model, "module") else model
    base = getattr(base, "_orig_mod", base)
    for name in blocks:
        module = getattr(base.backbone, name)
        if enabled:
            module.forward = types.MethodType(_checkpointed_forward, module)
        else:
            module.__dict__.pop("forward", None)
    for name in segmented_blocks:
        for module in getattr(base.backbone, name).modules():
            if isinstance(module, UnetResBlock):
                if enabled:
                    module.forward = types.MethodType(_segmented_res_block_forward, module)
                else:
                    module.__dict__.pop("forward", None)
    if integrator is not None:
        integrator.use_checkpoint = enabled
    return model


def _load_ssl_weights(model, in_channels):
    weights = torch.hub.load_state_dict_from_url(SSL_WEIGHTS_URL, map_location="cpu", progress=False)["model"]
    # The SSL patch embedding has 1 channel: it is adapted separately to the (moving, fixed) channels
    patch_weights = {k: weights.pop(k) for k in list(weights) if "patch_embed.proj" in k}
    copy_model_state(model.backbone, weights, filter_func=filter_swinunetr)

    proj = model.backbone.swinViT.patch_embed.proj
    with torch.no_grad():
        for k, v in patch_weights.items():
            if k.endswith("weight"):
                proj.weight.copy_(v.repeat(1, in_channels, *([1] * (v.ndim - 2))) / in_channels)
            elif k.endswith("bias") and proj.bias is not None:
                proj.bias.copy_(v)


def build_model(device, in_channels=2, feature_size=48, load_ssl=True, compile_model=False):
    """compile_model: torch.compile in place (checkpoints unchanged); the first iterations are slower."""
    model = PretrainedSwinUNETR_Reg(in_channels=in_channels, feature_size=feature_size)
    if load_ssl:
        try:
            _load_ssl_weights(model, in_channels)
            print("MONAI SSL weights loaded and adapted to 2 channels.")
        except Exception as e:
            print(f"[WARNING] SSL weights not loaded ({e}): the backbone starts from scratch.")
    model = model.to(device)
    if compile_model:
        model.compile()
    return model


def build_components(device, target_shape, integration_steps=9, feature_size=48, load_ssl=True,
                     compile_model=False):
    """Returns (model, warp_layer, integrator) ready on the device."""
    model = build_model(device, feature_size=feature_size, load_ssl=load_ssl, compile_model=compile_model)
    warp_layer = SpatialTransformer3D(size=target_shape).to(device)
    integrator = ScalingAndSquaring(steps=integration_steps).to(device)
    return model, warp_layer, integrator


def predict_dvf(model, integrator, warp_layer, moving, fixed, lcc_mask=None):
    """Full forward pass: velocity -> diffeomorphic DVF, optionally confined by the LCC mask."""
    velocity = model(torch.cat([moving, fixed], dim=1))
    dvf = integrator(velocity, warp_layer)
    if lcc_mask is not None:
        dvf = dvf * lcc_mask
    return dvf, velocity
