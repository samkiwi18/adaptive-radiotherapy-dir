"""General utilities: device, NIfTI I/O, resampling, collate, schedulers and checkpoints."""
import gc
import math
import os
import random
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

NON_STACKABLE_KEYS = {
    "lm_moving", "lm_fixed", "has_landmarks",
    "kp_moving", "kp_fixed", "has_keypoints",
    "lm_mov_orig", "lm_fix_orig", "spacing_orig", "shape_orig", "shape_orig_fixed",
}


def set_seed(seed):
    """Seeds Python, NumPy and PyTorch; the CUDA kernels of the TTO stay non-deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device in use: {device}")
    return device


def autocast_bf16(device):
    """bfloat16 mixed-precision context, valid on both GPU and CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


def extract_patient_id(filepath):
    name = os.path.basename(filepath)
    match = re.search(r"\d+", name)
    return match.group(0).zfill(4) if match else name


def read_nifti(path):
    """Returns (3D float32 volume, spacing in mm as a 3-element array)."""
    nii = nib.load(path)
    vol = nii.get_fdata(dtype=np.float32)
    if vol.ndim > 3:
        vol = vol[..., 0]
    spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
    if spacing.size < 3:
        spacing = np.ones(3, dtype=np.float32)
    return vol, spacing


def find_nifti(folder, base_name):
    """Looks for base_name.nii.gz or base_name.nii in the folder."""
    for extension in (".nii.gz", ".nii"):
        path = os.path.join(folder, base_name + extension)
        if os.path.exists(path):
            return path
    return None


def grid_scale(native_shape, network_shape):
    """Native -> network voxel factor with align_corners=True: i maps to i * (r - 1) / (n - 1), not i * r / n."""
    native = np.asarray(native_shape, dtype=np.float64)
    network = np.asarray(network_shape, dtype=np.float64)
    return (network - 1.0) / np.maximum(native - 1.0, 1.0)


def network_spacing(native_spacing, native_shape, network_shape):
    """Voxel spacing (mm) of the network grid obtained by resampling the native volume."""
    return np.asarray(native_spacing, dtype=np.float64) / grid_scale(native_shape, network_shape)


def resize(tensor, shape, mode="trilinear"):
    """3D resize of (B, C, X, Y, Z); 'nearest' uses the align_corners=True mapping, so labels stay aligned."""
    shape = tuple(int(s) for s in shape)
    if mode == "nearest":
        axes = [torch.linspace(-1.0, 1.0, s, device=tensor.device, dtype=tensor.dtype) for s in shape]
        # grid_sample takes the coordinates in (z, y, x) order with respect to the tensor axes
        grid = torch.stack(torch.meshgrid(axes, indexing="ij")[::-1], dim=-1)
        return F.grid_sample(tensor, grid.expand(tensor.shape[0], *grid.shape), mode="nearest", align_corners=True)
    align = True if mode in ("linear", "bilinear", "trilinear") else None
    return F.interpolate(tensor, size=shape, mode=mode, align_corners=align)


def volume_to_tensor(vol_np, shape, device, mode="trilinear"):
    t = torch.from_numpy(np.ascontiguousarray(vol_np, dtype=np.float32))[None, None].to(device)
    return resize(t, shape, mode)


def dvf_to_native_resolution(dvf, native_shape):
    """Network-grid DVF interpolated onto the native grid, components rescaled to native voxels."""
    network_shape = dvf.shape[2:]
    native_dvf = resize(dvf.float(), native_shape, "trilinear")
    factors = torch.tensor(1.0 / grid_scale(native_shape, network_shape),
                           device=dvf.device, dtype=torch.float32).view(1, 3, 1, 1, 1)
    return native_dvf * factors


def custom_collate(batch):
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key not in NON_STACKABLE_KEYS and isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def get_curriculum_weights(epoch, max_epochs, start_val=0.5, end_val=1.5):
    """Linear interpolation of a weight from start_val to end_val over the epochs."""
    progress = min(1.0, epoch / max(1, max_epochs - 1))
    return start_val + (end_val - start_val) * progress


def align_module_prefix(state_dict, model):
    """Adds or removes the 'module.' prefix (DataParallel) so the checkpoint matches the model."""
    ckpt_has_prefix = next(iter(state_dict)).startswith("module.")
    model_has_prefix = next(iter(model.state_dict())).startswith("module.")
    if ckpt_has_prefix and not model_has_prefix:
        return {k[len("module."):]: v for k, v in state_dict.items()}
    if model_has_prefix and not ckpt_has_prefix:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_state_dict_checked(model, state_dict, source):
    """load_state_dict(strict=False) that raises if no key matches and warns about skipped keys."""
    state_dict = align_module_prefix(state_dict, model)
    if not set(state_dict) & set(model.state_dict()):
        raise RuntimeError(f"No key of {source} matches the model: nothing would be loaded")
    result = model.load_state_dict(state_dict, strict=False)
    for kind, keys in (("missing", result.missing_keys), ("unexpected", result.unexpected_keys)):
        if keys:
            shown = ", ".join(keys[:5]) + (", ..." if len(keys) > 5 else "")
            print(f"[WARNING] {source}: {len(keys)} {kind} keys ({shown})")


_weights_cache = {}


def load_weights(model, path, device=None):
    """Loads a checkpoint (see load_state_dict_checked), kept in memory until the file changes on disk."""
    if not os.path.exists(path):
        print(f"[WARNING] Weights not found: {path}")
        return False
    stat = os.stat(path)
    version = (stat.st_mtime_ns, stat.st_size)
    cached = _weights_cache.get(path)
    if cached is None or cached[0] != version:
        _weights_cache.clear()
        cached = _weights_cache[path] = (version, torch.load(path, map_location="cpu"))
    load_state_dict_checked(model, cached[1], path)
    return True


def zero_init_output(model):
    """Initializes the last Conv3d to ~0 so that the initial DVF is close to the identity."""
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv3d):
            nn.init.normal_(module.weight, mean=0.0, std=1e-5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            break
    return model
