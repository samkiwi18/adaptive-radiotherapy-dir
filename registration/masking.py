"""Masks: LCC (patient body) to confine the loss, and structure masks on the network grid."""
import numpy as np
import torch
from scipy import ndimage

from .utils import volume_to_tensor


def build_lcc_mask(vol, hu_threshold=-800, sigma=1.5):
    """Body mask per Z slice: smoothing, threshold, closing, hole filling and largest component."""
    above_threshold = ndimage.gaussian_filter(vol, sigma=sigma) > hu_threshold
    mask = np.zeros(vol.shape, dtype=np.float32)
    structure = np.ones((5, 5))
    for z in range(vol.shape[2]):
        slice_ = ndimage.binary_fill_holes(ndimage.binary_closing(above_threshold[:, :, z], structure=structure))
        labels, n = ndimage.label(slice_)
        if n > 0:
            sizes = ndimage.sum(slice_, labels, range(1, n + 1))
            slice_ = labels == (np.argmax(sizes) + 1)
        mask[:, :, z] = slice_
    return mask


def _binarize_to_grid(vol_np, target_shape, device, mode):
    binary = (vol_np > 0.5).astype(np.float32)
    if mode == "nearest":
        return volume_to_tensor(binary, target_shape, device, "nearest")
    return (volume_to_tensor(binary, target_shape, device, "trilinear") > 0.5).float()


def structures_to_tensor(masks, names, target_shape, device, mode="trilinear", dtype=torch.float32):
    """Named (X, Y, Z) masks as a (1, len(names), *target_shape) tensor; bfloat16 is exact here."""
    return torch.cat([_binarize_to_grid(masks[n], target_shape, device, mode).to(dtype) for n in names], dim=1)


def lcc_mask_tensor(mask_np, target_shape, device, mode="nearest"):
    """Converts the LCC mask (numpy) into a tensor on the network grid; None if missing."""
    if mask_np is None:
        print("[WARNING] No LCC mask: optimization without confinement.")
        return None
    return _binarize_to_grid(mask_np, target_shape, device, mode)
