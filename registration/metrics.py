"""Metrics: TRE, Dice, Jacobian (folding and CT-ventilation)."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from .utils import grid_scale


def compute_tre(dvf, landmarks_fixed, landmarks_moving, spacing_mm):
    """TRE in mm. dvf: (3, X, Y, Z) numpy in voxels; landmarks (N, 3) in voxels of the same grid."""
    coords = landmarks_fixed.T
    displacement = np.stack([map_coordinates(dvf[c], coords, order=3, mode="nearest") for c in range(3)], axis=1)
    distances = np.linalg.norm((landmarks_fixed + displacement - landmarks_moving) * np.asarray(spacing_mm), axis=1)
    return float(distances.mean()), float(distances.std()), distances


def tre_on_network_grid(dvf, landmarks_fixed, landmarks_moving, spacing_mm, native_shape):
    """
    Fast TRE (mm) for validation: the network DVF is sampled at the landmarks mapped onto its grid
    instead of being upsampled as in compute_tre; the result differs only slightly.
    """
    device = dvf.device
    scale = torch.tensor(grid_scale(native_shape, dvf.shape[2:]), dtype=torch.float32, device=device)
    lm_f = torch.as_tensor(landmarks_fixed, dtype=torch.float32, device=device)
    lm_m = torch.as_tensor(landmarks_moving, dtype=torch.float32, device=device)
    points = lm_f * scale
    shape = torch.tensor(dvf.shape[2:], dtype=torch.float32, device=device)
    grid = (2.0 * points / (shape - 1.0) - 1.0)[:, [2, 1, 0]].view(1, 1, 1, -1, 3)
    displacement = F.grid_sample(dvf.float(), grid, padding_mode="border", align_corners=True)[0, :, 0, 0].T / scale
    spacing = torch.as_tensor(spacing_mm, dtype=torch.float32, device=device)
    return float(torch.linalg.norm((lm_f + displacement - lm_m) * spacing, dim=1).mean())


def _determinant(J):
    return (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))


def jacobian_map(dvf):
    """Determinant of J = I + grad(u) for each voxel. dvf: (3, X, Y, Z) numpy."""
    g = [np.gradient(dvf[c], axis=(0, 1, 2)) for c in range(3)]
    return _determinant([[g[r][c] + (1.0 if r == c else 0.0) for c in range(3)] for r in range(3)])


def compute_jacobian_determinant(dvf):
    """(mean det(J), % of voxels with det(J) <= 0); dvf (3, X, Y, Z) numpy array or tensor."""
    if isinstance(dvf, torch.Tensor):
        det = jacobian_torch(dvf[None].float())
        return float(det.mean()), float((det <= 0).float().mean() * 100.0)
    det = jacobian_map(dvf)
    return float(det.mean()), float((det <= 0).mean() * 100.0)


def jacobian_torch(dvf):
    """Same as jacobian_map but on a (1, 3, X, Y, Z) tensor; returns (X, Y, Z)."""
    g = [torch.gradient(dvf[0, c], dim=(0, 1, 2)) for c in range(3)]
    return _determinant([[g[r][c] + (1.0 if r == c else 0.0) for c in range(3)] for r in range(3)])


def multiclass_dice(pred_mask, target_mask):
    scores = []
    for c in torch.unique(target_mask):
        if c == 0:
            continue
        p, t = pred_mask == c, target_mask == c
        denominator = p.sum() + t.sum()
        if denominator > 0:
            scores.append((2.0 * (p & t).sum().float() / denominator).item())
    return float(np.mean(scores)) if scores else 0.0


def binary_dice(a, b):
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else 0.0
