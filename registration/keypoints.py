"""Automatic 3D keypoint extraction: Förstner corners + NCC block matching."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, gaussian_filter, maximum_filter

from .utils import grid_scale


def forstner_corners_3d(volume, mask=None, sigma_deriv=1.0, sigma_int=2.0, w_thresh_percentile=70, q_thresh=1e-6,
                        min_distance=4, max_points=3000, n_blocks=4):
    vol = volume.astype(np.float32)
    grad = [gaussian_filter(vol, sigma=sigma_deriv, order=o) for o in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    S = {(i, j): gaussian_filter(grad[i] * grad[j], sigma=sigma_int) for i in range(3) for j in range(i, 3)}
    a, b, c = S[0, 0], S[1, 1], S[2, 2]
    d, e, f = S[0, 1], S[0, 2], S[1, 2]
    det = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)
    trace = a + b + c

    w = det / (trace + 1e-8)
    q = 4.0 * det / (trace * trace + 1e-8)
    region = np.ones(vol.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    w_pos = w[(w > 0) & region]
    if w_pos.size == 0:
        return np.zeros((0, 3), dtype=int)

    peaks = (region & (w > np.percentile(w_pos, w_thresh_percentile)) & (q > q_thresh)
             & (maximum_filter(w, size=min_distance) == w))
    coords = np.argwhere(peaks)

    if len(coords) > max_points:
        block_idx = np.clip((coords / (np.array(vol.shape) / n_blocks)).astype(int), 0, n_blocks - 1)
        block_id = block_idx[:, 0] * n_blocks**2 + block_idx[:, 1] * n_blocks + block_idx[:, 2]
        strength = w[tuple(coords.T)]
        per_block = max(1, max_points // n_blocks**3)
        selected = []
        for block in np.unique(block_id):
            idx = np.where(block_id == block)[0]
            selected.extend(idx[np.argsort(strength[idx])[::-1][:per_block]])
        coords = coords[selected]
    return coords


def block_match_3d(moving, fixed, kp_moving, patch_radius=4, search_radius=10, device="cuda", min_ncc=0.3,
                   chunk_size=256):
    """Displacement maximizing the patch NCC for each moving keypoint, one grouped convolution per chunk."""
    mov = torch.from_numpy(moving.astype(np.float32)).to(device)
    fix = torch.from_numpy(fixed.astype(np.float32)).to(device)
    D, H, W = mov.shape
    rp, rs = patch_radius, search_radius
    margin = rp + rs
    side = 2 * rp + 1
    n = side ** 3

    kp = np.asarray(kp_moving, dtype=np.int64).reshape(-1, 3)
    inside = np.all((kp >= margin) & (kp < np.array([D, H, W]) - margin), axis=1)
    kp = kp[inside]
    empty = (np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.int64), np.zeros(0))
    if len(kp) == 0:
        return empty

    patch_offsets = torch.arange(-rp, rp + 1, device=device)
    window_offsets = torch.arange(-margin, margin + 1, device=device)
    best_scores, best_indices = [], []
    for start in range(0, len(kp), chunk_size):
        z, y, x = torch.from_numpy(kp[start:start + chunk_size]).to(device).T
        patches = mov[(z[:, None, None, None] + patch_offsets[:, None, None]),
                      (y[:, None, None, None] + patch_offsets[None, :, None]),
                      (x[:, None, None, None] + patch_offsets[None, None, :])]
        windows = fix[(z[:, None, None, None] + window_offsets[:, None, None]),
                      (y[:, None, None, None] + window_offsets[None, :, None]),
                      (x[:, None, None, None] + window_offsets[None, None, :])]

        flat = patches.reshape(len(z), -1)
        patches = (patches - flat.mean(1).view(-1, 1, 1, 1)) / (flat.std(1).view(-1, 1, 1, 1) + 1e-6)

        # Pearson correlation between the zero-mean, unit-variance patch and each candidate
        # window, which is centred on its own mean too: sum((w - mean(w))^2) = sum(w^2) - sum(w)^2 / n
        corr = F.conv3d(windows[None], patches[:, None], groups=len(z))[0]
        sum_w = F.avg_pool3d(windows[None], kernel_size=side, stride=1)[0] * n
        sum_w2 = F.avg_pool3d(windows[None] ** 2, kernel_size=side, stride=1)[0] * n
        energy = torch.sqrt((sum_w2 - sum_w ** 2 / n).clamp(min=1e-6))
        ncc = corr / (energy * (n - 1) ** 0.5 + 1e-6)
        best, best_idx = ncc.flatten(1).max(dim=1)
        best_scores.append(best)
        best_indices.append(best_idx)

    scores = torch.cat(best_scores).cpu().numpy()
    indices = torch.cat(best_indices).cpu().numpy()
    keep = scores >= min_ncc
    if not keep.any():
        return empty
    size = 2 * rs + 1
    offset = np.stack(np.unravel_index(indices[keep], (size, size, size)), axis=1) - rs
    kp_m = kp[keep]
    return kp_m, kp_m + offset, scores[keep]


def save_paired_keypoints(kp_moving, kp_fixed, case_id, phase_moving, phase_fixed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path_m = os.path.join(out_dir, f"LungCT_{case_id}_{phase_moving}.csv")
    path_f = os.path.join(out_dir, f"LungCT_{case_id}_{phase_fixed}.csv")
    np.savetxt(path_m, kp_moving.astype(np.float64), delimiter=",", fmt="%.3f")
    np.savetxt(path_f, kp_fixed.astype(np.float64), delimiter=",", fmt="%.3f")
    print(f"[Keypoints] {len(kp_moving)} pairs saved: {path_m} <-> {path_f}")
    return path_m, path_f


def extract_dataset_keypoints(dataset, out_dir, device, skip_existing=True, mask_erosion=2, **matching_kwargs):
    """
    Saves the keypoints of every pair in native voxels, as the Dataset reads them.
    Returns (keypoints of the last case on the network grid, its moving volume).
    """
    last = (np.zeros((0, 3)), None)
    for idx in range(len(dataset)):
        item = dataset[idx]
        name = item["case_name"]
        if skip_existing and item["has_keypoints"]:
            print(f"{name}: keypoints already present, skipping")
            continue

        moving_np, fixed_np = item["moving"][0].numpy(), item["fixed"][0].numpy()

        lung = binary_erosion(item["label_moving"][0].numpy() > 0.5, iterations=mask_erosion)
        corners = forstner_corners_3d(moving_np, lung)
        if len(corners) == 0:
            print(f"{name}: no corners found")
            continue
        kp_m, kp_f, scores = block_match_3d(moving_np, fixed_np, corners, device=device, **matching_kwargs)
        if len(kp_m) == 0:
            print(f"{name}: no match above the NCC threshold")
            continue
        print(f"{name}: {len(corners)} corners -> {len(kp_m)} matches (mean NCC {scores.mean():.2f})")

        native_shape = item["shape_orig"].numpy()
        scale = 1.0 / grid_scale(native_shape, moving_np.shape)
        path_m, path_f = dataset.point_paths(idx, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(path_m, dataset.convert_points(kp_m * scale, native_shape), delimiter=",", fmt="%.3f")
        np.savetxt(path_f, dataset.convert_points(kp_f * scale, native_shape), delimiter=",", fmt="%.3f")
        print(f"[Keypoints] {len(kp_m)} pairs saved: {path_m} <-> {path_f}")
        last = (kp_m, moving_np)
    return last
