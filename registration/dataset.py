"""Learn2Reg dataset for intra-patient registration (expiration/inspiration pairs)."""
import hashlib
import os
import re
import shutil
import uuid
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import extract_patient_id, grid_scale, network_spacing, read_nifti, resize

MOVING_PHASE, FIXED_PHASE = "0001", "0000"
MOVING_TAGS = ("ex", "exh", "exp")
FIXED_TAGS = ("in", "inh", "insp")


def _find_nifti_files(folder):
    found = []
    if os.path.exists(folder):
        for root, _, files in os.walk(folder):
            found += [os.path.join(root, f) for f in files if f.endswith((".nii.gz", ".nii"))]
    return sorted(found)


def _name_without_extension(path):
    return os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")


def _phase_of(path):
    """MOVING_PHASE, FIXED_PHASE or None from the file name: whole-word tags, else the _0000/_0001 suffix."""
    name = _name_without_extension(path).lower()
    words = set(re.split(r"[_\-.\s]+", name))
    moving, fixed = bool(words & set(MOVING_TAGS)), bool(words & set(FIXED_TAGS))
    if moving and fixed:
        raise ValueError(f"Ambiguous respiratory phase in the file name: {os.path.basename(path)}")
    if moving or fixed:
        return MOVING_PHASE if moving else FIXED_PHASE
    match = re.search(r"_(0000|0001)$", name)
    return match.group(1) if match else None


def _normalize(vol):
    """Clipping to the 1st-99th percentiles and rescaling to [-1, 1]."""
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi)
    return (vol - lo) / (hi - lo + 1e-5) * 2.0 - 1.0


def _read_points(path):
    return pd.read_csv(path, header=None, engine="python").iloc[:, :3].to_numpy(dtype=np.float64)


def _file_version(path):
    stat = os.stat(path)
    return f"{os.path.abspath(path)}:{stat.st_mtime_ns}:{stat.st_size}"


def _to_grid(vol, shape, mode):
    return resize(torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None], shape, mode)[0]


def _compact_label(label):
    """Integer labels in 0-255 are stored as uint8 (4 times smaller than float32)."""
    if label.min() >= 0 and label.max() <= 255 and torch.equal(label, label.round()):
        return label.to(torch.uint8)
    return label


class Learn2RegDataset(Dataset):
    """
    Moving (expiration, _0001) / fixed (inspiration, _0000) pairs with points rescaled to target_shape.
    cache_dir keeps the resampled volumes, rebuilt when a source file changes.
    """

    CACHE_VERSION = 1

    def __init__(self, images_dir, masks_dir, landmarks_dir, keypoints_dir,
                 target_shape=(128, 128, 128), point_flip_axes=(0, 1), swap_point_phases=True, cache_dir=None):
        self.masks_dir = masks_dir
        self.landmarks_dir = landmarks_dir
        self.keypoints_dir = keypoints_dir
        self.target_shape = tuple(target_shape)
        self.point_flip_axes = tuple(point_flip_axes)
        self.swap_point_phases = swap_point_phases
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self.label_files = _find_nifti_files(masks_dir)
        self.moving_cases, self.fixed_cases, self.patient_per_pair = [], [], []

        groups = defaultdict(list)
        for f in _find_nifti_files(images_dir):
            groups[extract_patient_id(f)].append(f)

        for pid, files in sorted(groups.items()):
            mov_file = next((f for f in files if _phase_of(f) == MOVING_PHASE), None)
            fix_file = next((f for f in files if _phase_of(f) == FIXED_PHASE), None)
            if len(files) == 2:
                fix_file = fix_file or files[0]
                mov_file = mov_file or files[1]
            if mov_file and fix_file and mov_file != fix_file:
                self.moving_cases.append(mov_file)
                self.fixed_cases.append(fix_file)
                self.patient_per_pair.append(pid)
        self.label_pairs = [(self._find_label(m), self._find_label(f))
                            for m, f in zip(self.moving_cases, self.fixed_cases)]

    def __len__(self):
        return len(self.moving_cases)

    def _find_label(self, img_path):
        pid, phase = extract_patient_id(img_path), _phase_of(img_path)
        for lbl in self.label_files:
            if extract_patient_id(lbl) == pid and (phase is None or _phase_of(lbl) == phase):
                return lbl
        return None

    def point_paths(self, idx, folder):
        pid = extract_patient_id(self.moving_cases[idx])
        mov_phase = _phase_of(self.moving_cases[idx]) or MOVING_PHASE
        fix_phase = _phase_of(self.fixed_cases[idx]) or FIXED_PHASE
        if self.swap_point_phases:
            mov_phase, fix_phase = fix_phase, mov_phase
        return (os.path.join(folder, f"LungCT_{pid}_{mov_phase}.csv"),
                os.path.join(folder, f"LungCT_{pid}_{fix_phase}.csv"))

    def convert_points(self, points, native_shape):
        points = np.array(points, dtype=np.float64, copy=True)
        for axis in self.point_flip_axes:
            points[:, axis] = (native_shape[axis] - 1) - points[:, axis]
        return points

    def _cache_path(self, idx):
        sources = [self.moving_cases[idx], self.fixed_cases[idx], *self.label_pairs[idx]]
        key = repr((self.CACHE_VERSION, self.target_shape, [_file_version(p) if p else None for p in sources]))
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        name = f"{_name_without_extension(self.moving_cases[idx])}_{digest}.pt"
        return os.path.join(self.cache_dir, name)

    def _load_volumes(self, idx):
        """Normalized images and labels on the target grid, with native shapes and spacing."""
        cache_path = self._cache_path(idx) if self.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            try:
                return torch.load(cache_path)
            except Exception as e:
                print(f"[Dataset] Unreadable cache {cache_path} ({e}): rebuilding it")

        (mov_path, fix_path), (lbl_mov_path, lbl_fix_path) = self.pair_paths(idx), self.label_pairs[idx]
        vol_mov, spacing_mov = read_nifti(mov_path)
        vol_fix, _ = read_nifti(fix_path)
        lbl_mov = read_nifti(lbl_mov_path)[0] if lbl_mov_path else np.zeros_like(vol_mov)
        lbl_fix = read_nifti(lbl_fix_path)[0] if lbl_fix_path else np.zeros_like(vol_fix)
        data = {
            "moving": _to_grid(_normalize(vol_mov), self.target_shape, "trilinear"),
            "fixed": _to_grid(_normalize(vol_fix), self.target_shape, "trilinear"),
            "label_moving": _compact_label(_to_grid(lbl_mov, self.target_shape, "nearest")),
            "label_fixed": _compact_label(_to_grid(lbl_fix, self.target_shape, "nearest")),
            "shape_moving": tuple(int(s) for s in vol_mov.shape),
            "shape_fixed": tuple(int(s) for s in vol_fix.shape),
            "spacing_moving": torch.from_numpy(spacing_mov),
        }
        if cache_path:
            tmp = f"{cache_path}.{uuid.uuid4().hex}.tmp"
            torch.save(data, tmp)
            os.replace(tmp, cache_path)
        return data

    def pair_paths(self, idx):
        return self.moving_cases[idx], self.fixed_cases[idx]

    def _read_point_pair(self, idx, folder, shape_mov, shape_fix, kind, pid):
        """Points (native voxels, converted) of moving and fixed, or None if missing/unreadable."""
        path_m, path_f = self.point_paths(idx, folder)
        if not (os.path.exists(path_m) and os.path.exists(path_f)):
            return None
        try:
            return (self.convert_points(_read_points(path_m), shape_mov),
                    self.convert_points(_read_points(path_f), shape_fix))
        except Exception as e:
            print(f"[Dataset] Unreadable {kind} for {pid}: {e}")
            return None

    def __getitem__(self, idx):
        mov_name, fix_name = (_name_without_extension(p) for p in self.pair_paths(idx))
        data = self._load_volumes(idx)
        shape_mov, shape_fix = data["shape_moving"], data["shape_fixed"]
        spacing_mov = data["spacing_moving"].numpy()
        scale_mov = grid_scale(shape_mov, self.target_shape)
        scale_fix = grid_scale(shape_fix, self.target_shape)
        pid = extract_patient_id(mov_name)

        empty = torch.zeros((1, 3))
        item = {
            "moving": data["moving"].float(), "fixed": data["fixed"].float(),
            "label_moving": data["label_moving"].float(), "label_fixed": data["label_fixed"].float(),
            "spacing_mm": torch.from_numpy(network_spacing(spacing_mov, shape_mov, self.target_shape)).float(),
            "case_name": f"{mov_name}_to_{fix_name}",
            "lm_moving": empty, "lm_fixed": empty, "has_landmarks": False,
            "kp_moving": empty, "kp_fixed": empty, "has_keypoints": False,
            "lm_mov_orig": torch.zeros((1, 3)), "lm_fix_orig": torch.zeros((1, 3)),
            "spacing_orig": torch.from_numpy(spacing_mov).float(),
            "shape_orig": torch.tensor(shape_mov), "shape_orig_fixed": torch.tensor(shape_fix),
        }

        landmarks = self._read_point_pair(idx, self.landmarks_dir, shape_mov, shape_fix, "landmarks", pid)
        if landmarks is not None:
            lm_m, lm_f = landmarks
            item.update(lm_moving=torch.tensor(lm_m * scale_mov, dtype=torch.float32),
                        lm_fixed=torch.tensor(lm_f * scale_fix, dtype=torch.float32), has_landmarks=True,
                        lm_mov_orig=torch.from_numpy(lm_m).float(), lm_fix_orig=torch.from_numpy(lm_f).float())

        keypoints = self._read_point_pair(idx, self.keypoints_dir, shape_mov, shape_fix, "keypoints", pid)
        if keypoints is not None:
            kp_m, kp_f = keypoints
            item.update(kp_moving=torch.tensor(kp_m * scale_mov, dtype=torch.float32),
                        kp_fixed=torch.tensor(kp_f * scale_fix, dtype=torch.float32), has_keypoints=True)
        return item


class InvertedPair(Dataset):
    """Moving and fixed swapped; spacing_orig/shape_orig are kept, so both phases must share the native grid."""

    SWAPS = [("moving", "fixed"), ("label_moving", "label_fixed"), ("lm_moving", "lm_fixed"),
             ("kp_moving", "kp_fixed"), ("lm_mov_orig", "lm_fix_orig")]

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        if not torch.equal(item["shape_orig"], item["shape_orig_fixed"]):
            raise ValueError(f"{item['case_name']}: moving and fixed have different native grids, "
                             "the inverted pair would use the wrong spacing and shape")
        for a, b in self.SWAPS:
            item[a], item[b] = item[b], item[a]
        item["case_name"] += "_INV"
        return item


def copy_to_local(src, name, local_root):
    """Copies a folder to a local disk once per session and returns the local path."""
    dst = os.path.join(local_root, name)
    if not os.path.isdir(dst):
        tmp = dst + "_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"Copying {src} -> {dst} ...")
        shutil.copytree(src, tmp)
        os.rename(tmp, dst)
    return dst
