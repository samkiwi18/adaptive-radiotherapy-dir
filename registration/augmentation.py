"""
Training augmentation applied to the batch directly on the GPU: flip, rigid motion (images,
labels and points kept consistent) and intensity transforms. The intensity transforms
reproduce MONAI's RandGaussianNoise, RandAdjustContrast, RandBiasField and
RandSimulateLowResolution, but run on the GPU tensors instead of NumPy on the CPU.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F

POINT_KEYS = (("lm_moving", "lm_fixed", "has_landmarks"), ("kp_moving", "kp_fixed", "has_keypoints"))


def random_rigid(max_rot, max_trans_vox):
    """Random rotation (angles in rad about the three axes) and translation (voxels)."""
    a0, a1, a2 = (random.uniform(-max_rot, max_rot) for _ in range(3))
    c0, s0, c1, s1, c2, s2 = np.cos(a0), np.sin(a0), np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2)
    Ra = torch.tensor([[1, 0, 0], [0, c0, -s0], [0, s0, c0]], dtype=torch.float32)
    Rb = torch.tensor([[c1, 0, s1], [0, 1, 0], [-s1, 0, c1]], dtype=torch.float32)
    Rc = torch.tensor([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]], dtype=torch.float32)
    t = torch.tensor([random.uniform(-max_trans_vox, max_trans_vox) for _ in range(3)], dtype=torch.float32)
    return Rc @ Rb @ Ra, t


def apply_rigid(img, lbl, points, rigid):
    """Rotation + translation about the centre of a (C, X, Y, Z) image, its label and (N, 3) points."""
    shape, device = img.shape[1:], img.device
    R, t = rigid[0].to(device), rigid[1].to(device)
    center = torch.tensor(shape, dtype=torch.float32, device=device) / 2.0
    R_inv = R.T
    axes = [torch.arange(s, dtype=torch.float32, device=device) for s in shape]
    grid = torch.stack(torch.meshgrid(axes, indexing="ij")).reshape(3, -1)

    coord = (R_inv @ (grid - center[:, None]) - (R_inv @ t)[:, None] + center[:, None]).reshape(3, *shape)
    scale = torch.tensor([s - 1 for s in shape], dtype=torch.float32, device=device).view(3, 1, 1, 1)
    coord = (2.0 * coord / scale - 1.0).permute(1, 2, 3, 0)[..., [2, 1, 0]].unsqueeze(0)

    img = F.grid_sample(img[None], coord, mode="bilinear", padding_mode="border", align_corners=True)[0]
    lbl = F.grid_sample(lbl[None], coord, mode="nearest", padding_mode="border", align_corners=True)[0]
    return img, lbl, [(p - center) @ R.T + t + center for p in points]


def gaussian_noise(img, std=0.05):
    """As RandGaussianNoise(mean=0, std): noise std drawn uniformly in [0, std]."""
    return img + torch.randn_like(img) * random.uniform(0.0, std)


def adjust_contrast(img, gamma=(0.7, 1.5)):
    """As RandAdjustContrast(gamma): gamma correction over the image min-max range."""
    g = random.uniform(*gamma)
    img_min = img.min()
    img_range = img.max() - img_min
    return ((img - img_min) / (img_range + 1e-7)) ** g * img_range + img_min


def _legendre(x, degree):
    """Legendre polynomials P_0..P_degree evaluated at x, shape (degree + 1, len(x))."""
    P = [torch.ones_like(x), x]
    for n in range(1, degree):
        P.append(((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1))
    return torch.stack(P[:degree + 1])


def bias_field(img, degree=3, coeff_range=(0.0, 0.1)):
    """As RandBiasField(degree, coeff_range): multiplies the image by exp of a random Legendre polynomial."""
    terms = [(i, j, k) for i in range(degree + 1) for j in range(degree + 1 - i) for k in range(degree + 1 - i - j)]
    coeff = torch.zeros((degree + 1,) * 3, dtype=torch.float32, device=img.device)
    for (i, j, k) in terms:
        coeff[i, j, k] = random.uniform(*coeff_range)
    P = [_legendre(torch.linspace(-1.0, 1.0, s, device=img.device), degree) for s in img.shape[1:]]
    field = torch.einsum("ijk,ix,jy,kz->xyz", coeff, *P)
    return img * torch.exp(field)


def simulate_low_resolution(img, zoom_range=(0.5, 1.0)):
    """As RandSimulateLowResolution(zoom_range): nearest downsampling, then trilinear upsampling."""
    zoom = random.uniform(*zoom_range)
    shape = img.shape[1:]
    low = F.interpolate(img[None], size=tuple(max(1, int(round(s * zoom))) for s in shape), mode="nearest")
    return F.interpolate(low, size=tuple(shape), mode="trilinear", align_corners=False)[0]


INTENSITY_TRANSFORMS = (gaussian_noise, adjust_contrast, bias_field, simulate_low_resolution)


class RegistrationAugmentation:
    """
    Random Y flip, rigid motion (the same for moving and fixed if same_affine) and intensity
    transforms; called on each training batch, returns a new batch on device.
    """

    def __init__(self, flip_prob=0.5, affine_prob=0.8, max_rot=0.05, max_trans_vox=4.0,
                 same_affine=False, intensity_prob=0.3):
        self.flip_prob = flip_prob
        self.affine_prob = affine_prob
        self.max_rot = max_rot
        self.max_trans_vox = max_trans_vox
        self.same_affine = same_affine
        self.intensity_prob = intensity_prob

    @torch.no_grad()
    def __call__(self, batch, device):
        out = dict(batch)
        images = {k: batch[k].to(device, non_blocking=True).float()
                  for k in ("moving", "fixed", "label_moving", "label_fixed")}
        samples = {k: [] for k in images}
        points_out = {k: list(batch[k]) for keys in POINT_KEYS for k in keys[:2]}

        for i in range(images["moving"].shape[0]):
            points = {}
            for key_m, key_f, flag in POINT_KEYS:
                if batch[flag][i]:
                    points[key_m] = batch[key_m][i].to(device).float().clone()
                    points[key_f] = batch[key_f][i].to(device).float().clone()
            augmented = self.augment_sample(*(images[k][i] for k in samples), points)
            for k, v in zip(samples, augmented[:4]):
                samples[k].append(v)
            for k, p in augmented[4].items():
                points_out[k][i] = p

        out.update({k: torch.stack(v) for k, v in samples.items()})
        out.update(points_out)
        return out

    def augment_sample(self, mov, fix, lbl_mov, lbl_fix, points):
        """(C, X, Y, Z) tensors; points: dict of (N, 3) tensors whose keys end in 'moving' or 'fixed'."""
        if random.random() > 1.0 - self.flip_prob:
            # Flip along Y (dim 2 of the (C, X, Y, Z) tensor), consistent with column 1 of the points
            mov, fix = torch.flip(mov, [2]), torch.flip(fix, [2])
            lbl_mov, lbl_fix = torch.flip(lbl_mov, [2]), torch.flip(lbl_fix, [2])
            for p in points.values():
                p[:, 1] = (mov.shape[2] - 1) - p[:, 1]

        shared_rigid = None
        if self.same_affine and random.random() < self.affine_prob:
            shared_rigid = random_rigid(self.max_rot, self.max_trans_vox)
        for side in ("moving", "fixed"):
            if self.same_affine:
                rigid = shared_rigid
            else:
                rigid = random_rigid(self.max_rot, self.max_trans_vox) if random.random() < self.affine_prob else None
            if rigid is not None:
                keys = [k for k in points if k.endswith(side)]
                if side == "moving":
                    mov, lbl_mov, new = apply_rigid(mov, lbl_mov, [points[k] for k in keys], rigid)
                else:
                    fix, lbl_fix, new = apply_rigid(fix, lbl_fix, [points[k] for k in keys], rigid)
                points.update(zip(keys, new))

        for transform in INTENSITY_TRANSFORMS:
            if random.random() < self.intensity_prob:
                mov = transform(mov)
            if random.random() < self.intensity_prob:
                fix = transform(fix)
        return mov, fix, lbl_mov, lbl_fix, points
