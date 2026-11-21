"""Registration losses: local NCC, MIND, DVF regularization, Dice and point-based losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.networks.utils import one_hot


def box_sum(x, win):
    """Sum over a win^3 window (zero padding) as three 1D convolutions: same result, 3 * win operations."""
    channels, pad = x.shape[1], win // 2
    kernel = torch.ones((channels, 1, win), dtype=x.dtype, device=x.device)
    x = F.conv3d(x, kernel.view(channels, 1, win, 1, 1), padding=(pad, 0, 0), groups=channels)
    x = F.conv3d(x, kernel.view(channels, 1, 1, win, 1), padding=(0, pad, 0), groups=channels)
    return F.conv3d(x, kernel.view(channels, 1, 1, 1, win), padding=(0, 0, pad), groups=channels)


class NCCLoss(nn.Module):
    """Local normalized cross-correlation (win^3 window), optionally masked."""

    def __init__(self, win=7, eps=1e-5):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, y_pred, y_true, mask=None):
        # Local variances are differences of large sums: in bfloat16 they lose precision
        with torch.autocast(device_type=y_true.device.type, enabled=False):
            return self._ncc(y_pred.float(), y_true.float(), mask)

    def _ncc(self, J, I, mask):
        n = self.win ** 3
        sums = box_sum(torch.cat([I, J, I * I, J * J, I * J], dim=1), self.win)
        I_sum, J_sum, I2_sum, J2_sum, IJ_sum = sums.split(I.shape[1], dim=1)
        u_I, u_J = I_sum / n, J_sum / n
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * n
        I_var = torch.clamp(I2_sum - 2 * u_I * I_sum + u_I * u_I * n, min=self.eps)
        J_var = torch.clamp(J2_sum - 2 * u_J * J_sum + u_J * u_J * n, min=self.eps)
        cc = cross * cross / (I_var * J_var + self.eps)

        if mask is not None:
            mask = mask.float()
            return -(cc * mask).sum() / mask.sum().clamp(min=1.0)
        return -cc.mean()


class MINDLoss(nn.Module):
    """MIND descriptor (self-similarity over 12 neighbours), robust to intensity changes."""

    SHIFTS = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0),
              (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0)]

    def __init__(self, patch_size=3, h=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.h = h

    def descriptor(self, img):
        k, (X, Y, Z) = self.patch_size, img.shape[2:]
        # Neighbour at offset s = voxel x - s, with the border replicated (torch.roll would wrap it around)
        padded = F.pad(img, (1,) * 6, mode="replicate")
        diffs = torch.cat([(img - padded[:, :, 1 - a:1 - a + X, 1 - b:1 - b + Y, 1 - c:1 - c + Z]) ** 2
                           for a, b, c in self.SHIFTS], dim=1)
        channels = diffs.shape[1]
        weight = torch.ones((channels, 1, k, k, k), dtype=img.dtype, device=img.device) / k**3
        distances = F.conv3d(diffs, weight, padding=k // 2, groups=channels)
        variance = distances.mean(dim=1, keepdim=True) + 1e-5
        return torch.exp(-distances / (variance * self.h**2))

    def error_map(self, y_pred, y_true, true_descriptor=None):
        """Voxel-wise MIND error (B, 1, X, Y, Z); true_descriptor avoids recomputing descriptor(y_true)."""
        if true_descriptor is None:
            true_descriptor = self.descriptor(y_true)
        return ((self.descriptor(y_pred) - true_descriptor) ** 2).mean(dim=1, keepdim=True)

    def forward(self, y_pred, y_true):
        return self.error_map(y_pred, y_true).mean()


def _finite_differences(dvf):
    d0 = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
    d1 = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
    d2 = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]
    return d0, d1, d2


class GradientSmoothingLoss(nn.Module):
    """Mean squared gradient of the DVF (diffusion regularizer)."""

    def forward(self, dvf):
        return sum(torch.mean(d ** 2) for d in _finite_differences(dvf.float())) / 3.0


def smoothness_map(dvf):
    """Voxel-wise version of the regularizer, shape (B, 1, X, Y, Z)."""
    d0, d1, d2 = _finite_differences(dvf)
    d0 = F.pad(d0, (0, 0, 0, 0, 0, 1))
    d1 = F.pad(d1, (0, 0, 0, 1, 0, 0))
    d2 = F.pad(d2, (0, 1, 0, 0, 0, 0))
    return ((d0 ** 2 + d1 ** 2 + d2 ** 2) / 3.0).mean(dim=1, keepdim=True)


def soft_dice_loss(pred, target, eps=1e-5):
    target = target.to(pred.dtype)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    total = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    return 1.0 - ((2.0 * intersection + eps) / (total + eps)).mean()


def _point_error_mm(dvf, pt_moving, pt_fixed, spacing_mm):
    """Error (in mm) between the fixed points displaced by the DVF and the corresponding moving points."""
    shape = torch.tensor(dvf.shape[2:], device=dvf.device, dtype=torch.float32)
    grid = (2.0 * pt_fixed / (shape - 1.0) - 1.0)[..., [2, 1, 0]].unsqueeze(1).unsqueeze(1)
    displacement = F.grid_sample(dvf, grid, align_corners=True)[:, :, 0, 0].permute(0, 2, 1)
    return (pt_fixed + displacement - pt_moving) * spacing_mm.view(1, 1, 3)


def landmark_loss(dvf, pt_moving, pt_fixed, spacing_mm):
    return torch.norm(_point_error_mm(dvf, pt_moving, pt_fixed, spacing_mm), dim=-1).mean()


def keypoint_loss(dvf, pt_moving, pt_fixed, spacing_mm):
    error = _point_error_mm(dvf, pt_moving, pt_fixed, spacing_mm)
    return F.smooth_l1_loss(error, torch.zeros_like(error))


def batch_point_loss(dvf, batch, kind, device):
    """Mean point loss ('lm' or 'kp') over the samples that have points; None if none has."""
    flag, key_m, key_f = (("has_landmarks", "lm_moving", "lm_fixed") if kind == "lm"
                          else ("has_keypoints", "kp_moving", "kp_fixed"))
    loss_fn = landmark_loss if kind == "lm" else keypoint_loss
    losses = [
        loss_fn(dvf[i:i + 1], batch[key_m][i].to(device)[None], batch[key_f][i].to(device)[None],
                batch["spacing_mm"][i].to(device))
        for i in range(dvf.shape[0]) if batch[flag][i]
    ]
    return torch.stack(losses).mean() if losses else None


class SuperRegistrationLoss(nn.Module):
    """
    Multiscale NCC + MIND, DVF regularization and label Dice. differentiable_dice=True: soft Dice
    on bilinear-warped masks, the only one that drives the DVF (warp labels with warp_label).
    """

    def __init__(self, lambda_smooth=1.5, lambda_label=0.4, lambda_mind=0.5, differentiable_dice=False):
        super().__init__()
        self.differentiable_dice = differentiable_dice
        self.ncc = NCCLoss(win=7)
        self.mind = MINDLoss()
        self.smooth = GradientSmoothingLoss()
        self.dice = DiceLoss(softmax=False, squared_pred=True)
        self.lambda_smooth = lambda_smooth
        self.lambda_label = lambda_label
        self.lambda_mind = lambda_mind

    def forward(self, warped_img, fixed_img, dvf, warped_lbl, fixed_lbl):
        mask_100 = (fixed_lbl > 0).float()
        w_50 = F.interpolate(warped_img, scale_factor=0.5, mode="trilinear", align_corners=True)
        f_50 = F.interpolate(fixed_img, scale_factor=0.5, mode="trilinear", align_corners=True)
        mask_50 = F.interpolate(mask_100, scale_factor=0.5, mode="nearest")

        loss_ncc = (self.ncc(warped_img, fixed_img, mask_100) + self.ncc(w_50, f_50, mask_50)) / 2.0
        loss_mind = (self.mind(warped_img, fixed_img) + self.mind(w_50, f_50)) / 2.0
        loss_sim = loss_ncc + self.lambda_mind * loss_mind
        loss_reg = self.smooth(dvf)

        loss_lbl = torch.zeros((), device=dvf.device)
        if self.differentiable_dice:
            loss_lbl = soft_dice_loss(warped_lbl.float(), (fixed_lbl > 0).float())
        elif fixed_lbl.sum() > 0:
            n_classes = int(max(fixed_lbl.max().item(), warped_lbl.max().item())) + 1
            loss_lbl = self.dice(one_hot(warped_lbl.long(), n_classes), one_hot(fixed_lbl.long(), n_classes))

        total = loss_sim + self.lambda_smooth * loss_reg + self.lambda_label * loss_lbl
        return total, loss_sim, loss_reg, loss_lbl


def warp_label(warp_layer, lbl_moving, dvf, criterion):
    """Warps the labels as the loss requires (binary + bilinear if the Dice is differentiable)."""
    if getattr(criterion, "differentiable_dice", False):
        return warp_layer((lbl_moving > 0).float(), dvf, mode="bilinear")
    return warp_layer(lbl_moving, dvf, mode="nearest")
