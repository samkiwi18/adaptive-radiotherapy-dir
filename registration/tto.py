"""Per-patient Test-Time Optimization (TTO) and post-registration evaluation."""
import os
from dataclasses import dataclass
from typing import Tuple

import nibabel as nib
import numpy as np
import torch

from .losses import MINDLoss, batch_point_loss, smoothness_map, soft_dice_loss
from .metrics import compute_jacobian_determinant, compute_tre, jacobian_torch, multiclass_dice
from .model import predict_dvf
from .utils import autocast_bf16, dvf_to_native_resolution, load_weights


@dataclass
class TTOConfig:
    """Global fit up to switch_step, then a local fit weighted by the MIND residual; tuples = (fit 1, fit 2)."""
    n_steps: int = 150
    switch_step: int = 100
    lr: float = 5e-3
    freeze_backbone: bool = False
    mind_mask_weight: float = 50.0
    lambda_lung: Tuple[float, float] = (1.5, 1.0)
    lambda_tumor: Tuple[float, float] = (0.0, 0.0)
    lambda_oar: float = 0.0
    lambda_bone: float = 0.0
    lambda_reg: float = 0.01
    lambda_kp: float = 0.1
    residual_weight_amplitude: float = 3.0
    log_every: int = 30
    mask_dvf: bool = True


def _prepare_parameters(model, freeze_backbone):
    base = model.module if hasattr(model, "module") else model
    for p in base.backbone.parameters():
        p.requires_grad = not freeze_backbone
    for p in base.reg_head.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]


def run_tto(model, warp_layer, integrator, batch, device, weights_path, cfg=None,
            lcc_mask=None, weight_mask=None, tumor_moving=None, tumor_fixed=None,
            oar_moving=None, oar_fixed=None, bone_moving=None, bone_fixed=None,
            heldout_moving=None, heldout_fixed=None, bone_mask=None, callback=None):
    """
    Adapts the model to the pair in batch from weights_path. weight_mask (fixed space) raises the MIND
    weight; tumor / OAR mask pairs enter their Dice; callback runs every cfg.log_every steps.
    """
    moving, fixed = batch["moving"].to(device), batch["fixed"].to(device)
    lbl_mov, lbl_fix = batch["label_moving"].to(device).float(), batch["label_fixed"].to(device).float()
    B = moving.shape[0]
    cfg = cfg or TTOConfig()

    load_weights(model, weights_path, device)
    model.train()
    optimizer = torch.optim.Adam(_prepare_parameters(model, cfg.freeze_backbone), lr=cfg.lr)
    mind = MINDLoss().to(device)

    lcc = lcc_mask if lcc_mask is not None else 1.0
    spatial_weight = 1.0
    if weight_mask is not None:
        spatial_weight = 1.0 + cfg.mind_mask_weight * (weight_mask > 0.5).float().expand(B, -1, -1, -1, -1)
    use_tumor = max(cfg.lambda_tumor) > 0 and tumor_moving is not None and tumor_fixed is not None
    if use_tumor:
        tumor_moving, tumor_fixed = tumor_moving.expand(B, -1, -1, -1, -1), tumor_fixed.expand(B, -1, -1, -1, -1)
    use_oar = cfg.lambda_oar > 0 and oar_moving is not None and oar_fixed is not None and oar_moving.shape[1] > 0
    if use_oar:
        oar_moving, oar_fixed = oar_moving.expand(B, -1, -1, -1, -1), oar_fixed.expand(B, -1, -1, -1, -1)
    use_bone = (cfg.lambda_bone > 0 and bone_moving is not None and bone_fixed is not None
                and bone_moving.shape[1] > 0)
    if use_bone:
        bone_moving, bone_fixed = bone_moving.expand(B, -1, -1, -1, -1), bone_fixed.expand(B, -1, -1, -1, -1)
    use_heldout = heldout_moving is not None and heldout_fixed is not None and heldout_moving.shape[1] > 0
    if use_heldout:
        heldout_moving = heldout_moving.expand(B, -1, -1, -1, -1)
        heldout_fixed = heldout_fixed.expand(B, -1, -1, -1, -1)
    bone_region = bone_mask[0, 0] > 0.5 if bone_mask is not None else None

    with torch.no_grad(), autocast_bf16(device):
        fixed_mind = mind.descriptor(fixed)

    residual_weight = None
    history = []
    for step in range(cfg.n_steps):
        optimizer.zero_grad()
        with autocast_bf16(device):
            dvf, _ = predict_dvf(model, integrator, warp_layer, moving, fixed, lcc_mask if cfg.mask_dvf else None)
            warped = warp_layer(moving, dvf, mode="bilinear")
            # 'bilinear' rather than 'nearest': only then does the mask Dice propagate gradient to the DVF
            loss_lung = soft_dice_loss(warp_layer(lbl_mov, dvf, mode="bilinear"), lbl_fix)
            mind_map = mind.error_map(warped, fixed, fixed_mind)
            reg_map = smoothness_map(dvf)

            local_stage = step >= cfg.switch_step
            if local_stage and residual_weight is None:
                with torch.no_grad():
                    r = mind_map.detach()
                    residual_weight = 1.0 + cfg.residual_weight_amplitude * (r - r.min()) / (r.max() - r.min() + 1e-8)
                print(f"   [TTO] step {step}: switching to the local fit "
                      f"(residual weights {residual_weight.min():.2f}-{residual_weight.max():.2f})")

            if local_stage:
                weight = residual_weight * spatial_weight
                loss_mind = (weight * mind_map * lcc).mean()
                loss_reg = (weight * reg_map * lcc).mean()
            else:
                loss_mind = (mind_map * spatial_weight * lcc).mean()
                loss_reg = (reg_map * lcc).mean()

            loss = loss_mind + cfg.lambda_lung[int(local_stage)] * loss_lung + cfg.lambda_reg * loss_reg

            loss_tumor = torch.zeros((), device=device)
            if use_tumor:
                loss_tumor = soft_dice_loss(warp_layer(tumor_moving, dvf, mode="bilinear"), tumor_fixed)
                loss = loss + cfg.lambda_tumor[int(local_stage)] * loss_tumor

            loss_oar = torch.zeros((), device=device)
            if use_oar:
                loss_oar = soft_dice_loss(warp_layer(oar_moving, dvf, mode="bilinear"), oar_fixed)
                loss = loss + cfg.lambda_oar * loss_oar

            loss_bone = torch.zeros((), device=device)
            if use_bone:
                loss_bone = soft_dice_loss(warp_layer(bone_moving, dvf, mode="bilinear"), bone_fixed)
                loss = loss + cfg.lambda_bone * loss_bone

            loss_kp = batch_point_loss(dvf, batch, "kp", device)
            if loss_kp is not None:
                loss = loss + cfg.lambda_kp * loss_kp

        loss.backward()
        optimizer.step()
        history.append(loss.item())

        if step == 0 or (step + 1) % cfg.log_every == 0 or step + 1 == cfg.n_steps:
            fold = compute_jacobian_determinant(dvf[0].detach())[1]
            line = (f"   Step {step + 1:3d}/{cfg.n_steps} | loss {loss.item():.4f} | MIND {loss_mind.item():.4f} | "
                    f"lung Dice {loss_lung.item():.4f} | tumor Dice {loss_tumor.item():.4f} | "
                    f"OAR Dice {loss_oar.item():.4f} | bone Dice {loss_bone.item():.4f} | fold {fold:.3f}%")
            if use_heldout:
                with torch.no_grad(), autocast_bf16(device):
                    heldout_warped = (warp_layer(heldout_moving, dvf.detach(), mode="bilinear") > 0.5).float()
                line += f" | held-out Dice {soft_dice_loss(heldout_warped, heldout_fixed.float()).item():.4f}"
                del heldout_warped
            if bone_region is not None:
                with torch.no_grad():
                    log_det = jacobian_torch(dvf[:1].detach().float()).clamp(min=1e-3).log().abs()
                line += f" | bone |log detJ| {log_det[bone_region].mean().item():.4f}"
            print(line)
            if callback is not None:
                callback(moving, fixed, warped.detach(), step + 1, loss.item())

    model.eval()
    return history


def export_dvf_nifti(native_dvf, spacing_mm, path):
    """Saves a DVF (1, 3, X, Y, Z) in voxels as a vector NIfTI (X, Y, Z, 3) for 3D Slicer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = nib.Nifti1Image(native_dvf[0].cpu().numpy().transpose(1, 2, 3, 0),
                          np.diag(list(np.asarray(spacing_mm, dtype=float)) + [1.0]))
    img.header.set_intent("vector")
    nib.save(img, path)
    return path


@torch.no_grad()
def evaluate_registration(model, warp_layer, integrator, batch, device, lcc_mask=None, dvf_dir=None):
    """Final inference: label Dice, TRE on the native grid and optional DVF export, as a dict."""
    model.eval()
    moving, fixed = batch["moving"].to(device), batch["fixed"].to(device)
    lbl_mov, lbl_fix = batch["label_moving"].to(device), batch["label_fixed"].to(device)
    with autocast_bf16(device):
        dvf, velocity = predict_dvf(model, integrator, warp_layer, moving, fixed, lcc_mask)
        warped = warp_layer(moving, dvf, mode="bilinear")
        warped_label = warp_layer(lbl_mov, dvf, mode="nearest")
    dvf, velocity, warped, warped_label = dvf.float(), velocity.float(), warped.float(), warped_label.float()

    dice, tre = [], []
    for i, name in enumerate(batch["case_name"]):
        dice.append(multiclass_dice(warped_label[i:i + 1], lbl_fix[i:i + 1]))
        native_dvf = dvf_to_native_resolution(dvf[i:i + 1], batch["shape_orig"][i].tolist())
        line = f"   {name}: Dice {dice[-1]:.4f}"
        if batch["has_landmarks"][i]:
            t, _, _ = compute_tre(native_dvf[0].cpu().numpy(), batch["lm_fix_orig"][i].numpy(),
                                  batch["lm_mov_orig"][i].numpy(), batch["spacing_orig"][i].numpy())
            tre.append(t)
            line += f" | TRE {t:.2f} mm"
        if dvf_dir:
            export_dvf_nifti(native_dvf, batch["spacing_orig"][i].numpy(),
                             os.path.join(dvf_dir, f"DVF_{name}.nii.gz"))
        print(line)

    return {"moving": moving, "fixed": fixed, "warped": warped, "dvf": dvf, "velocity": velocity,
            "warped_label": warped_label, "dice": dice, "tre": tre}
