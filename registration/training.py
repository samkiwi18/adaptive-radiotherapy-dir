"""Training loop with curriculum learning, gradient accumulation, early stopping and resume."""
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .losses import batch_point_loss, warp_label
from .metrics import compute_jacobian_determinant, tre_on_network_grid
from .model import predict_dvf
from .utils import autocast_bf16, get_curriculum_weights, load_state_dict_checked


@dataclass
class PhaseConfig:
    """
    Training phase: (start, end) ranges are interpolated over the epochs, kp_range=None and lambda_inv=0
    disable those losses. Resume checkpoint in resume_dir (None: the folder of the weights).
    """
    name: str
    num_epochs: int
    acc_steps: int = 3
    patience: int = 25
    smooth_range: Tuple[float, float] = (0.5, 1.5)
    lm_range: Tuple[float, float] = (0.5, 1.5)
    kp_range: Optional[Tuple[float, float]] = (0.1, 1.0)
    lambda_inv: float = 0.5
    resume_dir: Optional[str] = None
    checkpoint_every: int = 5


def _to_cpu(obj):
    """Copy of a (nested) state dict with every tensor on the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _atomic_save(obj, path):
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class BackgroundSaver:
    """Writes checkpoints in a background thread (training waits only for the CPU copy); close() waits."""

    def __init__(self, max_pending=2):
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = []

    def save(self, obj, path):
        while len(self._pending) >= self.max_pending:
            self._pending.pop(0).result()
        self._pending.append(self._executor.submit(_atomic_save, _to_cpu(obj), path))

    def close(self):
        try:
            while self._pending:
                self._pending.pop(0).result()
        finally:
            self._executor.shutdown(wait=True)


def _to_device(batch, device):
    return (batch["moving"].to(device), batch["fixed"].to(device),
            batch["label_moving"].to(device), batch["label_fixed"].to(device))


def _training_loss(cfg, epoch, batch, model, warp_layer, integrator, criterion, device):
    moving, fixed, lbl_mov, lbl_fix = _to_device(batch, device)
    with autocast_bf16(device):
        dvf, velocity = predict_dvf(model, integrator, warp_layer, moving, fixed)
        warped = warp_layer(moving, dvf, mode="bilinear")
        warped_lbl = warp_label(warp_layer, lbl_mov, dvf, criterion)
        loss, *_ = criterion(warped, fixed, dvf, warped_lbl, lbl_fix)

        if cfg.lambda_inv > 0:
            # Inverse consistency: composing the DVF with its inverse must give zero displacement
            inv_dvf = integrator(-velocity, warp_layer)
            composed = dvf + warp_layer(inv_dvf, dvf, mode="bilinear")
            loss = loss + cfg.lambda_inv * composed.pow(2).mean()

        loss_lm = batch_point_loss(dvf, batch, "lm", device)
        if loss_lm is not None:
            loss = loss + get_curriculum_weights(epoch, cfg.num_epochs, *cfg.lm_range) * loss_lm

        if cfg.kp_range is not None:
            loss_kp = batch_point_loss(dvf, batch, "kp", device)
            if loss_kp is not None:
                loss = loss + get_curriculum_weights(epoch, cfg.num_epochs, *cfg.kp_range) * loss_kp
    return loss


@torch.no_grad()
def validate(model, warp_layer, integrator, criterion, loader, device):
    """Validation loss, TRE on the native grid (mm) and folding percentage."""
    model.eval()
    sums = {"loss": 0.0, "sim": 0.0, "reg": 0.0, "fold": 0.0}
    tre_values, land_values = [], []

    for batch in loader:
        moving, fixed, lbl_mov, lbl_fix = _to_device(batch, device)
        with autocast_bf16(device):
            dvf, _ = predict_dvf(model, integrator, warp_layer, moving, fixed)
            warped = warp_layer(moving, dvf, mode="bilinear")
            warped_lbl = warp_label(warp_layer, lbl_mov, dvf, criterion)
            loss, loss_sim, loss_reg, _ = criterion(warped, fixed, dvf, warped_lbl, lbl_fix)
        dvf = dvf.float()

        sums["loss"] += loss.item()
        sums["sim"] += loss_sim.item()
        sums["reg"] += (criterion.lambda_smooth * loss_reg).item()
        sums["fold"] += compute_jacobian_determinant(dvf[0])[1]

        loss_lm = batch_point_loss(dvf, batch, "lm", device)
        if loss_lm is not None:
            land_values.append(loss_lm.item())

        for i in range(dvf.shape[0]):
            if batch["has_landmarks"][i]:
                tre_values.append(tre_on_network_grid(dvf[i:i + 1], batch["lm_fix_orig"][i], batch["lm_mov_orig"][i],
                                                      batch["spacing_orig"][i], batch["shape_orig"][i].tolist()))

    n = max(1, len(loader))
    results = {k: v / n for k, v in sums.items()}
    results["tre"] = sum(tre_values) / len(tre_values) if tre_values else 0.0
    results["land"] = sum(land_values) / len(land_values) if land_values else 0.0
    return results


def _load_resume(path, model, opt, sched, device):
    chk = torch.load(path, map_location=device)
    load_state_dict_checked(model, chk["model_state"], path)
    opt.load_state_dict(chk["opt_state"])
    if sched is not None and chk.get("sched_state"):
        sched.load_state_dict(chk["sched_state"])
    return chk["epoch"] + 1, chk["best_val"], chk["patience"]


def train_phase(cfg, model, warp_layer, integrator, criterion, opt, sched,
                train_loader, val_loader, device, weights_path, best_val=float("inf"), writer=None,
                augmentation=None):
    """
    One phase: saves the best-TRE model to weights_path and a resume checkpoint, both on disk at return.
    augmentation(batch, device) is applied to each training batch. Returns the best TRE.
    """
    resume_dir = cfg.resume_dir if cfg.resume_dir is not None else os.path.dirname(os.path.abspath(weights_path))
    resume_path = os.path.join(resume_dir, f"resume_{cfg.name}.pth")
    start_epoch, patience_counter = 0, 0
    if os.path.exists(resume_path):
        start_epoch, best_val, patience_counter = _load_resume(resume_path, model, opt, sched, device)
        print(f"[RESUME] {cfg.name} from epoch {start_epoch}, best TRE = {best_val:.2f} mm")

    saver = BackgroundSaver()
    try:
        n_batches = len(train_loader)
        for epoch in range(start_epoch, cfg.num_epochs):
            criterion.lambda_smooth = get_curriculum_weights(epoch, cfg.num_epochs, *cfg.smooth_range)
            model.train()
            opt.zero_grad()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                if augmentation is not None:
                    batch = augmentation(batch, device)
                loss = _training_loss(cfg, epoch, batch, model, warp_layer, integrator, criterion, device)
                group_start = batch_idx - batch_idx % cfg.acc_steps
                (loss / min(cfg.acc_steps, n_batches - group_start)).backward()
                train_loss += loss.item()
                if (batch_idx + 1) % cfg.acc_steps == 0 or batch_idx + 1 == n_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    opt.zero_grad()

            if sched is not None:
                sched.step()

            val = validate(model, warp_layer, integrator, criterion, val_loader, device)
            train_loss /= max(1, n_batches)
            print(f"[{cfg.name}] Ep {epoch + 1}/{cfg.num_epochs} | smooth {criterion.lambda_smooth:.2f} | "
                  f"train {train_loss:.4f} | val {val['loss']:.4f} (sim {val['sim']:.4f}, reg {val['reg']:.4f}) | "
                  f"land {val['land']:.2f} | TRE {val['tre']:.2f} mm | fold {val['fold']:.3f}%")

            if writer is not None:
                writer.add_scalar(f"{cfg.name}/train_loss", train_loss, epoch)
                for key, value in val.items():
                    writer.add_scalar(f"{cfg.name}/val_{key}", value, epoch)
                writer.add_scalar(f"{cfg.name}/lr", opt.param_groups[0]["lr"], epoch)

            improved = 0.0 < val["tre"] < best_val
            if improved:
                best_val, patience_counter = val["tre"], 0
                saver.save(model.state_dict(), weights_path)
                print(f"  -> model saved (best TRE {best_val:.2f} mm)")
            else:
                patience_counter += 1

            stop = patience_counter >= cfg.patience
            # Also written when the best model improves, so that a resumed run never has a best
            # TRE older than the weights already on disk
            if improved or stop or (epoch + 1) % cfg.checkpoint_every == 0 or epoch + 1 == cfg.num_epochs:
                saver.save({"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
                            "sched_state": sched.state_dict() if sched is not None else None,
                            "best_val": best_val, "patience": patience_counter}, resume_path)
            if stop:
                print(f"[{cfg.name}] Early stopping at epoch {epoch + 1}")
                break
    finally:
        saver.close()
    return best_val
