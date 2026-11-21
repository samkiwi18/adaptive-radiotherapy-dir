"""Register a moving NIfTI image onto a fixed one with the trained network, optionally refined by test-time optimization."""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch

from registration.dataset import _normalize
from registration.metrics import jacobian_torch
from registration.model import (SpatialTransformer3D, build_components, predict_dvf, set_activation_checkpointing,
                                set_window_attention)
from registration.tto import TTOConfig, export_dvf_nifti, run_tto
from registration.utils import (autocast_bf16, dvf_to_native_resolution, load_weights, read_nifti, set_seed,
                                volume_to_tensor)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", required=True, help="moving image (NIfTI)")
    parser.add_argument("--fixed", required=True, help="fixed image (NIfTI), same grid as the moving image")
    parser.add_argument("--weights", default="weights/best_reg_model.pth", help="trained weights")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--moving_mask", default=None, help="optional lung mask of the moving image (NIfTI)")
    parser.add_argument("--fixed_mask", default=None, help="optional lung mask of the fixed image (NIfTI)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tto", action="store_true", help="refine the network output with test-time optimization")
    parser.add_argument("--steps", type=int, default=120, help="TTO steps")
    parser.add_argument("--switch_step", type=int, default=90, help="TTO step where the local fit starts")
    parser.add_argument("--lr", type=float, default=5e-3, help="TTO learning rate")
    parser.add_argument("--lambda_reg", type=float, default=0.5, help="TTO smoothness weight")
    parser.add_argument("--grid", type=int, nargs=3, default=(192, 192, 192), help="network grid (multiples of 32)")
    parser.add_argument("--integration_steps", type=int, default=9)
    parser.add_argument("--low_memory", action="store_true",
                        help="efficient window attention and activation checkpointing (same weights, much less memory)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mask_on_grid(path, shape, device):
    if path is None:
        return torch.zeros((1, 1, *shape), device=device)
    return volume_to_tensor((read_nifti(path)[0] > 0.5).astype(np.float32), shape, device, mode="nearest")


def jacobian_stats(det, mask=None):
    values = det[mask] if mask is not None else det.reshape(-1)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "folding_pct": float((values <= 0).float().mean() * 100.0)}


def main():
    args = parse_args()
    if any(s % 32 for s in args.grid):
        raise ValueError(f"every dimension of --grid must be a multiple of 32, got {args.grid}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"{args.weights} not found: see weights/README.md")
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    moving_np, _ = read_nifti(args.moving)
    fixed_np, fixed_spacing = read_nifti(args.fixed)
    if moving_np.shape != fixed_np.shape:
        raise ValueError(f"moving {moving_np.shape} and fixed {fixed_np.shape} must share the same grid")
    grid = tuple(args.grid)
    moving = volume_to_tensor(_normalize(moving_np), grid, device)
    fixed = volume_to_tensor(_normalize(fixed_np), grid, device)

    model, warp_layer, integrator = build_components(device, grid, args.integration_steps, load_ssl=False)
    load_weights(model, args.weights, device)
    set_window_attention(model, efficient=args.low_memory)
    set_activation_checkpointing(model, integrator, enabled=args.low_memory)
    if args.tto:
        batch = {"moving": moving, "fixed": fixed, "label_moving": mask_on_grid(args.moving_mask, grid, device),
                 "label_fixed": mask_on_grid(args.fixed_mask, grid, device), "has_keypoints": [False]}
        cfg = TTOConfig(n_steps=args.steps, switch_step=args.switch_step, lr=args.lr, lambda_reg=args.lambda_reg,
                        lambda_lung=(1.5, 1.0))
        run_tto(model, warp_layer, integrator, batch, device, args.weights, cfg)
    model.eval()
    with torch.no_grad(), autocast_bf16(device):
        dvf, _ = predict_dvf(model, integrator, warp_layer, moving, fixed)

    native_shape = fixed_np.shape
    with torch.no_grad():
        native_dvf = dvf_to_native_resolution(dvf.float(), native_shape)
        warped = SpatialTransformer3D(native_shape).to(device)(
            torch.from_numpy(moving_np)[None, None].to(device), native_dvf, mode="bilinear")
        det = jacobian_torch(native_dvf)

    fixed_img = nib.load(args.fixed)
    warped_path = os.path.join(args.output_dir, "warped.nii.gz")
    nib.save(nib.Nifti1Image(warped[0, 0].cpu().numpy(), fixed_img.affine, fixed_img.header), warped_path)
    dvf_path = export_dvf_nifti(native_dvf, fixed_spacing, os.path.join(args.output_dir, "dvf.nii.gz"))

    stats = {"moving": args.moving, "fixed": args.fixed, "tto": args.tto,
             "tto_steps": args.steps if args.tto else 0, "grid": list(grid),
             "integration_steps": args.integration_steps, "low_memory": args.low_memory,
             "jacobian": jacobian_stats(det)}
    if args.fixed_mask is not None:
        mask = torch.from_numpy(read_nifti(args.fixed_mask)[0] > 0.5).to(device)
        stats["jacobian_in_fixed_mask"] = jacobian_stats(det, mask)
    with open(os.path.join(args.output_dir, "jacobian_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"warped image: {warped_path}\nDVF (voxels): {dvf_path}\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
