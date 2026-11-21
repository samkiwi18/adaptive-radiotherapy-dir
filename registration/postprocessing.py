"""Post-processing of the warped CT: mass-preserving HU correction."""
import numpy as np
import torch
from scipy import ndimage as ndi

from .metrics import jacobian_torch


@torch.no_grad()
def warp_binary_mask(mask, dvf, warp_layer):
    """Warps a (1, 1, X, Y, Z) mask and returns it as a binary numpy array (X, Y, Z)."""
    return (warp_layer(mask, dvf, mode="bilinear") > 0.5).float()[0, 0].cpu().numpy()


@torch.no_grad()
def mass_preserving_hu_correction(hu_warped, dvf, lung, tumor, clamp=(0.5, 2.0)):
    """
    Mass conservation in the lung, tumor excluded: ρ_fixed = ρ_moving(x + u) · det(J), with
    ρ = HU + 1000; elsewhere the warped HU are kept. lung / tumor: (X, Y, Z) boolean arrays.
    """
    det = jacobian_torch(dvf.float()).clamp(*clamp)[None, None]
    hu_corr = (hu_warped + 1000.0).clamp(min=0) * det - 1000.0
    region = torch.from_numpy(lung & ~tumor).to(hu_warped.device)[None, None]
    return torch.where(region, hu_corr, hu_warped), det


@torch.no_grad()
def mass_diagnostics(hu_mov, hu_warped, hu_corrected, dvf, lbl_moving, lung_warp, tumor_warp,
                     hu_fix=None, lbl_fixed=None, clamp=(0.5, 2.0), erosion=2):
    """Checks of the HU correction in the inner lung: mean HU before/after, expected value, mass test."""
    device = hu_mov.device

    def inner(m):
        return torch.from_numpy(ndi.binary_erosion(np.asarray(m), iterations=erosion)).to(device)

    lung_m = inner(np.asarray(lbl_moving) > 0.5)
    lung_w = inner(lung_warp) & ~torch.from_numpy(np.asarray(tumor_warp)).to(device)
    hu_mov, hu_warped, hu_corrected = hu_mov[0, 0], hu_warped[0, 0], hu_corrected[0, 0]
    det = jacobian_torch(dvf.float())

    volume_ratio = lung_w.sum().item() / max(lung_m.sum().item(), 1)
    expected = -1000 + (hu_mov[lung_m] + 1000).mean().item() / volume_ratio
    print(f"Inner lung HU | moving {hu_mov[lung_m].mean():.0f} | warped {hu_warped[lung_w].mean():.0f} | "
          f"corrected {hu_corrected[lung_w].mean():.0f} | expected from volume {expected:.0f}")
    if hu_fix is not None and lbl_fixed is not None:
        print(f"Inner lung HU of the real fixed CT: {hu_fix[0, 0][inner(np.asarray(lbl_fixed) > 0.5)].mean():.0f}")
    whole_lung = torch.from_numpy(np.asarray(lung_warp)).to(device)
    moving_lung = torch.from_numpy(np.asarray(lbl_moving) > 0.5).to(device)
    print(f"Mean det(J) in the lung (clamp {clamp}): {det.clamp(*clamp)[lung_w].mean():.3f} | "
          f"fixed/moving volume ratio: {volume_ratio:.3f}")
    print(f"Mass test: sum of det(J) over the warped lung / moving lung volume = "
          f"{det[whole_lung].sum().item() / max(moving_lung.sum().item(), 1):.3f}")
