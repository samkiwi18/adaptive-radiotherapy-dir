"""Plots: TTO diagnostics, interactive viewer, LCC mask, keypoints, structures and VRAM profile."""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D


def _np(x, idx=0):
    """(B, C, X, Y, Z) tensor or array -> numpy (X, Y, Z) of sample idx."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu()
        x = x[idx, 0] if x.ndim == 5 else x
        return x.numpy()
    return np.asarray(x)


def central_slice(mask, default):
    """Median Z index of the mask, or default if the mask is empty."""
    z = np.where(np.asarray(mask) > 0.5)[2]
    return int(np.median(z)) if z.size else default


def _contour(ax, mask_2d, color, lw=2.0, style="-"):
    m = np.asarray(mask_2d, dtype=np.float32)
    if np.any(m > 0.5):
        ax.contour(m, levels=[0.5], colors=color, linewidths=lw, linestyles=style)


def _legend(ax_or_fig, entries, **kwargs):
    lines = [Line2D([0], [0], color=c, lw=lw, linestyle=s) for _, c, lw, s in entries]
    ax_or_fig.legend(lines, [n for n, *_ in entries], **kwargs)


def structure_colors(names, rgb=None):
    """Colour of each structure: its RTSTRUCT display colour (0-255 RGB) if given, otherwise from tab20."""
    rgb = rgb or {}
    return {n: to_hex(np.asarray(rgb[n]) / 255.0) if n in rgb else to_hex(plt.cm.tab20(i % 20))
            for i, n in enumerate(names)}


def plot_tto_diagnostics(moving, fixed, warped, step, loss):
    """Central axial slice of moving, fixed, warped and red/green overlay (yellow = aligned)."""
    m, f, w = _np(moving), _np(fixed), _np(warped)
    z = m.shape[2] // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"TTO - step {step} | loss {loss:.4f}", fontsize=14, fontweight="bold")
    for ax, img, title in zip(axes, (m, f, w), ("Moving", "Fixed", "Warped")):
        ax.imshow(img[:, :, z].T, cmap="gray", origin="lower")
        ax.set_title(title)

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-5)

    f_s, w_s = normalize(f[:, :, z].T), normalize(w[:, :, z].T)
    overlay = np.zeros((*f_s.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1] = w_s, f_s
    axes[3].imshow(overlay, origin="lower")
    axes[3].set_title("Overlay (red = warped, green = fixed)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def _landmark_displacement(dvf, lm_fixed):
    """DVF (3, X, Y, Z) sampled at the fixed landmarks (N, 3) -> (N, 3)."""
    shape = torch.tensor(dvf.shape[1:], dtype=torch.float32)
    grid = (2.0 * lm_fixed / (shape - 1.0) - 1.0)[..., [2, 1, 0]].view(1, 1, 1, -1, 3)
    return F.grid_sample(dvf[None], grid, align_corners=True)[0, :, 0, 0].T.numpy()


def registration_viewer(result, batch, idx=0, grid_step=4, z_tolerance=10, structures=None):
    """
    Z-slider viewer: contours, overlay, DVF and landmark vectors (data copied, safe in loops).
    structures: list of (name, colour, moving, fixed, warped) masks (X, Y, Z).
    """
    import ipywidgets as widgets

    structures = [(n, c, _np(m), _np(f), _np(w)) for n, c, m, f, w in structures or []]
    mov, fix, warp = (_np(result[k], idx) for k in ("moving", "fixed", "warped"))
    dvf = result["dvf"][idx].detach().float().cpu()
    lbl_mov, lbl_fix = _np(batch["label_moving"], idx), _np(batch["label_fixed"], idx)
    landmarks = None
    if batch["has_landmarks"][idx]:
        lm_f = batch["lm_fixed"][idx].float()
        true_vec = (batch["lm_moving"][idx] - batch["lm_fixed"][idx]).numpy()
        landmarks = (lm_f.numpy(), true_vec, _landmark_displacement(dvf, lm_f))
    dvf = dvf.numpy()
    rows, cols = np.mgrid[0:mov.shape[0]:grid_step, 0:mov.shape[1]:grid_step]

    def wide_window(x):
        return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

    def lung_window(x):
        return (np.clip(x, -0.5, 1.0) + 0.5) / 1.5

    def draw(z):
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        m, f, w = wide_window(mov[:, :, z]), wide_window(fix[:, :, z]), wide_window(warp[:, :, z])
        for ax, img, lbl, title in ((axes[0, 0], m, lbl_mov, "Moving"), (axes[0, 1], f, lbl_fix, "Fixed"),
                                    (axes[0, 2], w, None, "Warped")):
            ax.imshow(img, cmap="gray")
            if lbl is not None:
                _contour(ax, lbl[:, :, z], "cyan", 1.5)
            ax.set_title(f"{title} (Z={z})" + (" [cyan: label]" if lbl is not None else ""))
        visible = []
        for name, color, *masks in structures:
            for ax, mask in zip(axes[0, :3], masks):
                _contour(ax, mask[:, :, z], color, 1.2)
            if any(np.any(mask[:, :, z] > 0.5) for mask in masks):
                visible.append((name, color, 1.5, "-"))
        if visible:
            _legend(fig, visible, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        rgb = np.zeros((*f.shape, 3))
        rgb[..., 0], rgb[..., 1] = f, w
        axes[0, 3].imshow(rgb)
        axes[0, 3].set_title("Overlay (red = fixed, green = warped)")

        axes[1, 0].imshow(lung_window(mov[:, :, z]), cmap="gray")
        axes[1, 0].set_title("Moving (lung window)")
        axes[1, 1].imshow(lung_window(warp[:, :, z]), cmap="gray")
        axes[1, 1].set_title("Warped (lung window)")

        # In the image rows are axis 0 and columns axis 1: the components go in this order
        axes[1, 2].imshow(lung_window(fix[:, :, z]), cmap="gray", alpha=0.5)
        axes[1, 2].quiver(cols, rows, dvf[1, ::grid_step, ::grid_step, z],
                          dvf[0, ::grid_step, ::grid_step, z], color="cyan",
                          angles="xy", scale_units="xy", scale=1, width=0.005, alpha=0.6)
        axes[1, 2].set_title("Predicted DVF")

        axes[1, 3].imshow(lung_window(fix[:, :, z]), cmap="gray", alpha=0.6)
        if landmarks is None:
            axes[1, 3].set_title("No landmarks available")
        else:
            lm, true_vec, pred_vec = landmarks
            near = np.abs(lm[:, 2] - z) < z_tolerance
            if near.any():
                for vectors, color, width, label in ((true_vec, "lime", 0.007, "True"), (pred_vec, "red", 0.005, "Predicted")):
                    axes[1, 3].quiver(lm[near, 1], lm[near, 0], vectors[near, 1], vectors[near, 0],
                                      color=color, angles="xy", scale_units="xy", scale=1, width=width, label=label)
                axes[1, 3].legend(loc="upper right", framealpha=0.9)
                axes[1, 3].set_title(f"Landmarks (Z ± {z_tolerance})")
            else:
                axes[1, 3].set_title("No landmarks near this slice")
        for ax in axes.ravel():
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    n_z = mov.shape[2]
    widgets.interact(draw, z=widgets.IntSlider(min=0, max=n_z - 1, value=n_z // 2, description="Slice Z:"))


def plot_lcc_mask(vol, mask, z=None):
    z = vol.shape[2] // 2 if z is None else z
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(vol[:, :, z].T, cmap="gray", origin="lower")
    _contour(ax, mask[:, :, z].T, "yellow", 2.5, "dashed")
    ax.set_title(f"LCC mask (yellow) - slice Z={z}", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_keypoint_coverage(kp, volume, n_slices=6):
    """Spatial density of the keypoints, per-slice distribution and overlay on a few slices."""
    if len(kp) == 0:
        print("No keypoints to show.")
        return
    extent = kp.max(axis=0) - kp.min(axis=0)
    axis = int(np.argmin(extent))
    plane = [c for c in range(3) if c != axis]
    print(f"{len(kp)} keypoints | volume {volume.shape} | axial axis: column {axis}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    h = axes[0].hist2d(kp[:, plane[1]], kp[:, plane[0]], bins=20, cmap="hot")
    plt.colorbar(h[3], ax=axes[0], label="keypoints per cell")
    axes[0].invert_yaxis()
    axes[0].set_title("Spatial density (projection)")
    axes[1].hist(kp[:, axis], bins=volume.shape[axis], color="crimson")
    axes[1].set_title("Keypoints per slice")
    axes[1].set_xlabel(f"slice index (axis {axis})")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    axes = np.atleast_1d(axes)
    for ax, z in zip(axes, np.linspace(kp[:, axis].min(), kp[:, axis].max(), n_slices).astype(int)):
        near = np.abs(kp[:, axis] - z) < 3
        ax.imshow(np.take(volume, z, axis=axis), cmap="gray")
        ax.scatter(kp[near, plane[1]], kp[near, plane[0]], c="red", s=8)
        ax.set_title(f"slice {z} ({near.sum()} keypoints)")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_ct_structures(ct, structures, z, title="Warped CT with structures"):
    """structures: list of (name, mask (X, Y, Z), color, line width, style)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ct[:, :, z], cmap="gray", origin="lower", vmin=-1000, vmax=400)
    for _, mask, color, lw, style in structures:
        _contour(ax, mask[:, :, z], color, lw, style)
    ax.set_title(f"{title} (Z={z})", fontsize=14, fontweight="bold")
    ax.axis("off")
    _legend(ax, [(n, c, lw, s) for n, mask, c, lw, s in structures if np.any(np.asarray(mask)[:, :, z] > 0.5)],
            loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_memory_profile(inference_blocks, training_blocks=None, title="VRAM per block"):
    """Bars per block (from profile_blocks): weights, no_grad retained memory, memory kept for backward."""
    series = [("Weights", inference_blocks, "weights_MiB", "tab:gray"),
              ("Retained, inference", inference_blocks, "retained_MiB", "tab:blue")]
    if training_blocks is not None:
        series.append(("Retained for the backward", training_blocks, "retained_MiB", "tab:red"))
    names = list(inference_blocks["block"])
    y = np.arange(len(names))
    height = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(names) + 2))
    for k, (label, df, column, color) in enumerate(series):
        values = df.set_index("block")[column].reindex(names).fillna(0.0).to_numpy()
        ax.barh(y + (k - (len(series) - 1) / 2) * height, values, height, label=label, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("MiB", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.show()
