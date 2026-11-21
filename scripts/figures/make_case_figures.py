"""Figures of the 4D-Lung case study (TCIA subject 107_HM10395) from the dosimetry inputs of each experiment.

Expected layout of --case_dir (see data/README.md):

  <case_dir>/cbct.nii.gz                         CBCT of the day on the common 448 x 288 x 32 grid
  <case_dir>/<run>/RTDose_<case>.dcm             dose of each case, <case> in real_ct, baseline, dir_ct, dir_cbct
  <case_dir>/<run>/RTstruct_<case>.dcm           structures of each case
  <case_dir>/<run>/ct_<case>/                    CT slices of each case

<run> is full_network (registration figure, doses, gamma, DVH). Gamma pass rates in the titles are read from
results/dosimetry/<run>/gamma_results.csv and gamma_in_structures.csv; the gamma maps are recomputed with the same
function and settings.
"""
import argparse
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
from matplotlib import patheffects
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from PIL import Image

from registration.dose_analysis.dose_analysis import _gamma_worker, load_dose, rasterize_structures

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASES = ["real_ct", "baseline", "dir_ct", "dir_cbct"]
LABELS = {"real_ct": "real CT (reference)", "baseline": "baseline (no registration)",
          "dir_ct": "planning CT registered to the CT", "dir_cbct": "planning CT registered to the CBCT"}
CT_WINDOW = (-1350.0, 150.0)
CBCT_WINDOW = (-1600.0, 200.0)
PRESCRIPTION_GY = 7.5
ISODOSES = [0.95, 0.80, 0.50, 0.20]
CREDIT = "Data: 4D-Lung, TCIA (CC BY 3.0)"
LOW_DOSE_CUTOFF = 0.10
DPI = 150
plt.rcParams.update({"font.size": 8, "savefig.dpi": DPI, "figure.facecolor": "white", "savefig.facecolor": "white"})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case_dir", default="data/clinical/4dlung_107", help="4D-Lung case folder (layout above)")
    parser.add_argument("--run", default="full_network", help="experiment folder inside --case_dir")
    parser.add_argument("--results_dir", default="results/dosimetry", help="dosimetric results of the experiments")
    parser.add_argument("--out_dir", default="docs/figures")
    parser.add_argument("--summary", default="results/case_figures.json", help="numbers quoted in the captions")
    parser.add_argument("--ram-gb", type=float, default=1.2, help="RAM available to pymedphys for the gamma maps")
    return parser.parse_args()


def path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_ct(folder, grid):
    slices = [pydicom.dcmread(f) for f in glob.glob(os.path.join(folder, "*.dcm"))]
    slices = sorted([s for s in slices if s.Modality == "CT"], key=lambda s: float(s.ImagePositionPatient[2]))
    hu = np.stack([s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in slices])
    iop = [round(float(v)) for v in slices[0].ImageOrientationPatient]
    if iop[0] < 0:
        hu = hu[:, :, ::-1]
    if iop[4] < 0:
        hu = hu[:, ::-1, :]
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    assert hu.shape == grid["dose"].shape and np.allclose(z, grid["axes"][0], atol=1e-2)
    return hu.astype(np.float32)


def load_cbct(nifti_path, grid):
    img = nib.load(nifti_path)
    vol = img.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    codes = nib.aff2axcodes(img.affine)
    if codes[0] == "R":
        vol = vol[:, :, ::-1]
    if codes[1] == "A":
        vol = vol[:, ::-1, :]
    assert vol.shape == grid["dose"].shape
    return vol


def load_case(case_dir, run):
    folder = os.path.join(case_dir, run)
    data = {}
    for case in CASES:
        d = load_dose(os.path.join(folder, f"RTDose_{case}.dcm"))
        struct = pydicom.dcmread(os.path.join(folder, f"RTstruct_{case}.dcm"))
        d["masks"] = rasterize_structures(struct, f"RTstruct_{case}", d, {"Tumor": "GTV"}, lambda *_: None)
        d["ct"] = load_ct(os.path.join(folder, f"ct_{case}"), d)
        data[case] = d
    data["cbct"] = load_cbct(os.path.join(case_dir, "cbct.nii.gz"), data["real_ct"])
    return data


def centroid(mask):
    return [int(round(c)) for c in np.array(np.nonzero(mask)).mean(axis=1)]


def axial(volume, z):
    return volume[z]


def coronal(volume, y):
    return volume[:, y, :][::-1]


def extents(grid):
    z, y, x = grid["axes"]
    dz, dy, dx = grid["spacing"]
    return ((0, len(x) * dx, len(y) * dy, 0), (0, len(x) * dx, 0, len(z) * dz))


def scale_bar(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xb, yb = x0 + 0.06 * (x1 - x0), y0 + 0.07 * (y1 - y0)
    ax.plot([xb, xb + 50], [yb, yb], color="white", lw=3, solid_capstyle="butt")
    ax.plot([xb, xb + 50], [yb, yb], color="black", lw=1.2, solid_capstyle="butt")
    ax.text(xb + 25, yb + 0.03 * (y1 - y0), "5 cm", ha="center", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))


def pixel_centers(shape2d, ext):
    ny, nx = shape2d
    sx, sy = (ext[1] - ext[0]) / nx, abs(ext[3] - ext[2]) / ny
    xs = ext[0] + (np.arange(nx) + 0.5) * sx
    ys = (np.arange(ny) + 0.5) * sy if ext[2] > ext[3] else ext[3] - (np.arange(ny) + 0.5) * sy
    return xs, ys


def contour(ax, mask2d, ext, color, style):
    if mask2d.any():
        xs, ys = pixel_centers(mask2d.shape, ext)
        ax.contour(xs, ys, mask2d.astype(float), levels=[0.5], colors=[color], linewidths=1.5, linestyles=style)


def show(ax, image, ext, window, title, cmap="gray"):
    im = ax.imshow(image, cmap=cmap, vmin=window[0], vmax=window[1], extent=ext, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return im


def crop(ax, mask2d, ext, half_width_mm=120.0, half_height_mm=85.0):
    xs, ys = pixel_centers(mask2d.shape, ext)
    rows, cols = np.nonzero(mask2d)
    cx, cy = xs[cols].mean(), ys[rows].mean()
    ax.set_xlim(cx - half_width_mm, cx + half_width_mm)
    if ext[2] > ext[3]:
        ax.set_ylim(cy + half_height_mm, cy - half_height_mm)


def registration_figure(data, out_dir):
    ref = data["real_ct"]
    z, y, _ = centroid(ref["masks"]["GTV"])
    axial_ext, coronal_ext = extents(ref)
    panels = [("planning CT", data["baseline"]["ct"], data["baseline"]["masks"], CT_WINDOW),
              ("CBCT of the day", data["cbct"], None, CBCT_WINDOW),
              ("planning CT registered to the CBCT", data["dir_cbct"]["ct"], data["dir_cbct"]["masks"], CT_WINDOW)]
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 4.9), layout="constrained", sharex="row", sharey="row",
                             gridspec_kw={"height_ratios": [160.0, float(len(ref["axes"][0]) * ref["spacing"][0])]})
    for row, (view, cut, ext) in enumerate((("axial", lambda v: axial(v, z), axial_ext),
                                            ("coronal", lambda v: coronal(v, y), coronal_ext))):
        for col, (title, volume, masks, window) in enumerate(panels):
            ax = axes[row, col]
            im = show(ax, cut(volume), ext, window, f"{title}, {view}")
            if col == 1:
                cbct_im = im
            else:
                ct_im = im
            for name, color in (("GTV", "#ee3377"), ("spinal cord", "#33bbee")):
                contour(ax, cut(ref["masks"][name]).astype(float), ext, color, "--")
                if masks is not None and name in masks:
                    contour(ax, cut(masks[name]).astype(float), ext, color, "-")
            crop(ax, cut(ref["masks"]["GTV"]), ext, half_width_mm=110.0, half_height_mm=80.0)
            scale_bar(ax)
    handles = [Line2D([], [], color="#ee3377", ls="-", label="GTV: expert contour on the planning CT, propagated by "
                      "the registration on the registered CT"),
               Line2D([], [], color="#ee3377", ls="--", label="GTV of the real CT (expert)"),
               Line2D([], [], color="#33bbee", ls="-", label="spinal cord: TotalSegmentator on the planning CT, "
                      "propagated by the registration on the registered CT"),
               Line2D([], [], color="#33bbee", ls="--", label="spinal cord of the real CT (TotalSegmentator)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=6.5, frameon=False)
    fig.colorbar(ct_im, ax=axes.ravel().tolist(), shrink=0.7, label="CT (HU)")
    fig.colorbar(cbct_im, ax=axes.ravel().tolist(), shrink=0.7, label="CBCT (uncalibrated)")
    fig.suptitle(f"{CREDIT}, subject 107_HM10395; radiological view", fontsize=7, color="0.3")
    target = os.path.join(out_dir, "case_registration.png")
    fig.savefig(target, bbox_inches="tight", pad_inches=0.04, metadata={"Software": None})
    plt.close(fig)
    return {"gtv_centroid_index_zyx": [int(v) for v in centroid(ref["masks"]["GTV"])]}


def dose_figure(data, out_dir):
    ref = data["real_ct"]
    z = centroid(ref["masks"]["GTV"])[0]
    ext = extents(ref)[0]
    vmax = max(float(data[c]["dose"].max()) for c in CASES)
    fig, axes = plt.subplots(2, 4, figsize=(10.2, 5.0), layout="constrained")
    levels = [f * PRESCRIPTION_GY for f in ISODOSES]
    iso_colors = ["#000000", "#ffffff", "#ff7f00", "#7fbfff"]
    for col, case in enumerate(CASES):
        d = data[case]
        ax = axes[0, col]
        show(ax, axial(d["ct"], z), ext, CT_WINDOW, LABELS[case])
        dim = ax.imshow(np.ma.masked_less(axial(d["dose"], z), 0.2 * PRESCRIPTION_GY), cmap="viridis", vmin=0,
                        vmax=vmax, extent=ext, alpha=0.6, interpolation="bilinear")
        xs, ys = pixel_centers(axial(d["dose"], z).shape, ext)
        ax.contour(xs, ys, axial(d["dose"], z), levels=levels[::-1], colors=iso_colors[::-1], linewidths=0.8)
        for name in ("GTV", "PTV"):
            if name in d["masks"]:
                contour(ax, axial(d["masks"][name], z).astype(float), ext, "#ee3377", "-" if name == "GTV" else ":")
        crop(ax, axial(ref["masks"]["PTV"], z), ext)
        scale_bar(ax)
    axes[1, 0].axis("off")
    region = axial((ref["dose"] >= LOW_DOSE_CUTOFF * float(ref["dose"].max())) & ref["masks"]["BODY"], z)
    for col, case in enumerate(CASES[1:], start=1):
        ax = axes[1, col]
        diff = axial(data[case]["dose"] - ref["dose"], z)
        show(ax, axial(ref["ct"], z), ext, CT_WINDOW, f"{case} − real_ct")
        lim = 1.0
        rim = ax.imshow(np.ma.masked_where(~region | (np.abs(diff) < 0.05), diff), cmap="RdBu_r", vmin=-lim,
                        vmax=lim, extent=ext, interpolation="nearest")
        contour(ax, axial(ref["masks"]["GTV"], z).astype(float), ext, "#000000", "-")
        crop(ax, axial(ref["masks"]["PTV"], z), ext)
        scale_bar(ax)
    fig.colorbar(dim, ax=axes[0, :].tolist(), shrink=0.8, label="dose per fraction (Gy)")
    fig.colorbar(rim, ax=axes[1, 1:].tolist(), shrink=0.8, extend="both", label="dose difference (Gy)")
    handles = [Line2D([], [], color=c, lw=1.2, label=f"{int(f * 100)}% isodose") for f, c in zip(ISODOSES, iso_colors)]
    handles += [Line2D([], [], color="#ee3377", ls="-", label="GTV (expert) of each image"),
                Line2D([], [], color="#ee3377", ls=":", label="PTV (GTV + 5 mm) of each image"),
                Line2D([], [], color="#000000", ls="-", label="GTV of the real CT (bottom row)")]
    legend = axes[1, 0].legend(handles=handles, loc="center", fontsize=6.5, frameon=True, facecolor="white",
                               edgecolor="0.6", labelcolor="0.1", framealpha=1.0,
                               title=f"isodoses: % of the prescription\n({PRESCRIPTION_GY} Gy per fraction)",
                               title_fontsize=7)
    for line in legend.get_lines()[1:2]:
        line.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground="0.3")])
    fig.suptitle(f"{CREDIT}; axial slice through the GTV centroid, radiological view", fontsize=7, color="0.3")
    target = os.path.join(out_dir, "case_dose.png")
    fig.savefig(target, bbox_inches="tight", pad_inches=0.04, metadata={"Software": None})
    plt.close(fig)
    return {"slice_z_index": int(z), "dose_max_gy": {c: float(data[c]["dose"].max()) for c in CASES}}


def gamma_figure(data, rates, ptv_rates, ram_bytes, out_dir):
    ref = data["real_ct"]
    z = centroid(ref["masks"]["GTV"])[0]
    ext = extents(ref)[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), layout="constrained")
    cmap = ListedColormap(plt.cm.Greens(np.linspace(0.25, 0.85, 128)).tolist()
                          + plt.cm.Purples(np.linspace(0.55, 1.0, 128)).tolist())
    out = {}
    for ax, case in zip(axes, ("baseline", "dir_cbct")):
        gamma = _gamma_worker((ref["axes"], ref["dose"], data[case]["axes"], data[case]["dose"], 2, 2, ram_bytes))
        g = axial(gamma, z)
        show(ax, axial(ref["ct"], z), ext, CT_WINDOW,
             f"{case}, 2%/2 mm pass rate\nglobal {rates[case]}%, inside the PTV {ptv_rates[case]}%")
        gim = ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=0, vmax=2, extent=ext, interpolation="nearest")
        contour(ax, axial(ref["masks"]["GTV"], z).astype(float), ext, "#ee3377", "-")
        contour(ax, axial(ref["masks"]["PTV"], z).astype(float), ext, "#ee3377", ":")
        crop(ax, axial(ref["masks"]["PTV"], z), ext, half_width_mm=90.0, half_height_mm=65.0)
        scale_bar(ax)
        out[case] = {"slice_failing_points": int(np.nansum(g > 1)), "slice_points": int(np.isfinite(g).sum())}
    fig.colorbar(gim, ax=axes.tolist(), shrink=0.8, label="gamma (> 1 fails, purple)")
    handles = [Line2D([], [], color="#ee3377", ls="-", label="GTV of the real CT (expert)"),
               Line2D([], [], color="#ee3377", ls=":", label="PTV of the real CT (GTV + 5 mm)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle(f"{CREDIT}; global gamma against the dose on the real CT, axial slice through the GTV", fontsize=7,
                 color="0.3")
    target = os.path.join(out_dir, "case_gamma.png")
    fig.savefig(target, bbox_inches="tight", pad_inches=0.04, metadata={"Software": None})
    plt.close(fig)
    return out


def dvh_figure(data, out_dir):
    structures = [("GTV", "GTV (expert)"), ("PTV", "PTV (GTV + 5 mm)"), ("spinal cord", "spinal cord (TotalSegmentator)"),
                  ("Lungs-GTV", "lungs − GTV (TotalSegmentator)")]
    style = {"real_ct": ("#000000", "-"), "baseline": ("#bbbbbb", ":"), "dir_ct": ("#0077bb", "--"),
             "dir_cbct": ("#ee7733", "-.")}
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.9), layout="constrained")
    for ax, (name, title) in zip(axes, structures):
        for case in CASES:
            d = data[case]
            values = np.sort(d["dose"][d["masks"][name]])[::-1]
            color, ls = style[case]
            ax.plot(values, np.arange(1, values.size + 1) / values.size * 100, color=color, ls=ls, lw=1.6,
                    label=case)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("dose per fraction (Gy)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 101)
    axes[0].set_ylabel("volume (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{CREDIT}; each dose with the structures of its own image", fontsize=7, color="0.3")
    target = os.path.join(out_dir, "case_dvh.png")
    fig.savefig(target, bbox_inches="tight", pad_inches=0.04, metadata={"Software": None})
    plt.close(fig)


def normalized(image, window):
    return np.clip((image - window[0]) / (window[1] - window[0]), 0.0, 1.0)


def pairs(data):
    return [("real CT", data["real_ct"]["ct"], CT_WINDOW, "planning CT (baseline)", data["baseline"]["ct"], CT_WINDOW),
            ("real CT", data["real_ct"]["ct"], CT_WINDOW, "planning CT registered to the CT", data["dir_ct"]["ct"],
             CT_WINDOW),
            ("CBCT", data["cbct"], CBCT_WINDOW, "planning CT registered to the CBCT", data["dir_cbct"]["ct"],
             CT_WINDOW)]


def flicker_gif(data, out_dir):
    ref = data["real_ct"]
    z = centroid(ref["masks"]["GTV"])[0]
    ext = extents(ref)[0]
    gtv = axial(ref["masks"]["GTV"], z)
    frames = []
    for name_a, vol_a, win_a, name_b, vol_b, win_b in pairs(data):
        for name, volume, window in ((name_a, vol_a, win_a), (name_b, vol_b, win_b)):
            fig, ax = plt.subplots(figsize=(6.0, 5.0), layout="constrained")
            show(ax, normalized(axial(volume, z), window), ext, (0.0, 1.0), "")
            ax.set_title(f"{name_a} vs {name_b}\nshown: {name}", fontsize=10)
            xs, ys = pixel_centers(gtv.shape, ext)
            ax.contour(xs, ys, gtv.astype(float), levels=[0.5], colors=["#ff2fa0"], linewidths=1.5,
                       path_effects=[patheffects.withStroke(linewidth=2.6, foreground="#202020")])
            crop(ax, gtv, ext, half_width_mm=95.0, half_height_mm=70.0)
            scale_bar(ax)
            ax.text(0.98, 0.02, "GTV of the real CT (expert)", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="#c0006e", bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.9))
            fig.supxlabel(f"{CREDIT}; axial, radiological view", fontsize=7.5, color="0.3")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
    sequence = []
    for k in range(0, len(frames), 2):
        sequence += [frames[k], frames[k + 1]] * 2
    swatch = np.zeros((40, frames[0].shape[1], 3), dtype=np.uint8)
    swatch[:20] = (0xff, 0x2f, 0xa0)
    swatch[20:] = (0x20, 0x20, 0x20)
    palette = Image.fromarray(np.concatenate(frames + [swatch] * 6, axis=0)).quantize(
        colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in sequence]
    target = os.path.join(out_dir, "case_flicker.gif")
    images[0].save(target, save_all=True, append_images=images[1:], duration=1000, loop=0, optimize=False)


def checkerboard_figure(data, out_dir, square_mm=20.0):
    ref = data["real_ct"]
    z = centroid(ref["masks"]["GTV"])[0]
    ext = extents(ref)[0]
    gtv = axial(ref["masks"]["GTV"], z)
    ny, nx = gtv.shape
    dy, dx = ref["spacing"][1], ref["spacing"][2]
    rows, cols = np.meshgrid(np.arange(ny) * dy // square_mm, np.arange(nx) * dx // square_mm, indexing="ij")
    tiles = (rows + cols) % 2 == 0
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), layout="constrained")
    for ax, (name_a, vol_a, win_a, name_b, vol_b, win_b) in zip(axes, pairs(data)):
        board = np.where(tiles, normalized(axial(vol_a, z), win_a), normalized(axial(vol_b, z), win_b))
        show(ax, board, ext, (0.0, 1.0), f"{name_a} / {name_b}")
        contour(ax, gtv, ext, "#ee3377", "--")
        crop(ax, gtv, ext, half_width_mm=95.0, half_height_mm=70.0)
        scale_bar(ax)
    fig.suptitle(f"{CREDIT}; 2 cm checkerboard, axial slice through the GTV centroid, radiological view; dashed: GTV of "
                 "the real CT (expert)", fontsize=7, color="0.3")
    target = os.path.join(out_dir, "case_checkerboard.png")
    fig.savefig(target, bbox_inches="tight", pad_inches=0.04, metadata={"Software": None})
    plt.close(fig)


def read_rates(results_dir, run):
    rates, ptv_rates = {}, {}
    with open(os.path.join(path(results_dir), run, "gamma_results.csv")) as f:
        for row in csv.DictReader(f):
            if row["criterion"] == "2%/2mm":
                rates[row["evaluated_dose"]] = row["pass_rate_pct"]
    with open(os.path.join(path(results_dir), run, "gamma_in_structures.csv")) as f:
        for row in csv.DictReader(f):
            if row["criterion"] == "2%/2mm" and row["structure"] == "PTV (real_ct)":
                ptv_rates[row["dataset"]] = row["pass_rate_pct"]
    return rates, ptv_rates


def main():
    args = parse_args()
    out_dir = path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    data = load_case(path(args.case_dir), args.run)
    summary = {"data": CREDIT, "subject": "107_HM10395", "run": args.run}
    summary["registration"] = registration_figure(data, out_dir)
    summary["dose"] = dose_figure(data, out_dir)
    rates, ptv_rates = read_rates(args.results_dir, args.run)
    summary["gamma_2_2_pass_rate_pct"] = {k: rates[k] for k in ("baseline", "dir_cbct")}
    summary["gamma_2_2_pass_rate_in_ptv_pct"] = {k: ptv_rates[k] for k in ("baseline", "dir_cbct")}
    summary["gamma_slice"] = gamma_figure(data, rates, ptv_rates, int(args.ram_gb * 1e9), out_dir)
    dvh_figure(data, out_dir)
    flicker_gif(data, out_dir)
    checkerboard_figure(data, out_dir)
    with open(path(args.summary), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
