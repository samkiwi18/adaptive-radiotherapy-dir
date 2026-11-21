import argparse
import csv
import json
import multiprocessing as mp
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from skimage import draw

GAMMA_CRITERIA = [(3, 3), (3, 2), (2, 2), (1, 1)]
MAIN_CRITERION = (2, 2)
LOW_DOSE_CUTOFF_PCT = 10
DOSE_REGIONS = [("dose>=10%", 0.10), ("dose>=50%", 0.50), ("dose>=80%", 0.80)]
EDGE_MARGIN_MM = 3.0

DEFAULT_ALIASES = {"Tumor": "GTV"}
TARGETS = ["GTV", "PTV"]
LUNGS = "Lungs-GTV"
LUNG_V_GY = [1.0, 2.0]

STYLE = [("black", "-", 2.8), ("tab:blue", "--", 2.2), ("tab:orange", "-.", 2.2), ("tab:gray", ":", 2.4),
         ("tab:green", "--", 2.2), ("tab:red", "-.", 2.2), ("tab:purple", ":", 2.2)]


class Logger:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def load_dose(path):
    ds = pydicom.dcmread(path)
    dose = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    if dose.ndim == 2:
        dose = dose[None]
    ipp = np.array([float(v) for v in ds.ImagePositionPatient])
    row_step, col_step = (float(v) for v in ds.PixelSpacing)
    offsets = [float(v) for v in ds.GridFrameOffsetVector] if hasattr(ds, "GridFrameOffsetVector") else [0.0]
    z = ipp[2] + np.array(offsets)
    y = ipp[1] + np.arange(dose.shape[1]) * row_step
    x = ipp[0] + np.arange(dose.shape[2]) * col_step
    dz = float(np.mean(np.diff(z))) if len(z) > 1 else float(getattr(ds, "SliceThickness", 1.0) or 1.0)
    return dict(dose=dose, axes=(z, y, x), spacing=np.array([abs(dz), row_step, col_step]),
                units=str(getattr(ds, "DoseUnits", "GY")), frame_uid=str(getattr(ds, "FrameOfReferenceUID", "")))


def same_grid(a, b):
    return a["dose"].shape == b["dose"].shape and all(np.allclose(p, q, atol=1e-3) for p, q in zip(a["axes"], b["axes"]))


def resample_to(src, dst):
    interp = RegularGridInterpolator(src["axes"], src["dose"], bounds_error=False, fill_value=0.0)
    return interp(np.stack(np.meshgrid(*dst["axes"], indexing="ij"), axis=-1))


def rasterize_roi(roi_contour, slice_z, slice_thickness, plane_shape, to_pixels):
    """
    (rows, columns, slices) mask of an ROI: each contour on its nearest slice, filled with XOR (holes);
    to_pixels(points, k) maps mm onto (row, column) of slice k. Returns (mask, skipped contours).
    """
    slice_z = np.asarray(slice_z, dtype=np.float64)
    mask = np.zeros((*plane_shape, len(slice_z)), dtype=bool)
    skipped = 0
    for contour in getattr(roi_contour, "ContourSequence", []):
        points = np.asarray(contour.ContourData, dtype=np.float64).reshape(-1, 3)
        k = int(np.argmin(np.abs(slice_z - points[0, 2])))
        if abs(slice_z[k] - points[0, 2]) > slice_thickness / 2:
            skipped += 1
            continue
        rows, cols = to_pixels(points, k)
        rr, cc = draw.polygon(rows, cols, shape=plane_shape)
        plane = np.zeros(plane_shape, dtype=bool)
        plane[rr, cc] = True
        mask[:, :, k] ^= plane
    return mask, skipped


def rasterize_structures(ds, path, grid, aliases, log):
    names = {r.ROINumber: str(r.ROIName).strip() for r in ds.StructureSetROISequence}
    z, y, x = grid["axes"]
    shape = grid["dose"].shape
    dz, dy, dx = grid["spacing"]
    to_pixels = lambda p, k: ((p[:, 1] - y[0]) / dy, (p[:, 0] - x[0]) / dx)
    masks = {}
    for roi in getattr(ds, "ROIContourSequence", []):
        raw_name = names.get(roi.ReferencedROINumber, str(roi.ReferencedROINumber))
        name = aliases.get(raw_name, raw_name)
        mask, skipped = rasterize_roi(roi, z, dz, shape[1:], to_pixels)
        mask = np.ascontiguousarray(np.moveaxis(mask, 2, 0))
        if skipped:
            log(f"  [warning] {os.path.basename(path)} / {raw_name}: {skipped} contours outside the dose grid slices")
        masks[name] = masks[name] | mask if name in masks else mask
    return masks


def _gamma_worker(task):
    import pymedphys
    ref_axes, ref_dose, ev_axes, ev_dose, dose_pct, dist_mm, ram_bytes = task
    return pymedphys.gamma(ref_axes, ref_dose, ev_axes, ev_dose, dose_percent_threshold=dose_pct,
                           distance_mm_threshold=dist_mm, lower_percent_dose_cutoff=LOW_DOSE_CUTOFF_PCT,
                           interp_fraction=10, max_gamma=2, local_gamma=False, quiet=True, ram_available=ram_bytes)


def run_all_gamma(ref, datasets, evaluated, ram_bytes):
    tasks = [(ref["axes"], ref["dose"], datasets[n]["axes"], datasets[n]["dose"], d_pct, dta, ram_bytes)
             for n in evaluated for d_pct, dta in GAMMA_CRITERIA]
    keys = [(n, d_pct, dta) for n in evaluated for d_pct, dta in GAMMA_CRITERIA]
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        for key, g in zip(keys, pool.imap(_gamma_worker, tasks)):
            yield key, g


def pass_rate(values):
    values = values[~np.isnan(values)]
    return (float(100 * (values <= 1).mean()) if values.size else float("nan")), int(values.size)


def surface(mask):
    return mask & ~ndi.binary_erosion(mask)


def geometry(a, b, spacing):
    dice = 2 * (a & b).sum() / (a.sum() + b.sum())
    surface_a, surface_b = surface(a), surface(b)
    da = ndi.distance_transform_edt(~surface_b, sampling=spacing)[surface_a]
    db = ndi.distance_transform_edt(~surface_a, sampling=spacing)[surface_b]
    dists = np.concatenate([da, db])
    shift = (np.array(np.nonzero(a)).mean(1) - np.array(np.nonzero(b)).mean(1)) * spacing
    voxel_cc = np.prod(spacing) / 1000
    return dict(dice=dice, hd95=np.percentile(dists, 95), msd=dists.mean(), centroid_shift_mm=np.linalg.norm(shift),
                shift_z=shift[0], shift_y=shift[1], shift_x=shift[2],
                volume_cc=a.sum() * voxel_cc, volume_ref_cc=b.sum() * voxel_cc)


def dose_at_volume(values, pct):
    return float(np.percentile(values, 100 - pct))


def dvh_metrics(dose, mask, structure, voxel_cc, exclude_zero):
    values = dose[mask]
    if exclude_zero:
        values = values[values > 0.01]
    if values.size == 0:
        return None
    m = dict(n_voxels=int(values.size), volume_cc=values.size * voxel_cc,
             Dmean=values.mean(), Dmin=values.min(), Dmax=values.max())
    if structure in TARGETS:
        d98, d95, d50, d2 = np.percentile(values, [2, 5, 50, 98])
        m.update(D98=float(d98), D95=float(d95), D50=float(d50), D2=float(d2))
    elif structure == LUNGS:
        m["D2"] = dose_at_volume(values, 2)
        for v in LUNG_V_GY:
            m[f"V{v:g}Gy_pct"] = 100 * (values >= v).mean()
    else:
        n_top = max(1, int(round(0.03 / voxel_cc)))
        top = np.partition(values, values.size - n_top)[values.size - n_top]
        m.update(D2=dose_at_volume(values, 2), D0_03cc=float(top))
    return m


def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def plot_gamma_analysis(ref_name, ref, evaluated, doses, gamma_maps, path):
    dmax = ref["dose"].max()
    z = int(np.unravel_index(ref["dose"].argmax(), ref["dose"].shape)[0])
    _, y, x = ref["axes"]
    X, Y = np.meshgrid(x - x[0], y - y[0])
    extent = [0, x[-1] - x[0], y[-1] - y[0], 0]
    ncol = len(evaluated) + 1
    ref_slice = ref["dose"][z]

    def contours(ax, d):
        ax.contour(X, Y, d, levels=[0.5 * dmax, 0.8 * dmax], colors=["white", "black"], linewidths=1.2)

    fig, axes = plt.subplots(3, ncol, figsize=(5 * ncol, 15), squeeze=False)
    for ax, name in zip(axes[0], [ref_name] + evaluated):
        d = doses[name]
        im = ax.imshow(d[z], cmap="jet", vmin=0, vmax=dmax * 1.1, extent=extent)
        contours(ax, d[z])
        ax.set_title(f"{name} (Dmax {d.max():.2f} Gy)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.015, pad=0.01, label="Dose (Gy)")

    axes[1, 0].axis("off")
    axes[1, 0].text(0.05, 0.5, f"Dose difference\n(evaluated - {ref_name})\nas % of the reference Dmax\n\n"
                    "Reference contours:\n50% (white), 80% (black)", fontsize=13, va="center")
    low = ref_slice < LOW_DOSE_CUTOFF_PCT / 100 * dmax
    for ax, name in zip(axes[1, 1:], evaluated):
        diff = 100 * (doses[name][z] - ref_slice) / dmax
        diff[low] = np.nan
        im2 = ax.imshow(diff, cmap="RdBu_r", vmin=-10, vmax=10, extent=extent)
        contours(ax, ref_slice)
        ax.set_title(f"{name} - {ref_name}", fontsize=13, fontweight="bold")
    fig.colorbar(im2, ax=axes[1].tolist(), fraction=0.015, pad=0.01, label="Difference (% Dmax)")

    d_pct, dta = MAIN_CRITERION
    axes[2, 0].axis("off")
    axes[2, 0].text(0.05, 0.5, f"Global gamma {d_pct}%/{dta}mm\n({LOW_DOSE_CUTOFF_PCT}% low-dose cutoff)\n\n"
                    "gamma <= 1: passing point\ngamma > 1: failing point", fontsize=13, va="center")
    cmap = ListedColormap(plt.cm.Greens(np.linspace(0.3, 0.9, 128)).tolist()
                          + plt.cm.Reds(np.linspace(0.4, 1, 128)).tolist())
    for ax, name in zip(axes[2, 1:], evaluated):
        g = gamma_maps[name]
        im3 = ax.imshow(g[z], cmap=cmap, vmin=0, vmax=2, extent=extent)
        contours(ax, ref_slice)
        ax.set_title(f"Gamma {name}: pass {pass_rate(g)[0]:.1f}%", fontsize=13, fontweight="bold")
    fig.colorbar(im3, ax=axes[2].tolist(), fraction=0.015, pad=0.01, label="gamma")
    for ax in axes.ravel():
        if ax.images:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
    fig.suptitle(f"Dose comparison - axial slice {z} (reference maximum dose)", fontsize=16, fontweight="bold")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_histogram(evaluated, gamma_maps, path):
    d_pct, dta = MAIN_CRITERION
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(evaluated):
        g = gamma_maps[name]
        ax.hist(g[~np.isnan(g)], bins=80, range=(0, 2), histtype="step", lw=2, color=STYLE[i + 1][0], density=True,
                label=f"{name} ({pass_rate(g)[0]:.1f}%)")
    ax.axvline(1, color="k", ls="--")
    ax.set_xlabel(f"gamma ({d_pct}%/{dta}mm)")
    ax.set_ylabel("density")
    ax.set_title(f"Gamma distribution (voxels with dose >= {LOW_DOSE_CUTOFF_PCT}% Dmax)")
    ax.legend()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_dvh(names, datasets, structures, path):
    ncols = min(4, len(structures))
    nrows = -(-len(structures) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.5 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(structures):]:
        ax.axis("off")
    xmax = max(datasets[n]["dose"].max() for n in names) * 1.02
    for ax, s in zip(axes, structures):
        curves = {}
        for i, n in enumerate(names):
            mask = datasets[n]["masks"].get(s)
            if mask is None or not mask.any():
                continue
            v = np.sort(datasets[n]["dose"][mask])[::-1]
            curves[n] = (v, np.arange(1, v.size + 1) / v.size * 100)
            c, ls, lw = STYLE[i]
            ax.plot(*curves[n], color=c, ls=ls, lw=lw, label=n)
        ax.set_title(s, fontsize=15, fontweight="bold")
        ax.set_xlabel("Dose per fraction (Gy)", fontsize=12)
        ax.set_ylabel("Volume (%)", fontsize=12)
        ax.grid(ls="--", alpha=0.5)
        ax.set_ylim(0, 102)
        if s in TARGETS and curves:
            ax.set_xlim(0, xmax)
            ax.legend(fontsize=11, loc="center", bbox_to_anchor=(0.72, 0.6))
            d50 = np.percentile(curves[names[0]][0], 50) if names[0] in curves else xmax / 2
            ins = ax.inset_axes([0.08, 0.08, 0.5, 0.5])
            for i, n in enumerate(names):
                if n in curves:
                    c, ls, lw = STYLE[i]
                    ins.plot(*curves[n], color=c, ls=ls, lw=lw)
            ins.set_xlim(0.8 * d50, xmax)
            ins.set_ylim(75, 101)
            ins.grid(ls="--", alpha=0.5)
            ins.set_title(f"zoom {0.8 * d50:.1f}-{xmax:.1f} Gy", fontsize=9)
        else:
            ax.legend(fontsize=11, loc="upper right")
    fig.suptitle("Cumulative DVHs - each dose with its own structures", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Dose, gamma, geometry and DVH comparison of RTDOSE/RTSTRUCT sets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None, help="Default: <input_dir>/results")
    parser.add_argument("--reference", default="real_ct")
    parser.add_argument("--evaluated", nargs="+", default=["dir_ct", "dir_cbct", "baseline"])
    parser.add_argument("--dose_pattern", default="RTDose_{name}.dcm")
    parser.add_argument("--struct_pattern", default="RTstruct_{name}.dcm")
    parser.add_argument("--aliases", default=None, help='JSON mapping ROI name -> standard name, e.g. \'{"Tumour": "GTV"}\'')
    parser.add_argument("--ram-gb", type=float, default=1.2, help="RAM available to pymedphys (GB)")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "results")
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)
    aliases = dict(DEFAULT_ALIASES, **(json.loads(args.aliases) if args.aliases else {}))
    log = Logger()
    names = [args.reference] + args.evaluated
    ram = int(args.ram_gb * 1e9)

    datasets = {}
    for n in names:
        d = load_dose(os.path.join(args.input_dir, args.dose_pattern.format(name=n)))
        datasets[n] = d
        log(f"{n:10s} dose {d['dose'].shape} spacing {np.round(d['spacing'], 4).tolist()} mm | "
            f"Dmax {d['dose'].max():.3f} {d['units']}")
    ref = datasets[args.reference]
    voxel_cc = np.prod(ref["spacing"]) / 1000

    for n in names:
        d = datasets[n]
        struct_path = os.path.join(args.input_dir, args.struct_pattern.format(name=n))
        struct_ds = pydicom.dcmread(struct_path)
        d["masks"] = rasterize_structures(struct_ds, struct_path, d, aliases, log)
        on_ref = same_grid(d, ref)
        d["masks_on_ref"] = d["masks"] if on_ref else rasterize_structures(struct_ds, struct_path, ref, aliases, log)
        d["dose_on_ref"] = d["dose"] if on_ref else resample_to(d, ref)
        if not on_ref:
            log(f"  [info] {n}: dose grid differs from the reference, resampled for voxel-wise analyses")
        d_cc = np.prod(d["spacing"]) / 1000
        log(f"{n:10s} structures: " + ", ".join(f"{k} {v.sum() * d_cc:.1f} cc" for k, v in d["masks"].items()))

    log("\nData-consistency checks")
    for n in names:
        d = datasets[n]
        for t in TARGETS:
            if t in d["masks"]:
                mask = d["masks"][t]
                v = d["dose"][mask]
                zero = v < 0.01
                slices = sorted({int(k) for k in np.nonzero(mask & (d["dose"] < 0.01))[0]})
                log(f"  {n:10s} {t}: {zero.sum()} voxels with dose ~0 ({100 * zero.mean():.1f}%), "
                    f"{(v < 0.5 * d['dose'].max()).sum()} below 50% Dmax, slices with zero dose {slices}")
    ref_pos = ref["dose"] > 0
    for n in args.evaluated:
        e_pos = datasets[n]["dose_on_ref"] > 0
        log(f"  calculation region {n} vs {args.reference}: voxels with dose only in the reference "
            f"{int((ref_pos & ~e_pos).sum())}, only in {n} {int((~ref_pos & e_pos).sum())}")

    log("\nGamma index (global, low-dose cutoff "
        f"{LOW_DOSE_CUTOFF_PCT}% of the reference Dmax, reference = {args.reference})")
    gamma_rows, gamma_maps = [], {}
    for (n, d_pct, dta), g in run_all_gamma(ref, datasets, args.evaluated, ram):
        v = g[~np.isnan(g)]
        rate, npts = pass_rate(g)
        mean_g, p95 = float(v.mean()), float(np.percentile(v, 95))
        gamma_rows.append(dict(evaluated_dose=n, reference_dose=args.reference, criterion=f"{d_pct}%/{dta}mm",
                               normalization="global", low_dose_cutoff_pct=LOW_DOSE_CUTOFF_PCT,
                               pass_rate_pct=round(rate, 2), mean_gamma=round(mean_g, 3),
                               gamma_p95=round(p95, 2), n_points=npts))
        if (d_pct, dta) == MAIN_CRITERION:
            gamma_maps[n] = g
        log(f"  {n:10s} {d_pct}%/{dta}mm: pass {rate:6.2f}% | mean gamma {mean_g:.3f} | "
            f"p95 {p95:.2f} | points {npts}")
    write_csv(out("gamma_results.csv"), gamma_rows)

    dmax = ref["dose"].max()
    margin = max(1, int(round(EDGE_MARGIN_MM / ref["spacing"][1])))
    region_masks = [(label, ref["dose"] >= thr * dmax) for label, thr in DOSE_REGIONS]
    edge_structure = np.ones((1, 2 * margin + 1, 2 * margin + 1))
    region_rows = []
    for n in args.evaluated:
        e = datasets[n]["dose_on_ref"]
        g = gamma_maps[n]
        diff = 100 * (e - ref["dose"]) / dmax
        for label, m in region_masks:
            dm = diff[m]
            region_rows.append(dict(evaluated_dose=n, region=label, n_voxels=int(m.sum()),
                                    gamma_3_2_pass_pct=round(pass_rate(g[m])[0], 2),
                                    mean_diff_pct_dmax=round(float(dm.mean()), 2),
                                    mean_abs_diff_pct_dmax=round(float(np.abs(dm).mean()), 2),
                                    voxels_within_3pct=round(float(100 * (np.abs(dm) <= 3).mean()), 1)))
        common = ref_pos & (e > 0)
        inner = ndi.binary_erosion(common, structure=edge_structure)
        region_rows.append(dict(evaluated_dose=n, region="common calculation region",
                                gamma_3_2_pass_pct=round(pass_rate(g[common])[0], 2)))
        region_rows.append(dict(evaluated_dose=n, region=f"common region >= {EDGE_MARGIN_MM:g} mm from edge",
                                gamma_3_2_pass_pct=round(pass_rate(g[inner])[0], 2)))
    write_csv(out("dose_comparison_regions.csv"), region_rows)

    doses_on_ref = {n: datasets[n]["dose_on_ref"] for n in names}
    plot_gamma_analysis(args.reference, ref, args.evaluated, doses_on_ref, gamma_maps, out("gamma_analysis.png"))
    plot_gamma_histogram(args.evaluated, gamma_maps, out("gamma_histogram.png"))

    ref_masks = ref["masks"]
    oars = [k for k in ref_masks if k not in TARGETS and k != LUNGS]
    geo_rows = []
    for n in args.evaluated:
        for s in TARGETS + oars + [LUNGS]:
            if s in ref_masks and s in datasets[n]["masks_on_ref"] and ref_masks[s].any() \
                    and datasets[n]["masks_on_ref"][s].any():
                r = geometry(datasets[n]["masks_on_ref"][s], ref_masks[s], ref["spacing"])
                geo_rows.append(dict(dataset=n, structure=s, **{k: round(float(v), 3) for k, v in r.items()}))
    write_csv(out("geometry_vs_reference.csv"), geo_rows)

    dvh_structures = [s for s in TARGETS + [LUNGS] + oars if s in ref_masks]
    metric_rows = []
    for exclude in (False, True):
        for s in dvh_structures:
            ref_m = dvh_metrics(ref["dose"], ref_masks[s], s, voxel_cc, exclude)
            for n in names:
                d = datasets[n]
                if s not in d["masks"]:
                    continue
                m = dvh_metrics(d["dose"], d["masks"][s], s, np.prod(d["spacing"]) / 1000, exclude)
                if m is None:
                    continue
                row = dict(voxels="without dose=0" if exclude else "all", structure=s, dataset=n)
                for k, v in m.items():
                    row[k] = round(float(v), 3)
                    if k not in ("n_voxels", "volume_cc") and n != args.reference and ref_m:
                        row[f"{k}_diff_pct"] = round(100 * (v - ref_m[k]) / ref_m[k], 2) if ref_m[k] else ""
                metric_rows.append(row)
    write_csv(out("dvh_metrics.csv"), metric_rows)

    gin_rows = []
    d_pct, dta = MAIN_CRITERION
    for n in args.evaluated:
        for s in dvh_structures:
            rate, npts = pass_rate(gamma_maps[n][ref_masks[s]])
            gin_rows.append(dict(dataset=n, structure=f"{s} ({args.reference})", criterion=f"{d_pct}%/{dta}mm",
                                 n_points=npts, pass_rate_pct=round(rate, 2) if npts else ""))
    write_csv(out("gamma_in_structures.csv"), gin_rows)

    plot_dvh(names, datasets, dvh_structures, out("dvh.png"))

    log("\nGeometry against the reference")
    for r in geo_rows:
        log(f"  {r['dataset']:10s} {r['structure']:30s} Dice {r['dice']:.3f} | HD95 {r['hd95']:.2f} mm | "
            f"centroid shift {r['centroid_shift_mm']:.2f} mm")
    log("\nGamma inside the reference structures")
    for r in gin_rows:
        log(f"  {r['dataset']:10s} {r['structure']:40s} {r['criterion']}: {r['pass_rate_pct']}% ({r['n_points']} points)")
    log(f"\nResults written to {os.path.abspath(args.output_dir)}")
    log.save(out("analysis_summary.txt"))


if __name__ == "__main__":
    sys.exit(main())
