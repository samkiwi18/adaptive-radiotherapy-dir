import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
LOG = os.path.join(RESULTS, "training")
RUNS = {"frozen": os.path.join(RESULTS, "dosimetry", "frozen_backbone"),
        "full": os.path.join(RESULTS, "dosimetry", "full_network")}
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs", "figures")

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 140,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
COL = {"baseline": "#7f7f7f", "frozen_dir_ct": "#9ecae1", "frozen_dir_cbct": "#fdae6b", "full_dir_ct": "#2171b5",
       "full_dir_cbct": "#d94801"}
LAB = {"baseline": "baseline (no registration)", "frozen_dir_ct": "DIR CT - frozen backbone",
       "frozen_dir_cbct": "DIR CBCT - frozen backbone", "full_dir_ct": "DIR CT - full network",
       "full_dir_cbct": "DIR CBCT - full network"}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def training_curves():
    phases = [("warmup.csv", "WARMUP"), ("finetune.csv", "FINETUNE"),
              ("finetune_D.csv", "FINETUNE_D"), ("finetune_E.csv", "FINETUNE_E")]
    colors = ["#6a51a3", "#2171b5", "#238b45", "#d94801"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    offset = 0
    for (fname, name), c in zip(phases, colors):
        rows = read_csv(os.path.join(LOG, fname))
        ep = np.array([int(r["epoch"]) for r in rows]) + offset
        tre = np.array([float(r["val_tre_mm"]) for r in rows])
        loss = np.array([float(r["train_loss"]) for r in rows])
        axes[0].plot(ep, tre, "-o", ms=2.5, lw=1.2, color=c, label=name)
        best = np.argmin(tre)
        axes[0].plot(ep[best], tre[best], "*", ms=11, color=c, mec="k", mew=0.6)
        axes[1].plot(ep, loss, "-", lw=1.3, color=c, label=name)
        axes[0].axvline(offset + 0.5, color="k", lw=0.6, ls=":")
        axes[1].axvline(offset + 0.5, color="k", lw=0.6, ls=":")
        offset = ep[-1]
    axes[0].axhline(14.65, color="gray", ls="--", lw=1)
    axes[0].text(2, 14.72, "identity (no registration) 14.65 mm", fontsize=8, color="gray")
    axes[0].set_xlabel("cumulative epoch")
    axes[0].set_ylabel("validation TRE (mm)")
    axes[0].set_title("Validation TRE (stars = best model of each phase)")
    axes[0].legend(fontsize=8, loc="center right", ncol=2)
    axes[1].set_xlabel("cumulative epoch")
    axes[1].set_ylabel("training loss")
    axes[1].set_title("Training loss (weights change between phases)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"))
    plt.close(fig)


def keypoint_diagnostic():
    train = {"004": (5.96, 4.05), "005": (14.44, 8.39), "006": (13.86, 9.72), "007": (8.08, 4.86),
             "008": (14.06, 6.22), "009": (16.69, 8.79), "010": (7.75, 5.19), "011": (8.37, 4.93),
             "012": (11.63, 7.23), "013": (7.61, 4.84), "014": (10.01, 5.22), "015": (9.54, 4.09),
             "016": (3.87, 2.27), "017": (22.09, 11.32), "018": (4.11, 2.68), "019": (18.56, 9.06),
             "020": (6.93, 4.86)}
    val = {"001": (20.24, 18.61), "002": (16.51, 11.73), "003": (9.60, 7.23)}
    fig, ax = plt.subplots(figsize=(11, 3.4))
    names = list(train) + list(val)
    x = np.arange(len(names))
    before = [train.get(n, val.get(n))[0] for n in names]
    after = [train.get(n, val.get(n))[1] for n in names]
    is_val = [n in val for n in names]
    ax.bar(x - 0.2, before, 0.4, color="#bdbdbd", label="identity")
    ax.bar(x + 0.2, after, 0.4, color=["#d94801" if v else "#2171b5" for v in is_val],
           label="network (blue: training, orange: validation)")
    for xi, b, a in zip(x, before, after):
        ax.text(xi, max(b, a) + 0.4, f"{100 * (1 - a / b):.0f}%", ha="center", fontsize=7)
    ax.axvline(len(train) - 0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, [f"case {n}" for n in names], rotation=60, fontsize=8)
    ax.set_ylabel("mean keypoint error (mm)")
    ax.set_title("Keypoint error after FINETUNE_E: mean reduction 42% on training, 21% on validation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "keypoint_diagnostic.png"))
    plt.close(fig)


def tto_validation():
    cases = ["case 001", "case 002", "case 003", "mean"]
    none = [20.18, 15.60, 8.14, 14.64]
    net = [18.72, 10.34, 6.48, 11.85]
    tto = [2.64, 3.29, 2.91, 2.94]
    fold = [3.83, 3.12, 8.72, None]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    x = np.arange(len(cases))
    for k, (vals, lab, c) in enumerate(((none, "no registration", "#bdbdbd"), (net, "network", "#2171b5"),
                                       (tto, "network + TTO", "#d94801"))):
        bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=lab, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    for xi, f in zip(x, fold):
        if f is not None:
            ax.text(xi + 0.27, 5.2, f"fold\n{f:.1f}%", ha="center", fontsize=7, color="#d94801")
    ax.set_xticks(x, cases)
    ax.set_ylabel("TRE (mm)")
    ax.set_title("TTO on the validation patients (lambda_reg = 0.01)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tto_validation.png"))
    plt.close(fig)


def geometry():
    data = {}
    for run, path in RUNS.items():
        for r in read_csv(os.path.join(path, "geometry_vs_reference.csv")):
            key = "baseline" if r["dataset"] == "baseline" else f"{run}_{r['dataset']}"
            data[(key, r["structure"])] = r
    structs = ["GTV", "PTV", "Lungs-GTV", "Heart_total", "GreatVessels", "ChestWall", "esophagus", "trachea",
               "spinal cord", "SpinalCord_PRV5"]
    labels = ["GTV", "PTV", "Lungs-GTV", "Heart", "Great vessels", "Chest wall", "Esophagus", "Trachea",
              "Spinal cord", "Cord PRV5"]
    keys = ["baseline", "frozen_dir_ct", "frozen_dir_cbct", "full_dir_ct", "full_dir_cbct"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    x = np.arange(len(structs))
    w = 0.16
    for k, key in enumerate(keys):
        dice = [float(data[(key, s)]["dice"]) for s in structs]
        hd = [float(data[(key, s)]["hd95"]) for s in structs]
        axes[0].bar(x + (k - 2) * w, dice, w, color=COL[key], label=LAB[key])
        axes[1].bar(x + (k - 2) * w, hd, w, color=COL[key], label=LAB[key])
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Dice vs real CT contours")
    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[1].set_ylabel("HD95 (mm)")
    axes[1].set_xticks(x, labels, rotation=25)
    axes[0].set_title("Geometric agreement of the propagated contours with the contours of the real CT")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles, labels_, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.savefig(os.path.join(OUT, "case_geometry_bars.png"))
    plt.close(fig)


def load_dvh(path):
    rows = []
    for row in read_csv(path):
        rows.append({k: v if k in ("voxels", "structure", "dataset") else (float(v) if v else None)
                     for k, v in row.items()})
    return rows


def dvh_metrics():
    rows = {}
    for run, path in RUNS.items():
        for r in load_dvh(os.path.join(path, "dvh_metrics.csv")):
            if r["voxels"] != "all":
                continue
            key = r["dataset"] if r["dataset"] in ("real_ct", "baseline") else f"{run}_{r['dataset']}"
            rows[(key, r["structure"])] = r
    metrics = [("GTV", "D98"), ("GTV", "D95"), ("GTV", "D50"), ("GTV", "D2"), ("PTV", "D98"), ("PTV", "D95"),
               ("PTV", "D50"), ("PTV", "D2")]
    keys = ["baseline", "frozen_dir_ct", "frozen_dir_cbct", "full_dir_ct", "full_dir_cbct"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    x = np.arange(len(metrics))
    w = 0.16
    for k, key in enumerate(keys):
        vals = [rows[(key, s)][f"{m}_diff_pct"] for s, m in metrics]
        axes[0].bar(x + (k - 2) * w, vals, w, color=COL[key], label=LAB[key])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x, [f"{s} {m}" for s, m in metrics], rotation=35, fontsize=8)
    axes[0].set_ylabel("difference vs real CT (%)")
    axes[0].set_title("Target dose-volume metrics")
    axes[0].set_ylim(-50, 5)
    handles, labels_ = axes[0].get_legend_handles_labels()

    oars = [("Lungs-GTV", "Dmean"), ("Lungs-GTV", "V2Gy_pct"), ("Heart_total", "D0_03cc"),
            ("GreatVessels", "D0_03cc"), ("ChestWall", "D0_03cc"), ("esophagus", "D0_03cc"),
            ("trachea", "D0_03cc"), ("spinal cord", "D0_03cc"), ("SpinalCord_PRV5", "D0_03cc")]
    olabels = ["Lungs Dmean", "Lungs V2Gy", "Heart D0.03cc", "Vessels D0.03cc", "Chest wall D0.03cc",
               "Esophagus D0.03cc", "Trachea D0.03cc", "Cord D0.03cc", "Cord PRV D0.03cc"]
    x = np.arange(len(oars))
    for k, key in enumerate(keys):
        vals = [rows[(key, s)][f"{m}_diff_pct"] for s, m in oars]
        axes[1].bar(x + (k - 2) * w, vals, w, color=COL[key], label=LAB[key])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x, olabels, rotation=35, fontsize=8)
    axes[1].set_ylabel("difference vs real CT (%)")
    axes[1].set_title("Organ-at-risk metrics")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels_, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.savefig(os.path.join(OUT, "case_dvh_metrics_bars.png"))
    plt.close(fig)
    return rows


def gamma():
    res = {}
    for run, path in RUNS.items():
        for r in read_csv(os.path.join(path, "gamma_results.csv")):
            key = "baseline" if r["evaluated_dose"] == "baseline" else f"{run}_{r['evaluated_dose']}"
            res[(key, r["criterion"])] = float(r["pass_rate_pct"])
    reg = {}
    for run, path in RUNS.items():
        for r in read_csv(os.path.join(path, "dose_comparison_regions.csv")):
            key = "baseline" if r["evaluated_dose"] == "baseline" else f"{run}_{r['evaluated_dose']}"
            reg[(key, r["region"])] = float(r["gamma_3_2_pass_pct"])
    crit = ["3%/3mm", "3%/2mm", "2%/2mm", "1%/1mm"]
    regions = ["dose>=10%", "dose>=50%", "dose>=80%", "common region >= 3 mm from edge"]
    rlab = ["dose >= 10%", "dose >= 50%", "dose >= 80%", "common region\n>= 3 mm from edge"]
    keys = ["baseline", "frozen_dir_ct", "frozen_dir_cbct", "full_dir_ct", "full_dir_cbct"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    w = 0.16
    for k, key in enumerate(keys):
        axes[0].bar(np.arange(4) + (k - 2) * w, [res[(key, c)] for c in crit], w, color=COL[key], label=LAB[key])
        axes[1].bar(np.arange(4) + (k - 2) * w, [reg[(key, r)] for r in regions], w, color=COL[key], label=LAB[key])
    axes[0].set_xticks(np.arange(4), crit)
    axes[0].set_ylim(90, 100.5)
    axes[0].set_ylabel("global gamma pass rate (%)")
    axes[0].set_title("Whole dose grid (10% low-dose cutoff)")
    axes[1].set_xticks(np.arange(4), rlab, fontsize=8)
    axes[1].set_ylim(88, 100.5)
    axes[1].set_ylabel("2%/2mm pass rate (%)")
    axes[1].set_title("By dose region (2%/2mm)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels_, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.savefig(os.path.join(OUT, "case_gamma_rates_bars.png"))
    plt.close(fig)


def warp(field, ux, uy):
    ny, nx = field.shape[1:]
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.stack([map_coordinates(field[c], [yy + uy, xx + ux], order=1, mode="nearest") for c in range(2)])


def scaling_squaring_demo():
    n = 128
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = n / 2, n / 2
    s = 0.11 * n
    bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))
    vx = 26.0 * bump
    vy = 8.0 * bump * (yy - cy) / s
    v = np.stack([vy, vx])
    u = v / 2 ** 7
    for _ in range(7):
        u = u + warp(u, u[1], u[0])

    def jac(uy, ux):
        duy_dy, duy_dx = np.gradient(uy)
        dux_dy, dux_dx = np.gradient(ux)
        return (1 + duy_dy) * (1 + dux_dx) - duy_dx * dux_dy

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    step = 6
    axes[0].quiver(xx[::step, ::step], yy[::step, ::step], vx[::step, ::step], vy[::step, ::step],
                   angles="xy", scale_units="xy", scale=1, color="#2171b5", width=0.004)
    axes[0].set_title("stationary velocity field v")
    for ax, (uy, ux), title in ((axes[1], (vy, vx), "phi = id + v  (folding)"),
                                (axes[2], (u[0], u[1]), "phi = exp(v), scaling & squaring")):
        det = jac(uy, ux)
        ax.imshow(det <= 0, cmap="Reds", alpha=0.35, origin="upper", extent=(-0.5, n - 0.5, n - 0.5, -0.5))
        for k in range(0, n, 4):
            ax.plot(xx[k, :] + ux[k, :], yy[k, :] + uy[k, :], color="k", lw=0.5)
            ax.plot(xx[:, k] + ux[:, k], yy[:, k] + uy[:, k], color="k", lw=0.5)
        ax.set_title(f"{title}\nvoxels with det(J) <= 0: {100 * (det <= 0).mean():.2f}%")
    for ax in axes:
        ax.set_xlim(0, n - 1)
        ax.set_ylim(n - 1, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaling_squaring.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    training_curves()
    keypoint_diagnostic()
    tto_validation()
    geometry()
    dvh_metrics()
    gamma()
    scaling_squaring_demo()
    print("done")
