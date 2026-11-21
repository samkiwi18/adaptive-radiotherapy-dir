import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
SRC = os.path.join(RESULTS, "tto")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs", "figures")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.dpi": 125,
                     "axes.spines.top": False, "axes.spines.right": False})


def validation_test():
    text = open(os.path.join(SRC, "tto_validation_test.txt")).read()
    rows = {}
    for group in ("VALIDATION", "TEST"):
        block = text.split(f"\n{group}:")[1].split("MEAN")[0]
        rows[group] = []
        for line in block.splitlines():
            m = re.match(r"(\S+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)%", line.strip())
            if m:
                name = m.group(1)
                short = name.split("_")[1] if name.startswith("case") else name.split("_")[1]
                rows[group].append((short, *[float(m.group(k)) for k in range(2, 6)]))
    return rows


def regularization_study():
    text = open(os.path.join(SRC, "tto_regularization_study.txt")).read()
    means = {}
    for line in text.splitlines():
        m = re.match(r"\s*([\d.]+)\s+\|\s+validation\s+\|\s+MEAN \(3 patients\)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)%", line)
        if m:
            means[float(m.group(1))] = tuple(float(m.group(k)) for k in (2, 3, 4))
    return means


rows = validation_test()
study = regularization_study()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), gridspec_kw={"width_ratios": [0.8, 1.7, 1.0]})
colors = ["#bdbdbd", "#6baed6", "#08519c"]
labels = ["no registration", "network", "network + TTO"]
for ax, group, title, ylabel in ((axes[0], "VALIDATION", "Validation: TRE on landmarks", "TRE (mm)"),
                                 (axes[1], "TEST", "Test: error on Learn2Reg keypoints", "keypoint error (mm)")):
    data = rows[group]
    x = np.arange(len(data) + 1)
    w = 0.27
    for k in range(3):
        vals = [r[1 + k] for r in data]
        vals.append(np.mean(vals))
        ax.bar(x + (k - 1) * w, vals, w, color=colors[k], label=labels[k])
    ax.set_xticks(x, [r[0] for r in data] + ["mean"], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
axes[0].legend(fontsize=7, loc="upper right")
lams = sorted(study)
ax = axes[2]
ax.plot(lams, [study[lam][0] for lam in lams], "o-", color="#08519c", label="TRE (mm)")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda_{reg}$")
ax.set_ylabel("mean TRE (mm)", color="#08519c")
ax.set_ylim(2.5, 3.6)
ax2 = ax.twinx()
ax2.plot(lams, [study[lam][1] for lam in lams], "s--", color="#e6550d", label="folding, whole volume")
ax2.plot(lams, [study[lam][2] for lam in lams], "^:", color="#a63603", label="folding, lungs")
ax2.set_ylabel("voxels with det J <= 0 (%)", color="#e6550d")
ax2.grid(False)
ax.set_title("Regularization study (validation)", fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tto_regularization.png"))
print(rows)
print(study)
