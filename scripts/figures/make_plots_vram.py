import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
SRC = os.path.join(RESULTS, "vram_profiling")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs", "figures")

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 140,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})

SETUPS = ["MONAI", "efficient", "MONAI + checkpointing", "efficient + checkpointing"]
SETUP_LABELS = ["MONAI", "efficient", "MONAI\n+ checkpointing", "efficient\n+ checkpointing"]
SUMMARY_COLS = ["peak_MiB", "baseline_MiB", "weights_MiB", "gradients_MiB", "adam_MiB", "activations_MiB",
                "forward_retained_MiB", "forward_s", "backward_s", "step_s"]
PHASE_COLS = ["network_forward", "losses", "network_backward", "optimizer", "other"]
SCENARIOS = ["inference", "forward + backward", "section 9 TTO", "section 9 TTO, frozen backbone"]


def parse_tables(path):
    text = open(path).read()
    summary_block = text.split("Summary (MiB")[1].split("TTO peak per phase")[0]
    phase_block = text.split("TTO peak per phase (MiB):")[1]
    summary, phases = {}, {}
    for line in summary_block.splitlines():
        for setup in sorted(SETUPS, key=len, reverse=True):
            stripped = line.strip()
            if stripped.startswith(setup + " "):
                rest = stripped[len(setup):].strip()
                for scenario in sorted(SCENARIOS, key=len, reverse=True):
                    if rest.startswith(scenario):
                        values = rest[len(scenario):].split()
                        summary[(setup, scenario)] = dict(zip(SUMMARY_COLS, [float(v) for v in values]))
                        break
                break
    for line in phase_block.splitlines():
        for setup in sorted(SETUPS, key=len, reverse=True):
            stripped = line.strip()
            if stripped.startswith(setup + " "):
                rest = stripped[len(setup):].strip()
                for scenario in sorted(SCENARIOS, key=len, reverse=True):
                    if rest.startswith(scenario):
                        values = rest[len(scenario):].split()
                        phases[(setup, scenario)] = dict(zip(PHASE_COLS, [float(v) for v in values]))
                        break
                break
    return summary, phases


def parse_blocks(path):
    text = open(path).read()
    blocks = {}
    sections = re.split(r"===== (.+?) =====", text)
    for k in range(1, len(sections), 2):
        setup, body = sections[k], sections[k + 1]
        for scenario in ("inference", "forward + backward"):
            part = body.split(f"Per-block memory, {scenario}:")[1].split("\n\n")[0]
            rows = {}
            for line in part.strip().splitlines()[1:]:
                f = line.split()
                rows[f[0]] = dict(weights=float(f[2]), output=float(f[3]), retained=float(f[4]),
                                  peak=float(f[5]), time=float(f[6]))
            blocks[(setup, scenario)] = rows
    return blocks


def tto_breakdown(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    parts = [("baseline_MiB", "already on the GPU (weights, images, masks)", "#bdbdbd"),
             ("gradients_MiB", "gradients", "#6baed6"),
             ("adam_MiB", "Adam states", "#2171b5"),
             ("activations_MiB", "activations + temporaries", "#e6550d")]
    x = np.arange(len(SETUPS))
    w = 0.38
    for k, (scenario, offset, hatch) in enumerate((("section 9 TTO", -w / 2, None),
                                                    ("section 9 TTO, frozen backbone", w / 2, "//"))):
        bottom = np.zeros(len(SETUPS))
        for col, label, color in parts:
            vals = np.array([summary[(s, scenario)][col] for s in SETUPS]) / 1024
            axes[0].bar(x + offset, vals, w, bottom=bottom, color=color, hatch=hatch, edgecolor="white",
                        label=label if k == 0 else None)
            bottom += vals
        for xi, s in zip(x, SETUPS):
            peak = summary[(s, scenario)]["peak_MiB"] / 1024
            axes[0].text(xi + offset, peak + 0.2, f"{peak:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, SETUP_LABELS, fontsize=8)
    axes[0].set_ylabel("TTO peak (GiB)")
    axes[0].set_title("Peak of the 4D-Lung TTO: full network (solid) / frozen backbone (hatched)", fontsize=9)
    axes[0].set_ylim(0, 16.5)
    handles, labels_ = axes[0].get_legend_handles_labels()

    for scenario, color, label in (("section 9 TTO", "#e6550d", "full network"),
                                   ("section 9 TTO, frozen backbone", "#2171b5", "frozen backbone")):
        peaks = [summary[(s, scenario)]["peak_MiB"] / 1024 for s in SETUPS]
        times = [summary[(s, scenario)]["step_s"] for s in SETUPS]
        axes[1].plot(times, peaks, "o", ms=7, color=color, label=label)
        offsets = [(-30, 6), (6, -4), (6, 6), (6, -10)] if label == "full network" else [(-12, 8), (8, 4), (8, 4), (8, -10)]
        for t, p, name, off in zip(times, peaks, ["MONAI", "eff.", "MONAI+ckpt", "eff.+ckpt"], offsets):
            if label == "frozen backbone" and name == "MONAI+ckpt":
                continue
            text = "MONAI (±ckpt)" if label == "frozen backbone" and name == "MONAI" else name
            axes[1].annotate(text, (t, p), textcoords="offset points", xytext=off, fontsize=7)
    axes[1].set_xlabel("time per TTO step (s)")
    axes[1].set_ylabel("TTO peak (GiB)")
    axes[1].set_title("Memory / time trade-off", fontsize=9)
    axes[1].legend(fontsize=8, loc="center right")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels_, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.savefig(os.path.join(OUT, "vram_tto.png"))
    plt.close(fig)


def phase_peaks(phases):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    names = ["network forward", "losses", "network backward", "optimizer step", "other"]
    x = np.arange(len(PHASE_COLS))
    w = 0.2
    for k, (s, c) in enumerate(zip(SETUPS, ["#7f7f7f", "#fd8d3c", "#6baed6", "#08519c"])):
        vals = [phases[(s, "section 9 TTO")][p] / 1024 for p in PHASE_COLS]
        ax.bar(x + (k - 1.5) * w, vals, w, label=s, color=c)
    ax.set_xticks(x, names)
    ax.set_ylabel("peak in the phase (GiB)")
    ax.set_title("4D-Lung TTO (full network): peak of each phase of a step", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vram_phases.png"))
    plt.close(fig)


def block_retained(blocks):
    names = ["backbone.swinViT", "backbone.encoder1", "backbone.encoder2", "backbone.decoder2",
             "backbone.decoder1", "integrator"]
    labels = ["swinViT", "encoder1", "encoder2", "decoder2", "decoder1", "integrator"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(names))
    w = 0.2
    for k, (s, c) in enumerate(zip(SETUPS, ["#7f7f7f", "#fd8d3c", "#6baed6", "#08519c"])):
        retained = [blocks[(s, "forward + backward")][n]["retained"] for n in names]
        peak = [blocks[(s, "inference")][n]["peak"] for n in names]
        axes[0].bar(x + (k - 1.5) * w, retained, w, color=c, label=s)
        axes[1].bar(x + (k - 1.5) * w, peak, w, color=c, label=s)
    axes[0].set_title("Memory kept alive for the backward (MiB)", fontsize=10)
    axes[1].set_title("Peak increase inside the block, inference (MiB)", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "vram_blocks.png"))
    plt.close(fig)


def quality_check(path):
    text = open(path).read()
    groups = ["lung Dice", "tumor Dice", "organs (loss) Dice", "bones (loss) Dice", "held-out Dice"]
    stats = {}
    block = text.split("Mean ± std per attention:")[1].split("DVF difference")[0]
    current = None
    for line in block.splitlines():
        f = line.split()
        if not f or f[0] == "attention":
            continue
        if f[-3] == "mean":
            current = " ".join(f[:-3])
            stats[(current, "mean")] = (float(f[-2]), float(f[-1]))
        elif f[0] == "std" and current is not None:
            stats[(current, "std")] = (float(f[1]), float(f[2]))
    pairs = {}
    pblock = text.split("DVF difference between pairs of runs")[1]
    for line in pblock.splitlines():
        m = re.match(r"(\w+ vs \w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line.strip())
        if m:
            pairs[m.group(1)] = [float(m.group(k)) for k in (2, 3, 4)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(groups))
    for k, (att, c) in enumerate((("MONAI", "#7f7f7f"), ("efficient", "#08519c"))):
        means = [stats[(g, "mean")][k] for g in groups]
        stds = [stats[(g, "std")][k] for g in groups]
        axes[0].errorbar(x + (k - 0.5) * 0.18, means, yerr=stds, fmt="o", color=c, capsize=4, label=att)
    axes[0].set_xticks(x, ["lungs", "tumor", "organs\n(loss)", "bones\n(loss)", "held-out"])
    axes[0].set_ylabel("Dice (mean ± std, 10 runs)")
    axes[0].set_title("4D-Lung TTO repeated 10 times per attention", fontsize=10)
    axes[0].legend(fontsize=8)
    order = ["MONAI vs MONAI", "efficient vs efficient", "MONAI vs efficient"]
    xx = np.arange(len(order))
    axes[1].bar(xx - 0.2, [pairs[p][0] for p in order], 0.4, color="#9ecae1", label="whole volume")
    axes[1].bar(xx + 0.2, [pairs[p][1] for p in order], 0.4, color="#2171b5", label="inside the lungs")
    for xi, p in zip(xx, order):
        axes[1].text(xi, max(pairs[p][:2]) + 0.08, f"max {pairs[p][2]:.0f} mm", ha="center", fontsize=7)
    axes[1].set_xticks(xx, ["MONAI\nvs MONAI", "efficient\nvs efficient", "MONAI\nvs efficient"], fontsize=8)
    axes[1].set_ylabel("mean |difference of DVF| (mm)")
    axes[1].set_title("DVF difference between pairs of runs", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "attention_quality.png"))
    plt.close(fig)
    return stats, pairs


if __name__ == "__main__":
    summary, phases = parse_tables(os.path.join(SRC, "vram_profiling.txt"))
    blocks = parse_blocks(os.path.join(SRC, "vram_profiling.txt"))
    assert len(summary) == 16 and len(phases) == 8, (len(summary), len(phases))
    tto_breakdown(summary)
    phase_peaks(phases)
    block_retained(blocks)
    stats, pairs = quality_check(os.path.join(SRC, "attention_quality.txt"))
    print(stats)
    print(pairs)
    for s in SETUPS:
        print(s, summary[(s, "section 9 TTO")]["peak_MiB"], summary[(s, "section 9 TTO")]["step_s"])
