# scripts/

Command-line entry points. Run them from the repository root after `pip install -e .`. The two dosimetry tools are also
installed as console commands (`dir-dose-analysis`, `dir-build-structures`) and can be run as modules
(`python -m registration.dose_analysis.dose_analysis`).

| Script | What it does | Output |
|---|---|---|
| `register_pair.py --moving M --fixed F --output_dir DIR` | registers a NIfTI pair with the trained network (`--weights`), optionally refined by test-time optimization (`--tto`, `--steps`, `--switch_step`, `--lr`, `--lambda_reg`, optional `--moving_mask` / `--fixed_mask`); `--low_memory` for CPUs and small GPUs | `DIR/warped.nii.gz`, `DIR/dvf.nii.gz` (voxels), `DIR/jacobian_stats.json` |
| `build_structures.py --ct_dir DIR` | planning structures (GTV, PTV, lungs minus GTV, spinal cord PRV, heart, great vessels, chest wall, PTV ring, BODY) from each TotalSegmentator RTSTRUCT in `DIR` and its CT series | `DIR/RTSTRUCT_TPS/TPS_<name>.dcm` (`--output_dir`, `--output_pattern`) |
| `dose_analysis.py --input_dir DIR` | gamma, dose differences, geometry and DVH of `RTDose_<case>.dcm` / `RTstruct_<case>.dcm` against the reference (`--reference real_ct`, `--evaluated dir_ct dir_cbct baseline`) | `DIR/results/` (`--output_dir`): `gamma_results.csv`, `dose_comparison_regions.csv`, `geometry_vs_reference.csv`, `dvh_metrics.csv`, `gamma_in_structures.csv`, `analysis_summary.txt`, `gamma_analysis.png`, `gamma_histogram.png`, `dvh.png` |
| `figures/make_plots.py`, `make_plots_tto.py`, `make_plots_vram.py` | plots of the reported results, read from `results/` | PNG files in `docs/figures/` (or the folder given as first argument) |
| `figures/make_case_figures.py --case_dir data/clinical/4dlung_107` | 4D-Lung figures from the existing outputs of an experiment (no registration or dose calculation is rerun): registration onto the CBCT, doses and differences, gamma maps (recomputed from the RTDOSE files), DVH, flicker GIF, checkerboards | `docs/figures/case_*`, `results/case_figures.json` |

`dose_analysis.py` runs the gamma index in a separate process with pymedphys; `--ram-gb` (default 1.2) sets the memory
pymedphys may use. ROI names are mapped with `--aliases` (JSON, for example `'{"Tumour": "GTV"}'`); `Tumor` is mapped to
`GTV` by default. On the 448 × 288 × 32 grid of the 4D-Lung case the whole analysis takes about 20 minutes on a
4-core CPU (indicative).
