# results/

Recorded outputs of the reported experiments: the source of every number in the main README and in `docs/`.
Training and test-time optimization ran on an NVIDIA A100; the dosimetric comparison on a CPU.

| Path | Content | Produced by |
|---|---|---|
| `training/warmup.*`, `finetune*.*` | per-epoch CSVs and reports of the training phases | notebook sections 3-4 |
| `training/keypoint_diagnostic.txt` | keypoint error of the trained network on training and validation pairs | notebook section 5 |
| `tto/tto_validation_test.txt` | TTO on the Learn2Reg validation and test pairs | notebook section 6 |
| `tto/tto_regularization_study.txt` | TRE and folding against the TTO smoothness weight | notebook section 7 |
| `keypoints/` | automatic keypoint extraction on the Learn2Reg test pairs, plots of the Learn2Reg case `LungCT_0030` (Learn2Reg lung CT, CC BY 4.0) | notebook section 8 |
| `vram_profiling/vram_profiling.txt`, `vram_per_block_<setup>.png` | GPU memory of the 4D-Lung TTO for four attention / checkpointing setups | notebook section 10 |
| `vram_profiling/attention_quality.txt` | 4D-Lung TTO repeated 10 times with each attention implementation | notebook section 10.1 |
| `vram_profiling/memory_snapshot_analysis.txt` | analysis of the CUDA memory snapshot of the optimized TTO | notebook section 10.2 and memory_viz |
| `dosimetry/frozen_backbone/` | dosimetric comparison of the 4D-Lung case study, TTO with `freeze_backbone=True` | `scripts/dose_analysis.py` |
| `dosimetry/full_network/` | the same with `freeze_backbone=False` | `scripts/dose_analysis.py` |
| `dosimetry/treatment_plan.txt` | report of the matRad SBRT plan shared by both runs | matRad plan of `matlab/compute_dose.m` |
| `case_figures.json` | numbers quoted in the captions of the 4D-Lung figures | `scripts/figures/make_case_figures.py` |

## Notes

- The training CSVs and the `.txt` reports of the training phases were compiled from the printed output of the
  training; the notebook now writes that output to `training/training_log.txt`.
- The `dosimetry/` folders derive from subject `107_HM10395` of the TCIA collection 4D-Lung (CC BY 3.0,
  [hugo2016_4dlung_data](../REFERENCES.md#hugo2016_4dlung_data)). They were produced on 2026-11-21 by a single run of the
  current `dose_analysis.py` on the RTDOSE / RTSTRUCT files of each experiment, with the case names `real_ct`,
  `dir_ct`, `dir_cbct`, `baseline`. The organ-at-risk references are TotalSegmentator contours, the GTV reference is
  the expert contour of 4D-Lung (`data/README.md`, *Contours*).
- Dose recalculation: in `frozen_backbone` the dose of `dir_ct` was computed on the HU-corrected warped CT
  (`CT_Warped_TTO`), the dose of `dir_cbct` on the warped CT without correction (`CT_Warped_TTO_noJac`); in
  `full_network` both on the warped CT without correction. The RTDOSE files do not reference their CT series; the
  matRad input of each dose (`ct.cubeHU`) is identical voxel by voxel to the CT series stored with it.
- In `dose_comparison_regions.csv` the column `gamma_3_2_pass_pct` holds the pass rate of the main criterion of
  `dose_analysis.py`, 2%/2 mm, despite its name.
- The TTO is not deterministic on the GPU: repeated runs of the same configuration differ. For example the mean
  validation TRE with `lambda_reg = 0.5` is 3.29 mm in `tto/tto_validation_test.txt` and 2.98 mm in
  `tto/tto_regularization_study.txt`.
- `vram_profiling/tto_memory_snapshot.pickle` (8.7 MB) is not included; notebook section 10.2 rebuilds it.
- Some reports refer to notebook cells (`In[n]`) and sections of the notebook that produced them.
