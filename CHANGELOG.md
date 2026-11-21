# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-21

### Added

- `registration` package: SwinUNETR registration network with a velocity head, scaling-and-squaring integration,
  spatial transformer, MIND / NCC / Dice / inverse-consistency / smoothness / point losses, training loop with
  curriculum and resume, two-stage test-time optimization with structure supervision, mass-preserving HU correction,
  DICOM CT and RTSTRUCT export, keypoint extraction, memory-efficient window attention and activation checkpointing,
  GPU memory profiling.
- `registration.dose_analysis`: planning structures from TotalSegmentator RTSTRUCTs and gamma / DVH / geometry
  comparison of RTDOSE and RTSTRUCT sets.
- `scripts/register_pair.py` for the registration of a NIfTI pair, with optional test-time optimization.
- matRad scripts for a fixed-plan dose recalculation and DICOM export.
- Notebook covering training, test-time optimization, keypoint extraction, 4D-Lung export and GPU profiling.
- Documentation of method, architecture, GPU memory and pipeline; results of the reported experiments; figures of the
  4D-Lung case study regenerated from the existing outputs by `scripts/figures/make_case_figures.py`.
- CPU unit tests and a GitHub Actions workflow (ruff and pytest on Python 3.11 and 3.12).
