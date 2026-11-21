# matlab/: dose recalculation with matRad

These scripts optimize one treatment plan on the real CT of the day and recalculate it, with the same beams and the
same weights, on the other anatomies (baseline CT, registered CT, registered CBCT). They are not part of matRad and do
not contain matRad.

| File | What it does |
|---|---|
| `compute_dose.m` | optimizes the plan on `patient_real_ct.mat` once and saves it to `plan_reference.mat`, then runs a forward dose calculation with the fixed weights on each anatomy and writes `<case>_dose.mat` |
| `show_isocenter.m` | optional check: isocenter against the PTV centroid on three orthogonal views, saved as `isocenter_check.png` |
| `export_dose.m` | checks that every `<case>_dose.mat` uses the weights of `plan_reference.mat`, then exports CT, RTSTRUCT and RTDOSE of each anatomy to `<case>_dicom/` |
| `isgpuarray.m`, `dicomuid.m` | stand-ins for two MATLAB functions that GNU Octave lacks; needed only under Octave |

The cases are `real_ct`, `baseline`, `dir_cbct` and `dir_ct`, the names used by `scripts/dose_analysis.py`.

## Version

- matRad 3.2.3 (`matRad/matRad_version.m`), from <https://github.com/e0404/matRad>. matRad is GPL-3.0 and is not
  redistributed here.
- The reported plan was built and tested in GNU Octave 8.4; MATLAB may give slightly different weights
  (notes at the end of `results/dosimetry/treatment_plan.txt`).

## Dose grid

The reported doses were computed on a 3 × 3 × 3 mm dose grid: it is the grid stored in the plan (`pln`) of the input
`.mat` files and the one stated in `results/dosimetry/treatment_plan.txt`. matRad interpolates the dose back to the CT
grid for export: every RTDOSE file of the reported runs has `PixelSpacing` 0.9766 × 0.9766 mm, `SliceThickness` 3 mm and
a `GridFrameOffsetVector` step of 3 mm on 32 frames of 288 × 448 pixels. `compute_dose.m` uses `doseGridRes = 3` (mm) by
default; ICRU 91 and AAPM recommend at most 2 mm for SBRT.

## Execution order

1. Build the planning structures from the TotalSegmentator RTSTRUCTs:
   `python scripts/build_structures.py --ct_dir <folder with CT slices and RTSTRUCT>`.
2. Import each anatomy (real CT, baseline CT, registered CT and CBCT exported by the notebook) into matRad with its
   DICOM import and save `ct`, `cst` and `pln` as `patient_real_ct.mat`, `patient_baseline.mat`, `patient_dir_ct.mat`
   and `patient_dir_cbct.mat`.
3. Put these scripts and the four `.mat` files in the matRad root folder (the scripts call `matRad_rc`), then run, in
   MATLAB or Octave:

   ```matlab
   compute_dose     % plan on patient_real_ct.mat + forward dose on the 4 anatomies
   show_isocenter   % optional
   export_dose      % <case>_dicom/ folders with CT, RTSTRUCT and RTDOSE
   ```

   Set `forceReoptimization` / `forceRecalculation` to `true` in `compute_dose.m`, or delete `plan_reference.mat` and
   `*_dose.mat`, to recompute.
4. Copy the RTDOSE and RTSTRUCT of each anatomy into one folder as `RTDose_<case>.dcm` and `RTstruct_<case>.dcm` and
   run `python scripts/dose_analysis.py --input_dir <folder>`.

## Notes

- `dicomuid.m` returns random UIDs under the DICOM standard root `1.2.840.10008`: acceptable for research exports, not
  globally unique.
- `*.mat` files hold patient data and are excluded by `.gitignore`.
