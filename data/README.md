# data/

Not in git, except this file. The notebook and the scripts expect:

```
data/
├── LungCT/                         Learn2Reg lung CT (main README, "Data")
│   ├── training/
│   │   ├── scans/                  case_001_exp.nii.gz, case_001_insp.nii.gz, ...
│   │   └── lungMasks/
│   ├── imagesTr/, masksTr/         LungCT_0001_0000.nii.gz, ...
│   ├── landmarksTr/                LungCT_0001_0000.csv, ...
│   ├── keypointsTr/
│   ├── imagesTs/                   LungCT_0021_0000.nii.gz, ...
│   ├── masksTs/
│   └── keypointsTs/
└── clinical/4dlung_107/            4D-Lung case study, TCIA subject 107_HM10395
    ├── images/                     4dlung_107_0000.nii.gz (fixed), 4dlung_107_0001.nii.gz (moving)
    ├── masks/                      lung masks with the same names
    ├── real_ct_dicom/              DICOM series of the real CT (reference grid of the export)
    ├── baseline/                   planning CT slices + RTSTRUCT
    ├── real_ct/                    real CT slices + RTSTRUCT
    ├── cbct.nii.gz                 CBCT on the common grid (case figures)
    └── <run>/                      dosimetry inputs of one experiment (full_network, frozen_backbone):
                                    RTDose_<case>.dcm, RTstruct_<case>.dcm, ct_<case>/ for
                                    <case> in real_ct, baseline, dir_ct, dir_cbct
```

Every path, and the case name `CLINICAL_CASE`, can be changed in the first code cell of `notebooks/main.ipynb`. The
suffixes `_0000` and `_0001` mark the fixed and the moving image.

## 4D-Lung case study

The case study uses subject `107_HM10395` of the TCIA collection **4D-Lung** ("Data from 4D Lung Imaging of NSCLC
Patients", version 2, doi:[10.7937/K9/TCIA.2016.ELN8YGLE](https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE)), distributed
under the [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/) license and the
[TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/). Cite the data set
and its descriptor ([hugo2016_4dlung_data](../REFERENCES.md#hugo2016_4dlung_data),
[hugo2017_4dlung](../REFERENCES.md#hugo2017_4dlung)) when you use it, and do not attempt to identify the subject.

Series used (TCIA series descriptions; study dates in the files are shifted by TCIA and are not used):

| Role | Series description | Modality | Files | Series Instance UID |
|---|---|---|---:|---|
| planning CT (moving, `baseline`) | `P4^P107^S300^I00003, Gated, 0.0%A` | CT | 84 | `1.3.6.1.4.1.14519.5.2.1.6834.5010.143988608531594213289446897958` |
| contours of the planning CT | `P4^P107^S300^I00003, Gated, 0.0%A` | RTSTRUCT | 1 | `2.25.309267149608437150045269105686764431388.1` |
| real CT (reference, `real_ct`, fixed of the CT case) | `P4^P107^S302^I00003, Gated, 0.0%A` | CT | 110 | `1.3.6.1.4.1.14519.5.2.1.6834.5010.315261063272600094208003975975` |
| contours of the real CT | `P4^P107^S302^I00003, Gated, 0.0%A` | RTSTRUCT | 1 | `2.25.309267149608437150045269105686764431388.21` |
| CBCT (fixed of the CBCT case) | `P4^P107^S107^I0, Gated, 0.0%` | CT (4D-CBCT phase) | 50 | `1.3.6.1.4.1.14519.5.2.1.6834.5010.719264704051895933838862732754` |

All three images are the 0% phase of the respective 4D acquisition (S300 and S302: 4D fan-beam CT studies; S107: 4D
cone-beam CT). 4D-Lung provides no RTSTRUCT on the CBCT studies of this subject. The series were resampled onto a
common grid of 448 × 288 × 32 voxels of 0.9766 × 0.9766 × 3 mm before registration and planning; that preprocessing
step is not part of this repository.

### Download

From the NCI Imaging Data Commons (collection `4d_lung`, checked with `idc-index` 0.13.5, IDC data release v25):

```bash
pip install idc-index
python - <<'EOF'
from idc_index import IDCClient
IDCClient().download_from_selection(
    seriesInstanceUID=[
        "1.3.6.1.4.1.14519.5.2.1.6834.5010.143988608531594213289446897958",
        "2.25.309267149608437150045269105686764431388.1",
        "1.3.6.1.4.1.14519.5.2.1.6834.5010.315261063272600094208003975975",
        "2.25.309267149608437150045269105686764431388.21",
        "1.3.6.1.4.1.14519.5.2.1.6834.5010.719264704051895933838862732754",
    ],
    downloadDir="data/clinical/4dlung_107/tcia",
)
EOF
```

Alternatively, from the [4D-Lung collection page](https://www.cancerimagingarchive.net/collection/4d-lung/) build a
manifest restricted to subject `107_HM10395` and download it with the NBIA Data Retriever, keeping the five series
above. The whole subject is 290 series (7.6 GB); only these five are needed.

### Contours

| Structure | Source | Image | Role in the TTO | Role in the evaluation |
|---|---|---|---|---|
| GTV (`Tumor_c00` of the 4D-Lung RTSTRUCT) | 4D-Lung, expert radiation oncologist | planning CT, real CT | Dice term (tumor) | reference GTV (Dice, HD95, DVH) |
| lung lobes, heart, great vessels, esophagus, trachea, spinal cord, deep back muscles, vertebrae, ribs, scapulae, clavicles, sternum, costal cartilage | TotalSegmentator 2.14.0 (installed version at the time of writing; nnU-Net 2.8.1), task `total` (CT), run on CPU in the 3D Slicer 5.12 SlicerTotalSegmentator extension; fast mode not recorded | planning CT, real CT | organ or bone Dice terms; smaller vessels, left atrial appendage, back muscles, costal cartilage and right ribs 1-7 held out | references of the organs at risk |
| PTV, Lungs-GTV, SpinalCord_PRV5, Heart_total, GreatVessels, ChestWall, Ring_PTV_10-30mm, BODY | derived by `scripts/build_structures.py` | planning CT, real CT, warped CTs | not used | planning and evaluation |

For both the CT and the CBCT case the fixed labels of the TTO are the structures of the real CT: there are no
contours on the CBCT.
