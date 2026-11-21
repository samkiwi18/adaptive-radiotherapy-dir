"""DICOM export of the warped CT (using the real CT as template) and RTSTRUCT creation."""
import datetime
import glob
import os

import nibabel as nib
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from scipy import ndimage as ndi
from skimage import draw, measure

RT_STRUCTURE_SET_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
ROI_COLORS = {"Lungs": [0, 255, 255], "Tumor": [255, 0, 0], "Vertebra": [0, 255, 0], "Patient": [128, 128, 128],
              "BODY": [128, 128, 128]}
TUMOR_KEYWORDS = ("tumor", "tumour", "gtv")
LUNG_KEYWORDS = ("lung", "polmon")
BONE_WORDS = {"vertebra", "vertebrae", "rib", "sternum", "scapula", "clavicle",
              "vertebre", "costa", "coste", "sterno", "scapola", "clavicola"}


def is_tumor(name):
    return any(k in name.lower() for k in TUMOR_KEYWORDS)


def is_lung(name):
    return any(k in name.lower() for k in LUNG_KEYWORDS)


def is_bone(name):
    return any(w in BONE_WORDS for w in name.lower().replace("_", " ").split())


def read_dicom_series(folder, with_pixels=False):
    """Reads the slices of a DICOM series sorted by increasing Z position."""
    slices = []
    for f in sorted(glob.glob(os.path.join(folder, "*.dcm"))):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=not with_pixels)
        except Exception:
            continue
        if hasattr(ds, "ImagePositionPatient") and (not with_pixels or hasattr(ds, "PixelData")):
            slices.append(ds)
    slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    return slices


def read_reference_grid(dicom_dir, nifti_fallback=None):
    """Shape (X, Y, Z), spacing and origin of the reference CT (DICOM, otherwise NIfTI)."""
    slices = [ds for ds in read_dicom_series(dicom_dir) if hasattr(ds, "PixelSpacing")]
    if slices:
        zs = [float(ds.ImagePositionPatient[2]) for ds in slices]
        ds0 = slices[0]
        grid = {
            "shape": (int(ds0.Columns), int(ds0.Rows), len(slices)),
            "spacing": [float(ds0.PixelSpacing[1]), float(ds0.PixelSpacing[0]),
                        abs(zs[1] - zs[0]) if len(zs) > 1 else 3.0],
            "origin": [float(ds0.ImagePositionPatient[0]), float(ds0.ImagePositionPatient[1]), zs[0]],
        }
        print(f"Reference grid from {len(slices)} DICOM slices in {dicom_dir}")
    elif nifti_fallback and os.path.exists(nifti_fallback):
        nii = nib.load(nifti_fallback)
        grid = {"shape": nii.shape[:3], "spacing": [float(z) for z in nii.header.get_zooms()[:3]],
                "origin": [float(v) for v in nii.affine[:3, 3]]}
        print(f"No DICOM in {dicom_dir}: grid read from {nifti_fallback}")
    else:
        raise FileNotFoundError(f"No reference grid in {dicom_dir} nor {nifti_fallback}")
    print(f"-> shape {grid['shape']} | spacing {grid['spacing']} | origin {grid['origin']}")
    return grid


def _find_rtstruct(folder):
    for f in sorted(glob.glob(os.path.join(folder, "*.dcm"))):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") == "RTSTRUCT":
            return ds
    raise FileNotFoundError(f"No RTSTRUCT in {folder}")


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


def read_rtstruct_masks(path):
    """Every ROI of an RTSTRUCT as an (X, Y, Z) mask on the CT grid next to it; returns (masks, colours)."""
    folder = os.path.dirname(path) if os.path.isfile(path) else path
    ds = pydicom.dcmread(path) if os.path.isfile(path) else _find_rtstruct(folder)
    slices = [s for s in read_dicom_series(folder) if hasattr(s, "PixelSpacing")]
    if not slices:
        raise FileNotFoundError(f"No CT series next to the RTSTRUCT in {folder}")
    ref = slices[0]
    shape = (int(ref.Rows), int(ref.Columns), len(slices))
    zs = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    dz = abs(zs[1] - zs[0]) if len(zs) > 1 else float(getattr(ref, "SliceThickness", 1.0) or 1.0)
    row_step, col_step = float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1])
    iop = np.array([float(v) for v in ref.ImageOrientationPatient])
    col_dir, row_dir = iop[:3], iop[3:]
    positions = [np.array([float(v) for v in s.ImagePositionPatient]) for s in slices]

    def to_pixels(points, k):
        delta = points - positions[k]
        return delta @ row_dir / row_step, delta @ col_dir / col_step

    names = {r.ROINumber: str(r.ROIName).strip() for r in ds.StructureSetROISequence}

    masks, colors = {}, {}
    for roi in getattr(ds, "ROIContourSequence", []):
        name = names.get(roi.ReferencedROINumber, str(roi.ReferencedROINumber))
        volume, skipped = rasterize_roi(roi, zs, dz, shape[:2], to_pixels)
        if skipped:
            print(f"[WARNING] {name}: {skipped} contours outside the CT slices")
        masks[name] = volume.transpose(1, 0, 2)
        if hasattr(roi, "ROIDisplayColor"):
            colors[name] = [int(v) for v in roi.ROIDisplayColor]
    print(f"RTSTRUCT {folder}: {len(masks)} ROIs on the grid {(shape[1], shape[0], shape[2])}")
    return masks, colors


def common_structures(masks_moving, masks_fixed):
    """ROIs with the same name and non-empty in both RTSTRUCTs, as (tumor name, other names)."""
    names = [n for n in masks_moving if n in masks_fixed and masks_moving[n].any() and masks_fixed[n].any()]
    excluded = sorted((set(masks_moving) | set(masks_fixed)) - set(names))
    if excluded:
        print(f"Structures not in both RTSTRUCTs (excluded): {', '.join(excluded)}")
    tumors = [n for n in names if is_tumor(n)]
    if not tumors:
        raise ValueError(f"No tumor ROI among the common structures: {names}")
    return tumors[0], [n for n in names if n != tumors[0]]


def to_fixed_grid(vol_np, fixed_shape, is_mask=False):
    """Resamples a volume (X, Y, Z) onto the fixed CT grid."""
    t = torch.from_numpy(np.ascontiguousarray(vol_np, dtype=np.float32))[None, None]
    out = F.interpolate(t, size=tuple(fixed_shape), mode="trilinear", align_corners=True)[0, 0].numpy()
    return (out > 0.5).astype(np.float32) if is_mask else out


def export_dicom_from_template(volume_np, output_dir, ref_dicom_dir, series_name="CT_Warped_TTO",
                               patient_name="Patient_TTO_Warped", patient_id="TTO_0001"):
    """Writes volume_np (X, Y, Z, HU) as a DICOM series with the geometric tags of the real CT."""
    template = read_dicom_series(ref_dicom_dir, with_pixels=True)
    if not template:
        raise FileNotFoundError(f"No DICOM template in {ref_dicom_dir}")
    if len(template) != volume_np.shape[2]:
        print(f"[WARNING] {len(template)} template slices but the volume has {volume_np.shape[2]}: "
              f"exporting the first {min(len(template), volume_np.shape[2])}")

    os.makedirs(output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(output_dir, "*.dcm")):
        os.remove(f)

    study_uid, series_uid = generate_uid(), generate_uid()
    volume_dcm = volume_np.transpose(1, 0, 2)  # (rows=Y, columns=X, Z)

    for z, ds in enumerate(template[:volume_dcm.shape[2]]):
        sop_uid = generate_uid()
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = study_uid, series_uid, sop_uid
        if hasattr(ds, "file_meta"):
            ds.file_meta.MediaStorageSOPInstanceUID = sop_uid  # required by 3D Slicer
        ds.SeriesDescription = series_name
        ds.PatientName, ds.PatientID = patient_name, patient_id
        ds.InstanceNumber = z + 1

        intercept = float(getattr(ds, "RescaleIntercept", -1024))
        slope = float(getattr(ds, "RescaleSlope", 1))
        pixel = np.round((volume_dcm[:, :, z] - intercept) / slope)
        if getattr(ds, "PixelRepresentation", 1) == 1:
            pixel = np.clip(pixel, -32768, 32767).astype(np.int16)
        else:
            pixel = np.clip(pixel, 0, 65535).astype(np.uint16)
        ds.PixelData = pixel.tobytes()
        pydicom.dcmwrite(os.path.join(output_dir, f"slice_{z:03d}.dcm"), ds)

    print(f"[OK] CT exported to {output_dir} ({min(len(template), volume_dcm.shape[2])} slices)")


def _slice_contours(mask_2d, fill_holes=True, tolerance=0.5):
    """Closed (row, column) contours of a slice; padding closes them at the border, filling removes holes."""
    m = ndi.binary_opening(mask_2d, structure=np.ones((2, 2)))
    if fill_holes:
        m = ndi.binary_fill_holes(m)
    if not m.any():
        return []
    contours = []
    for c in measure.find_contours(np.pad(m.astype(np.uint8), 1), level=0.5):
        c = measure.approximate_polygon(c - 1.0, tolerance=tolerance)
        if len(c) < 4:  # approximate_polygon repeats the first point at the end
            continue
        contours.append(c[:-1] if np.allclose(c[0], c[-1]) else c)
    return contours


def create_rtstruct(segmentations, output_filepath, ref_dicom_dir, fill_holes=True, colors=None):
    """RTSTRUCT with one ROI per (X, Y, Z) mask, referenced to the DICOM series in ref_dicom_dir."""
    slices = read_dicom_series(ref_dicom_dir)
    if not slices:
        raise FileNotFoundError(f"No reference DICOM in {ref_dicom_dir}")
    ref = slices[0]
    now = datetime.datetime.now()

    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = RT_STRUCTURE_SET_STORAGE
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(output_filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.PatientName, ds.PatientID = ref.PatientName, ref.PatientID
    ds.PatientBirthDate = getattr(ref, "PatientBirthDate", "19800101")
    ds.PatientSex = getattr(ref, "PatientSex", "O")
    ds.StudyInstanceUID = ref.StudyInstanceUID
    ds.StudyDate = getattr(ref, "StudyDate", now.strftime("%Y%m%d"))
    ds.StudyTime = getattr(ref, "StudyTime", now.strftime("%H%M%S"))
    ds.Modality = "RTSTRUCT"
    ds.Manufacturer = "SwinUNETR_Model"
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = RT_STRUCTURE_SET_STORAGE
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.ApprovalStatus = "UNAPPROVED"
    ds.StructureSetLabel = ds.StructureSetName = "Seg_Warped"
    ds.StructureSetDate, ds.StructureSetTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")

    def image_reference(f):
        item = Dataset()
        item.ReferencedSOPClassUID, item.ReferencedSOPInstanceUID = f.SOPClassUID, f.SOPInstanceUID
        return item

    series = Dataset()
    series.SeriesInstanceUID = ref.SeriesInstanceUID
    series.ContourImageSequence = Sequence([image_reference(f) for f in slices])
    study = Dataset()
    study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    study.ReferencedSOPInstanceUID = ref.StudyInstanceUID
    study.RTReferencedSeriesSequence = Sequence([series])
    frame = Dataset()
    frame.FrameOfReferenceUID = ref.FrameOfReferenceUID
    frame.RTReferencedStudySequence = Sequence([study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([frame])

    ds.StructureSetROISequence, ds.ROIContourSequence, ds.RTROIObservationsSequence = Sequence(), Sequence(), Sequence()

    for number, (name, volume) in enumerate(segmentations.items(), start=1):
        volume = volume.transpose(1, 0, 2)  # (rows=Y, columns=X, Z)

        roi = Dataset()
        roi.ROINumber, roi.ReferencedFrameOfReferenceUID = number, ref.FrameOfReferenceUID
        roi.ROIName, roi.ROIGenerationAlgorithm = name, "AUTOMATIC"
        ds.StructureSetROISequence.append(roi)

        contour_seq = Sequence()
        for z, slice_ds in enumerate(slices[:volume.shape[2]]):
            ipp = [float(v) for v in slice_ds.ImagePositionPatient]
            row_step, col_step = float(slice_ds.PixelSpacing[0]), float(slice_ds.PixelSpacing[1])
            iop = np.array([float(v) for v in slice_ds.ImageOrientationPatient])
            col_dir, row_dir = iop[:3], iop[3:]  # direction along the columns (x) and along the rows (y)

            for contour in _slice_contours(volume[:, :, z] > 0.5, fill_holes):
                rows, cols = contour[:, 0:1], contour[:, 1:2]
                points_mm = np.array(ipp) + cols * col_step * col_dir + rows * row_step * row_dir
                item = Dataset()
                item.ContourGeometricType = "CLOSED_PLANAR"
                item.NumberOfContourPoints = len(points_mm)
                item.ContourData = [round(float(v), 3) for v in points_mm.ravel()]
                item.ContourImageSequence = Sequence([image_reference(slice_ds)])
                contour_seq.append(item)

        if len(contour_seq):
            roi_contour = Dataset()
            roi_contour.ROIDisplayColor = (colors or {}).get(name, ROI_COLORS.get(name, [255, 255, 255]))
            roi_contour.ReferencedROINumber = number
            roi_contour.ContourSequence = contour_seq
            ds.ROIContourSequence.append(roi_contour)

        observation = Dataset()
        observation.ObservationNumber = observation.ReferencedROINumber = number
        observation.ROIObservationLabel = name
        observation.RTROIInterpretedType = "GTV" if is_tumor(name) else "ORGAN"
        ds.RTROIObservationsSequence.append(observation)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    pydicom.dcmwrite(output_filepath, ds, enforce_file_format=True)
    print(f"[OK] RTSTRUCT saved to {output_filepath}")
