import argparse
import copy
import datetime
import glob
import os
import re
import sys

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from scipy import ndimage as ndi
from skimage import draw, measure

TUMOR = "Tumor"
LOBES = ["superior lobe of left lung", "inferior lobe of left lung", "superior lobe of right lung",
         "middle lobe of right lung", "inferior lobe of right lung"]
HEART = ["heart", "left atrial appendage"]
GREAT_VESSELS = ["aorta", "pulmonary vein", "brachiocephalic trunk", "right subclavian artery", "left subclavian artery",
                 "right common carotid artery", "left common carotid artery", "left brachiocephalic vein",
                 "right brachiocephalic vein", "superior vena cava"]
RIB = re.compile(r"^(left|right) rib \d+$")
CHEST_WALL = ["sternum", "costal cartilage"]
SPINAL_CORD = "spinal cord"
KEPT = ["esophagus", "trachea", "thyroid", "T8 vertebra", "T7 vertebra", "T6 vertebra", "T5 vertebra", "T4 vertebra",
        "T3 vertebra", "T2 vertebra", "T1 vertebra", "left scapula", "right scapula", "left clavicle", "right clavicle", SPINAL_CORD,
        "left deep back muscle", "right deep back muscle"]

BODY_HU = -400
PTV_MARGIN_MM = 5.0
PRV_MARGIN_MM = 5.0
RING_MM = (10.0, 30.0)
NEW_STRUCTURES = ["PTV", "Lungs-GTV", "SpinalCord_PRV5", "Heart_total", "GreatVessels", "ChestWall",
                  "Ring_PTV_10-30mm", "BODY"]
COLORS = {"PTV": [255, 140, 0], "Lungs-GTV": [0, 200, 255], "SpinalCord_PRV5": [255, 255, 0],
          "Heart_total": [200, 0, 100], "GreatVessels": [160, 0, 255], "ChestWall": [230, 230, 180],
          "Ring_PTV_10-30mm": [0, 255, 0], "BODY": [128, 128, 128]}
TYPES = {"GTV": "GTV", "PTV": "PTV", "Ring_PTV_10-30mm": "CONTROL", "BODY": "EXTERNAL"}


def scan_dicom(folder):
    series, structs = {}, []
    for f in sorted(glob.glob(os.path.join(folder, "*.dcm"))):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        modality = getattr(ds, "Modality", "")
        if modality == "RTSTRUCT":
            structs.append(f)
        elif modality == "CT" and hasattr(ds, "ImagePositionPatient"):
            series.setdefault(str(ds.SeriesInstanceUID), dict(frame=str(getattr(ds, "FrameOfReferenceUID", "")),
                                                              files=[]))["files"].append(f)
    if not structs:
        raise FileNotFoundError(f"No RTSTRUCT in {folder}")
    if not series:
        raise FileNotFoundError(f"No CT slices in {folder}")
    return series, structs


def referenced_series(ds, series):
    uids, frames = set(), set()
    for frame in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
        frames.add(str(frame.FrameOfReferenceUID))
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for ref in getattr(study, "RTReferencedSeriesSequence", []):
                uids.add(str(ref.SeriesInstanceUID))
    for roi in getattr(ds, "StructureSetROISequence", []):
        frames.add(str(getattr(roi, "ReferencedFrameOfReferenceUID", "")))
    matches = [u for u in series if u in uids] or [u for u in series if series[u]["frame"] in frames]
    if len(matches) == 1:
        return matches[0]
    if not matches and len(series) == 1:
        return next(iter(series))
    raise ValueError(f"cannot match the RTSTRUCT to a single CT series ({len(matches)} candidates)")


def load_ct(files):
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
        except Exception:
            continue
        if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelData"):
            slices.append(ds)
    if not slices:
        raise FileNotFoundError("No CT slices with pixel data")
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    hu = np.stack([s.pixel_array * float(getattr(s, "RescaleSlope", 1)) + float(getattr(s, "RescaleIntercept", 0))
                   for s in slices], axis=-1)
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    ref = slices[0]
    row_step, col_step = (float(v) for v in ref.PixelSpacing)
    dz = abs(z[1] - z[0]) if len(z) > 1 else float(getattr(ref, "SliceThickness", 1.0) or 1.0)
    iop = np.array([float(v) for v in ref.ImageOrientationPatient])
    return dict(slices=slices, hu=hu, z=z, spacing=np.array([row_step, col_step, dz]),
                col_dir=iop[:3], row_dir=iop[3:])


def rasterize(roi_contour, ct, name):
    shape = ct["hu"].shape
    mask = np.zeros(shape, bool)
    skipped = 0
    for contour in getattr(roi_contour, "ContourSequence", []):
        p = np.asarray(contour.ContourData, float).reshape(-1, 3)
        k = int(np.argmin(np.abs(ct["z"] - p[0, 2])))
        if abs(ct["z"][k] - p[0, 2]) > ct["spacing"][2] / 2:
            skipped += 1
            continue
        d = p - np.array([float(v) for v in ct["slices"][k].ImagePositionPatient])
        rr, cc = draw.polygon(d @ ct["row_dir"] / ct["spacing"][0], d @ ct["col_dir"] / ct["spacing"][1],
                              shape=shape[:2])
        plane = np.zeros(shape[:2], bool)
        plane[rr, cc] = True
        mask[:, :, k] ^= plane
    if skipped:
        print(f"  [warning] {name}: {skipped} contours outside the CT slices")
    return mask


def union(masks, names, shape):
    present = [masks[n] for n in names if n in masks]
    return np.logical_or.reduce(present) if present else np.zeros(shape, bool)


def expand(mask, margin_mm, spacing):
    if not mask.any():
        return mask
    return ndi.distance_transform_edt(~mask, sampling=spacing) <= margin_mm


def body_mask(hu, structures):
    structures = np.logical_or.reduce(structures)
    body = (hu > BODY_HU) | structures
    for k in range(body.shape[2]):
        body[:, :, k] = ndi.binary_fill_holes(body[:, :, k])
    labels, n = ndi.label(body)
    if n > 1:
        body = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return body | structures


def image_reference(ds):
    item = Dataset()
    item.ReferencedSOPClassUID, item.ReferencedSOPInstanceUID = ds.SOPClassUID, ds.SOPInstanceUID
    return item


def mask_to_contours(mask, ct):
    items = []
    row_step, col_step = ct["spacing"][:2]
    for k, slice_ds in enumerate(ct["slices"]):
        if not mask[:, :, k].any():
            continue
        ipp = np.array([float(v) for v in slice_ds.ImagePositionPatient])
        for c in measure.find_contours(np.pad(mask[:, :, k].astype(np.uint8), 1), 0.5):
            c = measure.approximate_polygon(c - 1.0, tolerance=0.1)
            if np.allclose(c[0], c[-1]):
                c = c[:-1]
            if len(c) < 3:
                continue
            points = ipp + c[:, 1:2] * col_step * ct["col_dir"] + c[:, 0:1] * row_step * ct["row_dir"]
            item = Dataset()
            item.ContourImageSequence = Sequence([image_reference(slice_ds)])
            item.ContourGeometricType = "CLOSED_PLANAR"
            item.NumberOfContourPoints = len(points)
            item.ContourData = [round(float(v), 3) for v in points.ravel()]
            items.append(item)
    return items


def build(src, ct_files, dst):
    ds = pydicom.dcmread(src)
    ct = load_ct(ct_files)
    rois = {str(r.ROIName).strip(): r for r in ds.StructureSetROISequence}
    numbers = {r.ROINumber: n for n, r in rois.items()}
    contours = {numbers[c.ReferencedROINumber]: c for c in getattr(ds, "ROIContourSequence", [])
                if c.ReferencedROINumber in numbers}
    if TUMOR not in contours:
        raise ValueError(f"{src}: no '{TUMOR}' ROI")

    ribs = [n for n in contours if RIB.match(n)]
    replaced = [TUMOR] + LOBES + HEART + GREAT_VESSELS + ribs + CHEST_WALL
    used = replaced + KEPT
    masks = {n: rasterize(contours[n], ct, n) for n in used if n in contours}
    if not masks[TUMOR].any():
        raise ValueError(f"{src}: '{TUMOR}' has no contours on the CT slices")

    shape, spacing = ct["hu"].shape, ct["spacing"]
    gtv = masks[TUMOR]
    body = body_mask(ct["hu"], list(masks.values()))
    ptv = expand(gtv, PTV_MARGIN_MM, spacing) & body
    d_ptv = ndi.distance_transform_edt(~ptv, sampling=spacing)
    new = {
        "PTV": ptv,
        "Lungs-GTV": union(masks, LOBES, shape) & ~gtv,
        "SpinalCord_PRV5": expand(union(masks, [SPINAL_CORD], shape), PRV_MARGIN_MM, spacing) & body,
        "Heart_total": union(masks, HEART, shape),
        "GreatVessels": union(masks, GREAT_VESSELS, shape),
        "ChestWall": union(masks, ribs + CHEST_WALL, shape),
        "Ring_PTV_10-30mm": (d_ptv > RING_MM[0]) & (d_ptv <= RING_MM[1]) & body,
        "BODY": body,
    }

    entries = [("GTV", TUMOR)]
    for name in NEW_STRUCTURES:
        if new[name].any():
            entries.append((name, None))
        else:
            print(f"  [warning] {name} is empty: not written")
    entries += [(n, n) for n in KEPT if n in contours]

    frame_uid = rois[TUMOR].ReferencedFrameOfReferenceUID
    roi_seq, contour_seq, observation_seq = Sequence(), Sequence(), Sequence()
    for number, (name, source) in enumerate(entries, start=1):
        if source is not None:
            roi = copy.deepcopy(rois[source])
            item = copy.deepcopy(contours[source])
        else:
            roi = Dataset()
            roi.ReferencedFrameOfReferenceUID = frame_uid
            roi.ROIGenerationAlgorithm = "AUTOMATIC"
            item = Dataset()
            item.ROIDisplayColor = COLORS[name]
            item.ContourSequence = Sequence(mask_to_contours(new[name], ct))
        roi.ROINumber, roi.ROIName = number, name
        item.ReferencedROINumber = number
        roi_seq.append(roi)
        contour_seq.append(item)
        observation = Dataset()
        observation.ObservationNumber = observation.ReferencedROINumber = number
        observation.ROIObservationLabel = name[:16]
        observation.RTROIInterpretedType = TYPES.get(name, "ORGAN")
        observation.ROIInterpreter = ""
        observation_seq.append(observation)

    out = copy.deepcopy(ds)
    out.StructureSetROISequence, out.ROIContourSequence, out.RTROIObservationsSequence = \
        roi_seq, contour_seq, observation_seq
    out.SOPInstanceUID, out.SeriesInstanceUID = generate_uid(), generate_uid()
    if hasattr(out, "file_meta"):
        out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    now = datetime.datetime.now()
    out.StructureSetLabel = "TPS_Structures"
    out.StructureSetDate, out.StructureSetTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    out.save_as(dst)

    voxel_cc = np.prod(spacing) / 1000
    print(f"  CT {shape[1]}x{shape[0]}x{shape[2]} | spacing {np.round(spacing[[1, 0, 2]], 3).tolist()} mm")
    for name, source in entries:
        print(f"  {name:28s} {(masks[source] if source else new[name]).sum() * voxel_cc:9.1f} cc")
    missing = [n for n in used if n not in contours]
    if missing:
        print(f"  missing in the input: {', '.join(missing)}")
    print(f"  replaced: {', '.join(n for n in replaced if n in contours)}")
    dropped = [n for n in contours if n not in used]
    if dropped:
        print(f"  not described, not written: {', '.join(dropped)}")
    print(f"  -> {dst}")


def parse_args():
    parser = argparse.ArgumentParser(description="Planning structures (GTV, PTV, PRV, ring, BODY, ...) "
                                                 "from TotalSegmentator RTSTRUCTs and their CT.")
    parser.add_argument("--ct_dir", required=True, help="folder with the CT slices and the RTSTRUCT files")
    parser.add_argument("--output_dir", default=None, help="default: <ct_dir>/RTSTRUCT_TPS")
    parser.add_argument("--output_pattern", default="TPS_{name}.dcm")
    return parser.parse_args()


def main():
    args = parse_args()
    ct_dir = os.path.abspath(args.ct_dir)
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"{ct_dir} not found")
    output_dir = args.output_dir or os.path.join(ct_dir, "RTSTRUCT_TPS")
    series, structs = scan_dicom(ct_dir)
    for src in structs:
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(output_dir, args.output_pattern.format(name=name))
        if os.path.abspath(src) == os.path.abspath(dst):
            raise ValueError(f"{name}: the output would overwrite the source RTSTRUCT {src}")
        try:
            uid = referenced_series(pydicom.dcmread(src, stop_before_pixels=True), series)
        except ValueError as e:
            print(f"{name}: {e}, skipped")
            continue
        print(f"{name}: {src}\n  CT series {uid} ({len(series[uid]['files'])} slices)")
        build(src, series[uid]["files"], dst)


if __name__ == "__main__":
    sys.exit(main())
