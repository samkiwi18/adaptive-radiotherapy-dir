import numpy as np
import pytest

from registration.dose_analysis.dose_analysis import _gamma_worker, dvh_metrics, geometry, pass_rate

pytest.importorskip("pymedphys")


def synthetic_dose():
    z, y, x = np.arange(10) * 3.0, np.arange(16) * 2.0, np.arange(18) * 2.0
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dose = 8.0 * np.exp(-((zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 17) ** 2) / 200.0)
    return (z, y, x), dose


def test_gamma_is_100_percent_for_identical_doses():
    axes, dose = synthetic_dose()
    gamma = _gamma_worker((axes, dose, axes, dose, 2, 2, int(5e8)))
    rate, points = pass_rate(gamma)
    assert points > 0
    assert rate == 100.0
    assert np.nanmax(gamma) == 0.0


def test_gamma_detects_a_dose_change():
    axes, dose = synthetic_dose()
    gamma = _gamma_worker((axes, dose, axes, dose * 1.1, 2, 2, int(5e8)))
    assert pass_rate(gamma)[0] < 100.0


def test_geometry_and_dvh_of_identical_structures():
    _, dose = synthetic_dose()
    mask = dose > 4.0
    g = geometry(mask, mask, np.array([3.0, 2.0, 2.0]))
    assert g["dice"] == 1.0 and g["hd95"] == 0.0 and g["centroid_shift_mm"] == 0.0
    m = dvh_metrics(dose, mask, "PTV", 0.012, exclude_zero=False)
    assert m["D98"] <= m["D95"] <= m["D50"] <= m["D2"] <= m["Dmax"]
