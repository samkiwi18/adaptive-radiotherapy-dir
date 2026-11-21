import numpy as np
import torch

from conftest import smooth_field
from registration.losses import GradientSmoothingLoss, MINDLoss, NCCLoss, landmark_loss, soft_dice_loss
from registration.metrics import compute_jacobian_determinant, jacobian_map, jacobian_torch
from registration.model import ScalingAndSquaring, SpatialTransformer3D
from registration.postprocessing import mass_preserving_hu_correction
from registration.utils import dvf_to_native_resolution

SHAPE = (24, 28, 20)


def test_zero_velocity_gives_identity():
    warp = SpatialTransformer3D(SHAPE)
    dvf = ScalingAndSquaring(steps=7)(torch.zeros(1, 3, *SHAPE), warp)
    assert torch.count_nonzero(dvf) == 0
    assert torch.allclose(jacobian_torch(dvf), torch.ones(SHAPE))
    assert compute_jacobian_determinant(dvf[0].numpy()) == (1.0, 0.0)
    assert np.allclose(jacobian_map(dvf[0].numpy()), 1.0)


def test_warp_with_zero_dvf_returns_input():
    image = torch.rand(1, 1, *SHAPE)
    warp = SpatialTransformer3D(SHAPE)
    zero = torch.zeros(1, 3, *SHAPE)
    assert torch.allclose(warp(image, zero), image, atol=1e-5)
    labels = (image > 0.5).float()
    assert torch.equal(warp(labels, zero, mode="nearest"), labels)


def test_losses_on_identical_images():
    image = torch.rand(1, 1, *SHAPE)
    mask = (torch.rand(1, 1, *SHAPE) > 0.5).float()
    assert NCCLoss()(image, image).item() < -0.99
    assert abs(MINDLoss()(image, image).item()) < 1e-9
    assert soft_dice_loss(mask, mask).item() < 1e-6
    assert GradientSmoothingLoss()(torch.zeros(1, 3, *SHAPE)).item() == 0.0
    points = torch.rand(1, 10, 3) * (torch.tensor(SHAPE) - 1)
    assert landmark_loss(torch.zeros(1, 3, *SHAPE), points, points, torch.ones(3)).item() == 0.0


def test_ncc_is_lower_for_aligned_images():
    image = torch.rand(1, 1, *SHAPE)
    other = torch.rand(1, 1, *SHAPE)
    assert NCCLoss()(image, image).item() < NCCLoss()(image, other).item()


def test_hu_correction_is_identity_for_unit_jacobian():
    hu = torch.rand(1, 1, *SHAPE) * 1500.0 - 1000.0
    lung = np.ones(SHAPE, dtype=bool)
    tumor = np.zeros(SHAPE, dtype=bool)
    corrected, det = mass_preserving_hu_correction(hu, torch.zeros(1, 3, *SHAPE), lung, tumor)
    assert torch.allclose(det, torch.ones_like(det))
    assert torch.allclose(corrected, hu, atol=1e-3)


def test_hu_correction_scales_density_in_lung_only():
    hu = torch.full((1, 1, *SHAPE), -700.0)
    lung = np.zeros(SHAPE, dtype=bool)
    lung[:12] = True
    dvf = torch.zeros(1, 3, *SHAPE)
    dvf[:, 0] = 0.1 * torch.arange(SHAPE[0], dtype=torch.float32)[:, None, None]
    corrected, _ = mass_preserving_hu_correction(hu, dvf, lung, np.zeros(SHAPE, dtype=bool))
    assert torch.allclose(corrected[0, 0, 1:11], torch.full_like(corrected[0, 0, 1:11], 300.0 * 1.1 - 1000.0))
    assert torch.equal(corrected[0, 0, 12:], hu[0, 0, 12:])


def test_inverse_velocity_composes_to_identity():
    warp = SpatialTransformer3D(SHAPE)
    integrator = ScalingAndSquaring(steps=9)
    velocity = smooth_field(SHAPE, amplitude=2.0)
    forward = integrator(velocity, warp)
    backward = integrator(-velocity, warp)
    composed = forward + warp(backward, forward)
    assert forward.abs().max() > 1.0
    assert composed.abs().max() < 0.05 * forward.abs().max()
    assert compute_jacobian_determinant(forward[0])[1] == 0.0


def test_dvf_to_native_resolution_rescales_components():
    dvf = torch.ones(1, 3, 16, 16, 16)
    native = dvf_to_native_resolution(dvf, (31, 46, 16))
    assert torch.allclose(native[0, 0], torch.full((31, 46, 16), 2.0))
    assert torch.allclose(native[0, 1], torch.full((31, 46, 16), 3.0))
    assert torch.allclose(native[0, 2], torch.ones(31, 46, 16))
