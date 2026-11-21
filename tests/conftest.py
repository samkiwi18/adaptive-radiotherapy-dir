import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def smooth_field(shape, amplitude):
    grid = torch.stack(torch.meshgrid(*[torch.linspace(-1.0, 1.0, s) for s in shape], indexing="ij"))
    bump = torch.exp(-(grid ** 2).sum(0) / 0.3)
    return torch.stack([amplitude * bump, -0.5 * amplitude * bump, 0.25 * amplitude * bump])[None]
