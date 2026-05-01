import pytest
import torch

@pytest.fixture
def dummy_input():
    """Provides a dummy input tensor of shape (B, C, H, W)."""
    return torch.randn(2, 3, 32, 32)

@pytest.fixture
def dummy_spatiotemporal_input():
    """Provides a dummy spatio-temporal input tensor of shape (T, B, C, H, W)."""
    return torch.randn(4, 2, 3, 32, 32)
