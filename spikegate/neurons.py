"""
Leaky Integrate-and-Fire (LIF) neuron for Spiking Neural Networks.

This module provides a standard LIF neuron implementation with soft reset
and configurable membrane threshold and time constant. Suitable for use
in STBP-trained spiking transformers.
"""

import torch
import torch.nn as nn
from typing import Optional


class LIFNode(nn.Module):
    """Standard Leaky Integrate-and-Fire neuron for Spiking Neural Networks.

    Converts continuous activations into binary spike trains using a
    membrane potential mechanism with leaky integration and soft reset.

    Args:
        v_th: Membrane voltage threshold for spike generation.
        tau: Membrane time constant controlling the leak rate.
             Decay factor = ``1.0 - (1.0 / tau)``.
    """

    def __init__(self, v_th: float = 0.5, tau: float = 4.0) -> None:
        super().__init__()
        self.v_th = v_th
        self.decay = 1.0 - (1.0 / tau)
        self.u: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Clears membrane potential. Must be called between temporal passes."""
        self.u = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrates input and emits binary spikes.

        Args:
            x: Input tensor of any shape.

        Returns:
            Binary spike tensor (same shape as ``x``), with 1.0 where
            membrane potential exceeded threshold.
        """
        if self.u is None:
            self.u = torch.zeros_like(x)

        # Leaky integrate
        self.u = self.u.detach() * self.decay + x

        # Fire (Heaviside step function)
        spike = (self.u >= self.v_th).float()

        # Soft reset
        self.u = self.u - spike.detach() * self.v_th

        return spike
