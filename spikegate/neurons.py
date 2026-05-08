"""
Leaky Integrate-and-Fire (LIF) neuron for Spiking Neural Networks.

This module provides a standard LIF neuron implementation with soft reset
and configurable membrane threshold and time constant. Suitable for use
in STBP-trained spiking transformers.
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class ATanSurrogate(torch.autograd.Function):
    """
    ATan surrogate gradient for spiking neural networks.
    Replaces the non-differentiable Heaviside step function.
    """
    alpha = 2.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        surrogate = ATanSurrogate.alpha / 2 / (1 + (math.pi / 2 * ATanSurrogate.alpha * input).pow(2))
        return grad_input * surrogate


class IntrinsicDopamineTracker(nn.Module):
    """
    Autonomous Dopamine generator using Shannon Entropy to track the
    network's Intrinsic Reward Prediction Error.
    """
    def __init__(self, beta: float = 0.9):
        super().__init__()
        self.beta = beta
        self.V_t = 0.0
        self.prev_H = None

    def reset_state(self):
        self.V_t = 0.0
        self.prev_H = None

    def step(self, logits: torch.Tensor) -> float:
        """
        Calculates dopamine D[t] based on the entropy of the current prediction.
        Args:
            logits: Output logits of the classification head at time t.
        Returns:
            D_t: The scalar intrinsic dopamine broadcast signal.
        """
        probs = torch.softmax(logits, dim=-1)
        # H[t] = - sum(P * log(P))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        
        if self.prev_H is None:
            self.prev_H = entropy
            return 0.0

        # Intrinsic reward R[t] = -(H[t] - H[t-1])
        R_t = -(entropy - self.prev_H)
        self.prev_H = entropy

        # Dopamine D[t] = R[t] - V[t-1]
        D_t = R_t - self.V_t

        # Update expected baseline V[t]
        self.V_t = self.beta * self.V_t + (1 - self.beta) * R_t

        return D_t


class DopaminergicLIFNode(nn.Module):
    """
    Dopamine-modulated LIF Node with dynamic parametric leak and hard reset.
    Implements the formulation:
    u[t] = lambda[t] * u[t-1] * (1 - o[t-1]) + x[t]
    lambda[t] = sigmoid(w_d * D[t] + b_lambda)
    """
    def __init__(self, v_th: float = 0.5, features: int = 1, default_leak: float = 0.5):
        super().__init__()
        self.v_th = v_th
        self.w_d = nn.Parameter(torch.zeros(features))
        
        # Initialize b_lambda so sigmoid(b_lambda) == default_leak
        init_b = math.log(default_leak / (1 - default_leak)) if default_leak < 1.0 else 10.0
        self.b_lambda = nn.Parameter(torch.ones(features) * init_b)
        
        self.u: Optional[torch.Tensor] = None
        self.prev_spike: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self.u = None
        self.prev_spike = None

    def forward(self, x: torch.Tensor, D_t: float = 0.0) -> torch.Tensor:
        if self.u is None:
            self.u = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)

        # Dynamic leak controlled by dopamine
        # lambda_t shape will broadcast to (B, C, H, W) or (B, N, D)
        lambda_t = torch.sigmoid(self.w_d * D_t + self.b_lambda)

        # Reshape lambda_t to match x's dimensions if necessary
        while lambda_t.dim() < x.dim():
            if lambda_t.shape[0] == x.shape[1]:  # Assuming features match channels (dim 1) or features (dim -1)
                 lambda_t = lambda_t.unsqueeze(-1) if x.dim() == 3 else lambda_t.view(1, -1, 1, 1)
            else:
                 lambda_t = lambda_t.unsqueeze(0)

        # Hard reset integration (No detach, full BPTT)
        self.u = lambda_t * self.u * (1.0 - self.prev_spike) + x

        # Fire using Surrogate Gradient
        spike = ATanSurrogate.apply(self.u - self.v_th)
        self.prev_spike = spike

        return spike




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
