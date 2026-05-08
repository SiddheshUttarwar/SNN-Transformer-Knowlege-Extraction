"""
Spiking Gated Attention module.

Drop-in replacement for standard MultiHead Attention that wraps Q, K, V
projections with LIF spiking nodes and applies dynamic hardware gating
to physically bypass computation on inactive heads.
"""

import torch
import torch.nn as nn
from typing import Optional

from .neurons import LIFNode
from .gating import DynamicGateController


class SpikingGatedAttention(nn.Module):
    """Drop-in replacement for standard MultiHead Attention.

    Wraps existing Q, K, V projections with LIF spiking nodes and
    applies Dynamic Hardware Gating to bypass computation on heads
    that the ``DynamicGateController`` determines are inactive.

    Args:
        original_attn: The original attention module being replaced.
            Used to extract ``num_heads``.
        block_idx: Index of this transformer block (for policy lookup).
        gate_controller: Shared controller that makes bypass decisions.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        block_idx: int,
        gate_controller: DynamicGateController,
    ) -> None:
        super().__init__()
        self.block_idx = block_idx
        self.gate_controller = gate_controller

        self.num_heads: int = getattr(original_attn, "num_heads", 8)

        # Projections — populated by the converter after construction
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None

        # Spiking Nodes (one per head for Q, K, V; one shared for post-attention)
        self.q_lif = nn.ModuleList([LIFNode() for _ in range(self.num_heads)])
        self.k_lif = nn.ModuleList([LIFNode() for _ in range(self.num_heads)])
        self.v_lif = nn.ModuleList([LIFNode() for _ in range(self.num_heads)])
        self.attn_lif = LIFNode()

        # Internal timestep counter
        self.timestep: int = 0

    def reset_state(self) -> None:
        """Resets all spiking neuron states and the gate controller."""
        for lif in self.q_lif:
            lif.reset_state()
        for lif in self.k_lif:
            lif.reset_state()
        for lif in self.v_lif:
            lif.reset_state()
        self.attn_lif.reset_state()
        self.gate_controller.reset_state()
        self.timestep = 0

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for a single timestep.

        Args:
            x: Input tensor of shape ``(B, N, D)``.
            **kwargs: Extra arguments like attn_mask (ignored for now).

        Returns:
            Output tensor of shape ``(B, N, D)``.
        """
        B, N, D = x.shape
        head_dim = D // self.num_heads

        # Compute Q projection for all heads
        q_full = self.q_proj(x).view(B, N, self.num_heads, head_dim)

        out_heads = []

        for h in range(self.num_heads):
            # Q Spike Generation
            q_h = q_full[:, :, h, :]
            q_spike = self.q_lif[h](q_h)

            # Ask Gate Controller if we should compute K and V for this head
            if self.gate_controller.should_compute_kv(
                self.block_idx, h, self.timestep, q_spike
            ):
                # NOTE: This computes the full projection then slices.
                # A hardware-optimized implementation would compute only
                # the head-specific slice of the weight matrix.
                k_h = self.k_proj(x).view(B, N, self.num_heads, head_dim)[:, :, h, :]
                v_h = self.v_proj(x).view(B, N, self.num_heads, head_dim)[:, :, h, :]

                k_spike = self.k_lif[h](k_h)
                v_spike = self.v_lif[h](v_h)

                # Spike-driven Attention: Q @ K.T, scaled
                attn_scores = (q_spike @ k_spike.transpose(-2, -1)) * (1.0 / N)
                head_out = attn_scores @ v_spike
            else:
                # GATED: Head is asleep or pruned — bypass all K/V/Attention compute
                head_out = torch.zeros(B, N, head_dim, device=x.device)

            out_heads.append(head_out)

        # Concatenate heads and apply post-attention LIF + output projection
        concat_out = torch.cat(out_heads, dim=-1)
        attn_out_spike = self.attn_lif(concat_out)
        out = self.out_proj(attn_out_spike)

        self.timestep += 1
        return out
