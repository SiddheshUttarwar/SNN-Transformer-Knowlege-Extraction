"""
Model converter for the SpikeGate framework.

Provides utilities to convert standard transformer models (e.g., ``timm`` ViTs)
into spiking gated models by replacing attention modules and wrapping inference
with a temporal loop.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .attention import SpikingGatedAttention
from .gating import DynamicGateController

logger = logging.getLogger(__name__)


def replace_attention_with_spiking_gated(
    model: nn.Module, gate_controller: DynamicGateController
) -> nn.Module:
    """Recursively replaces standard attention modules with SpikingGatedAttention.

    Targets ``timm`` Vision Transformer ``Attention`` blocks. Unpacks the
    fused ``qkv`` linear layer into separate Q, K, V projections to enable
    per-head compute bypass.

    Args:
        model: The model whose attention modules will be replaced in-place.
        gate_controller: Shared controller for gating decisions.

    Returns:
        The modified model (same object, mutated in-place).
    """
    block_idx = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Attention":
            gated_attn = SpikingGatedAttention(module, block_idx, gate_controller)

            # Unpack timm's fused qkv linear into separate projections
            if hasattr(module, "qkv"):
                embed_dim = module.qkv.in_features
                has_bias = module.qkv.bias is not None

                gated_attn.q_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
                gated_attn.k_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
                gated_attn.v_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)

                gated_attn.q_proj.weight.data.copy_(module.qkv.weight[:embed_dim, :])
                gated_attn.k_proj.weight.data.copy_(module.qkv.weight[embed_dim : 2 * embed_dim, :])
                gated_attn.v_proj.weight.data.copy_(module.qkv.weight[2 * embed_dim :, :])

                if has_bias:
                    gated_attn.q_proj.bias.data.copy_(module.qkv.bias[:embed_dim])
                    gated_attn.k_proj.bias.data.copy_(module.qkv.bias[embed_dim : 2 * embed_dim])
                    gated_attn.v_proj.bias.data.copy_(module.qkv.bias[2 * embed_dim :])

            # Copy output projection
            if hasattr(module, "proj"):
                gated_attn.out_proj = module.proj

            _set_module(model, name, gated_attn)
            block_idx += 1
            logger.debug("Replaced block %d (%s) with SpikingGatedAttention", block_idx - 1, name)

    logger.info("Converted %d attention blocks to SpikingGatedAttention", block_idx)
    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replaces a module in a hierarchy given its dot-separated name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class SpikingModelWrapper(nn.Module):
    """Wraps an arbitrary converted Spiking Model to handle the time-step loop.

    For static images, repeats the input ``T`` times and averages the
    output logits across all timesteps.

    Args:
        model: The converted spiking model.
        T: Number of simulation timesteps.
    """

    def __init__(self, model: nn.Module, T: int = 4) -> None:
        super().__init__()
        self.model = model
        self.T = T

    def reset_net(self) -> None:
        """Recursively calls ``reset_state`` on all spiking components."""
        for _, module in self.model.named_modules():
            if hasattr(module, "reset_state"):
                module.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs T timesteps of inference and averages the logits.

        Args:
            x: Static image batch of shape ``(B, C, H, W)``.

        Returns:
            Averaged logits of shape ``(B, num_classes)``.
        """
        self.reset_net()
        outputs = []
        for _ in range(self.T):
            outputs.append(self.model(x))

        # Clear SNN states after batch completes to prevent CUDA OOM
        self.reset_net()

        return torch.stack(outputs, dim=0).mean(dim=0)


def convert_to_gated_snn(
    model: nn.Module,
    profile_path: Optional[str] = None,
    T: int = 4,
) -> SpikingModelWrapper:
    """High-level API to convert a standard model into a spiking gated model.

    Args:
        model: A standard transformer model (e.g., from ``timm``).
        profile_path: Optional path to a pre-generated gating profile JSON.
        T: Number of simulation timesteps.

    Returns:
        A ``SpikingModelWrapper`` ready for dynamic inference.
    """
    gate_controller = DynamicGateController(profile_path)
    model = replace_attention_with_spiking_gated(model, gate_controller)
    return SpikingModelWrapper(model, T=T)
