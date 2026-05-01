"""
AutoProfiler for spiking transformer head analysis.

Hooks into a model to automatically profile Q/K/V sparsity, temporal
dynamics, dead/ghost neurons, cross-head correlations, and generate
hardware gating policy JSON profiles.
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .policies import GatingPolicy

logger = logging.getLogger(__name__)

# ── Configurable thresholds ──────────────────────────────────────────
DEFAULT_STATIC_PRUNE_SPARSITY = 0.99
DEFAULT_LATE_WAKEUP_SPARSITY = 0.85
DEFAULT_LATE_WAKEUP_FST = 2.0
DEFAULT_TEMPORAL_GAP_THRESHOLD = 1.5
DEFAULT_CORRELATION_THRESHOLD = 0.85
DEFAULT_GHOST_NEURON_FIRING_RATE = 0.95
# ─────────────────────────────────────────────────────────────────────


class AutoProfiler:
    """Hooks into a spiking model to profile head behaviour and generate gating policies.

    Supports both:
    - **Wrapped ANNs** (via ``SpikingModelWrapper``): hooks capture per-timestep
      outputs of shape ``(B, N, d)`` that are stacked into ``(T, B, N, d)``.
    - **Native SNNs** (e.g., MaxFormer): hooks capture full spatio-temporal
      outputs of shape ``(T, B, C, N)`` that are dynamically unpacked.

    Args:
        model: The spiking model to profile.
        T: Number of simulation timesteps.
        correlation_threshold: Pearson-r above which two heads are deemed redundant.
        static_prune_sparsity: Q-sparsity above which a head is statically pruned.
        late_wakeup_sparsity: Q-sparsity above which a head gets ``LATE_WAKEUP_GATE``.
        ghost_firing_rate: Firing rate above which a neuron is classified as a ghost.
    """

    def __init__(
        self,
        model: nn.Module,
        T: int = 4,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        static_prune_sparsity: float = DEFAULT_STATIC_PRUNE_SPARSITY,
        late_wakeup_sparsity: float = DEFAULT_LATE_WAKEUP_SPARSITY,
        ghost_firing_rate: float = DEFAULT_GHOST_NEURON_FIRING_RATE,
    ) -> None:
        self.model = model
        self.T = T
        self.correlation_threshold = correlation_threshold
        self.static_prune_sparsity = static_prune_sparsity
        self.late_wakeup_sparsity = late_wakeup_sparsity
        self.ghost_firing_rate = ghost_firing_rate

        self.extracted_spikes: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.handles: List[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    # ── Hook registration ────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """Dynamically discovers Q/K/V LIF nodes and attaches forward hooks."""
        block_idx = 0
        for name, module in self.model.named_modules():
            has_lif = (
                hasattr(module, "q_lif")
                and hasattr(module, "k_lif")
                and hasattr(module, "v_lif")
            )
            if not has_lif:
                continue

            def make_hook(b_idx: int, l_type: str):
                def hook_fn(mod, inputs, output):
                    self.extracted_spikes[f"b{b_idx}_{l_type}"].append(
                        output.detach().cpu()
                    )
                return hook_fn

            if isinstance(module.q_lif, nn.ModuleList):
                num_heads = len(module.q_lif)
                for h in range(num_heads):
                    self.handles.append(module.q_lif[h].register_forward_hook(make_hook(block_idx, f"q_h{h}")))
                    self.handles.append(module.k_lif[h].register_forward_hook(make_hook(block_idx, f"k_h{h}")))
                    self.handles.append(module.v_lif[h].register_forward_hook(make_hook(block_idx, f"v_h{h}")))
            else:
                # Single node handling packed heads (e.g., MaxFormer)
                self.handles.append(module.q_lif.register_forward_hook(make_hook(block_idx, "q_h0")))
                self.handles.append(module.k_lif.register_forward_hook(make_hook(block_idx, "k_h0")))
                self.handles.append(module.v_lif.register_forward_hook(make_hook(block_idx, "v_h0")))

            block_idx += 1

        logger.info("Registered hooks on %d attention blocks", block_idx)

    def remove_hooks(self) -> None:
        """Removes all registered forward hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def first_spike_time(spike_tensor: torch.Tensor) -> float:
        """Computes the mean first-spike-time across a spike tensor.

        Args:
            spike_tensor: Shape ``(T, B, N, d)`` or ``(T, B, d)``.

        Returns:
            Scalar mean first spike time.
        """
        T_sz = spike_tensor.shape[0]
        indices = torch.arange(1, T_sz + 1, device=spike_tensor.device, dtype=torch.float32)
        shape_ones = [1] * len(spike_tensor.shape)
        shape_ones[0] = T_sz
        mask = indices.view(*shape_ones) * spike_tensor
        mask[mask == 0] = 999
        fst_times = mask.min(dim=0)[0]
        fst_times[fst_times == 999] = T_sz
        # Mean across Batch, Sequence, Feature
        return fst_times.mean(dim=tuple(range(len(fst_times.shape)))).item()

    # ── Main calibration ─────────────────────────────────────────────

    def calibrate(
        self,
        dataloader,
        device: str,
        num_blocks: int,
        num_heads: int,
        out_file: str = "gating_profile.json",
    ) -> Dict:
        """Runs calibration over a dataloader and generates a gating policy.

        Args:
            dataloader: Iterable of ``(images, labels)`` batches.
            device: Device string (``'cuda'`` or ``'cpu'``).
            num_blocks: Number of transformer blocks to profile.
            num_heads: Number of attention heads per block.
            out_file: Output path for the JSON gating profile.

        Returns:
            The generated gating policy dictionary.

        Raises:
            RuntimeError: If no spike data is captured (hooks may be misconfigured).
        """
        logger.info("Starting Auto-Calibration for Spiking Gating...")
        print("Starting Auto-Calibration for Spiking Gating...")
        self.model.eval()

        # ── Accumulators ─────────────────────────────────────────────
        acc_stbp_delta = np.zeros((num_blocks, num_heads))
        acc_q_fst = np.zeros((num_blocks, num_heads))

        acc_sparsity_q_T = np.zeros((num_blocks, num_heads, self.T))
        acc_sparsity_k_T = np.zeros((num_blocks, num_heads, self.T))
        acc_sparsity_v_T = np.zeros((num_blocks, num_heads, self.T))

        dead_neurons_q = np.zeros((num_blocks, num_heads))
        ghost_neurons_q = np.zeros((num_blocks, num_heads))
        dead_neurons_k = np.zeros((num_blocks, num_heads))
        ghost_neurons_k = np.zeros((num_blocks, num_heads))
        dead_neurons_v = np.zeros((num_blocks, num_heads))
        ghost_neurons_v = np.zeros((num_blocks, num_heads))

        head_activities: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_blocks)}

        batches = 0
        with torch.no_grad():
            for x, _ in tqdm(dataloader):
                x = x.to(device)
                self.model(x)

                for block_i in range(num_blocks):
                    block_act: List[np.ndarray] = []

                    # ── Inner helper (captures block_i, block_act by default arg) ──
                    def process_head_spikes(
                        h: int,
                        q_t: torch.Tensor,
                        k_t: torch.Tensor,
                        v_t: torch.Tensor,
                        _bi: int = block_i,
                        _ba: list = block_act,
                    ) -> None:
                        q_f = self.first_spike_time(q_t)
                        k_f = self.first_spike_time(k_t)

                        acc_q_fst[_bi, h] += q_f
                        acc_stbp_delta[_bi, h] += abs(q_f - k_f)

                        for t in range(self.T):
                            acc_sparsity_q_T[_bi, h, t] += (1.0 - q_t[t].mean().item())
                            acc_sparsity_k_T[_bi, h, t] += (1.0 - k_t[t].mean().item())
                            acc_sparsity_v_T[_bi, h, t] += (1.0 - v_t[t].mean().item())

                        # Dead and Ghost Neuron Analysis
                        fr_q = q_t.mean(dim=(0, 1, 2))
                        fr_k = k_t.mean(dim=(0, 1, 2))
                        fr_v = v_t.mean(dim=(0, 1, 2))

                        d = fr_q.shape[0]
                        dead_neurons_q[_bi, h] += (fr_q == 0).sum().item() / d
                        ghost_neurons_q[_bi, h] += (fr_q > self.ghost_firing_rate).sum().item() / d
                        dead_neurons_k[_bi, h] += (fr_k == 0).sum().item() / d
                        ghost_neurons_k[_bi, h] += (fr_k > self.ghost_firing_rate).sum().item() / d
                        dead_neurons_v[_bi, h] += (fr_v == 0).sum().item() / d
                        ghost_neurons_v[_bi, h] += (fr_v > self.ghost_firing_rate).sum().item() / d

                        _ba.append(q_t.sum(dim=(2, 3)).flatten().numpy())

                    # ── Dispatch to appropriate unpacking strategy ────
                    self._dispatch_head_extraction(
                        block_i, num_heads, process_head_spikes
                    )

                    if len(block_act) == num_heads:
                        head_activities[block_i].append(np.stack(block_act, axis=0))

                self.extracted_spikes.clear()
                batches += 1

        if batches == 0:
            raise RuntimeError("No batches processed — check dataloader.")

        self.remove_hooks()

        # ── Compile policy ───────────────────────────────────────────
        spec = self._compile_policy(
            num_blocks, num_heads, batches,
            acc_stbp_delta, acc_q_fst,
            acc_sparsity_q_T, acc_sparsity_k_T, acc_sparsity_v_T,
            dead_neurons_q, ghost_neurons_q,
            dead_neurons_k, ghost_neurons_k,
            dead_neurons_v, ghost_neurons_v,
            head_activities,
        )

        with open(out_file, "w") as f:
            json.dump(spec, f, indent=4)

        print(f"Calibration Complete! Written policy to {out_file}")
        logger.info("Calibration complete. Profile saved to %s", out_file)
        return spec

    # ── Private helpers ──────────────────────────────────────────────

    def _dispatch_head_extraction(
        self,
        block_i: int,
        num_heads: int,
        process_fn,
    ) -> None:
        """Routes spike data to the correct unpacking path."""
        has_h0 = f"b{block_i}_q_h0" in self.extracted_spikes
        has_h1 = f"b{block_i}_q_h1" in self.extracted_spikes

        if has_h1:
            # Per-head hooks (SpikingGatedAttention with ModuleList LIFs)
            self._unpack_per_head_hooks(block_i, num_heads, process_fn)
        elif has_h0:
            # Single-node hooks (packed heads — e.g., MaxFormer or wrapped ANN)
            latest_q = self.extracted_spikes[f"b{block_i}_q_h0"][-1]
            latest_k = self.extracted_spikes[f"b{block_i}_k_h0"][-1]
            latest_v = self.extracted_spikes[f"b{block_i}_v_h0"][-1]

            if latest_q.ndim == 4 and latest_q.shape[0] == self.T:
                # Native SNN (MaxFormer) → (T, B, C, N)
                self._unpack_native_snn(latest_q, latest_k, latest_v, num_heads, process_fn)
            else:
                # Wrapped ANN → (T items of B, N, C)
                self._unpack_wrapped_ann(block_i, num_heads, process_fn)

    def _unpack_native_snn(self, q_raw, k_raw, v_raw, num_heads, process_fn):
        """Unpacks (T, B, C, N) → per-head (T, B, N, D)."""
        T, B, C, seq_len = q_raw.shape
        D = C // num_heads
        q_raw = q_raw.view(T, B, num_heads, D, seq_len).permute(0, 1, 2, 4, 3)
        k_raw = k_raw.view(T, B, num_heads, D, seq_len).permute(0, 1, 2, 4, 3)
        v_raw = v_raw.view(T, B, num_heads, D, seq_len).permute(0, 1, 2, 4, 3)
        for h in range(num_heads):
            process_fn(h, q_raw[:, :, h], k_raw[:, :, h], v_raw[:, :, h])

    def _unpack_wrapped_ann(self, block_i, num_heads, process_fn):
        """Unpacks stacked (T, B, N, C) → per-head (T, B, N, D)."""
        q_raw = torch.stack(self.extracted_spikes[f"b{block_i}_q_h0"][-self.T:])
        k_raw = torch.stack(self.extracted_spikes[f"b{block_i}_k_h0"][-self.T:])
        v_raw = torch.stack(self.extracted_spikes[f"b{block_i}_v_h0"][-self.T:])
        T, B, seq_len, C = q_raw.shape
        D = C // num_heads
        q_raw = q_raw.view(T, B, seq_len, num_heads, D).permute(0, 1, 3, 2, 4)
        k_raw = k_raw.view(T, B, seq_len, num_heads, D).permute(0, 1, 3, 2, 4)
        v_raw = v_raw.view(T, B, seq_len, num_heads, D).permute(0, 1, 3, 2, 4)
        for h in range(num_heads):
            process_fn(h, q_raw[:, :, h], k_raw[:, :, h], v_raw[:, :, h])

    def _unpack_per_head_hooks(self, block_i, num_heads, process_fn):
        """Processes data from per-head hooks (SpikingGatedAttention)."""
        for h in range(num_heads):
            q_list = self.extracted_spikes.get(f"b{block_i}_q_h{h}")
            k_list = self.extracted_spikes.get(f"b{block_i}_k_h{h}")
            v_list = self.extracted_spikes.get(f"b{block_i}_v_h{h}")
            if not q_list or len(q_list) < self.T:
                continue
            q_t = torch.stack(q_list[-self.T:], dim=0)
            k_t = torch.stack(k_list[-self.T:], dim=0)
            v_t = torch.stack(v_list[-self.T:], dim=0)
            process_fn(h, q_t, k_t, v_t)

    def _compile_policy(
        self,
        num_blocks, num_heads, batches,
        acc_stbp_delta, acc_q_fst,
        acc_sparsity_q_T, acc_sparsity_k_T, acc_sparsity_v_T,
        dead_neurons_q, ghost_neurons_q,
        dead_neurons_k, ghost_neurons_k,
        dead_neurons_v, ghost_neurons_v,
        head_activities,
    ) -> Dict:
        """Compiles accumulated statistics into a gating policy dictionary."""
        spec: Dict = {}
        for i in range(num_blocks):
            spec[f"block_{i}"] = {}

            # Correlation analysis
            redundant_heads: set = set()
            if head_activities[i]:
                all_acts = np.concatenate(head_activities[i], axis=1)
                corr_matrix = np.corrcoef(all_acts)
                for h1 in range(num_heads):
                    for h2 in range(h1 + 1, num_heads):
                        if corr_matrix[h1, h2] > self.correlation_threshold:
                            avg_h1 = acc_sparsity_q_T[i, h1].mean()
                            avg_h2 = acc_sparsity_q_T[i, h2].mean()
                            redundant_heads.add(h1 if avg_h1 >= avg_h2 else h2)

            for h in range(num_heads):
                spars_q_t = (acc_sparsity_q_T[i, h] / batches).tolist()
                spars_k_t = (acc_sparsity_k_T[i, h] / batches).tolist()
                spars_v_t = (acc_sparsity_v_T[i, h] / batches).tolist()

                avg_spars_q = sum(spars_q_t) / self.T
                delta = acc_stbp_delta[i, h] / batches
                q_fst = acc_q_fst[i, h] / batches

                # Policy assignment
                if h in redundant_heads:
                    gate_rec = GatingPolicy.STATICALLY_GATED_BY_REDUNDANCY.value
                elif avg_spars_q >= self.static_prune_sparsity:
                    gate_rec = GatingPolicy.STATICALLY_PRUNE_OR_EARLY_EXIT_T1.value
                elif delta > DEFAULT_TEMPORAL_GAP_THRESHOLD:
                    gate_rec = GatingPolicy.DYNAMIC_KEY_EXIT_WAIT_T2.value
                elif q_fst > DEFAULT_LATE_WAKEUP_FST or avg_spars_q > self.late_wakeup_sparsity:
                    gate_rec = GatingPolicy.LATE_WAKEUP_GATE.value
                else:
                    gate_rec = GatingPolicy.ACTIVE_NO_GATE.value

                spec[f"block_{i}"][f"head_{h}"] = {
                    "sparsity_q_per_timestep": [round(s, 4) for s in spars_q_t],
                    "sparsity_k_per_timestep": [round(s, 4) for s in spars_k_t],
                    "sparsity_v_per_timestep": [round(s, 4) for s in spars_v_t],
                    "average_sparsity_q": round(avg_spars_q, 4),
                    "stbp_qk_temporal_gap_abs": round(delta, 3),
                    "dead_neurons_pct_q": round(dead_neurons_q[i, h] / batches * 100, 2),
                    "ghost_neurons_pct_q": round(ghost_neurons_q[i, h] / batches * 100, 2),
                    "dead_neurons_pct_k": round(dead_neurons_k[i, h] / batches * 100, 2),
                    "ghost_neurons_pct_k": round(ghost_neurons_k[i, h] / batches * 100, 2),
                    "dead_neurons_pct_v": round(dead_neurons_v[i, h] / batches * 100, 2),
                    "ghost_neurons_pct_v": round(ghost_neurons_v[i, h] / batches * 100, 2),
                    "is_highly_correlated": h in redundant_heads,
                    "HARDWARE_GATING_POLICY": gate_rec,
                }

        return spec
