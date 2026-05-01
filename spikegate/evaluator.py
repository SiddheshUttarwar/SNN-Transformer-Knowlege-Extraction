"""
Gating Ablation Study and Head Importance Estimation.

Evaluates the impact of different gating policies on model fidelity
and compute savings using label-agnostic Baseline Fidelity.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .policies import GatingPolicy

logger = logging.getLogger(__name__)


class GatingAblationStudy:
    """Evaluates the impact of gating policies on model fidelity and compute savings.

    Uses **Baseline Fidelity** (matching the predictions of the ungated model)
    rather than true dataset labels, making it completely dataset- and label-agnostic.

    Args:
        model: The spiking model to evaluate.
        gate_controller: The gate controller whose policies will be ablated.
    """

    def __init__(self, model, gate_controller) -> None:
        self.model = model
        self.gate_controller = gate_controller

    def _run_pass(
        self, dataloader, device: str, desc: str = "Evaluating"
    ) -> Tuple[torch.Tensor, int, int, float]:
        """Runs a full pass over the dataloader and returns predictions + compute stats."""
        self.model.eval()
        self.gate_controller.reset_compute_counters()

        all_preds: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                logits = self.model(x)
                all_preds.append(logits.argmax(dim=-1).cpu())

        bypassed = self.gate_controller.bypassed_kv_computations
        total = self.gate_controller.total_kv_computations
        savings_pct = (bypassed / total * 100) if total > 0 else 0.0

        return torch.cat(all_preds, dim=0), bypassed, total, savings_pct

    def run_study(
        self,
        dataloader,
        device: str = "cuda",
        out_file: str = "gating_impact_report.json",
    ) -> Dict:
        """Runs a full ablation study comparing each gating policy individually.

        Args:
            dataloader: Iterable of batches.
            device: Device string.
            out_file: Output path for the JSON report.

        Returns:
            Dictionary of results keyed by policy name.
        """
        print("--- Starting Gating Ablation Study ---")
        results: Dict = {}

        # 1. Baseline Pass (Force NO GATING)
        print("\n[1/6] Running Baseline (No Gating)...")
        self.gate_controller.override_policy = GatingPolicy.ACTIVE_NO_GATE.value
        baseline_preds, _, _, _ = self._run_pass(dataloader, device, "Baseline")
        results["BASELINE"] = {"fidelity": 100.0, "compute_savings_pct": 0.0}

        def evaluate_policy(name: str, override_val: str) -> None:
            print(f"\nEvaluating: {name}")
            self.gate_controller.override_policy = override_val
            preds, bypassed, total, savings = self._run_pass(dataloader, device, name)
            matches = (preds == baseline_preds).sum().item()
            fidelity = (matches / len(baseline_preds)) * 100
            results[name] = {
                "fidelity": round(fidelity, 2),
                "compute_savings_pct": round(savings, 2),
                "bypassed_computations": bypassed,
                "total_computations": total,
            }
            print(f"   -> Fidelity to Baseline: {fidelity:.2f}%")
            print(f"   -> Compute Savings:      {savings:.2f}%")

        # 2–5. Individual policy ablations
        evaluate_policy("ONLY_STATICALLY_PRUNE", GatingPolicy.STATICALLY_PRUNE_OR_EARLY_EXIT_T1.value)
        evaluate_policy("ONLY_LATE_WAKEUP_GATE", GatingPolicy.LATE_WAKEUP_GATE.value)
        evaluate_policy("ONLY_REDUNDANCY_GATE", GatingPolicy.STATICALLY_GATED_BY_REDUNDANCY.value)
        evaluate_policy("ONLY_DYNAMIC_ONLINE_PRUNING", GatingPolicy.DYNAMIC_ONLINE_PRUNING.value)

        # 6. Full Configured Policy from JSON profile
        print("\n[6/6] Evaluating FULL Combined Policy from JSON Profile...")
        self.gate_controller.override_policy = None
        preds, bypassed, total, savings = self._run_pass(dataloader, device, "Full Policy")
        matches = (preds == baseline_preds).sum().item()
        fidelity = (matches / len(baseline_preds)) * 100
        results["FULL_POLICY"] = {
            "fidelity": round(fidelity, 2),
            "compute_savings_pct": round(savings, 2),
            "bypassed_computations": bypassed,
            "total_computations": total,
        }
        print(f"   -> Fidelity to Baseline: {fidelity:.2f}%")
        print(f"   -> Compute Savings:      {savings:.2f}%")

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nStudy Complete! Full report saved to {out_file}")
        return results

    def run_head_importance_study(
        self,
        dataloader,
        device: str = "cuda",
        num_blocks: int = 12,
        num_heads: int = 12,
        out_file: str = "head_importance_report.json",
        checkpoint_file: Optional[str] = None,
    ) -> Dict:
        """Runs MASK-ONE-OUT and ISOLATE-ONE studies to estimate head importance.

        Supports **checkpoint/resume**: if ``checkpoint_file`` is provided and
        exists, the study resumes from the last completed head.

        Args:
            dataloader: Iterable of batches.
            device: Device string.
            num_blocks: Number of transformer blocks.
            num_heads: Number of attention heads per block.
            out_file: Output path for the final JSON report.
            checkpoint_file: Optional path for intermediate progress saves.

        Returns:
            Dictionary containing per-head importance scores and aggregated insights.
        """
        print("\n--- Starting Head Importance Estimation Study ---")

        # ── Load checkpoint or start fresh ───────────────────────────
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                results = json.load(f)
            logger.info("Resumed from checkpoint: %s", checkpoint_file)
            print(f"Resumed from checkpoint: {checkpoint_file}")
        else:
            results = {"MASK_ONE_OUT": {}, "ISOLATE_ONE": {}}

        # 1. Baseline Pass
        print("[1/3] Running Baseline (No Gating) to establish Ground Truth...")
        self.gate_controller.override_policy = GatingPolicy.ACTIVE_NO_GATE.value
        self.gate_controller.override_mode = None
        baseline_preds, _, _, _ = self._run_pass(dataloader, device, "Baseline")

        # 2. MASK-ONE-OUT Study
        print("\n[2/3] Running MASK-ONE-OUT Study (Disabling heads one by one)...")
        self.gate_controller.override_mode = "MASK_ONE"
        self.gate_controller.override_policy = None

        for b in range(num_blocks):
            for h in range(num_heads):
                head_key = f"B{b}H{h}"
                if head_key in results["MASK_ONE_OUT"]:
                    continue  # Already completed in a previous run

                self.gate_controller.override_target = (b, h)
                preds, _, _, _ = self._run_pass(dataloader, device, f"Mask {head_key}")
                matches = (preds == baseline_preds).sum().item()
                fidelity = (matches / len(baseline_preds)) * 100
                importance_score = 100.0 - fidelity

                try:
                    orig_policy = self.gate_controller.policies[f"block_{b}"][f"head_{h}"]["HARDWARE_GATING_POLICY"]
                except (KeyError, TypeError):
                    orig_policy = "UNKNOWN"

                results["MASK_ONE_OUT"][head_key] = {
                    "fidelity": round(fidelity, 2),
                    "importance_score_drop": round(importance_score, 2),
                    "original_policy": orig_policy,
                }

                # Save checkpoint after each head
                if checkpoint_file:
                    with open(checkpoint_file, "w") as f:
                        json.dump(results, f, indent=4)

        # 3. ISOLATE-ONE Study
        print("\n[3/3] Running ISOLATE-ONE Study (Disabling all heads EXCEPT one)...")
        self.gate_controller.override_mode = "ISOLATE_ONE"

        for b in range(num_blocks):
            for h in range(num_heads):
                head_key = f"B{b}H{h}"
                if head_key in results["ISOLATE_ONE"]:
                    continue  # Already completed

                self.gate_controller.override_target = (b, h)
                preds, _, _, _ = self._run_pass(dataloader, device, f"Isolate {head_key}")
                matches = (preds == baseline_preds).sum().item()
                fidelity = (matches / len(baseline_preds)) * 100

                try:
                    orig_policy = self.gate_controller.policies[f"block_{b}"][f"head_{h}"]["HARDWARE_GATING_POLICY"]
                except (KeyError, TypeError):
                    orig_policy = "UNKNOWN"

                results["ISOLATE_ONE"][head_key] = {
                    "isolated_fidelity": round(fidelity, 2),
                    "original_policy": orig_policy,
                }

                if checkpoint_file:
                    with open(checkpoint_file, "w") as f:
                        json.dump(results, f, indent=4)

        # Clean up overrides
        self.gate_controller.override_mode = None
        self.gate_controller.override_target = None

        # ── Aggregate insights ───────────────────────────────────────
        insights: Dict = {
            "MASK_ONE_OUT_AVG_IMPORTANCE_BY_POLICY": {},
            "ISOLATE_ONE_AVG_FIDELITY_BY_POLICY": {},
        }

        for mode, target_metric, out_key in [
            ("MASK_ONE_OUT", "importance_score_drop", "MASK_ONE_OUT_AVG_IMPORTANCE_BY_POLICY"),
            ("ISOLATE_ONE", "isolated_fidelity", "ISOLATE_ONE_AVG_FIDELITY_BY_POLICY"),
        ]:
            policy_groups: Dict[str, List[float]] = defaultdict(list)
            for head_id, data in results[mode].items():
                policy_groups[data["original_policy"]].append(data[target_metric])
            for policy, values in policy_groups.items():
                insights[out_key][policy] = round(sum(values) / len(values), 2)

        results["AGGREGATED_INSIGHTS"] = insights

        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        # Clean up checkpoint
        if checkpoint_file and os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        print(f"\nHead Importance Study Complete! Report saved to {out_file}")
        return results
