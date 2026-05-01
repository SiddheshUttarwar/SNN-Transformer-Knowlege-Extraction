"""
Comprehensive Report Generator for the SpikeGate framework.

Produces publication-quality visualizations and a detailed Markdown report
from AutoProfiler calibration data and multi-run gating stability results.
"""

import os
import json
import time
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
from tqdm import tqdm

from .policies import GatingPolicy

logger = logging.getLogger(__name__)


# =====================================================================
#  Dark Theme
# =====================================================================

def _set_dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.family": "sans-serif",
    })


def _save(fig, directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return path


# =====================================================================
#  Visualization Functions
# =====================================================================

def plot_sparsity_heatmap(
    profile: dict, num_blocks: int, num_heads: int, out_dir: str
) -> str:
    """Block x Head Q-sparsity heatmap."""
    mat = np.zeros((num_blocks, num_heads))
    for b in range(num_blocks):
        for h in range(num_heads):
            mat[b, h] = profile[f"block_{b}"][f"head_{h}"]["average_sparsity_q"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="magma_r",
                xticklabels=[f"H{h}" for h in range(num_heads)],
                yticklabels=[f"B{b}" for b in range(num_blocks)],
                ax=ax, linewidths=0.4, linecolor="#30363d",
                cbar_kws={"label": "Q Sparsity"})
    ax.set_title("Average Q-Sparsity per Head", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Transformer Block")
    return _save(fig, out_dir, "sparsity_heatmap.png")


def plot_dead_ghost_heatmap(
    profile: dict, num_blocks: int, num_heads: int, out_dir: str
) -> str:
    """Dead + Ghost neuron percentage heatmap (Q projection)."""
    dead = np.zeros((num_blocks, num_heads))
    ghost = np.zeros((num_blocks, num_heads))
    for b in range(num_blocks):
        for h in range(num_heads):
            d = profile[f"block_{b}"][f"head_{h}"]
            dead[b, h] = d["dead_neurons_pct_q"]
            ghost[b, h] = d["ghost_neurons_pct_q"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title, cmap in [
        (axes[0], dead, "Dead Neurons (Q) %", "Blues"),
        (axes[1], ghost, "Ghost Neurons (Q) %", "Reds"),
    ]:
        sns.heatmap(mat, annot=True, fmt=".1f", cmap=cmap, ax=ax,
                    xticklabels=[f"H{h}" for h in range(num_heads)],
                    yticklabels=[f"B{b}" for b in range(num_blocks)],
                    linewidths=0.4, linecolor="#30363d")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Block")
    fig.suptitle("Neuron Health Analysis (Q Projection)", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "dead_ghost_heatmap.png")


def plot_temporal_sparsity(
    profile: dict, num_blocks: int, num_heads: int, out_dir: str
) -> str:
    """Per-timestep Q sparsity evolution for early/mid/late blocks."""
    selected = [0, num_blocks // 2, num_blocks - 1]
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4),
                             sharey=True)
    palette = sns.color_palette("cool", num_heads)
    for idx, b in enumerate(selected):
        ax = axes[idx]
        for h in range(num_heads):
            vals = profile[f"block_{b}"][f"head_{h}"]["sparsity_q_per_timestep"]
            ax.plot(range(1, len(vals) + 1), vals, marker="o", color=palette[h],
                    label=f"H{h}", linewidth=2, markersize=5)
        ax.set_title(f"Block {b}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestep")
        if idx == 0:
            ax.set_ylabel("Q Sparsity")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Temporal Q-Sparsity Evolution", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "temporal_sparsity.png")


def plot_policy_distribution(
    profile: dict, num_blocks: int, num_heads: int, out_dir: str
) -> str:
    """Pie chart of gating policy distribution."""
    counts: Dict[str, int] = defaultdict(int)
    for b in range(num_blocks):
        for h in range(num_heads):
            p = profile[f"block_{b}"][f"head_{h}"]["HARDWARE_GATING_POLICY"]
            counts[p] += 1

    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = sns.color_palette("Set2", len(labels))

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%", colors=colors,
        startangle=140, pctdistance=0.8,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("#c9d1d9")
    short = [l.replace("STATICALLY_", "S_").replace("DYNAMIC_", "D_") for l in labels]
    ax.legend(wedges, short, loc="lower left", fontsize=9)
    ax.set_title("Hardware Gating Policy Distribution", fontsize=14,
                 fontweight="bold", pad=15)
    return _save(fig, out_dir, "policy_pie.png")


def plot_run_stability(run_results: list, out_dir: str) -> str:
    """Bar chart of fidelity and savings across N runs."""
    runs = list(range(1, len(run_results) + 1))
    fids = [r["fidelity"] for r in run_results]
    savs = [r["savings"] for r in run_results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(runs))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, fids, w, label="Fidelity %", color="#58a6ff",
                    edgecolor="#0d1117")
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w / 2, savs, w, label="Savings %", color="#f78166",
                    edgecolor="#0d1117")

    ax1.set_xlabel("Inference Run")
    ax1.set_ylabel("Baseline Fidelity (%)", color="#58a6ff")
    ax2.set_ylabel("Compute Savings (%)", color="#f78166")
    ax1.set_xticks(x)
    ax1.set_xticklabels(runs)
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)

    lines = [bars1, bars2]
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="upper right")
    ax1.set_title("Gating Stability Across Runs", fontsize=14,
                  fontweight="bold", pad=12)
    ax1.grid(True, axis="y", alpha=0.2)
    return _save(fig, out_dir, "run_stability.png")


def plot_stbp_gap(
    profile: dict, num_blocks: int, num_heads: int, out_dir: str
) -> str:
    """STBP Q-K temporal gap bar chart."""
    mat = np.zeros((num_blocks, num_heads))
    for b in range(num_blocks):
        for h in range(num_heads):
            mat[b, h] = profile[f"block_{b}"][f"head_{h}"]["stbp_qk_temporal_gap_abs"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(num_blocks)
    w = 0.8 / num_heads
    palette = sns.color_palette("cool", num_heads)
    for h in range(num_heads):
        ax.bar(x + h * w, mat[:, h], w, label=f"H{h}", color=palette[h],
               edgecolor="#0d1117")
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("|Q_FST - K_FST| (timesteps)")
    ax.set_xticks(x + w * num_heads / 2)
    ax.set_xticklabels([f"B{b}" for b in range(num_blocks)])
    ax.legend(fontsize=8, ncol=num_heads)
    ax.set_title("STBP Q-K Temporal Gap", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="y", alpha=0.2)
    return _save(fig, out_dir, "stbp_gap.png")


# =====================================================================
#  Report Generator
# =====================================================================

class ReportGenerator:
    """Generates comprehensive analysis reports from SpikeGate profiling data.

    Orchestrates multi-run stability evaluation, produces 6 publication-quality
    figures, and assembles a detailed Markdown report.

    Args:
        model: The spiking model to evaluate.
        gate_controller: Gate controller loaded with a profile.
        profile: The gating profile dict from ``AutoProfiler.calibrate()``.
        num_blocks: Number of transformer blocks.
        num_heads: Number of attention heads per block.
        output_dir: Root directory for the report and figures.
    """

    def __init__(
        self,
        model,
        gate_controller,
        profile: dict,
        num_blocks: int,
        num_heads: int,
        output_dir: str = "analysis_outputs/comprehensive_report",
    ) -> None:
        self.model = model
        self.gate_controller = gate_controller
        self.profile = profile
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")

        os.makedirs(self.figures_dir, exist_ok=True)
        _set_dark_style()

    def run_stability_study(
        self,
        data_chunks: list,
        device: str = "cuda",
    ) -> List[Dict]:
        """Runs the gating profile across multiple data chunks and records fidelity/savings.

        Args:
            data_chunks: List of dataloaders (one per run). Each is a list of
                ``(images, labels)`` tuples.
            device: Device string.

        Returns:
            List of dicts with ``fidelity`` and ``savings`` per run.
        """
        run_results: List[Dict] = []

        for i, chunk in enumerate(data_chunks):
            print(f"  Run {i + 1}/{len(data_chunks)}...")
            self.model.eval()
            all_baseline, all_gated = [], []

            with torch.no_grad():
                for batch in tqdm(chunk, desc=f"Run {i + 1}"):
                    x = batch[0].to(device)

                    # Baseline (no gating)
                    self.gate_controller.override_policy = GatingPolicy.ACTIVE_NO_GATE.value
                    self.gate_controller.reset_compute_counters()
                    logits_base = self.model(x)
                    all_baseline.append(logits_base.argmax(dim=-1).cpu())

                    # Full policy
                    self.gate_controller.override_policy = None
                    self.gate_controller.reset_compute_counters()
                    logits_gated = self.model(x)
                    all_gated.append(logits_gated.argmax(dim=-1).cpu())

            baseline = torch.cat(all_baseline)
            gated = torch.cat(all_gated)
            fidelity = (gated == baseline).float().mean().item() * 100

            bypassed = self.gate_controller.bypassed_kv_computations
            total = self.gate_controller.total_kv_computations
            savings = (bypassed / total * 100) if total > 0 else 0.0

            run_results.append({"fidelity": fidelity, "savings": savings})
            print(f"    Fidelity: {fidelity:.2f}%  Savings: {savings:.2f}%")

        return run_results

    def generate_figures(self, run_results: List[Dict]) -> Dict[str, str]:
        """Generates all 6 report visualizations.

        Returns:
            Dict mapping figure name to its absolute file path.
        """
        print("Generating visualizations...")
        d = self.figures_dir
        paths = {
            "sparsity": plot_sparsity_heatmap(self.profile, self.num_blocks, self.num_heads, d),
            "dead_ghost": plot_dead_ghost_heatmap(self.profile, self.num_blocks, self.num_heads, d),
            "temporal": plot_temporal_sparsity(self.profile, self.num_blocks, self.num_heads, d),
            "policy_pie": plot_policy_distribution(self.profile, self.num_blocks, self.num_heads, d),
            "stability": plot_run_stability(run_results, d),
            "stbp_gap": plot_stbp_gap(self.profile, self.num_blocks, self.num_heads, d),
        }
        print(f"  Generated {len(paths)} figures in {d}/")
        return paths

    def generate_report(
        self,
        run_results: List[Dict],
        figure_paths: Dict[str, str],
        images_per_run: int,
        elapsed_sec: float = 0.0,
    ) -> str:
        """Assembles a comprehensive Markdown report.

        Args:
            run_results: Output from ``run_stability_study()``.
            figure_paths: Output from ``generate_figures()``.
            images_per_run: Number of images evaluated per run.
            elapsed_sec: Total wall time in seconds.

        Returns:
            Path to the written Markdown file.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        num_runs = len(run_results)
        total_images = num_runs * images_per_run

        fids = [r["fidelity"] for r in run_results]
        savs = [r["savings"] for r in run_results]
        mean_fid, std_fid = np.mean(fids), np.std(fids)
        mean_sav, std_sav = np.mean(savs), np.std(savs)

        policy_counts: Dict[str, int] = defaultdict(int)
        for b in range(self.num_blocks):
            for h in range(self.num_heads):
                p = self.profile[f"block_{b}"][f"head_{h}"]["HARDWARE_GATING_POLICY"]
                policy_counts[p] += 1

        lines: List[str] = []
        w = lines.append

        w("# SpikeGate — Comprehensive MaxFormer Analysis Report")
        w(f"\n> Generated on **{ts}** | {total_images} images across {num_runs} runs "
          f"| Elapsed: {elapsed_sec / 60:.1f} min\n")

        # Executive Summary
        w("## Executive Summary\n")
        w("| Metric | Value |")
        w("|---|---|")
        w(f"| Total Images Evaluated | **{total_images}** |")
        w(f"| Inference Runs | **{num_runs}** |")
        w(f"| Images per Run | **{images_per_run}** |")
        w(f"| Architecture | MaxFormer 10-384-T4 |")
        total_heads = self.num_blocks * self.num_heads
        w(f"| Blocks x Heads | {self.num_blocks} x {self.num_heads} = {total_heads} total heads |")
        w(f"| Mean Baseline Fidelity | **{mean_fid:.2f}% +/- {std_fid:.2f}%** |")
        w(f"| Mean Compute Savings | **{mean_sav:.2f}% +/- {std_sav:.2f}%** |")
        w("")

        # Policy Distribution
        w("## 1. Hardware Gating Policy Distribution\n")
        w(f"![Policy Distribution]({os.path.basename(figure_paths['policy_pie'])})\n")
        w("| Policy | Count | Percentage |")
        w("|---|---|---|")
        for p, c in sorted(policy_counts.items(), key=lambda x: -x[1]):
            w(f"| `{p}` | {c} | {c / total_heads * 100:.1f}% |")
        w("")

        # Q-Sparsity
        w("## 2. Q-Sparsity Heatmap (Block x Head)\n")
        w(f"![Sparsity Heatmap]({os.path.basename(figure_paths['sparsity'])})\n")

        # Temporal
        w("## 3. Temporal Q-Sparsity Evolution\n")
        w(f"![Temporal Sparsity]({os.path.basename(figure_paths['temporal'])})\n")

        # STBP Gap
        w("## 4. STBP Q-K Temporal Gap\n")
        w(f"![STBP Gap]({os.path.basename(figure_paths['stbp_gap'])})\n")

        # Dead / Ghost
        w("## 5. Dead & Ghost Neuron Analysis\n")
        w(f"![Dead Ghost]({os.path.basename(figure_paths['dead_ghost'])})\n")

        # Run Stability
        w("## 6. Gating Stability Across Runs\n")
        w(f"![Run Stability]({os.path.basename(figure_paths['stability'])})\n")
        w("| Run | Fidelity (%) | Savings (%) |")
        w("|---|---|---|")
        for i, r in enumerate(run_results, 1):
            w(f"| {i} | {r['fidelity']:.2f} | {r['savings']:.2f} |")
        w(f"| **Mean +/- std** | **{mean_fid:.2f} +/- {std_fid:.2f}** | "
          f"**{mean_sav:.2f} +/- {std_sav:.2f}** |")
        w("")

        # Per-Head Detail
        w("## 7. Per-Head Detail Table\n")
        w("| Block | Head | Q Sparsity | K Sparsity | V Sparsity | "
          "STBP Gap | Dead Q% | Ghost Q% | Policy |")
        w("|---|---|---|---|---|---|---|---|---|")
        for b in range(self.num_blocks):
            for h in range(self.num_heads):
                d = self.profile[f"block_{b}"][f"head_{h}"]
                sq = d["average_sparsity_q"]
                sk = np.mean(d["sparsity_k_per_timestep"])
                sv = np.mean(d["sparsity_v_per_timestep"])
                gap = d["stbp_qk_temporal_gap_abs"]
                dq = d["dead_neurons_pct_q"]
                gq = d["ghost_neurons_pct_q"]
                pol = d["HARDWARE_GATING_POLICY"]
                w(f"| {b} | {h} | {sq:.4f} | {sk:.4f} | {sv:.4f} | "
                  f"{gap:.3f} | {dq:.1f} | {gq:.1f} | `{pol}` |")
        w("")

        out_path = os.path.join(self.output_dir, "SpikeGate_Comprehensive_Report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Report written to {out_path}")
        logger.info("Report saved to %s", out_path)
        return out_path

    def run_full_pipeline(
        self,
        data_chunks: list,
        device: str = "cuda",
        images_per_run: int = 500,
    ) -> str:
        """Convenience method: runs stability study, generates figures, writes report.

        Args:
            data_chunks: List of dataloaders (one per run).
            device: Device string.
            images_per_run: Number of images per chunk.

        Returns:
            Path to the generated Markdown report.
        """
        t0 = time.time()

        print("=" * 60)
        print("  Phase: Multi-Run Gating Stability Study")
        print("=" * 60)
        run_results = self.run_stability_study(data_chunks, device)

        with open(os.path.join(self.output_dir, "run_stability_results.json"), "w") as f:
            json.dump(run_results, f, indent=4)

        print("\n" + "=" * 60)
        print("  Phase: Generating Visualizations")
        print("=" * 60)
        figure_paths = self.generate_figures(run_results)

        print("\n" + "=" * 60)
        print("  Phase: Assembling Report")
        print("=" * 60)
        elapsed = time.time() - t0
        report_path = self.generate_report(run_results, figure_paths, images_per_run, elapsed)

        print(f"\nAll done in {elapsed / 60:.1f} minutes!")
        return report_path
