"""
SpikeGate Comprehensive Report Generator.

Runs large-scale empirical analysis (10 inference runs × 500 images each)
on MaxFormer and produces a detailed Markdown report with visualizations.
"""

import os
import sys
import json
import time
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

# ── Local imports ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from qkv_sparsity_extractor import load_maxformer, IMG_SIZE, BATCH_SIZE, NUM_HEADS
from spikegate.profiler import AutoProfiler
from spikegate.gating import DynamicGateController
from spikegate.evaluator import GatingAblationStudy

# ── Configuration ────────────────────────────────────────────────────
NUM_BLOCKS = 10
T_STEPS = 4
REPORT_DIR = "analysis_outputs/comprehensive_report"
IMAGES_DIR = os.path.join(REPORT_DIR, "figures")


# =====================================================================
#  Data Loading
# =====================================================================

def create_data_chunks(num_runs: int, images_per_run: int, batch_size: int):
    """Streams ImageNet-1K and yields lists of batches (one list per run)."""
    dataset = load_dataset(
        "mrm8488/ImageNet1K-val", split="train",
        streaming=True, trust_remote_code=True,
    )
    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    batches_per_run = images_per_run // batch_size
    buf = []
    run_batches = []

    for item in dataset:
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf.append(tfs(img).unsqueeze(0))

        if len(buf) == batch_size:
            run_batches.append((torch.cat(buf, dim=0), torch.zeros(batch_size)))
            buf = []

            if len(run_batches) == batches_per_run:
                yield run_batches
                run_batches = []
                num_runs -= 1
                if num_runs <= 0:
                    return


# =====================================================================
#  Visualization Helpers
# =====================================================================

def _save(fig, name):
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return path


def set_dark_style():
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


def plot_sparsity_heatmap(profile: dict, num_blocks: int, num_heads: int):
    """Block × Head Q-sparsity heatmap."""
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
    return _save(fig, "sparsity_heatmap.png")


def plot_dead_ghost_heatmap(profile: dict, num_blocks: int, num_heads: int):
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
    return _save(fig, "dead_ghost_heatmap.png")


def plot_temporal_sparsity(profile: dict, num_blocks: int, num_heads: int):
    """Per-timestep Q sparsity evolution for selected blocks."""
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
    return _save(fig, "temporal_sparsity.png")


def plot_policy_distribution(profile: dict, num_blocks: int, num_heads: int):
    """Pie chart of gating policy distribution."""
    counts = defaultdict(int)
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
    return _save(fig, "policy_pie.png")


def plot_run_stability(run_results: list):
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
    return _save(fig, "run_stability.png")


def plot_stbp_gap(profile: dict, num_blocks: int, num_heads: int):
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
    ax.set_ylabel("|Q_FST − K_FST| (timesteps)")
    ax.set_xticks(x + w * num_heads / 2)
    ax.set_xticklabels([f"B{b}" for b in range(num_blocks)])
    ax.legend(fontsize=8, ncol=num_heads)
    ax.set_title("STBP Q−K Temporal Gap", fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="y", alpha=0.2)
    return _save(fig, "stbp_gap.png")


# =====================================================================
#  Markdown Report Assembly
# =====================================================================

def generate_report(
    profile: dict,
    run_results: list,
    figure_paths: dict,
    num_blocks: int,
    num_heads: int,
    num_runs: int,
    images_per_run: int,
    elapsed_sec: float,
):
    """Writes a comprehensive Markdown report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_images = num_runs * images_per_run

    # Aggregate run stats
    fids = [r["fidelity"] for r in run_results]
    savs = [r["savings"] for r in run_results]
    mean_fid = np.mean(fids)
    std_fid = np.std(fids)
    mean_sav = np.mean(savs)
    std_sav = np.std(savs)

    # Policy counts
    policy_counts = defaultdict(int)
    for b in range(num_blocks):
        for h in range(num_heads):
            p = profile[f"block_{b}"][f"head_{h}"]["HARDWARE_GATING_POLICY"]
            policy_counts[p] += 1

    lines = []
    w = lines.append

    w("# SpikeGate — Comprehensive MaxFormer Analysis Report")
    w(f"\n> Generated on **{ts}** | {total_images} images across {num_runs} runs "
      f"| Elapsed: {elapsed_sec/60:.1f} min\n")

    # ── Executive Summary ────────────────────────────────────────────
    w("## Executive Summary\n")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Total Images Evaluated | **{total_images}** |")
    w(f"| Inference Runs | **{num_runs}** |")
    w(f"| Images per Run | **{images_per_run}** |")
    w(f"| Architecture | MaxFormer 10-384-T4 |")
    w(f"| Blocks × Heads | {num_blocks} × {num_heads} = {num_blocks*num_heads} total heads |")
    w(f"| Mean Baseline Fidelity | **{mean_fid:.2f}% ± {std_fid:.2f}%** |")
    w(f"| Mean Compute Savings | **{mean_sav:.2f}% ± {std_sav:.2f}%** |")
    w("")

    # ── Policy Distribution ──────────────────────────────────────────
    w("## 1. Hardware Gating Policy Distribution\n")
    w(f"![Policy Distribution]({os.path.basename(figure_paths['policy_pie'])})\n")
    w("| Policy | Count | Percentage |")
    w("|---|---|---|")
    total_heads = num_blocks * num_heads
    for p, c in sorted(policy_counts.items(), key=lambda x: -x[1]):
        w(f"| `{p}` | {c} | {c/total_heads*100:.1f}% |")
    w("")

    # ── Q-Sparsity Heatmap ───────────────────────────────────────────
    w("## 2. Q-Sparsity Heatmap (Block × Head)\n")
    w(f"![Sparsity Heatmap]({os.path.basename(figure_paths['sparsity'])})\n")

    # ── Temporal Evolution ───────────────────────────────────────────
    w("## 3. Temporal Q-Sparsity Evolution\n")
    w(f"![Temporal Sparsity]({os.path.basename(figure_paths['temporal'])})\n")

    # ── STBP Gap ─────────────────────────────────────────────────────
    w("## 4. STBP Q−K Temporal Gap\n")
    w(f"![STBP Gap]({os.path.basename(figure_paths['stbp_gap'])})\n")

    # ── Dead / Ghost Neurons ─────────────────────────────────────────
    w("## 5. Dead & Ghost Neuron Analysis\n")
    w(f"![Dead Ghost]({os.path.basename(figure_paths['dead_ghost'])})\n")

    # ── Run Stability ────────────────────────────────────────────────
    w("## 6. Gating Stability Across Runs\n")
    w(f"![Run Stability]({os.path.basename(figure_paths['stability'])})\n")
    w("| Run | Fidelity (%) | Savings (%) |")
    w("|---|---|---|")
    for i, r in enumerate(run_results, 1):
        w(f"| {i} | {r['fidelity']:.2f} | {r['savings']:.2f} |")
    w(f"| **Mean ± σ** | **{mean_fid:.2f} ± {std_fid:.2f}** | "
      f"**{mean_sav:.2f} ± {std_sav:.2f}** |")
    w("")

    # ── Per-Head Detail Table ────────────────────────────────────────
    w("## 7. Per-Head Detail Table\n")
    w("| Block | Head | Q Sparsity | K Sparsity | V Sparsity | "
      "STBP Gap | Dead Q% | Ghost Q% | Policy |")
    w("|---|---|---|---|---|---|---|---|---|")
    for b in range(num_blocks):
        for h in range(num_heads):
            d = profile[f"block_{b}"][f"head_{h}"]
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

    out_path = os.path.join(REPORT_DIR, "SpikeGate_Comprehensive_Report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Report written to {out_path}")
    return out_path


# =====================================================================
#  Main Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="SpikeGate Comprehensive Report")
    parser.add_argument("--runs", type=int, default=10, help="Number of inference runs")
    parser.add_argument("--images", type=int, default=500, help="Images per run")
    parser.add_argument("--fast", action="store_true", help="Quick test (2 runs × 50 images)")
    args = parser.parse_args()

    if args.fast:
        args.runs, args.images = 2, 50

    os.makedirs(IMAGES_DIR, exist_ok=True)
    set_dark_style()
    t0 = time.time()

    # ── 1. Load Model ────────────────────────────────────────────────
    print("=" * 60)
    print("  SpikeGate — Comprehensive MaxFormer Analysis")
    print("=" * 60)
    model, _, device = load_maxformer()

    # ── 2. Stream Data ───────────────────────────────────────────────
    print(f"\n📦 Streaming {args.runs} × {args.images} = "
          f"{args.runs * args.images} ImageNet images...")
    chunks = list(create_data_chunks(args.runs, args.images, BATCH_SIZE))
    print(f"   Loaded {len(chunks)} data chunks "
          f"({len(chunks[0])} batches each)\n")

    # ── 3. Calibration (first chunk) ─────────────────────────────────
    print("─" * 60)
    print("Phase 1: AutoProfiler Calibration")
    print("─" * 60)
    profiler = AutoProfiler(model, T=T_STEPS)
    profile = profiler.calibrate(
        chunks[0], device=device,
        num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS,
        out_file=os.path.join(REPORT_DIR, "gating_profile.json"),
    )

    # ── 4. Multi-Run Stability Study ─────────────────────────────────
    print("\n" + "─" * 60)
    print("Phase 2: Multi-Run Gating Stability Study")
    print("─" * 60)
    gate_ctrl = DynamicGateController(
        os.path.join(REPORT_DIR, "gating_profile.json")
    )
    evaluator = GatingAblationStudy(model, gate_ctrl)

    run_results = []
    for i, chunk in enumerate(chunks):
        print(f"\n  Run {i+1}/{len(chunks)}...")
        gate_ctrl.override_policy = None
        gate_ctrl.reset_compute_counters()

        model.eval()
        all_baseline, all_gated = [], []
        with torch.no_grad():
            for batch in tqdm(chunk, desc=f"Run {i+1}"):
                x = batch[0].to(device)

                # Baseline (no gating)
                gate_ctrl.override_policy = "ACTIVE_NO_GATE"
                gate_ctrl.reset_compute_counters()
                logits_base = model(x)
                all_baseline.append(logits_base.argmax(dim=-1).cpu())

                # Full policy
                gate_ctrl.override_policy = None
                gate_ctrl.reset_compute_counters()
                logits_gated = model(x)
                all_gated.append(logits_gated.argmax(dim=-1).cpu())

        baseline = torch.cat(all_baseline)
        gated = torch.cat(all_gated)
        fidelity = (gated == baseline).float().mean().item() * 100

        bypassed = gate_ctrl.bypassed_kv_computations
        total = gate_ctrl.total_kv_computations
        savings = (bypassed / total * 100) if total > 0 else 0.0

        run_results.append({"fidelity": fidelity, "savings": savings})
        print(f"  → Fidelity: {fidelity:.2f}%  Savings: {savings:.2f}%")

    # ── 5. Generate Visualizations ───────────────────────────────────
    print("\n" + "─" * 60)
    print("Phase 3: Generating Visualizations")
    print("─" * 60)
    figure_paths = {
        "sparsity": plot_sparsity_heatmap(profile, NUM_BLOCKS, NUM_HEADS),
        "dead_ghost": plot_dead_ghost_heatmap(profile, NUM_BLOCKS, NUM_HEADS),
        "temporal": plot_temporal_sparsity(profile, NUM_BLOCKS, NUM_HEADS),
        "policy_pie": plot_policy_distribution(profile, NUM_BLOCKS, NUM_HEADS),
        "stability": plot_run_stability(run_results),
        "stbp_gap": plot_stbp_gap(profile, NUM_BLOCKS, NUM_HEADS),
    }
    print(f"  Generated {len(figure_paths)} figures in {IMAGES_DIR}/")

    # ── 6. Assemble Report ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Phase 4: Assembling Markdown Report")
    print("─" * 60)
    elapsed = time.time() - t0
    generate_report(
        profile, run_results, figure_paths,
        NUM_BLOCKS, NUM_HEADS,
        len(chunks), args.images, elapsed,
    )

    # ── Save raw JSON results ────────────────────────────────────────
    with open(os.path.join(REPORT_DIR, "run_stability_results.json"), "w") as f:
        json.dump(run_results, f, indent=4)

    print(f"\n🎉 All done in {elapsed/60:.1f} minutes!")
    print(f"   Report:  {REPORT_DIR}/SpikeGate_Comprehensive_Report.md")
    print(f"   Profile: {REPORT_DIR}/gating_profile.json")
    print(f"   Figures: {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
