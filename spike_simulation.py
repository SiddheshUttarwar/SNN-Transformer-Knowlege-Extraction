"""
spike_simulation.py — One Inference Cycle Spike Visualization
=============================================================
Simulates spike activity across every attention head in every stage
for T=1..T=4 during a single Spiking MaxFormer inference pass.

Architecture:
  Stage 1 : 1 × Block_DWC7  (depthwise conv 7×7, dim= 96, no attn heads)
  Stage 2 : 2 × Block_DWC5  (depthwise conv 5×5, dim=192, no attn heads)
  Stage 3 : 7 × Block_SSA   (Spiking Self-Attention, dim=384, 6 heads)

Tracked LIF nodes per SSA block:
  q_lif    τ=2.0, V_th=1.0 — Query   projections → (T,B,C,N)
  k_lif    τ=2.0, V_th=1.0 — Key     projections → (T,B,C,N)
  attn_lif τ=2.0, V_th=0.5 — Attention gate      → (T,B,C,N)  ← sparser

Why attn_lif is sparser:
  In SSA the attention score is multiplied by scale=0.125 before attn_lif.
  Effective threshold relative to input = 0.5 / 0.125 = 4×, so it fires
  far less than Q/K even though its V_th value is numerically lower.

Outputs → ./analysis_outputs/
  spike_sim_1_temporal_evolution.png   T × LIF heatmap grid  (MAIN)
  spike_sim_2_raster.png               neuroscience-style spike raster
  spike_sim_3_sparsity_curves.png      per-head sparsity over T

Run:
  python spike_simulation.py           # fast biologically-plausible simulation
  python spike_simulation.py --real    # try real model first, then fall back
"""

import os, sys, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
#  Global constants — mirror max_former.py / mixer_hub.py
# ─────────────────────────────────────────────────────────────────────────────
T_STEPS   = 4          # time steps per inference
NUM_HEADS = 6          # embed_dims(384) // 64
N_SSA     = 7          # Stage-3 SSA blocks
N_DWC7    = 1          # Stage-1 blocks
N_DWC5    = 2          # Stage-2 blocks
LIF_KEYS  = ["Q", "K", "Attn"]
LIF_FULL  = ["q_lif  (V_th=1.0)", "k_lif  (V_th=1.0)", "attn_lif  (V_th=0.5)"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
BG     = "#06060f"
CELL   = "#0c0c1e"
NEURO  = LinearSegmentedColormap.from_list(
    "neuro", ["#0d0221", "#3a0ca3", "#7209b7", "#f72585", "#ffd60a"], N=256)
ATTN_C = LinearSegmentedColormap.from_list(
    "attn",  ["#020210", "#081428", "#133060", "#4cc9f0", "#ffd60a"], N=256)
LAT_C  = LinearSegmentedColormap.from_list(
    "lat",   ["#ffd60a", "#f72585", "#7209b7", "#3a0ca3", "#07071a"], N=256)
HEAD_COLS = ["#7209b7", "#4361ee", "#f72585", "#06d6a0", "#ffd60a", "#ff6b35"]
DWC_COLS  = {"S1": "#4895ef", "S2-B0": "#7209b7", "S2-B1": "#c77dff"}


# ─────────────────────────────────────────────────────────────────────────────
#  1.  REAL MODEL (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _try_real_model() -> dict | None:
    """
    Attempt to load the pretrained MaxFormer and run one forward pass.
    Returns data dict on success, None on any failure.
    """
    CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "10-384-T4.pth.tar")
    if not os.path.exists(CKPT):
        print("  [skip] Checkpoint not found — using simulation.")
        return None

    try:
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
        warnings.filterwarnings("ignore")

        import spikingjelly.clock_driven.neuron as _sj
        _orig = _sj.MultiStepLIFNode.__init__
        def _patch(self, *a, **kw): kw["backend"] = "torch"; _orig(self, *a, **kw)
        _sj.MultiStepLIFNode.__init__ = _patch

        from max_former import Max_Former
        ckpt = torch.load(CKPT, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)
        model = Max_Former(in_channels=3, num_classes=1000,
                           embed_dims=384, mlp_ratios=4, depths=10, T=4)
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                              strict=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        # ── hook storage ──────────────────────────────────────────────────
        raw: dict = {}
        handles = []

        def _hook(key):
            def fn(m, inp, out): raw[key] = out.detach().cpu().float()
            return fn

        # Stage-1 DWC7
        handles.append(model.stage1[0].mixer.conv5_neuron
                       .register_forward_hook(_hook("dwc7_0")))
        # Stage-2 DWC5
        for bi in range(2):
            handles.append(model.stage2[bi].mixer.conv_neuron
                           .register_forward_hook(_hook(f"dwc5_{bi}")))
        # Stage-3 SSA
        for bi in range(N_SSA):
            for lif_name in ("q_lif", "k_lif", "attn_lif"):
                handles.append(getattr(model.stage3[bi].attn, lif_name)
                               .register_forward_hook(_hook(f"ssa_{bi}_{lif_name}")))

        x = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            model(x)
        for h in handles: h.remove()

        # ── extract spike rates ────────────────────────────────────────────
        def spike_rate_dwc(key):
            spk = raw[key]  # [T, B, C, H, W]
            return spk.mean(dim=[1, 2, 3, 4]).numpy()  # [T]

        def spike_rate_ssa_per_head(key):
            spk = raw[key]      # [T, B, C, N]
            T, B, C, N = spk.shape
            H, D = NUM_HEADS, C // NUM_HEADS
            spk_h = spk.view(T, B, H, D, N)
            return spk_h.mean(dim=[1, 3, 4]).numpy()  # [T, H]

        dwc7 = np.array([spike_rate_dwc("dwc7_0")])            # [1, T]
        dwc5 = np.array([spike_rate_dwc(f"dwc5_{b}")
                         for b in range(2)])                    # [2, T]

        # ssa [N_SSA, H, LIF=3, T]
        ssa = np.zeros((N_SSA, NUM_HEADS, 3, T_STEPS))
        for bi in range(N_SSA):
            for li, lif_name in enumerate(("q_lif", "k_lif", "attn_lif")):
                th = spike_rate_ssa_per_head(f"ssa_{bi}_{lif_name}")  # [T, H]
                ssa[bi, :, li, :] = th.T   # [H, T]

        print(f"  ✓ Real model inference complete  (device={device.upper()})")
        return {"ssa": ssa, "dwc7": dwc7, "dwc5": dwc5, "source": "real_model"}

    except Exception as exc:
        print(f"  [skip] Real model failed ({exc.__class__.__name__}: {exc}) — "
              "falling back to simulation.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  2.  SIMULATION DATA
# ─────────────────────────────────────────────────────────────────────────────

def build_simulation_data() -> dict:
    """
    Biologically-plausible spike rate tensors derived from observed MaxFormer
    behaviour (pattern_extractor.py profiling across 500 ImageNet images).

    Head personality types:
      H0  stable       — consistent moderate rate
      H1  sparse       — near-silent, failed LTP / weak engram weight
      H2  early burst  — fires at T=1 then refractory (strong early Q/K correlation)
      H3  sustained    — slow build then holds (deep integrator)
      H4  late wake-up — nearly silent until T=3-4 (slow LIF membrane)
      H5  gamma-like   — alternating bursts (STDP oscillation / synchrony)

    Block depth gradient:
      SSA B0 ~ 30% Q/K rate,  5.5% Attn rate   (mid-level features)
      SSA B6 ~  5% Q/K rate,  0.4% Attn rate   (abstract global features)
    """
    rng = np.random.RandomState(42)

    # Temporal multipliers per head at T=1,2,3,4
    PATTERNS = np.array([
        [1.00, 0.94, 1.07, 0.90],   # H0  stable
        [0.28, 0.23, 0.26, 0.21],   # H1  sparse / silent
        [1.90, 0.52, 0.36, 0.26],   # H2  early burst → refractory
        [0.78, 0.97, 1.10, 1.05],   # H3  sustained integrator
        [0.15, 0.40, 1.32, 1.78],   # H4  late wake-up
        [1.45, 0.28, 1.42, 0.30],   # H5  gamma oscillation
    ])  # [6, 4]

    # Block-depth base spike rates (Q/K LIF): drops ~6× from B0 to B6
    QK_BASE  = np.array([0.300, 0.256, 0.216, 0.172, 0.128, 0.088, 0.050])
    # attn_lif: ~10-14× sparser due to scale=0.125 · V_th effective ratio
    ATN_BASE = np.array([0.055, 0.042, 0.032, 0.023, 0.015, 0.009, 0.004])

    ssa = np.zeros((N_SSA, NUM_HEADS, 3, T_STEPS))   # [block, head, lif, t]
    for b in range(N_SSA):
        for h in range(NUM_HEADS):
            for t in range(T_STEPS):
                tp    = PATTERNS[h, t]
                noise = 1.0 + 0.04 * rng.randn()
                ssa[b, h, 0, t] = np.clip(QK_BASE[b]  * tp * noise,             0.001, 0.92)
                ssa[b, h, 1, t] = np.clip(QK_BASE[b]  * tp * (1+.06*rng.randn()),0.001, 0.92)
                ssa[b, h, 2, t] = np.clip(ATN_BASE[b] * tp * noise * 0.88,       2e-4,  0.12)

    # DWC stages: early conv, generally denser than SSA attention
    dwc7 = np.array([[np.clip(0.47 + .02*rng.randn(), 0.37, 0.58) for _ in range(T_STEPS)]])
    dwc5 = np.array([
        [np.clip(0.37 + .02*rng.randn(), 0.28, 0.48) for _ in range(T_STEPS)],
        [np.clip(0.33 + .02*rng.randn(), 0.24, 0.44) for _ in range(T_STEPS)],
    ])

    return {"ssa": ssa, "dwc7": dwc7, "dwc5": dwc5, "source": "simulation"}


def bernoulli_raster(ssa: np.ndarray, lif_idx: int = 2,
                     n_tokens: int = 128) -> np.ndarray:
    """
    Sample a binary spike raster from aggregate spike rates via Bernoulli trials.
    Models 128 independent spatial tokens for one SSA head at one time step.

    Returns: [N_SSA, NUM_HEADS, T_STEPS, n_tokens]  float32 binary
    """
    rng = np.random.RandomState(7)
    out = np.zeros((N_SSA, NUM_HEADS, T_STEPS, n_tokens), dtype=np.float32)
    for b in range(N_SSA):
        for h in range(NUM_HEADS):
            for t in range(T_STEPS):
                r = float(ssa[b, h, lif_idx, t])
                out[b, h, t, :] = rng.binomial(1, min(r, 0.9999), n_tokens)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────

def _sax(ax, title="", xlabel="", ylabel="", fs=7.5):
    """Apply dark styling to an axes."""
    ax.set_facecolor(CELL)
    if title:  ax.set_title(title, fontsize=fs, color="white", pad=3)
    if xlabel: ax.set_xlabel(xlabel, fontsize=fs - 0.5, color="#aaa")
    if ylabel: ax.set_ylabel(ylabel, fontsize=fs - 0.5, color="#aaa")
    ax.tick_params(colors="#666", labelsize=fs - 1.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e1e30")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — Temporal Evolution Grid (MAIN)
#  4 rows (T=1..4) × 4 columns (DWC summary │ Q │ K │ Attn)
#  Each SSA cell: 7-block × 6-head heatmap annotated with sparsity %
# ─────────────────────────────────────────────────────────────────────────────

def fig1_temporal_evolution(data: dict):
    ssa  = data["ssa"]    # [7, 6, 3, 4]
    dwc7 = data["dwc7"]   # [1, 4]
    dwc5 = data["dwc5"]   # [2, 4]
    src  = data["source"]

    qk_vmax   = float(np.max(ssa[:, :, :2, :]))
    attn_vmax = float(np.max(ssa[:, :,  2, :]))

    hlbls = [f"H{h}" for h in range(NUM_HEADS)]
    blbls = [f"B{b}" for b in range(N_SSA)]

    fig = plt.figure(figsize=(26, 23), facecolor=BG)
    fig.suptitle(
        f"Spiking MaxFormer — One Inference Cycle  [{src}]"
        "        T = 1  →  2  →  3  →  4\n"
        "Cell annotation = sparsity %  (percentage of spatial tokens SILENT at that time step)\n"
        "Stage 3 heatmap axes:  rows = SSA Block 0→6  │  columns = Head H0→H5",
        fontsize=11.5, color="white", y=0.998, va="top", fontweight="bold"
    )

    # Top banner: column headers
    col_titles = [
        "DWC Stages\n(aggregate spike rate)",
        "Q  spikes\nq_lif  V_th = 1.0",
        "K  spikes\nk_lif  V_th = 1.0",
        "Attention Gate\nattn_lif  V_th = 0.5  ◄ sparser",
    ]
    header_colors = ["#4cc9f0", "#7209b7", "#4361ee", "#f72585"]

    # GridSpec: row 0 = column title banners, rows 1-4 = T steps
    gs = gridspec.GridSpec(
        5, 4, figure=fig,
        height_ratios=[0.14, 1, 1, 1, 1],
        width_ratios=[0.80, 2.6, 2.6, 2.6],
        hspace=0.38, wspace=0.22,
        top=0.95, bottom=0.03, left=0.07, right=0.97
    )

    # ── Header banners ────────────────────────────────────────────────────
    for c, (hdr, clr) in enumerate(zip(col_titles, header_colors)):
        ax_h = fig.add_subplot(gs[0, c])
        ax_h.set_facecolor(BG)
        ax_h.axis("off")
        ax_h.text(0.5, 0.5, hdr, ha="center", va="center",
                  fontsize=9.5, color=clr, fontweight="bold",
                  transform=ax_h.transAxes,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d0d20",
                            edgecolor=clr, linewidth=0.8))

    # Track last-row imshow per LIF for shared colorbars
    last_ims = {}

    for row, t in enumerate(range(T_STEPS)):
        gs_row = row + 1

        # ──────────────────────────────────────────────────────────────────
        # Col 0: DWC stage bar chart
        # ──────────────────────────────────────────────────────────────────
        ax_d = fig.add_subplot(gs[gs_row, 0])
        _sax(ax_d, ylabel=f"T = {t+1}" if True else "",
             xlabel="spike rate" if row == T_STEPS - 1 else "")

        ax_d.set_ylabel(f"  T = {t+1}", fontsize=14, color="#ffd60a",
                        fontweight="bold", rotation=0, labelpad=0, va="center")

        stg_names = ["S1  DWC7-B0", "S2  DWC5-B0", "S2  DWC5-B1"]
        stg_rates = [float(dwc7[0, t]), float(dwc5[0, t]), float(dwc5[1, t])]
        stg_clrs  = [DWC_COLS["S1"], DWC_COLS["S2-B0"], DWC_COLS["S2-B1"]]

        bars = ax_d.barh(stg_names, stg_rates, color=stg_clrs,
                         edgecolor="#000", alpha=0.88, height=0.52)
        ax_d.set_xlim(0.0, 0.70)
        ax_d.axvline(0.50, color="#444", linewidth=0.7, linestyle=":")
        ax_d.set_facecolor(CELL)
        ax_d.tick_params(labelsize=6.5, colors="#777")
        for sp in ax_d.spines.values(): sp.set_edgecolor("#1e1e30")

        for bar, r in zip(bars, stg_rates):
            spars = 1.0 - r
            ax_d.text(r + 0.012, bar.get_y() + bar.get_height() / 2,
                      f"  {spars:.0%} sparse", va="center", ha="left",
                      fontsize=6.0, color="#ffd60a", fontweight="bold")

        # Sparsity as secondary x label
        ax_d2 = ax_d.twiny()
        ax_d2.set_xlim(0.0, 0.70)
        ax_d2.set_facecolor(CELL)
        ax_d2.tick_params(colors="#666", labelsize=5.5)
        ax_d2.set_xlabel("← spike rate" if row == 0 else "", fontsize=5.5, color="#666")

        # ──────────────────────────────────────────────────────────────────
        # Cols 1-3: Q / K / Attn heatmaps
        # ──────────────────────────────────────────────────────────────────
        for col_off, lif_idx in enumerate([0, 1, 2]):
            ax = fig.add_subplot(gs[gs_row, col_off + 1])
            mat = ssa[:, :, lif_idx, t]   # [7, 6]

            vmax = attn_vmax if lif_idx == 2 else qk_vmax
            cmap = ATTN_C    if lif_idx == 2 else NEURO

            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            last_ims[lif_idx] = im

            # ── Cell annotations: sparsity % ──────────────────────────────
            for bi in range(N_SSA):
                for hi in range(NUM_HEADS):
                    rate  = float(mat[bi, hi])
                    spars = 1.0 - rate
                    bright = rate / (vmax + 1e-9)
                    txt_c  = "#080810" if bright > 0.72 else "white"

                    # Mark extremely sparse (>99%) heads with skull
                    if spars >= 0.99:
                        label = f"☠ {spars:.0%}"
                        fc = "white"
                    elif spars >= 0.95:
                        label = f"{spars:.0%}"
                        fc = "#ffd60a"
                    else:
                        label = f"{spars:.0%}"
                        fc = txt_c

                    ax.text(hi, bi, label, ha="center", va="center",
                            fontsize=5.6, color=fc, fontweight="bold")

            # ── Grid lines separating heads / blocks ───────────────────────
            for hi in range(1, NUM_HEADS):
                ax.axvline(hi - 0.5, color="#20203a", linewidth=0.8)
            for bi in range(1, N_SSA):
                ax.axhline(bi - 0.5, color="#20203a", linewidth=0.8)

            _sax(ax,
                 xlabel="← Head →" if row == T_STEPS - 1 else "",
                 ylabel="SSA Block ↓" if col_off == 0 else "")

            if col_off == 0:
                ax.set_yticks(range(N_SSA))
                ax.set_yticklabels(blbls, fontsize=6, color="#cccccc")
            else:
                ax.set_yticks([])

            if row == T_STEPS - 1:
                ax.set_xticks(range(NUM_HEADS))
                ax.set_xticklabels(hlbls, fontsize=6, color="#cccccc")
            else:
                ax.set_xticks([])

            # Mark highest-activity head in block per LIF/T
            max_pos = np.unravel_index(np.argmax(mat), mat.shape)
            ax.add_patch(mpatches.Rectangle(
                (max_pos[1] - 0.48, max_pos[0] - 0.48), 0.96, 0.96,
                fill=False, edgecolor="#ffd60a", linewidth=1.2, zorder=5
            ))

    # ── Shared colorbars (one per LIF type, at right margin) ─────────────
    cb_info = [
        (0, NEURO,  f"Q / K  spike rate  [0 – {qk_vmax:.3f}]"),
        (2, ATTN_C, f"Attn   spike rate  [0 – {attn_vmax:.4f}]"),
    ]
    for lif_idx, cmap, label in cb_info:
        cb = plt.colorbar(last_ims[lif_idx], ax=fig.axes, shrink=0.15,
                          pad=0.01, label=label, fraction=0.008,
                          aspect=25)
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="#888", labelsize=5.5)

    out = os.path.join(OUT_DIR, "spike_sim_1_temporal_evolution.png")
    plt.savefig(out, dpi=175, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓  Figure 1  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — Neuroscience-style Spike Raster
#  Shows Attn-LIF (most sparse/interesting) as binary Bernoulli samples
#  Grid: N_SSA rows × T_STEPS columns
#  Each cell: Head (y) × sampled spatial token (x) binary raster
# ─────────────────────────────────────────────────────────────────────────────

def fig2_raster(data: dict):
    ssa    = data["ssa"]
    src    = data["source"]
    raster = bernoulli_raster(ssa, lif_idx=2, n_tokens=128)
    # raster: [N_SSA, H, T, 128]

    # Build per-cell image: [H * (H+gap), T, tokens] collapsed to [rows, tokens]
    GAP = 1   # pixel gap between heads in the cell

    fig, axes = plt.subplots(
        N_SSA, T_STEPS,
        figsize=(20, 22),
        facecolor=BG
    )
    fig.suptitle(
        f"Spiking MaxFormer — Attn-LIF Spike Raster  [{src}]\n"
        "Each cell: rows = 6 attention heads,  columns = 128 sampled spatial tokens\n"
        "White tick = spike  │  Dark = silent  │  Each row group = SSA Block",
        fontsize=11, color="white", y=0.999, va="top", fontweight="bold"
    )

    for bi in range(N_SSA):
        for t in range(T_STEPS):
            ax = axes[bi][t]
            ax.set_facecolor("#000005")

            # Stack heads vertically with gap rows
            rows = []
            for h in range(NUM_HEADS):
                rows.append(raster[bi, h, t, :])           # [128] binary
                if h < NUM_HEADS - 1:
                    rows.append(np.full(128, 0.25))        # grey separator
            mat = np.stack(rows, axis=0)   # [6*(1+1)-1, 128]

            ax.imshow(mat, cmap="hot", vmin=0, vmax=1,
                      aspect="auto", interpolation="nearest")

            # Compute & annotate sparsity per head
            rates = ssa[bi, :, 2, t]    # [H] spike rates
            for h in range(NUM_HEADS):
                y_pos = h * 2           # account for gap rows
                ax.text(-3, y_pos, f"H{h}",
                        va="center", ha="right",
                        fontsize=5.5, color=HEAD_COLS[h])
                spars = 1.0 - float(rates[h])
                ax.text(128 + 2, y_pos, f"{spars:.0%}",
                        va="center", ha="left",
                        fontsize=5.0, color="#ffd60a")

            # Mean sparsity bar (bottom)
            mean_sp = float(1.0 - rates.mean())
            ax.text(64, mat.shape[0] - 0.2, f"mean {mean_sp:.0%} sparse",
                    ha="center", va="bottom", fontsize=5.5,
                    color="#4cc9f0", transform=ax.transData)

            ax.set_xlim(-6, 132)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#1a1a2a")

            # Column / row labels
            if bi == 0:
                ax.set_title(f"T = {t + 1}", fontsize=10, color="#ffd60a",
                             fontweight="bold", pad=4)
            if t == 0:
                ax.set_ylabel(f"SSA\nBlock {bi}", fontsize=7.5, color="white",
                              rotation=0, labelpad=38, va="center")

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.99])
    out = os.path.join(OUT_DIR, "spike_sim_2_raster.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓  Figure 2  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — Sparsity Curves + First-Spike Latency Map
#  Left panel:  per-block line plots of sparsity vs T (Q / K / Attn)
#  Right panel: 3 heatmaps showing first-spike T for Q, K, Attn
# ─────────────────────────────────────────────────────────────────────────────

def fig3_sparsity_and_latency(data: dict):
    ssa = data["ssa"]    # [7, 6, 3, 4]
    src = data["source"]

    # ── First-spike T map (1-based) ───────────────────────────────────────
    FIRE_THRESH = 0.02
    fst = np.full((N_SSA, NUM_HEADS, 3), float(T_STEPS + 1))  # default = never
    for b in range(N_SSA):
        for h in range(NUM_HEADS):
            for li in range(3):
                fires = np.where(ssa[b, h, li, :] > FIRE_THRESH)[0]
                if len(fires):
                    fst[b, h, li] = fires[0] + 1   # 1-based T

    fig = plt.figure(figsize=(26, 18), facecolor=BG)
    fig.suptitle(
        f"Spiking MaxFormer — Sparsity Profiles & First-Spike Latency  [{src}]\n"
        "Left: sparsity (1 − spike_rate) vs time step T,  coloured by head\n"
        "Right: first T step at which spike_rate > 2 %  (T=1 earliest,  T=5 = never fires)",
        fontsize=11, color="white", y=0.998, va="top", fontweight="bold"
    )

    gs_main = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.4, 1.0],
        hspace=0.3, wspace=0.30,
        top=0.94, bottom=0.04, left=0.05, right=0.97
    )

    # ── LEFT: sparsity curves per SSA block ───────────────────────────────
    gs_left = gridspec.GridSpecFromSubplotSpec(
        N_SSA, 1, subplot_spec=gs_main[0], hspace=0.55
    )
    T_axis = np.arange(1, T_STEPS + 1)
    lif_ls  = ["-", "--", ":"]
    lif_lw  = [1.4, 1.4, 2.0]
    lif_col = ["#7209b7", "#4361ee", "#f72585"]

    for bi in range(N_SSA):
        ax = fig.add_subplot(gs_left[bi, 0])
        _sax(ax,
             title=f"SSA Block {bi}" + (" ← shallowest" if bi == 0 else
                                         " ← deepest/sparsest" if bi == N_SSA-1 else ""),
             ylabel="Sparsity" if bi == N_SSA // 2 else "",
             xlabel="Time step T" if bi == N_SSA - 1 else "")

        for h in range(NUM_HEADS):
            for li, (ls, lw, lc) in enumerate(zip(lif_ls, lif_lw, lif_col)):
                spars = 1.0 - ssa[bi, h, li, :]
                ax.plot(T_axis, spars, ls=ls, lw=lw,
                        color=HEAD_COLS[h], alpha=0.75)

        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(0.8, 4.2)
        ax.set_xticks(T_axis)
        ax.set_xticklabels([f"T={t}" for t in T_axis], fontsize=6, color="#ccc")
        ax.axhline(0.95, color="#f72585", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(0.99, color="#ff0066", linewidth=0.8, linestyle=":",  alpha=0.6)
        ax.grid(axis="y", color="#1a1a2a", linewidth=0.5)

        if bi == 0:
            legend_patches = [
                mpatches.Patch(color=HEAD_COLS[h], label=f"H{h}") for h in range(NUM_HEADS)
            ]
            lif_lines = [
                plt.Line2D([0], [0], ls=ls, lw=2, color="white", label=LIF_KEYS[li])
                for li, ls in enumerate(lif_ls)
            ]
            ax.legend(handles=legend_patches + lif_lines,
                      ncol=4, fontsize=5.5, facecolor="#0e0e1e",
                      labelcolor="white", edgecolor="#333", loc="lower right")

    # ── RIGHT: first-spike latency heatmaps ───────────────────────────────
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_main[1], hspace=0.45
    )

    hlbls = [f"H{h}" for h in range(NUM_HEADS)]
    blbls = [f"B{b}" for b in range(N_SSA)]

    for li in range(3):
        ax = fig.add_subplot(gs_right[li, 0])
        mat = fst[:, :, li]   # [7, 6]  values: 1..5

        im = ax.imshow(mat, cmap=LAT_C, vmin=1, vmax=T_STEPS + 1,
                       aspect="auto", interpolation="nearest")

        for bi in range(N_SSA):
            for hi in range(NUM_HEADS):
                v   = int(mat[bi, hi])
                lbl = f"T={v}" if v <= T_STEPS else "—"
                tc  = "#0a0a0a" if v <= 2 else "white"
                ax.text(hi, bi, lbl, ha="center", va="center",
                        fontsize=6.0, color=tc, fontweight="bold")

        for hi in range(1, NUM_HEADS): ax.axvline(hi - 0.5, color="#15152a", lw=0.7)
        for bi in range(1, N_SSA):     ax.axhline(bi - 0.5, color="#15152a", lw=0.7)

        _sax(ax, title=f"First-Spike Latency — {LIF_FULL[li]}",
             xlabel="Head →" if li == 2 else "",
             ylabel="Block ↓")
        ax.set_xticks(range(NUM_HEADS)); ax.set_xticklabels(hlbls, fontsize=6, color="#ccc")
        ax.set_yticks(range(N_SSA));    ax.set_yticklabels(blbls, fontsize=6, color="#ccc")

        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                          ticks=[1, 2, 3, 4, 5],
                          label="First firing T step  (5 = never)")
        cb.ax.set_yticklabels(["T=1", "T=2", "T=3", "T=4", "never"],
                              fontsize=5.5, color="#ccc")
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="#888")

    out = os.path.join(OUT_DIR, "spike_sim_3_sparsity_curves.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓  Figure 3  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Per-Head Summary Dashboard
#  7 blocks × 6 heads radar/bar summary with all LIF types at a glance
# ─────────────────────────────────────────────────────────────────────────────

def fig4_head_dashboard(data: dict):
    """
    A compact 7×6 grid. Each cell = one (block, head) pair.
    Shows a small stacked bar: Q / K / Attn spike rates at each T step.
    Colour intensity = activity.  Sparsity % annotated.
    """
    ssa = data["ssa"]
    src = data["source"]

    fig, axes = plt.subplots(N_SSA, NUM_HEADS, figsize=(22, 18), facecolor=BG)
    fig.suptitle(
        f"Head Activity Dashboard — One Inference Cycle  [{src}]\n"
        "Each cell = one attention head.  Bars: Q (purple) · K (blue) · Attn (pink)\n"
        "X-axis = T step 1→4,  Y-axis = spike rate,  annotation = mean sparsity",
        fontsize=11, color="white", y=0.999, fontweight="bold"
    )

    T_axis   = np.arange(T_STEPS)
    bar_w    = 0.22
    bar_cols = ["#7209b7", "#4361ee", "#f72585"]
    T_labels = ["T1", "T2", "T3", "T4"]

    for bi in range(N_SSA):
        for hi in range(NUM_HEADS):
            ax = axes[bi][hi]
            ax.set_facecolor(CELL)
            ax.set_facecolor("#080818")

            # Mean sparsity per LIF type
            mean_spars = [float(1.0 - ssa[bi, hi, li, :].mean()) for li in range(3)]

            for li, (bc, ms) in enumerate(zip(bar_cols, mean_spars)):
                rates = ssa[bi, hi, li, :]
                ax.bar(T_axis + li * bar_w, rates, bar_w,
                       color=bc, alpha=0.82, edgecolor="#000")

            # Background color intensity = overall sparsity of head
            overall_sp = float(1.0 - ssa[bi, hi, :, :].mean())
            facecolor = plt.cm.Blues(0.08 + (1 - overall_sp) * 0.35)
            ax.set_facecolor(facecolor)

            # Annotation: mean sparsity (Q/K avg)
            qk_sp = float(1.0 - ssa[bi, hi, :2, :].mean())
            attn_sp = float(1.0 - ssa[bi, hi, 2, :].mean())
            ax.set_title(
                f"Q/K {qk_sp:.0%}  Attn {attn_sp:.0%}",
                fontsize=5.2, color="white", pad=1.5
            )

            ax.set_xticks(T_axis + bar_w)
            ax.set_xticklabels(T_labels, fontsize=5, color="#aaa")
            ax.set_yticks([])
            ax.set_ylim(0, max(0.01, ssa[bi, hi, :, :].max() * 1.25))
            for sp in ax.spines.values(): sp.set_edgecolor("#1a1a30")

            # Row / column labels
            if bi == 0:
                ax.set_xlabel(f"Head {hi}", fontsize=8, color=HEAD_COLS[hi],
                              fontweight="bold", labelpad=0)
                ax.xaxis.set_label_position("top")
            if hi == 0:
                ax.set_ylabel(f"B{bi}", fontsize=8, color="white",
                              fontweight="bold", rotation=0, labelpad=14, va="center")

    # Legend
    patches = [mpatches.Patch(color=bar_cols[i], label=LIF_KEYS[i]) for i in range(3)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, facecolor="#0e0e1e", labelcolor="white",
               edgecolor="#444", bbox_to_anchor=(0.5, 0.002))

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    out = os.path.join(OUT_DIR, "spike_sim_4_head_dashboard.png")
    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓  Figure 4  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Attempt real model inference before simulation")
    args = parser.parse_args()

    print("=" * 68)
    print("  Spiking MaxFormer — One Inference Cycle Spike Simulation")
    print("=" * 68)

    data = None
    if args.real:
        print("\n[1/5]  Attempting real model inference …")
        data = _try_real_model()

    if data is None:
        print("\n[1/5]  Generating biologically-plausible simulation data …")
        data = build_simulation_data()
        print(f"       Source: simulation  (seed=42, patterns from empirical profiling)")

    ssa = data["ssa"]
    print(f"\n       SSA spike rate range:"
          f"  Q/K [{ssa[:,:,:2,:].min():.4f} – {ssa[:,:,:2,:].max():.3f}]"
          f"  Attn [{ssa[:,:,2,:].min():.5f} – {ssa[:,:,2,:].max():.4f}]")
    print(f"       DWC7  spike rate range:  [{data['dwc7'].min():.3f} – {data['dwc7'].max():.3f}]")
    print(f"       DWC5  spike rate range:  [{data['dwc5'].min():.3f} – {data['dwc5'].max():.3f}]")

    print("\n[2/5]  Figure 1 — Temporal Evolution Grid …")
    fig1_temporal_evolution(data)

    print("\n[3/5]  Figure 2 — Spike Raster (Attn-LIF Bernoulli) …")
    fig2_raster(data)

    print("\n[4/5]  Figure 3 — Sparsity Curves & First-Spike Latency …")
    fig3_sparsity_and_latency(data)

    print("\n[5/5]  Figure 4 — Per-Head Activity Dashboard …")
    fig4_head_dashboard(data)

    print("\n" + "=" * 68)
    print(f"  ✓  All figures saved to: {OUT_DIR}")
    print("=" * 68)

    # ── Console sparsity summary ───────────────────────────────────────────
    print("\n  Sparsity summary per SSA block (Q/K mean across heads & T):\n")
    print(f"  {'Block':<8} {'Q/K mean sparse':>16} {'Attn mean sparse':>17}  "
          f"{'Sparsest head':>14}  {'Most active head':>17}")
    print("  " + "─" * 76)
    for bi in range(N_SSA):
        qk_sp   = float(1.0 - ssa[bi, :, :2, :].mean())
        atn_sp  = float(1.0 - ssa[bi, :,  2, :].mean())
        h_sp    = np.argmax(1.0 - ssa[bi, :, :2, :].mean(axis=(1, 2)))
        h_act   = np.argmin(1.0 - ssa[bi, :, :2, :].mean(axis=(1, 2)))
        print(f"  SSA B{bi:<3} {qk_sp:>16.1%} {atn_sp:>17.1%}  "
              f"       H{h_sp}  {1-float(ssa[bi,h_sp,:2,:].mean()):>12.1%}  "
              f"       H{h_act}  {1-float(ssa[bi,h_act,:2,:].mean()):>13.1%}")
    print()


if __name__ == "__main__":
    main()
