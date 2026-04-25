"""
visualize_model.py — Biological Learning Analysis for Spiking MaxFormer
=======================================================================
Applies biological-learning intuitions to a pretrained Spiking Transformer:

  1. Engram (Hebbian) Analysis      — W_q / W_k / W_v magnitude heatmaps
  2. STDP Profiling (ISI Proxy)     — inter-spike timing tightness for Q and K
  3. Predictive Coding / Sparsity   — per-head spike density across time steps
  4. Attentional Synchrony          — head-pairwise correlation of attn outputs

All figures are written to ./analysis_outputs/.
Run from the MaxFormer root directory:
    python visualize_model.py

Checkpoint: checkpoints/10-384-T4.pth.tar
  arch: maxformer_10_384  (ImageNet, embed_dims=384, num_classes=1000, T=4)
  Stage blocks: stage1=1 (DWC7), stage2=2 (DWC5), stage3=7 (SSA)
  num_heads = 384 // 64 = 6
"""

import sys, os, warnings
# Use the imagenet subfolder — this is where 10-384-T4.pth.tar was trained from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from datasets import load_dataset
from torchvision import transforms

# ── colour palette ────────────────────────────────────────────────────────────
NEURO_CMAP = LinearSegmentedColormap.from_list(
    "neuro", ["#0d0221", "#3a0ca3", "#7209b7", "#f72585", "#ffd60a"], N=256
)
SYNC_CMAP  = LinearSegmentedColormap.from_list(
    "sync",  ["#023e8a", "#48cae4", "#ffffff", "#f77f00", "#d62828"], N=256
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "10-384-T4.pth.tar")

# ==============================================================================
# 1 ── MODEL LOADING
# ==============================================================================

# ── Architecture constants (from imagenet/max_former.py: maxformer_10_384) ────
NUM_HEADS   = 6          # embed_dims(384) // 64
NUM_CLASSES = 1000       # ImageNet
EMBED_DIMS  = 384
T_STEPS     = 4
IMG_SIZE    = 224        # ImageNet input resolution

def load_model_and_weights(ckpt_path: str):
    """
    Load the checkpoint.  We import Max_Former lazily so sys.path can be
    patched first.  Backend falls back to 'torch' if 'cupy' is unavailable.
    arch: maxformer_10_384  →  embed_dims=384, num_classes=1000, T=4
    Stage layout: stage1=1×DWC7, stage2=2×DWC5, stage3=7×SSA
    """
    print("="*70)
    print("  Spiking MaxFormer — Biological Learning Analysis")
    print("="*70)
    print(f"\n[1/5]  Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    print(f"   Checkpoint epoch={ckpt.get('epoch','?')}  "
          f"top-1={ckpt.get('metric','?')}%  arch={ckpt.get('arch','?')}")

    # ── patch cupy → torch backend so we can run on CPU without CUDA ──────────
    import spikingjelly.clock_driven.neuron as _sj_neuron
    _orig_init = _sj_neuron.MultiStepLIFNode.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["backend"] = "torch"
        _orig_init(self, *args, **kwargs)

    _sj_neuron.MultiStepLIFNode.__init__ = _patched_init

    # Import from imagenet/ subfolder (already in sys.path[0])
    from max_former import Max_Former

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Exact architecture matching maxformer_10_384:
    #   stage1=1×DWC7, stage2=2×DWC5, stage3=7×SSA, embed_dims=384, T=4
    model = Max_Former(
        in_channels=3, num_classes=NUM_CLASSES,
        embed_dims=EMBED_DIMS, mlp_ratios=4,
        depths=10, T=T_STEPS
    )

    # strip 'module.' prefix if trained with DataParallel
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"   [!] Missing keys  ({len(missing)}): {missing[:3]} …")
    if unexpected:
        print(f"   [!] Unexpected ({len(unexpected)}): first few shown above")
    else:
        print("   ✓  State dict loaded perfectly (strict match)")

    model.to(device).eval()
    print(f"   Model on {device.upper()} | params: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    return model, state_dict, device


# ==============================================================================
# 2 ── ENGRAM ANALYSIS (Hebbian / LTP weight magnitudes)
# ==============================================================================

def analyse_engrams(state_dict: dict):
    """
    Examine W_q, W_k, W_v magnitude per SSA head.
    High-magnitude slices ≡ Long-Term Potentiation — strong pattern pathways.
    """
    print("\n[2/5]  Engram Analysis  (W_q / W_k / W_v magnitude per head)")

    # collect SSA attention weight sub-tensors
    ssa_records = defaultdict(dict)            # block_id → {q, k, v} tensors

    for key, val in state_dict.items():
        key_clean = key.replace("module.", "")
        for proj in ("q_conv", "k_conv", "v_conv"):
            if f"attn.{proj}.weight" in key_clean and "stage3" in key_clean:
                # key like: stage3.0.attn.q_conv.weight
                parts = key_clean.split(".")
                block_id = int(parts[1])       # stage3.{block_id}.attn…
                ssa_records[block_id][proj] = val.float().cpu()

    if not ssa_records:
        print("   [!] No SSA weights found — skipping.")
        return

    num_blocks = len(ssa_records)
    num_heads  = NUM_HEADS                     # 384 // 64 = 6 in maxformer_10_384
    fig, axes = plt.subplots(
        num_blocks, 3,
        figsize=(14, 4 * num_blocks),
        squeeze=False,
        facecolor="#0a0a0a"
    )
    fig.suptitle(
        "Engram Analysis — SSA Head Weight Magnitudes (Hebbian / LTP Proxy)",
        fontsize=14, color="white", y=1.01
    )

    proj_labels = {"q_conv": "Query  $W_q$", "k_conv": "Key  $W_k$",
                   "v_conv": "Value  $W_v$"}

    for row, (blk, projs) in enumerate(sorted(ssa_records.items())):
        for col, proj in enumerate(("q_conv", "k_conv", "v_conv")):
            ax = axes[row][col]
            ax.set_facecolor("#111111")
            if proj not in projs:
                ax.axis("off")
                continue

            w = projs[proj]                    # shape: (C_out, C_in, 1) conv1d
            C = w.shape[0]
            head_dim = C // num_heads
            # reshape → (num_heads, head_dim, C_in) → mean abs per head
            w_heads = w.view(num_heads, head_dim, -1)
            mag = w_heads.abs().mean(dim=[1, 2]).numpy()   # (num_heads,)

            # full weight slice heatmap
            w_2d = w.squeeze(-1).numpy()       # (C_out, C_in)
            im = ax.imshow(
                np.abs(w_2d), aspect="auto", cmap=NEURO_CMAP,
                interpolation="nearest"
            )
            plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                         label="|w|").ax.yaxis.label.set_color("white")

            # overlay head-boundary lines
            for h in range(1, num_heads):
                ax.axhline(h * head_dim - 0.5, color="#ffd60a",
                           linewidth=0.6, alpha=0.7)

            # annotate mean magnitude per head on the right
            for h in range(num_heads):
                ax.text(
                    w_2d.shape[1] + 1,
                    h * head_dim + head_dim / 2,
                    f"{mag[h]:.3f}",
                    va="center", ha="left",
                    fontsize=6.5, color="#ffd60a"
                )

            ax.set_title(
                f"Block {blk} — {proj_labels[proj]}",
                fontsize=10, color="white", pad=4
            )
            ax.set_xlabel("Input Channel", fontsize=8, color="#aaaaaa")
            ax.set_ylabel("Output Channel", fontsize=8, color="#aaaaaa")
            ax.tick_params(colors="#888888", labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "1_engram_weight_heatmaps.png")
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    plt.close()
    print(f"   Saved → {out}")

    # ── per-head bar chart ────────────────────────────────────────────────────
    fig2, axb = plt.subplots(
        1, num_blocks, figsize=(6 * num_blocks, 4),
        sharey=True, facecolor="#0a0a0a"
    )
    if num_blocks == 1:
        axb = [axb]

    for col, (blk, projs) in enumerate(sorted(ssa_records.items())):
        ax = axb[col]
        ax.set_facecolor("#111111")
        x = np.arange(num_heads)
        width = 0.25
        colors = ["#7209b7", "#4361ee", "#f72585"]
        for i, (proj, colour) in enumerate(
                zip(("q_conv", "k_conv", "v_conv"), colors)):
            if proj not in projs:
                continue
            w = projs[proj]
            w_heads = w.squeeze(-1).view(num_heads, -1)
            mag = w_heads.abs().mean(dim=1).numpy()
            ax.bar(x + i * width, mag, width, label=proj_labels[proj],
                   color=colour, alpha=0.85, edgecolor="#000")

        ax.set_title(f"Block {blk} — Head Magnitude",
                     fontsize=11, color="white")
        ax.set_xlabel("Head index", color="#aaaaaa")
        ax.set_ylabel("|W| mean", color="#aaaaaa")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)],
                           fontsize=7, color="#cccccc")
        ax.tick_params(colors="#888888")
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
                  edgecolor="#444")
        ax.set_facecolor("#111111")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.grid(axis="y", color="#333333", linewidth=0.5)

    plt.suptitle("Per-Head Mean Weight Magnitude — LTP Bandwidth Proxy",
                 fontsize=13, color="white", y=1.02)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "2_engram_head_bandwidth.png")
    plt.savefig(out2, dpi=180, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    plt.close()
    print(f"   Saved → {out2}")


# ==============================================================================
# 3 ── HOOK INFRASTRUCTURE (reusable)
# ==============================================================================

class SpikeRecorder:
    """Attaches a forward hook and accumulates raw spike tensors."""

    def __init__(self):
        self.records: list[torch.Tensor] = []
        self._handle = None

    def attach(self, module: nn.Module):
        self._handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        self.records.append(output.detach().cpu().float())

    def detach(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.records.clear()

    @property
    def stacked(self) -> torch.Tensor | None:
        """Stack all recorded outputs along a new leading dim."""
        if not self.records:
            return None
        return torch.stack(self.records, dim=0)


def attach_ssa_hooks(model: nn.Module):
    """Register recorders on every SSA q_lif / k_lif / attn_lif."""
    recorders = {}
    for name, module in model.named_modules():
        # only SSA blocks (stage3 in max_former.py)
        if not ("stage3" in name and "attn" in name):
            continue
        for lif_name in ("q_lif", "k_lif", "attn_lif"):
            child = getattr(module, lif_name, None)
            if child is not None and f"{name}.{lif_name}" not in recorders:
                key = f"{name}.{lif_name}"
                recorders[key] = SpikeRecorder().attach(child)
    return recorders


# ==============================================================================
# 4 ── STDP PROFILING (ISI-proxy)
# ==============================================================================

def analyse_stdp(model: nn.Module, x_sample: torch.Tensor, device: str):
    """
    STDP profiling via ISI proxy.
    For each head, measure the first-spike time step of Q and K, then compute
    delta_t = |t_Q_first − t_K_first|.  Tighter delta_t → stronger correlation.
    """
    print("\n[3/5]  STDP Profiling  (ISI-proxy: Q vs K first-spike timing)")

    recorders = attach_ssa_hooks(model)
    with torch.no_grad():
        _ = model(x_sample.to(device))

    # group by block
    block_data: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, rec in recorders.items():
        if rec.stacked is None:
            rec.detach()
            continue
        # key: stage3.{b}.attn.{lif_name}
        parts = key.split(".")
        block_id  = parts[1]
        lif_label = parts[-1]
        # stacked shape: (1, T, B, C, N)  — one forward call
        spike = rec.stacked[0]             # (T, B, C, N)
        block_data[block_id][lif_label] = spike
        rec.detach()
        rec.clear()

    num_heads  = 8
    fig, axes  = plt.subplots(
        len(block_data), 2,
        figsize=(12, 4 * len(block_data)),
        facecolor="#0a0a0a", squeeze=False
    )
    fig.suptitle(
        "STDP Profiling — Q/K Inter-Spike Timing (ISI Proxy)",
        fontsize=13, color="white", y=1.01
    )

    for row, (blk, lifs) in enumerate(sorted(block_data.items())):
        q_spikes = lifs.get("q_lif")
        k_spikes = lifs.get("k_lif")

        if q_spikes is None or k_spikes is None:
            for c in range(2): axes[row][c].axis("off")
            continue

        T, B, C, N = q_spikes.shape
        H = NUM_HEADS
        D = C // H
        q_h = q_spikes.view(T, B, H, D, N)
        k_h = k_spikes.view(T, B, H, D, N)

        # first spike time per head (mean over batch, dims, tokens)
        def first_spike_time(spk_h):
            # spk_h: (T, B, H, D, N) binary
            # for each (B,H,D,N) position, find first T where spike==1
            mask = spk_h > 0.5          # bool
            has_spike = mask.any(dim=0) # (B, H, D, N)
            # argmax on T gives first True index; if no spike, fill T
            fst = torch.argmax(mask.float(), dim=0).float()
            fst[~has_spike] = float(T)  # no spike → T (late)
            return fst.mean(dim=[0, 2, 3]).numpy()   # (H,)

        fst_q = first_spike_time(q_h)
        fst_k = first_spike_time(k_h)
        delta  = np.abs(fst_q - fst_k)
        heads  = np.arange(H)   # H == NUM_HEADS

        # — left panel: first-spike times per head ────────────────────────────
        ax0 = axes[row][0]; ax0.set_facecolor("#111111")
        ax0.plot(heads, fst_q, "o-", color="#7209b7", lw=2,
                 label="Q first-spike")
        ax0.plot(heads, fst_k, "s--", color="#4361ee", lw=2,
                 label="K first-spike")
        ax0.axhline(1.0, color="#ffd60a", linewidth=0.8,
                    linestyle=":", label="T=1 (earliest)")
        ax0.set_title(f"Block {blk} — Q & K First-Spike Time",
                      color="white", fontsize=10)
        ax0.set_xlabel("Head index", color="#aaaaaa"); ax0.set_ylabel("Time step", color="#aaaaaa")
        ax0.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
                   edgecolor="#444")
        ax0.set_xticks(heads); ax0.set_xticklabels([f"H{h}" for h in heads], color="#ccc")
        ax0.tick_params(colors="#888"); ax0.grid(color="#333", linewidth=0.5)
        for sp in ax0.spines.values(): sp.set_edgecolor("#333")

        # — right panel: Δt (ISI proxy) per head ─────────────────────────────
        ax1 = axes[row][1]; ax1.set_facecolor("#111111")
        bar_colours = [
            "#00b4d8" if d < 0.5 else "#f77f00" if d < 1.5 else "#d62828"
            for d in delta
        ]
        ax1.bar(heads, delta, color=bar_colours, edgecolor="#000", alpha=0.9)
        ax1.set_title(f"Block {blk} — |Δt| Q–K ISI (lower = tighter STDP)",
                      color="white", fontsize=10)
        ax1.set_xlabel("Head index", color="#aaaaaa")
        ax1.set_ylabel("|Δt| (time steps)", color="#aaaaaa")
        ax1.set_xticks(heads); ax1.set_xticklabels([f"H{h}" for h in heads], color="#ccc")
        ax1.tick_params(colors="#888"); ax1.grid(axis="y", color="#333", linewidth=0.5)
        for sp in ax1.spines.values(): sp.set_edgecolor("#333")

        # annotation legend
        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#00b4d8", label="Tight  (< 0.5 ts)"),
            Patch(facecolor="#f77f00", label="Moderate (0.5–1.5 ts)"),
            Patch(facecolor="#d62828", label="Loose  (> 1.5 ts)"),
        ]
        ax1.legend(handles=legend_els, fontsize=8, facecolor="#1a1a1a",
                   labelcolor="white", edgecolor="#444")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "3_stdp_isi_profiling.png")
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    plt.close()
    print(f"   Saved → {out}")


# ==============================================================================
# 5 ── TEMPORAL SPARSITY (Predictive Coding / Pruning Proxy)
# ==============================================================================

def analyse_temporal_sparsity(model: nn.Module, x_sample: torch.Tensor,
                               device: str, n_passes: int = 8,
                               real_batches: list | None = None):
    """
    Run n_passes forward passes with independent random inputs in the same
    distribution as training data.  Track spike fraction of attn_lif per head.
    High sparsity = efficient feature detector.  Low sparsity = redundant path.
    """
    using_real = real_batches is not None and len(real_batches) > 0
    n_passes   = len(real_batches) if using_real else n_passes
    src        = "real ImageNet" if using_real else "random noise"
    print(f"\n[4/5]  Temporal Sparsity  ({n_passes} forward passes — {src})")

    num_heads = NUM_HEADS   # 6 for maxformer_10_384
    all_sparsity: dict[str, list[float]] = defaultdict(list)

    for i in range(n_passes):
        x_rand = real_batches[i].to(device) if using_real \
                 else torch.randn_like(x_sample).to(device)

        # fresh recorders each pass
        recs = {}
        for name, module in model.named_modules():
            if "stage3" in name and "attn" in name:
                child = getattr(module, "attn_lif", None)
                if child is not None:
                    key = name
                    if key not in recs:
                        recs[key] = SpikeRecorder().attach(child)

        with torch.no_grad():
            _ = model(x_rand)

        for key, rec in recs.items():
            if rec.stacked is None:
                rec.detach(); rec.clear(); continue
            spk = rec.stacked[0].float()       # (T, B, C, N)
            T, B, C, N = spk.shape
            H = num_heads; D = C // H
            spk_h = spk.view(T, B, H, D, N)
            # spike rate per head: fraction of (T*B*D*N) positions that fired
            sr = spk_h.mean(dim=[0, 1, 3, 4]).numpy()   # (H,)
            for h in range(H):
                all_sparsity[f"{key}_H{h}"].append(1.0 - float(sr[h]))

            rec.detach(); rec.clear()

        if (i + 1) % max(1, n_passes // 4) == 0:
            print(f"   Pass {i+1}/{n_passes} done")

    # ── aggregate ─────────────────────────────────────────────────────────────
    # group by block
    block_head_sparsity: dict[str, np.ndarray] = {}
    for key in list(all_sparsity.keys()):
        parts = key.split("_H")
        blk_key = parts[0].split(".")
        blk_id  = blk_key[1]
        h_idx   = int(parts[1])
        bk = f"Block {blk_id}"
        if bk not in block_head_sparsity:
            block_head_sparsity[bk] = np.zeros((num_heads, n_passes))
        idx = h_idx
        block_head_sparsity[bk][idx] = all_sparsity[key]

    n_blks = len(block_head_sparsity)
    if n_blks == 0:
        print("   [!] No sparsity data collected — skipping.")
        return

    fig = plt.figure(figsize=(16, 5 * n_blks), facecolor="#0a0a0a")
    fig.suptitle(
        "Temporal Sparsity per Attention Head — Predictive Coding Proxy\n"
        "(Higher = fewer spikes = more efficient feature detector)",
        fontsize=13, color="white", y=1.01
    )
    gs = gridspec.GridSpec(n_blks, 2, figure=fig, hspace=0.45, wspace=0.35)

    PRUNING_THRESHOLD = 0.90      # if mean sparsity > this → head is "prunable"

    for row, (blk, mat) in enumerate(sorted(block_head_sparsity.items())):
        # mat: (H, n_passes)
        mean_sp = mat.mean(axis=1)      # (H,)
        std_sp  = mat.std(axis=1)

        # — left: violin / box ─────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[row, 0]); ax0.set_facecolor("#111111")
        vp = ax0.violinplot(
            [mat[h] for h in range(num_heads)],
            positions=range(num_heads),
            showmeans=True, showextrema=True
        )
        for body in vp["bodies"]:
            body.set_facecolor("#7209b7"); body.set_alpha(0.7)
        vp["cmeans"].set_color("#ffd60a"); vp["cmaxes"].set_color("#f72585")
        vp["cmins"].set_color("#4361ee"); vp["cbars"].set_color("#888")

        ax0.axhline(PRUNING_THRESHOLD, color="#f72585", linewidth=1.2,
                    linestyle="--", label=f"Pruning threshold ({PRUNING_THRESHOLD:.0%})")
        ax0.set_title(f"{blk} — Sparsity Distribution",
                      color="white", fontsize=10)
        ax0.set_xlabel("Head index", color="#aaaaaa")
        ax0.set_ylabel("Sparsity (1 − spike_rate)", color="#aaaaaa")
        ax0.set_xticks(range(num_heads))
        ax0.set_xticklabels([f"H{h}" for h in range(num_heads)], color="#ccc", fontsize=7)
        ax0.set_ylim(-0.05, 1.05)
        ax0.tick_params(colors="#888"); ax0.grid(axis="y", color="#333", lw=0.5)
        ax0.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
        for sp in ax0.spines.values(): sp.set_edgecolor("#333")

        # — right: temporal heatmap (head × pass) ─────────────────────────
        ax1 = fig.add_subplot(gs[row, 1]); ax1.set_facecolor("#111111")
        im = ax1.imshow(mat, aspect="auto", cmap=NEURO_CMAP,
                        vmin=0, vmax=1, interpolation="nearest")
        plt.colorbar(im, ax=ax1, fraction=0.035, pad=0.04,
                     label="Sparsity").ax.yaxis.label.set_color("white")
        ax1.set_title(f"{blk} — Sparsity over Passes (heatmap)",
                      color="white", fontsize=10)
        ax1.set_xlabel("Forward pass (different random input)", color="#aaaaaa")
        ax1.set_ylabel("Head index", color="#aaaaaa")
        ax1.set_yticks(range(num_heads))
        ax1.set_yticklabels([f"H{h}" for h in range(num_heads)], color="#ccc", fontsize=7)
        ax1.tick_params(colors="#888")
        # annotate prunable heads
        for h in range(num_heads):
            if mean_sp[h] > PRUNING_THRESHOLD:
                ax1.annotate("✂", xy=(n_passes - 0.5, h),
                             va="center", ha="center",
                             fontsize=12, color="#f72585")
        for sp in ax1.spines.values(): sp.set_edgecolor("#333")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "4_temporal_sparsity.png")
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    plt.close()
    print(f"   Saved → {out}")

    # ── print summary table ───────────────────────────────────────────────────
    print("\n   ── Sparsity Summary ──────────────────────────────────────────")
    print(f"   {'Block':<18} {'Head':<6} {'Mean Sp':>9} {'Std':>7} {'Status':>12}")
    print("   " + "─" * 56)
    for blk, mat in sorted(block_head_sparsity.items()):
        for h in range(num_heads):
            ms = mat[h].mean();  sd = mat[h].std()
            status = "PRUNABLE ✂" if ms > PRUNING_THRESHOLD else "ACTIVE  ●"
            print(f"   {blk:<18} H{h:<5} {ms:>9.4f} {sd:>7.4f} {status:>12}")
    print()


# ==============================================================================
# 6 ── ATTENTIONAL SYNCHRONY
# ==============================================================================

def analyse_attention_synchrony(model: nn.Module, x_sample: torch.Tensor,
                                 device: str):
    """
    Measure head-pairwise Pearson correlation of attn_lif outputs across
    spatial tokens and time steps.  High off-diagonal correlation = heads that
    fire synchronously → they attend to the same features (neural synchrony).
    """
    print("\n[5/5]  Attentional Synchrony  (head pairwise correlation)")

    num_heads = NUM_HEADS   # 6 for maxformer_10_384
    block_corr: dict[str, np.ndarray] = {}

    recs = {}
    for name, module in model.named_modules():
        if "stage3" in name and "attn" in name:
            child = getattr(module, "attn_lif", None)
            if child is not None and name not in recs:
                recs[name] = SpikeRecorder().attach(child)

    with torch.no_grad():
        _ = model(x_sample.to(device))

    for key, rec in recs.items():
        if rec.stacked is None:
            rec.detach(); rec.clear(); continue
        spk = rec.stacked[0].float()                   # (T, B, C, N)
        T, B, C, N = spk.shape
        H = num_heads; D = C // H
        spk_h = spk.view(T, B, H, D, N)
        # collapse to per-head signal: (H, T*D*N*B)
        sig = spk_h.permute(2, 0, 3, 4, 1).reshape(H, -1).numpy()
        # Pearson correlation matrix
        mu  = sig.mean(axis=1, keepdims=True)
        sig_c = sig - mu
        norms = np.linalg.norm(sig_c, axis=1, keepdims=True) + 1e-8
        sig_n = sig_c / norms
        corr = sig_n @ sig_n.T          # (H, H)
        blk_id = key.split(".")[1]
        block_corr[f"Block {blk_id}"] = corr
        rec.detach(); rec.clear()

    n_blks = len(block_corr)
    if n_blks == 0:
        print("   [!] No synchrony data collected — skipping.")
        return

    fig, axes = plt.subplots(
        1, n_blks,
        figsize=(7 * n_blks, 6),
        squeeze=False,
        facecolor="#0a0a0a"
    )
    fig.suptitle(
        "Attentional Synchrony — Head-Pair Correlation (Neural Gamma-Band Proxy)\n"
        "Off-diagonal = shared feature coding between heads",
        fontsize=13, color="white", y=1.03
    )

    for col, (blk, corr) in enumerate(sorted(block_corr.items())):
        ax = axes[0][col]; ax.set_facecolor("#111111")
        im = ax.imshow(corr, cmap=SYNC_CMAP, vmin=-1, vmax=1,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Pearson r").ax.yaxis.label.set_color("white")
        # annotate correlation values
        for i in range(num_heads):
            for j in range(num_heads):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(corr[i,j]) < 0.6 else "#0a0a0a")
        ax.set_title(f"{blk} — Head Synchrony", color="white", fontsize=11)
        ax.set_xlabel("Head index", color="#aaaaaa")
        ax.set_ylabel("Head index", color="#aaaaaa")
        ax.set_xticks(range(num_heads)); ax.set_yticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)], color="#ccc", fontsize=7)
        ax.set_yticklabels([f"H{h}" for h in range(num_heads)], color="#ccc", fontsize=7)
        ax.tick_params(colors="#888")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "5_attention_synchrony.png")
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    plt.close()
    print(f"   Saved → {out}")


# ==============================================================================
# 7 ── SUMMARY DASHBOARD
# ==============================================================================

def generate_summary_dashboard():
    """Assemble a single overview figure from saved panels."""
    print("\n[★]   Generating summary dashboard …")
    panels = [
        ("1 — Engram Heatmaps",      "1_engram_weight_heatmaps.png"),
        ("2 — Head Bandwidth",       "2_engram_head_bandwidth.png"),
        ("3 — STDP ISI Profiling",   "3_stdp_isi_profiling.png"),
        ("4 — Temporal Sparsity",    "4_temporal_sparsity.png"),
        ("5 — Attention Synchrony",  "5_attention_synchrony.png"),
    ]
    imgs = []
    for label, fname in panels:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            import matplotlib.image as mpimg
            imgs.append((label, mpimg.imread(path)))

    if not imgs:
        print("   [!] No panels found for dashboard.")
        return

    fig, axes = plt.subplots(
        len(imgs), 1,
        figsize=(18, 6 * len(imgs)),
        facecolor="#060610"
    )
    if len(imgs) == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_title(label, fontsize=11, color="#ffd60a",
                     loc="left", pad=6, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "MaxFormer — Biological Learning Analysis Dashboard",
        fontsize=16, color="white", fontweight="bold", y=1.005
    )
    plt.tight_layout(pad=1.5)
    out = os.path.join(OUT_DIR, "0_DASHBOARD.png")
    plt.savefig(out, dpi=120, bbox_inches="tight",
                facecolor="#060610", edgecolor="none")
    plt.close()
    print(f"   Saved → {out}")


# ==============================================================================
# MAIN
# ==============================================================================

# ==============================================================================
# 8 ── REAL IMAGENET-1K DATA LOADER (200 images)
# ==============================================================================

N_IMAGENET_SAMPLES = 200
IMAGENET_BATCH_SIZE = 8

def load_imagenet_batches(n_samples: int = N_IMAGENET_SAMPLES,
                          batch_size: int = IMAGENET_BATCH_SIZE):
    """
    Stream N_IMAGENET_SAMPLES real images from the ImageNet-1K validation split
    via HuggingFace datasets.  Returns a list of (B, 3, 224, 224) tensors.
    """
    print(f"\n[DATA]  Streaming {n_samples} real ImageNet-1K images "
          f"(batch_size={batch_size}) …")

    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = load_dataset(
        'mrm8488/ImageNet1K-val',
        split='train',
        streaming=True,
        trust_remote_code=True
    )

    batches  = []
    buf      = []
    collected = 0

    for item in dataset:
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf.append(tfs(img).unsqueeze(0))
        collected += 1
        if len(buf) == batch_size:
            batches.append(torch.cat(buf, dim=0))
            buf = []
        if collected >= n_samples:
            break

    if buf:                              # leftover partial batch
        batches.append(torch.cat(buf, dim=0))

    total = sum(b.shape[0] for b in batches)
    print(f"   ✓  Loaded {total} images in {len(batches)} batches")
    return batches


def main():
    model, state_dict, device = load_model_and_weights(CHECKPOINT)

    # ── Load 200 real ImageNet-1K images ──────────────────────────────────────
    imagenet_batches = load_imagenet_batches()

    # Use first batch as the representative single-image sample for
    # analyses that still need one forward pass (STDP, Synchrony)
    x_sample = imagenet_batches[0][:1]   # shape: (1, 3, 224, 224)

    # ── Analysis modules ───────────────────────────────────────────────────────
    analyse_engrams(state_dict)
    analyse_stdp(model, x_sample, device)
    # Temporal sparsity: pass all 200 images as individual passes
    analyse_temporal_sparsity(
        model, x_sample, device,
        n_passes=len(imagenet_batches),
        real_batches=imagenet_batches
    )
    analyse_attention_synchrony(model, x_sample, device)
    generate_summary_dashboard()

    print("\n" + "="*70)
    print(f"  ✓  All outputs saved to: {os.path.abspath(OUT_DIR)}")
    print("="*70)


if __name__ == "__main__":
    main()
