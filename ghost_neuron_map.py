"""
ghost_neuron_map.py — Ghost Neuron Analysis for Spiking MaxFormer
================================================================
A neuron (channel-slot within a head) is declared a "GHOST" if it fires
at least once despite the model receiving completely zeroed input images.

Outputs
-------
analysis_outputs/ghost_neuron_maps.png   — grid plot
analysis_outputs/ghost_neurons.json     — quantitative JSON
"""

import sys, os, warnings
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

OUT_DIR    = os.path.join(os.path.dirname(__file__), "analysis_outputs")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "10-384-T4.pth.tar")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_HEADS   = 6
NUM_CLASSES = 1000
EMBED_DIMS  = 384
T_STEPS     = 4
IMG_SIZE    = 224
N_SAMPLES   = 1    # 1 zero image is enough to check deterministic ghost behavior
BATCH_SIZE  = 1    # forward-pass batch size

from visualize_model import load_model_and_weights, SpikeRecorder

def main():
    # ── 1. Load model ──────────────────────────────────────────────────────────
    model, state_dict, device = load_model_and_weights(CHECKPOINT)

    print(f"\nAnalyzing ghost neurons with zero-image input (N={N_SAMPLES})...")

    # ── 3. Attach hooks on stage3 attn_lif (once) ─────────────────────────────
    recs: dict[str, SpikeRecorder] = {}
    for name, module in model.named_modules():
        if "stage3" in name and "attn" in name:
            child = getattr(module, "attn_lif", None)
            if child is not None and name not in recs:
                recs[name] = SpikeRecorder().attach(child)

    print(f"Hooks registered on {len(recs)} attn_lif modules.")

    # ── 4. Accumulate fired_ever mask over N_SAMPLES images ───────────────────
    fired_ever: dict[str, torch.Tensor] = {}

    print(f"Running inference ...")
    x_batch = torch.zeros((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), device=device)

    # Clear stale records
    for rec in recs.values():
        rec.clear()

    with torch.no_grad():
        _ = model(x_batch)

    # OR-accumulate fired mask
    for name, rec in recs.items():
        if not rec.records:
            continue
        spk = rec.records[0].float()       # (T, B, C, N)
        T, B, C, N = spk.shape
        H = NUM_HEADS
        D = C // H
        spk_h = spk.view(T, B, H, D, N)
        # Any spike over T and B → this neuron fired for this batch
        fired_batch = (spk_h.sum(dim=(0, 1)) > 0)   # (H, D, N) bool
        if name not in fired_ever:
            fired_ever[name] = fired_batch.clone()
        else:
            fired_ever[name] = fired_ever[name] | fired_batch

    # Remove hooks
    for rec in recs.values():
        rec.detach()

    # ── 5. Build plot + JSON ───────────────────────────────────────────────────
    print("\nBuilding ghost neuron maps ...")
    ghost_neuron_data = {}
    blocks    = sorted(fired_ever.keys())
    num_blocks = len(blocks)

    fig, axes = plt.subplots(
        num_blocks, NUM_HEADS,
        figsize=(NUM_HEADS * 3 + 2, num_blocks * 2 + 1),
        facecolor="#0a0a0a"
    )
    if num_blocks == 1:
        axes = [axes]

    im = None
    for row, name in enumerate(blocks):
        ghost_map = fired_ever[name].float()   # (H, D, N) - 1=ghost(fired), 0=silent
        H, D, N_tokens = ghost_map.shape
        block_id  = name.split(".")[1]
        ghost_neuron_data[f"Block_{block_id}"] = {}

        for h in range(H):
            m_np = ghost_map[h].cpu().numpy()   # (D, N)

            # JSON stats
            ghost_tokens_per_d = m_np.sum(axis=1)       # (D,)
            completely_ghost_channels = [int(d) for d in range(D)
                                         if ghost_tokens_per_d[d] == N_tokens]
            ghost_neuron_data[f"Block_{block_id}"][f"Head_{h}"] = {
                "n_samples_evaluated":               N_SAMPLES,
                "completely_ghost_channels_list":    completely_ghost_channels,
                "completely_ghost_channels_count":   len(completely_ghost_channels),
                "total_channels_D":                  int(D),
                "total_spatial_tokens_N":            int(N_tokens),
                "ghost_fraction_overall":            round(float(m_np.mean()), 4),
            }

            # Plot
            ax = axes[row][h]
            ax.set_facecolor("#111111")
            im = ax.imshow(m_np, aspect="auto", cmap="plasma", vmin=0, vmax=1)

            if row == 0:
                ax.set_title(f"Head {h}", color="white", fontsize=10)
            if h == 0:
                ax.set_ylabel(f"Block {block_id}\nChannels (D)",
                              color="#aaaaaa", fontsize=8)
            else:
                ax.set_yticks([])

            ax.set_xlabel("Spatial Tokens (N)", color="#aaaaaa", fontsize=7)
            ax.set_xticks([])
            ax.tick_params(colors="#888")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")

    # Global colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("0 = Silent (no spikes), 1 = Ghost (fired despite zero input)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("#333")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    plt.suptitle(
        f"Ghost Neuron Maps per Attention Head  |  T=1..4  |  Zeroed Image Input\n"
        "Yellow = GHOST (fired)  -  Dark = SILENT",
        color="white", fontsize=14
    )

    png_out  = os.path.join(OUT_DIR, "ghost_neuron_maps.png")
    json_out = os.path.join(OUT_DIR, "ghost_neurons.json")

    plt.savefig(png_out, facecolor="#0a0a0a", edgecolor="none",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  - Graph -> {png_out}")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(ghost_neuron_data, f, indent=4)
    print(f"  - JSON  -> {json_out}")

if __name__ == "__main__":
    main()
