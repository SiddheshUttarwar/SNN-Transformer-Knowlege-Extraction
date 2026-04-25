"""
dead_neuron_map.py — Dead Neuron Analysis for Spiking MaxFormer
================================================================
A neuron (channel-slot within a head) is declared "DEAD" if it fires
ZERO spikes across ALL T={1..4} timesteps, for EVERY image in the
evaluated ImageNet sample.

Accumulates a fired_ever boolean mask over N_SAMPLES real Tiny-ImageNet
validation images, then plots one D×N heatmap per (block, head).

Outputs
-------
analysis_outputs/dead_neuron_maps.png   — grid plot
analysis_outputs/dead_neurons.json     — quantitative JSON
"""

import sys, os, warnings
import json
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
N_SAMPLES   = 100   # total ImageNet images to evaluate
BATCH_SIZE  = 10    # forward-pass batch size

from visualize_model import load_model_and_weights, SpikeRecorder


def main():
    # ── 1. Load model ──────────────────────────────────────────────────────────
    model, state_dict, device = load_model_and_weights(CHECKPOINT)

    # ── 2. Load Tiny-ImageNet validation split (real ImageNet images) ──────────
    print(f"\nLoading Tiny-ImageNet val ({N_SAMPLES} images, batch={BATCH_SIZE})...")
    dataset = load_dataset('zh-plus/tiny-imagenet', split='valid')

    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── 3. Attach hooks on stage3 attn_lif (once) ─────────────────────────────
    recs: dict[str, SpikeRecorder] = {}
    for name, module in model.named_modules():
        if "stage3" in name and "attn" in name:
            child = getattr(module, "attn_lif", None)
            if child is not None and name not in recs:
                recs[name] = SpikeRecorder().attach(child)

    print(f"Hooks registered on {len(recs)} attn_lif modules.")

    # ── 4. Accumulate fired_ever mask over N_SAMPLES images ───────────────────
    # fired_ever[name] : Tensor(H, D, N) bool — True if the neuron fired at
    #   least once in at least one image across the entire evaluated corpus.
    fired_ever: dict[str, torch.Tensor] = {}

    n_batches   = N_SAMPLES // BATCH_SIZE
    dataset_iter = iter(dataset)

    print(f"Running inference over {n_batches} batches × {BATCH_SIZE} images …")
    for _ in tqdm(range(n_batches), desc="ImageNet batches"):
        # Build batch
        imgs = []
        for _ in range(BATCH_SIZE):
            item = next(dataset_iter)
            img  = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(tfs(img).unsqueeze(0))
        x_batch = torch.cat(imgs, dim=0).to(device)   # (B, 3, 224, 224)

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
            # Any spike over T and B → this neuron is NOT dead for this batch
            fired_batch = (spk_h.sum(dim=(0, 1)) > 0)   # (H, D, N) bool
            if name not in fired_ever:
                fired_ever[name] = fired_batch.clone()
            else:
                fired_ever[name] = fired_ever[name] | fired_batch

    # Remove hooks
    for rec in recs.values():
        rec.detach()

    # ── 5. Build plot + JSON ───────────────────────────────────────────────────
    print("\nBuilding dead neuron maps …")
    dead_neuron_data = {}
    blocks    = sorted(fired_ever.keys())
    num_blocks = len(blocks)

    fig, axes = plt.subplots(
        num_blocks, NUM_HEADS,
        figsize=(NUM_HEADS * 3 + 2, num_blocks * 2 + 1),
        facecolor="#0a0a0a"
    )
    if num_blocks == 1:
        axes = [axes]

    im = None   # keep last imshow for colorbar
    for row, name in enumerate(blocks):
        fired_map = fired_ever[name].float()   # (H, D, N)
        dead_map  = 1.0 - fired_map            # 1=dead, 0=active
        H, D, N   = dead_map.shape
        block_id  = name.split(".")[1]
        dead_neuron_data[f"Block_{block_id}"] = {}

        for h in range(H):
            m_np = dead_map[h].cpu().numpy()   # (D, N)

            # JSON stats
            dead_tokens_per_d       = m_np.sum(axis=1)          # (D,)
            completely_dead_channels = [int(d) for d in range(D)
                                        if dead_tokens_per_d[d] == N]
            dead_neuron_data[f"Block_{block_id}"][f"Head_{h}"] = {
                "n_samples_evaluated":              N_SAMPLES,
                "completely_dead_channels_list":    completely_dead_channels,
                "completely_dead_channels_count":   len(completely_dead_channels),
                "total_channels_D":                 D,
                "total_spatial_tokens_N":           int(N),
                "dead_fraction_overall":            round(float(m_np.mean()), 4),
            }

            # Plot
            ax = axes[row][h]
            ax.set_facecolor("#111111")
            im = ax.imshow(m_np, aspect="auto", cmap="magma", vmin=0, vmax=1)

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
    cbar.set_label("0 = Active (fired), 1 = Dead (never fired)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("#333")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    plt.suptitle(
        f"Dead Neuron Maps per Attention Head  |  T=1..4  |  {N_SAMPLES} real ImageNet images\n"
        "Yellow = DEAD (never fired across all samples)  ·  Dark = ACTIVE",
        color="white", fontsize=14
    )

    png_out  = os.path.join(OUT_DIR, "dead_neuron_maps.png")
    json_out = os.path.join(OUT_DIR, "dead_neurons.json")

    plt.savefig(png_out, facecolor="#0a0a0a", edgecolor="none",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ Graph → {png_out}")

    with open(json_out, "w") as f:
        json.dump(dead_neuron_data, f, indent=4)
    print(f"  ✓ JSON  → {json_out}")


if __name__ == "__main__":
    main()
