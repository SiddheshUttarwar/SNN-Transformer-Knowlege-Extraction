"""
absolutely_ghost_neuron_map.py — Absolute Ghost Neuron Analysis
==============================================================
A neuron (channel-slot within a head) is declared an "ABSOLUTELY GHOST NEURON" if
for EVERY given image (out of 100 benchmark images), it fires exactly the SAME
number of spikes across the T timesteps, and that number of spikes is > 0.
This renders it completely devoid of mutual information regarding the image.

Outputs
-------
analysis_outputs/absolutely_ghost_neuron_maps.png   — grid plot
analysis_outputs/absolutely_ghost_neurons.json     — quantitative JSON
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
    model, state_dict, device = load_model_and_weights(CHECKPOINT)

    print(f"\nLoading Tiny-ImageNet val ({N_SAMPLES} images, batch={BATCH_SIZE})...")
    dataset = load_dataset('zh-plus/tiny-imagenet', split='valid')

    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    recs: dict[str, SpikeRecorder] = {}
    for name, module in model.named_modules():
        if "stage3" in name and "attn" in name:
            child = getattr(module, "attn_lif", None)
            if child is not None and name not in recs:
                recs[name] = SpikeRecorder().attach(child)

    print(f"Hooks registered on {len(recs)} attn_lif modules.")

    # Tracking tensors per module:
    min_spikes_ever: dict[str, torch.Tensor] = {}
    max_spikes_ever: dict[str, torch.Tensor] = {}

    n_batches   = N_SAMPLES // BATCH_SIZE
    dataset_iter = iter(dataset)

    print(f"Running inference over {n_batches} batches x {BATCH_SIZE} images ...")
    for _ in tqdm(range(n_batches), desc="ImageNet batches"):
        imgs = []
        for _ in range(BATCH_SIZE):
            item = next(dataset_iter)
            img  = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(tfs(img).unsqueeze(0))
        x_batch = torch.cat(imgs, dim=0).to(device)

        for rec in recs.values():
            rec.clear()

        with torch.no_grad():
            _ = model(x_batch)

        for name, rec in recs.items():
            if not rec.records:
                continue
            
            # (T, B, C, N) -> sum over T -> total spikes per image evaluated
            spk = rec.records[0].float()      
            batch_spikes = spk.sum(dim=0)          # (B, C, N)
            
            # Min and max across this batch
            batch_min = batch_spikes.min(dim=0).values # (C, N)
            batch_max = batch_spikes.max(dim=0).values # (C, N)
            
            if name not in min_spikes_ever:
                min_spikes_ever[name] = batch_min.clone()
                max_spikes_ever[name] = batch_max.clone()
            else:
                min_spikes_ever[name] = torch.min(min_spikes_ever[name], batch_min)
                max_spikes_ever[name] = torch.max(max_spikes_ever[name], batch_max)

    for rec in recs.values():
        rec.detach()

    print("\nBuilding absolutely ghost neuron maps ...")
    abs_ghost_neuron_data = {}
    blocks    = sorted(min_spikes_ever.keys())
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
        _min_t = min_spikes_ever[name] # (C, N)
        _max_t = max_spikes_ever[name] # (C, N)
        
        # Absolute ghost condition: min == max AND max > 0
        cond = (_min_t == _max_t) & (_max_t > 0) # (C, N) bool
        
        C, N = cond.shape
        H = NUM_HEADS
        D = C // H
        
        ghost_map = cond.view(H, D, N).float() # (H, D, N)
        
        block_id  = name.split(".")[1]
        abs_ghost_neuron_data[f"Block_{block_id}"] = {}

        for h in range(H):
            m_np = ghost_map[h].cpu().numpy()   # (D, N)

            ghost_tokens_per_d = m_np.sum(axis=1)          # (D,)
            completely_ghost_channels = [int(d) for d in range(D)
                                        if ghost_tokens_per_d[d] == N]
            abs_ghost_neuron_data[f"Block_{block_id}"][f"Head_{h}"] = {
                "n_samples_evaluated":                 N_SAMPLES,
                "completely_abs_ghost_channels_list":   completely_ghost_channels,
                "completely_abs_ghost_channels_count":  len(completely_ghost_channels),
                "total_channels_D":                   int(D),
                "total_spatial_tokens_N":             int(N),
                "abs_ghost_fraction_overall":         round(float(m_np.mean()), 4),
            }

            ax = axes[row][h]
            ax.set_facecolor("#111111")
            # Yellow/Purple heatmap "plasma"
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

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("0 = Normal (variance > 0), 1 = Absolutely Ghost (variance == 0)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("#333")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    plt.suptitle(
        f"Absolutely Ghost Neuron Maps per Attention Head  |  T=1..4  |  {N_SAMPLES} ImageNet images\n"
        "Yellow = ABSOLUTELY GHOST (steady invariant firing on all images)  -  Dark = FUNCTIONAL",
        color="white", fontsize=14
    )

    png_out  = os.path.join(OUT_DIR, "absolutely_ghost_neuron_maps.png")
    json_out = os.path.join(OUT_DIR, "absolutely_ghost_neurons.json")

    plt.savefig(png_out, facecolor="#0a0a0a", edgecolor="none",
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  - Graph -> {png_out}")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(abs_ghost_neuron_data, f, indent=4)
    print(f"  - JSON  -> {json_out}")


if __name__ == "__main__":
    main()
