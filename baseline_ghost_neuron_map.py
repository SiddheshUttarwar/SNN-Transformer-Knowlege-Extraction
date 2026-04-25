"""
baseline_ghost_neuron_map.py — Baseline-Anchored Ghost Neuron Analysis
==============================================================
A neuron is declared a "BASELINE-ANCHORED GHOST" if it rigidly maintains its
zeros-input specific firing rate identically across all real-world data samples,
meaning its response is perfectly agnostic to real stimulus (mutual information = 0).

Outputs
-------
analysis_outputs/baseline_ghost_neuron_maps.png   — grid plot
analysis_outputs/baseline_ghost_neurons.json     — quantitative JSON
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
EMBED_DIMS  = 384
T_STEPS     = 4
IMG_SIZE    = 224
N_SAMPLES   = 100
BATCH_SIZE  = 10

from visualize_model import load_model_and_weights, SpikeRecorder

def main():
    model, state_dict, device = load_model_and_weights(CHECKPOINT)

    recs: dict[str, SpikeRecorder] = {}
    for name, module in model.named_modules():
        if "stage3" in name and "attn" in name:
            child = getattr(module, "attn_lif", None)
            if child is not None and name not in recs:
                recs[name] = SpikeRecorder().attach(child)

    print(f"Hooks registered on {len(recs)} attn_lif modules.")

    # 1. Establish the Reference Baseline Map (Blank Image)
    print("\n[Phase 1] Establishing Reference Zero-Input Firing Map...")
    for rec in recs.values(): rec.clear()
    
    x_blank = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    with torch.no_grad():
        _ = model(x_blank)
        
    ref_maps: dict[str, torch.Tensor] = {}
    locked_masks: dict[str, torch.Tensor] = {}

    for name, rec in recs.items():
        if not rec.records: continue
        # sum spikes out of T for the single batch
        spk = rec.records[0].float()               # (T, 1, C, N)
        ref_spikes = spk.sum(dim=0).squeeze(0)     # (C, N)
        
        ref_maps[name] = ref_spikes.clone()
        # Initialize locked mask: Must fire at baseline to even be considered a ghost!
        locked_masks[name] = (ref_spikes > 0)      # (C, N) bool
        
    # 2. Evaluation Streaming
    print(f"\n[Phase 2] Loading Tiny-ImageNet val ({N_SAMPLES} images, batch={BATCH_SIZE})...")
    dataset = load_dataset('zh-plus/tiny-imagenet', split='valid')

    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n_batches = N_SAMPLES // BATCH_SIZE
    dataset_iter = iter(dataset)

    print(f"Running rigid anchor comparison over {n_batches} batches ...")
    for _ in tqdm(range(n_batches), desc="ImageNet batches"):
        imgs = []
        for _ in range(BATCH_SIZE):
            item = next(dataset_iter)
            img  = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(tfs(img).unsqueeze(0))
        x_batch = torch.cat(imgs, dim=0).to(device)

        for rec in recs.values(): rec.clear()

        with torch.no_grad():
            _ = model(x_batch)

        for name, rec in recs.items():
            if not rec.records: continue
            
            # (T, B, C, N) -> sum over T -> (B, C, N)
            spk = rec.records[0].float()      
            batch_spikes = spk.sum(dim=0)          
            
            # Compare every image in the batch against the ref map
            # ref_maps[name] is (C, N), unsqueeze to broadcast across batch dim (1, C, N)
            is_match = (batch_spikes == ref_maps[name].unsqueeze(0)) # (B, C, N) bool
            
            # For a neuron to stay valid, it must match the ref in ALL images of this batch
            batch_locked = is_match.all(dim=0) # (C, N) bool
            
            # Update running mask
            locked_masks[name] &= batch_locked

    for rec in recs.values():
        rec.detach()

    print("\n[Phase 3] Building globally anchored ghost neuron maps ...")
    output_data = {}
    blocks = sorted(ref_maps.keys())
    num_blocks = len(blocks)

    fig, axes = plt.subplots(
        num_blocks, NUM_HEADS,
        figsize=(NUM_HEADS * 3 + 2, num_blocks * 2 + 1),
        facecolor="#0a0a0a"
    )
    if num_blocks == 1: axes = [axes]

    im = None
    for row, name in enumerate(blocks):
        mask      = locked_masks[name]   # (C, N) bool
        ref_rates = ref_maps[name]       # (C, N) float

        C, N = mask.shape
        H = NUM_HEADS
        D = C // H
        
        # (H, D, N)
        ghost_map = mask.view(H, D, N).float() 
        rates_map = ref_rates.view(H, D, N).float()
        
        block_id = name.split(".")[1]
        output_data[f"Block_{block_id}"] = {}

        for h in range(H):
            m_np = ghost_map[h].cpu().numpy()         # (D, N) bool map
            r_np = rates_map[h].cpu().numpy()         # (D, N) float rates

            # Extract specific rates for the truly anchored ghosts
            anchored_rates = {}
            for d in range(D):
                for n in range(N):
                    if m_np[d, n] == 1.0: # Is actively anchored
                        channel_rate = int(r_np[d, n])
                        anchored_rates[f"D{d}_N{n}"] = channel_rate

            ghost_tokens_per_d = m_np.sum(axis=1)     # (D,)
            completely_ghost_channels = [int(d) for d in range(D) if ghost_tokens_per_d[d] == N]
            
            output_data[f"Block_{block_id}"][f"Head_{h}"] = {
                "n_samples_evaluated":                 N_SAMPLES,
                "completely_baseline_ghost_channels":   completely_ghost_channels,
                "baseline_ghost_fraction_overall":      round(float(m_np.mean()), 4),
                "anchored_neurons_count":               len(anchored_rates),
                "anchored_neurons_firing_rates":        anchored_rates
            }

            ax = axes[row][h]
            ax.set_facecolor("#111111")
            im = ax.imshow(m_np, aspect="auto", cmap="plasma", vmin=0, vmax=1)

            if row == 0:
                ax.set_title(f"Head {h}", color="white", fontsize=10)
            if h == 0:
                ax.set_ylabel(f"Block {block_id}\nChannels (D)", color="#aaaaaa", fontsize=8)
            else:
                ax.set_yticks([])

            ax.set_xlabel("Spatial Tokens (N)", color="#aaaaaa", fontsize=7)
            ax.set_xticks([])
            ax.tick_params(colors="#888")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("0 = Functional, 1 = Baseline-Anchored (Zero Mutual Info)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("#333")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    plt.suptitle(
        f"Baseline-Anchored Ghost Neuron Maps  |  T=1..4  |  {N_SAMPLES} ImageNet images\n"
        "Yellow = PERFECTLY ANCHORED (Spike rate perfectly equals zero-input baseline on all images)",
        color="white", fontsize=14
    )

    png_out  = os.path.join(OUT_DIR, "baseline_ghost_neuron_maps.png")
    json_out = os.path.join(OUT_DIR, "baseline_ghost_neurons.json")

    plt.savefig(png_out, facecolor="#0a0a0a", edgecolor="none", bbox_inches="tight", dpi=150)
    plt.close()
    
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"  - Graph -> {png_out}")
    print(f"  - JSON  -> {json_out}")

if __name__ == "__main__":
    main()
