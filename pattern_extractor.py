"""
pattern_extractor.py - Automated Neuromorphic Hardware Profiling
Runs 500 real ImageNet subset samples (Imagenette) through Spiking MaxFormer
and outputs a deterministic Hardware Gating Specification json.
"""

import sys
import os
import json
import warnings
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Architecture parameters
NUM_HEADS   = 6
NUM_CLASSES = 1000
EMBED_DIMS  = 384
T_STEPS     = 4
IMG_SIZE    = 224
CHECKPOINT  = "./checkpoints/10-384-T4.pth.tar"
N_SAMPLES   = 500
BATCH_SIZE  = 10

def load_maxformer():
    print(f"[1/4] Loading MaxFormer CPKT: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    import spikingjelly.clock_driven.neuron as _sj_neuron
    _orig_init = _sj_neuron.MultiStepLIFNode.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["backend"] = "torch"
        _orig_init(self, *args, **kwargs)
    _sj_neuron.MultiStepLIFNode.__init__ = _patched_init

    from max_former import Max_Former
    model = Max_Former(
        in_channels=3, num_classes=NUM_CLASSES,
        embed_dims=EMBED_DIMS, mlp_ratios=4,
        depths=10, T=T_STEPS
    )

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"      Model loaded on: {device.upper()}")
    return model, state_dict, device

def first_spike_time(spike_tensor):
    # spike_tensor: [T, B, H, D, N]
    # returns mean first spike time per head (shape [H])
    T_sz = spike_tensor.shape[0]
    indices = torch.arange(1, T_sz + 1, device=spike_tensor.device, dtype=torch.float32)
    shape_ones = [1]*len(spike_tensor.shape)
    shape_ones[0] = T_sz
    mask = indices.view(*shape_ones) * spike_tensor
    # Replace 0 with infinity for taking the min
    mask[mask == 0] = 999
    fst_times = mask.min(dim=0)[0] # [B, H, D, N]
    fst_times[fst_times == 999] = T_sz # if never spikes, treat as firing at end T

    # Average over Batch, feature-dim D, and sequence N
    return fst_times.mean(dim=(0, 2, 3)).cpu().numpy()

def main():
    model, state_dict, device = load_maxformer()

    # 1. Setup Data Loader
    print("[2/4] Initializing ImageNet-Subset streaming...")
    dataset = load_dataset('mrm8488/ImageNet1K-val', split='train', streaming=True, trust_remote_code=True)
    
    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def data_generator():
        b = []
        for item in dataset:
            img = item['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            b.append(tfs(img).unsqueeze(0))
            if len(b) == BATCH_SIZE:
                yield torch.cat(b, dim=0)
                b = []

    # 2. Setup Hooks
    extracted_spikes = defaultdict(list)
    def _make_hook(block_idx: int, l_type: str):
        def hook_fn(module, inputs, output):
            # output is single tensor if return sequence, just detach to cpu
            extracted_spikes[f"b{block_idx}_{l_type}"].append(output.detach())
        return hook_fn

    # Register hooks on Stage 3 SSA blocks
    blocks = model.stage3
    handles = []
    num_blocks = len(blocks)
    for i, blk in enumerate(blocks):
        handles.append(blk.attn.q_lif.register_forward_hook(_make_hook(i, 'q')))
        handles.append(blk.attn.k_lif.register_forward_hook(_make_hook(i, 'k')))
        # handles.append(blk.attn.v_lif.register_forward_hook(_make_hook(i, 'v')))
        handles.append(blk.attn.attn_lif.register_forward_hook(_make_hook(i, 'attn')))

    # 3. Accumulate Statistics Variables
    # shape per head: block_id -> [H] array for running means
    acc_stdp_delta = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    acc_q_fst      = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    acc_sparsity   = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    
    # Store aggregated activity across batches to compute correlation later
    head_activities = {i: [] for i in range(num_blocks)}

    # 4. Inference Loop
    print(f"[3/4] Running Inference over {N_SAMPLES} ImageNet samples...")
    gen = data_generator()
    batches = N_SAMPLES // BATCH_SIZE
    
    with torch.no_grad():
        for b_idx in tqdm(range(batches), total=batches):
            x = next(gen).to(device)
            # MaxFormer requires [T, B, C, H, W] if len < 5 in forward, wait checking maxformer:
            # model handles unsqueeze(0).repeat(...) internally.
            model(x)

            # Process hooks immediately to save memory
            for i in range(num_blocks):
                q = extracted_spikes[f"b{i}_q"][-1]
                k = extracted_spikes[f"b{i}_k"][-1]
                attn = extracted_spikes[f"b{i}_attn"][-1]

                T, B, C, N = q.shape
                H = NUM_HEADS
                D = C // H

                q_h = q.view(T, B, H, D, N)
                k_h = k.view(T, B, H, D, N)
                attn_h = attn.view(T, B, H, D, N)

                # STDP Profiling
                q_f = first_spike_time(q_h)
                k_f = first_spike_time(k_h)
                acc_q_fst[i] += q_f
                acc_stdp_delta[i] += np.abs(q_f - k_f)

                # Sparsity Profiling (1 - spike_rate)
                # spike rate = total_spikes / (T * D * N)
                rates = q_h.mean(dim=(0, 3, 4)).cpu().numpy() # [B, H]
                acc_sparsity[i] += (1.0 - rates).mean(axis=0)

                # Synchrony (store [B, H] mean act to correlate later)
                b_act = attn_h.mean(dim=(0, 3, 4)).cpu().numpy() # [B, H]
                head_activities[i].append(b_act)

            # Clear memory
            extracted_spikes.clear()

    # Clean up hooks
    for h in handles: h.remove()

    # 5. Build Gating Specification JSON
    print("[4/4] Extracting Biological Patterns and Compiling Gating JSON...")
    spec = {}
    
    for i in range(num_blocks):
        spec[f"block_{i}"] = {}
        
        # average tracked metrics
        avg_delta    = acc_stdp_delta[i] / batches
        avg_q_fst    = acc_q_fst[i] / batches
        avg_sparsity = acc_sparsity[i] / batches
        
        # compute intra-block synchrony
        all_acts = np.concatenate(head_activities[i], axis=0) # [500, H]
        corr_matrix = np.corrcoef(all_acts.T) # [H, H]
        np.fill_diagonal(corr_matrix, 0)
        max_corrs = np.max(np.abs(corr_matrix), axis=1) # max off-diagonal r per head
        
        # Extract Engram Weights sizes directly
        wq = state_dict[f'stage3.{i}.attn.q_conv.weight'].float().abs().mean().item()
        
        for h in range(NUM_HEADS):
            spars = float(avg_sparsity[h])
            delta = float(avg_delta[h])
            sync  = float(max_corrs[h])
            
            # Neuromorphic Algorithm Thresholds to Gating Recommendation
            gate_rec = "ACTIVE_NO_GATE"
            
            if spars >= 0.99:
                # Head is >99% silent over 500 random inputs
                gate_rec = "STATICALLY_PRUNE_OR_EARLY_EXIT_T1"
            elif sync > 0.4:
                # Highly correlated with another head
                gate_rec = "STATICALLY_GATED_BY_REDUNDANCY"
            elif delta > 1.5:
                # Head waits > 1.5 time steps between Q and K
                gate_rec = "DYNAMIC_KEY_EXIT_WAIT_T2"
            elif avg_q_fst[h] > 3.0:
                gate_rec = "LATE_WAKEUP_GATE"

            spec[f"block_{i}"][f"head_{h}"] = {
                "engram_mean_Wq": float(wq), 
                "sparsity_rate": round(spars, 4),
                "stdp_timing_gap_abs": round(delta, 3),
                "q_first_spike_time": round(float(avg_q_fst[h]), 3),
                "max_Pearson_sync_with_another_head": round(sync, 3),
                "HARDWARE_GATING_POLICY": gate_rec
            }

    # Save format
    os.makedirs("analysis_outputs", exist_ok=True)
    out_file = "analysis_outputs/gating_profile.json"
    with open(out_file, "w") as f:
        json.dump(spec, f, indent=4)
        
    print(f"    ✓ Profiling Complete! Written to {out_file}")

if __name__ == "__main__":
    main()
