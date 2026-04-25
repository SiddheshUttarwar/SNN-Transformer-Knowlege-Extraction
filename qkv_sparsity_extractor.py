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
NUM_INFERENCES = 10

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
    T_sz = spike_tensor.shape[0]
    indices = torch.arange(1, T_sz + 1, device=spike_tensor.device, dtype=torch.float32)
    shape_ones = [1]*len(spike_tensor.shape)
    shape_ones[0] = T_sz
    mask = indices.view(*shape_ones) * spike_tensor
    mask[mask == 0] = 999
    fst_times = mask.min(dim=0)[0] 
    fst_times[fst_times == 999] = T_sz 
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
            extracted_spikes[f"b{block_idx}_{l_type}"].append(output.detach())
        return hook_fn

    blocks = model.stage3
    handles = []
    num_blocks = len(blocks)
    for i, blk in enumerate(blocks):
        handles.append(blk.attn.q_lif.register_forward_hook(_make_hook(i, 'q')))
        handles.append(blk.attn.k_lif.register_forward_hook(_make_hook(i, 'k')))
        handles.append(blk.attn.v_lif.register_forward_hook(_make_hook(i, 'v')))
        handles.append(blk.attn.attn_lif.register_forward_hook(_make_hook(i, 'attn')))

    acc_stdp_delta = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    acc_q_fst      = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    acc_sparsity_q = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    acc_sparsity_k = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    acc_sparsity_v = {i: np.zeros(NUM_HEADS) for i in range(num_blocks)}
    
    head_activities = {i: [] for i in range(num_blocks)}

    print(f"[3/4] Running {NUM_INFERENCES} inferences over {N_SAMPLES} ImageNet samples each...")
    gen = data_generator()
    batches_per_inf = N_SAMPLES // BATCH_SIZE
    total_batches = NUM_INFERENCES * batches_per_inf
    
    with torch.no_grad():
        for inf_idx in range(NUM_INFERENCES):
            print(f"--- Inference {inf_idx+1}/{NUM_INFERENCES} ---")
            for b_idx in tqdm(range(batches_per_inf), total=batches_per_inf):
                x = next(gen).to(device)
                model(x)

            for i in range(num_blocks):
                q = extracted_spikes[f"b{i}_q"][-1]
                k = extracted_spikes[f"b{i}_k"][-1]
                v = extracted_spikes[f"b{i}_v"][-1]
                attn = extracted_spikes[f"b{i}_attn"][-1]

                T, B, C, N = q.shape
                H = NUM_HEADS
                D = C // H

                q_h = q.view(T, B, H, D, N)
                k_h = k.view(T, B, H, D, N)
                v_h = v.view(T, B, H, D, N)
                attn_h = attn.view(T, B, H, D, N)

                q_f = first_spike_time(q_h)
                k_f = first_spike_time(k_h)
                acc_q_fst[i] += q_f
                acc_stdp_delta[i] += np.abs(q_f - k_f)

                rates_q = q_h.mean(dim=(0, 3, 4)).cpu().numpy()
                rates_k = k_h.mean(dim=(0, 3, 4)).cpu().numpy()
                rates_v = v_h.mean(dim=(0, 3, 4)).cpu().numpy()

                acc_sparsity_q[i] += (1.0 - rates_q).mean(axis=0)
                acc_sparsity_k[i] += (1.0 - rates_k).mean(axis=0)
                acc_sparsity_v[i] += (1.0 - rates_v).mean(axis=0)

                b_act = attn_h.mean(dim=(0, 3, 4)).cpu().numpy()
                head_activities[i].append(b_act)

            extracted_spikes.clear()

    for h in handles: h.remove()

    print("[4/4] Extracting Biological Patterns and Compiling QKV JSON...")
    spec = {}
    
    for i in range(num_blocks):
        spec[f"block_{i}"] = {}
        
        avg_delta    = acc_stdp_delta[i] / total_batches
        avg_q_fst    = acc_q_fst[i] / total_batches
        avg_sparsity_q = acc_sparsity_q[i] / total_batches
        avg_sparsity_k = acc_sparsity_k[i] / total_batches
        avg_sparsity_v = acc_sparsity_v[i] / total_batches
        
        all_acts = np.concatenate(head_activities[i], axis=0) 
        corr_matrix = np.corrcoef(all_acts.T)
        np.fill_diagonal(corr_matrix, 0)
        max_corrs = np.max(np.abs(corr_matrix), axis=1)
        
        for h in range(NUM_HEADS):
            spars_q = float(avg_sparsity_q[h])
            spars_k = float(avg_sparsity_k[h])
            spars_v = float(avg_sparsity_v[h])
            delta = float(avg_delta[h])
            sync  = float(max_corrs[h])
            
            gate_rec = "ACTIVE_NO_GATE"
            
            if spars_q >= 0.99:
                gate_rec = "STATICALLY_PRUNE_OR_EARLY_EXIT_T1"
            elif sync > 0.4:
                gate_rec = "STATICALLY_GATED_BY_REDUNDANCY"
            elif delta > 1.5:
                gate_rec = "DYNAMIC_KEY_EXIT_WAIT_T2"
            elif avg_q_fst[h] > 3.0:
                gate_rec = "LATE_WAKEUP_GATE"

            spec[f"block_{i}"][f"head_{h}"] = {
                "sparsity_q": round(spars_q, 4),
                "sparsity_k": round(spars_k, 4),
                "sparsity_v": round(spars_v, 4),
                "stdp_timing_gap_abs": round(delta, 3),
                "max_Pearson_sync_with_another_head": round(sync, 3),
                "HARDWARE_GATING_POLICY": gate_rec
            }

    os.makedirs("analysis_outputs/JSONs", exist_ok=True)
    out_file = "analysis_outputs/JSONs/qkv_sparsity_profile.json"
    with open(out_file, "w") as f:
        json.dump(spec, f, indent=4)
        
    print(f"    [OK] Profiling Complete! Written to {out_file}")

    md_content = "# Q, K, V Sparsity by Gating Category\n\n"
    md_content += "| Block | Head | Policy | Q Sparsity | K Sparsity | V Sparsity |\n"
    md_content += "|---|---|---|---|---|---|\n"
    for i in range(num_blocks):
        for h in range(NUM_HEADS):
            data = spec[f"block_{i}"][f"head_{h}"]
            md_content += f"| {i} | {h} | {data['HARDWARE_GATING_POLICY']} | {data['sparsity_q']:.4f} | {data['sparsity_k']:.4f} | {data['sparsity_v']:.4f} |\n"
            
    with open("analysis_outputs/qkv_sparsity_summary.md", "w") as f:
        f.write(md_content)

if __name__ == "__main__":
    main()
