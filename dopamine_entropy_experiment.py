"""
Dopamine Entropy Hardware Experiment
Simulates a Dopaminergic Controller (Tonic Search vs. Phasic Power-Gating) natively on MaxFormer.
"""
import sys, os, warnings, json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use the imagenet subfolder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagenet"))
warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

OUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_outputs")
OUT_JSON = os.path.join(OUT_DIR, "JSONs", "dopamine_experiment_results.json")
OUT_IMG = os.path.join(OUT_DIR, "Images", "6_dopamine_hardware_scaling.png")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "10-384-T4.pth.tar")

NUM_HEADS = 6
IMG_SIZE = 224

class SpikeRecorder:
    def __init__(self):
        self.records = []
        self._handle = None

    def attach(self, module):
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
    def stacked(self):
        if not self.records: return None
        return torch.stack(self.records, dim=0)

def load_model():
    print("="*80)
    print("  [INIT] Loading MaxFormer for Dopaminergic Hardware Simulation")
    
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    import spikingjelly.clock_driven.neuron as _sj_neuron
    _orig_init = _sj_neuron.MultiStepLIFNode.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["backend"] = "torch"
        _orig_init(self, *args, **kwargs)
    _sj_neuron.MultiStepLIFNode.__init__ = _patched_init

    from max_former import Max_Former
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Max_Former(in_channels=3, num_classes=1000, embed_dims=384, mlp_ratios=4, depths=10, T=4)

    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    print(f"  [OK] Model successfully bridged to CPU/GPU on {device.upper()}")
    return model, device

def calculate_entropy(spike_tensor, num_heads):
    T, B, C, N = spike_tensor.shape
    H = num_heads
    D = C // H
    spk_h = spike_tensor.view(T, B, H, D, N).sum(dim=(0, 3))
    
    entropies = []
    total_spikes = 0
    for h in range(H):
        spatial_profile = spk_h[0, h, :].numpy()
        tot = spatial_profile.sum()
        total_spikes += tot
        if tot < 1e-4:
            entropies.append(0.0)
            continue
        p = spatial_profile / tot
        p = p[p > 0]
        ent = -np.sum(p * np.log2(p + 1e-9))
        entropies.append(ent)
        
    return entropies, total_spikes

def main():
    model, device = load_model()
    
    ssa_modules = []
    for name, module in model.named_modules():
        if "stage3" in name and "attn" in name and hasattr(module, "attn_lif"):
            ssa_modules.append((name, module.attn_lif))
            
    if not ssa_modules:
         print("Error: Could not locate SSA attn_lif blocks.")
         return

    print("="*80)
    print("  [EXPERIMENT] Executing Two-Pass Dynamic Dopaminergic Evaluations")
    
    results = {}
    N_SAMPLES = 500
    ENTROPY_THRESHOLD = 5.9  # Adjusted to forcefully partition Gaussian bounds
    
    plot_data = {"entropy": [], "shift_pct": [], "phase": [], "base_mac": [], "act_mac": []}

    for sample_id in range(1, N_SAMPLES + 1):
        x_sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        
        # ── PASS 1: Baseline Readout ──
        recorders = {name: SpikeRecorder().attach(lif) for name, lif in ssa_modules}
        with torch.no_grad():
            model(x_sample)
            
        block_entropies = []
        baseline_model_spikes = 0
        
        for name, lif in ssa_modules:
            rec = recorders[name]
            if rec.stacked is not None:
                spk = rec.stacked[0]
                ents, tot_spikes = calculate_entropy(spk, NUM_HEADS)
                block_entropies.append(np.mean(ents))
                baseline_model_spikes += tot_spikes
            rec.detach()
            rec.clear()
            
        global_entropy = float(np.mean(block_entropies))
        phase = "TONIC_SEARCH" if global_entropy > ENTROPY_THRESHOLD else "PHASIC_LOCK"
        
        # ── PASS 2: DOPAMINERGIC ACTUATION ──
        hooks = []
        
        if phase == "TONIC_SEARCH":
            for name, lif in ssa_modules:
                lif.v_threshold = 0.5 
        else:
            def make_power_gate_hook(essential_heads=[0,1]):
                def hook(mdl, inputs, output):
                    T, B, C, N = output.shape
                    H = NUM_HEADS; D = C // H
                    out_v = output.clone().view(T, B, H, D, N)
                    for h in range(H):
                        if h not in essential_heads:
                            out_v[:, :, h, :, :] = 0.0
                    return out_v.view(T, B, C, N)
                return hook

            for name, lif in ssa_modules:
                lif.v_threshold = 1.0 
                hdl = lif.register_forward_hook(make_power_gate_hook(essential_heads=[0,1]))
                hooks.append(hdl)
                
        recorders = {name: SpikeRecorder().attach(lif) for name, lif in ssa_modules}
        with torch.no_grad():
             model(x_sample)
             
        actuated_model_spikes = 0
        for name, lif in ssa_modules:
            rec = recorders[name]
            if rec.stacked is not None:
                spk = rec.stacked[0]
                _, tot_spikes = calculate_entropy(spk, NUM_HEADS)
                actuated_model_spikes += tot_spikes
            rec.detach()
            rec.clear()
            
        for h in hooks: h.remove()
        for name, lif in ssa_modules: lif.v_threshold = 1.0 
        
        delta_spikes = actuated_model_spikes - baseline_model_spikes
        delta_pct = (delta_spikes / max(1, baseline_model_spikes)) * 100.0
        
        if sample_id % 20 == 0:
            print(f"Sample {sample_id:03d}/{N_SAMPLES} | Entropy: {global_entropy:.3f} | Phase: {phase:<12} | Shift: {delta_pct:+.1f}%")

        results[f"Sample_{sample_id:03d}"] = {
            "global_entropy": float(global_entropy),
            "dopamine_phase": phase,
            "baseline_spike_operations": float(baseline_model_spikes),
            "actuated_spike_operations": float(actuated_model_spikes),
            "dynamic_power_shift_pct": float(delta_pct)
        }
        
        plot_data["entropy"].append(global_entropy)
        plot_data["shift_pct"].append(delta_pct)
        plot_data["phase"].append(phase)
        plot_data["base_mac"].append(baseline_model_spikes)
        plot_data["act_mac"].append(actuated_model_spikes)
        
    # --- Visualization ---
    print("\n  [VIZ] Generating Dopamine Scaling visual artifacts...")
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0a0a0a")
    fig.suptitle("Dopamine Hardware Optimization Profile (500 Samples)", fontsize=18, color="white", fontweight="bold")
    
    # Left Plot: MAC Shift relative to Entropy
    ax1.set_facecolor("#111111")
    tonic_mask = np.array(plot_data["phase"]) == "TONIC_SEARCH"
    phasic_mask = np.array(plot_data["phase"]) == "PHASIC_LOCK"
    
    ent_arr = np.array(plot_data["entropy"])
    shift_arr = np.array(plot_data["shift_pct"])
    
    ax1.scatter(ent_arr[tonic_mask], shift_arr[tonic_mask], color="#f72585", alpha=0.7, label="Tonic Search ($V_{th}$ Drop)", s=50, edgecolors="white", linewidth=0.5)
    ax1.scatter(ent_arr[phasic_mask], shift_arr[phasic_mask], color="#4cc9f0", alpha=0.7, label="Phasic Lock (Power Gate)", s=50, edgecolors="white", linewidth=0.5)
    
    ax1.axvline(ENTROPY_THRESHOLD, color="#ffd60a", linestyle="--", label="Entropy Phase Threshold")
    ax1.axhline(0, color="gray", linewidth=1.5)
    
    ax1.set_xlabel("Spatial Attention Entropy $H(A_t)$", color="#aaaaaa", fontsize=12)
    ax1.set_ylabel("MAC Shift Percentage (%)", color="#aaaaaa", fontsize=12)
    ax1.set_title("Dynamic Sparsity Modulation per Sample", color="white", fontsize=14)
    ax1.tick_params(colors="#aaaaaa")
    ax1.legend(facecolor="#111111", labelcolor="white")
    ax1.grid(color="#333333", alpha=0.5)
    for sp in ax1.spines.values(): sp.set_edgecolor("#333333")

    # Right Plot: Base vs Actuated Spike Densities
    ax2.set_facecolor("#111111")
    tonic_base = np.mean(np.array(plot_data["base_mac"])[tonic_mask]) if np.any(tonic_mask) else 0
    tonic_act = np.mean(np.array(plot_data["act_mac"])[tonic_mask]) if np.any(tonic_mask) else 0
    phasic_base = np.mean(np.array(plot_data["base_mac"])[phasic_mask]) if np.any(phasic_mask) else 0
    phasic_act = np.mean(np.array(plot_data["act_mac"])[phasic_mask]) if np.any(phasic_mask) else 0

    x = np.arange(2)
    width = 0.35

    ax2.bar(x - width/2, [tonic_base, phasic_base], width, label="Baseline MACs", color="#7209b7")
    ax2.bar(x + width/2, [tonic_act, phasic_act], width, label="Actuated MACs", color="#ffd60a")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Tonic Search (Exploration)", "Phasic Lock (Exploitation)"], color="#aaaaaa", fontsize=12)
    ax2.set_ylabel("Average Active Spike OPs (MACs)", color="#aaaaaa", fontsize=12)
    ax2.set_title("Macroscopic Execution Cost per Phase", color="white", fontsize=14)
    ax2.tick_params(colors="#aaaaaa")
    ax2.legend(facecolor="#111111", labelcolor="white")
    ax2.grid(axis='y', color="#333333", alpha=0.5)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
         json.dump(results, f, indent=4)
         
    print("="*80)
    print(f"  [SUCCESS] JSON saved to {OUT_JSON}")
    print(f"  [SUCCESS] IMAGE saved to {OUT_IMG}")
    print("="*80)

if __name__ == "__main__":
     main()
