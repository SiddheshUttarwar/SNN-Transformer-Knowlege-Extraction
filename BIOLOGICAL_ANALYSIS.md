# Biological Learning Analysis — Spiking MaxFormer

## Overview

These two scripts analyze a pretrained **Spiking MaxFormer** (MaxFormer-10-384, ImageNet, T=4 time steps) through the lens of biological neuroscience.  
The model is a **Spiking Neural Network (SNN)** Transformer: instead of floating-point activations it fires discrete binary spikes across `T=4` time steps, mimicking biological neurons.

| Script | Purpose | Output |
|---|---|---|
| `visualize_model.py` | Visualization of biological patterns from a single forward pass | `analysis_outputs/*.png` |
| `pattern_extractor.py` | Statistical profiling over 500 real ImageNet samples → hardware gating decisions | `analysis_outputs/gating_profile.json` |

---

## Model Architecture Quick Reference

```
MaxFormer-10-384 (ImageNet, 1000 classes)
  embed_dims = 384
  T (time steps) = 4
  Stage 1 : 1 × DWC7 block   (depthwise conv, 7×7 kernel)
  Stage 2 : 2 × DWC5 blocks  (depthwise conv, 5×5 kernel)
  Stage 3 : 7 × SSA blocks   ← analysis targets (Spiking Self-Attention)
  num_heads per SSA = 384 // 64 = 6
```

The analyses focus exclusively on **Stage 3 SSA blocks** because they contain the learned temporal attention mechanism — the closest computational analogy to cortical processing.

---

## `visualize_model.py` — Visualization Pipeline

Runs **5 sequential analysis modules** on a single random forward pass (or a small set of passes). Outputs publication-quality dark-theme figures.

### Module 1 — Engram Analysis (Hebbian / LTP Proxy)

**File:** `1_engram_weight_heatmaps.png`, `2_engram_head_bandwidth.png`

```
For each SSA block in stage3:
  Extract W_q, W_k, W_v conv weights  →  shape (C_out, C_in, 1)
  Reshape to (num_heads, head_dim, C_in)
  Compute mean |W| per head  →  "engram strength"
  Plot full weight matrix as heatmap + annotate head boundaries
  Plot per-head bar chart of mean magnitude
```

**Biological connection — Long-Term Potentiation (LTP):**  
In the brain, LTP strengthens synaptic connections that fire repeatedly together ("neurons that fire together, wire together" — Hebb's rule). High-magnitude weight slices in W_q / W_k / W_v indicate which attention heads have formed strong *engrams* — stable, reinforced feature-detection pathways. A head with consistently high |W| is the model's analog of a potentiated cortical column.

---

### Module 2 — STDP Profiling (ISI Proxy)

**File:** `3_stdp_isi_profiling.png`

```
Hook q_lif and k_lif outputs  →  spike tensors (T, B, C, N)
Reshape to (T, B, H, D, N)
For each head h:
  fst_Q[h] = mean first time-step where Q spike == 1
  fst_K[h] = mean first time-step where K spike == 1
  Δt[h]    = |fst_Q[h] − fst_K[h]|

Color-coded bars:
  Δt < 0.5  → Tight   (blue)
  Δt < 1.5  → Moderate (orange)
  Δt ≥ 1.5  → Loose   (red)
```

**Biological connection — Spike-Timing Dependent Plasticity (STDP):**  
STDP is the biological learning rule in which the relative timing of pre- and post-synaptic spikes determines whether a synapse is strengthened (LTP) or weakened (LTD). In SSA, Q spikes act as "pre-synaptic" signals and K spikes as "post-synaptic" signals. A small Δt means Q and K fire nearly synchronously, analogous to a strongly potentiated synapse. Large Δt means Q and K are temporally misaligned — a weaker or unreliable synaptic pathway. The ISI (Inter-Spike Interval) metric here is a proxy for the classic STDP timing window.

---

### Module 3 — Temporal Sparsity (Predictive Coding Proxy)

**File:** `4_temporal_sparsity.png`

```
Run 8 random forward passes with different random inputs
For each pass, hook attn_lif outputs per SSA block
Compute spike_rate per head = mean(spikes) over (T, B, D, N)
sparsity = 1 − spike_rate

Aggregate over 8 passes:
  Violin plot  →  distribution of sparsity per head
  Heatmap      →  head × pass (shows consistency)
  Threshold = 0.90 → heads above marked as PRUNABLE (✂)
```

**Biological connection — Predictive Coding / Sparse Coding:**  
The brain operates under metabolic constraints and encodes information as sparsely as possible (Olshausen & Field, 1996). Neurons in V1 and beyond fire only when their preferred stimulus is present — the rest of the time they are silent. A high-sparsity head in the SNN is doing the same: it only activates for specific input patterns. Low-sparsity heads fire densely regardless of input — they are less discriminative and are candidates for pruning, analogous to neurons that do not develop feature selectivity.

---

### Module 4 — Attentional Synchrony (Neural Gamma-Band Proxy)

**File:** `5_attention_synchrony.png`

```
Hook attn_lif outputs for all SSA blocks in stage3
For each block:
  Spike tensor (T, B, C, N) → reshape → (H, T*D*N*B) signal per head
  Pearson correlation matrix between all head pairs  →  (H, H)
  High off-diagonal r = heads that fire synchronously
  Heatmap with SYNC_CMAP (blue=anti-corr, white=zero, red=corr)
```

**Biological connection — Neural Synchrony / Gamma Oscillations:**  
Synchronized firing across neural populations — particularly in the gamma band (30–80 Hz) — is associated with feature binding, attention, and working memory in the cortex. When two attention heads show high Pearson correlation in their spike outputs, they are the model's analog of synchronously firing neural ensembles attending to the same stimulus feature. Perfect synchrony (r ≈ 1) between two heads means they are functionally redundant — encoding the same representation.

---

### Module 5 — Summary Dashboard

**File:** `0_DASHBOARD.png`

Assembles all 5 saved panel images into a single vertically-stacked overview figure for presentation.

---

## `pattern_extractor.py` — Statistical Profiling Pipeline

Runs inference over **500 real ImageNet images** (streamed from HuggingFace `mrm8488/ImageNet1K-val`) and computes **per-head biological statistics** aggregated across all batches. The result is a deterministic JSON specification used to drive hardware gating decisions.

### Data Pipeline

```
HuggingFace streaming dataset (500 samples, batch=10)
  → Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet stats)
  → 50 batches of shape (10, 3, 224, 224)
```

### Hook Strategy

```
Register forward hooks on stage3 blocks:
  blk.attn.q_lif    → captures Query spike tensor  (T, B, C, N)
  blk.attn.k_lif    → captures Key spike tensor    (T, B, C, N)
  blk.attn.attn_lif → captures Attention output    (T, B, C, N)
Clear hooks after each batch to save memory
```

### Metrics Computed Per Head (Accumulated Over 50 Batches)

| Metric | Computation | Biological Analog |
|---|---|---|
| `q_first_spike_time` | Mean first T-step where Q fires, averaged over batch/D/N | Neural latency / response onset |
| `stdp_timing_gap_abs` | `mean\|fst_Q − fst_K\|` per head | STDP pre/post timing window |
| `sparsity_rate` | `1 − mean(Q spikes)` over T, D, N | Sparse coding / metabolic efficiency |
| `max_Pearson_sync` | Max off-diagonal Pearson r in head correlation matrix | Neural synchrony / feature binding |
| `engram_mean_Wq` | Mean absolute value of `stage3.i.attn.q_conv.weight` | LTP engram strength |

### Hardware Gating Decision Logic

```python
if sparsity >= 0.99:
    → STATICALLY_PRUNE_OR_EARLY_EXIT_T1
    # Head is >99% silent over 500 real images — it has learned no useful feature.
    # Biological analog: a silent neuron that never fires; can be ablated.

elif sync > 0.4:
    → STATICALLY_GATED_BY_REDUNDANCY
    # Head fires in near-lockstep with another head — redundant encoding.
    # Biological analog: two neurons with identical receptive fields; one is redundant.

elif delta > 1.5:
    → DYNAMIC_KEY_EXIT_WAIT_T2
    # Q and K are temporally misaligned by >1.5 time steps — weak STDP coupling.
    # Biological analog: pre/post timing outside the STDP window → no potentiation.

elif avg_q_fst > 3.0:
    → LATE_WAKEUP_GATE
    # Head doesn't fire until very late in the T=4 sequence.
    # Biological analog: slow-onset neuron / late cortical layer response.

else:
    → ACTIVE_NO_GATE
    # Head is sparse, well-timed, non-redundant, early-firing → keep fully active.
```

### Output JSON Structure

```json
{
  "block_0": {
    "head_0": {
      "engram_mean_Wq": 0.043,
      "sparsity_rate": 0.9612,
      "stdp_timing_gap_abs": 0.812,
      "q_first_spike_time": 1.234,
      "max_Pearson_sync_with_another_head": 0.217,
      "HARDWARE_GATING_POLICY": "ACTIVE_NO_GATE"
    },
    "head_1": { ... }
  },
  ...
}
```

---

## How the Two Scripts Relate

```
visualize_model.py                       pattern_extractor.py
─────────────────                        ────────────────────
Single random input (qualitative)   →    500 real ImageNet samples (quantitative)
Visual heatmaps / plots             →    Deterministic JSON specification
Intuition / exploration             →    Hardware implementation target
Identifies patterns                 →    Translates patterns to gating rules
```

Both scripts share the same 4 biological analysis axes (engram, STDP, sparsity, synchrony). `visualize_model.py` is the **research tool** — you explore and understand what the model is doing biologically. `pattern_extractor.py` is the **engineering tool** — it takes those findings and converts them into concrete, statistically validated hardware gating policies across a large real-world dataset.

---

## Biological Model Correspondence Table

| SNN / MaxFormer Concept | Biological Neuroscience Analog |
|---|---|
| Spike tensor (binary, T steps) | Action potential train |
| T = 4 time steps | Temporal integration window (e.g., 4 ms bins) |
| Q/K/V LIF neurons (`q_lif`, `k_lif`) | Pre/post-synaptic neurons in a cortical circuit |
| First spike time | Neural response latency |
| W_q / W_k magnitude | Synaptic weight (LTP-strengthened connection) |
| Δt = \|fst_Q − fst_K\| | STDP timing window (pre-before-post = LTP) |
| Sparsity (1 − spike_rate) | Sparse coding / metabolic efficiency |
| Head-pair Pearson r | Neural synchrony / gamma-band binding |
| Head pruning (sparsity > 99%) | Synaptic pruning / silent neuron ablation |
| Redundant head gating (sync > 0.4) | Functional redundancy removal in cortical maps |
| Late-wakeup head (fst_Q > 3.0) | Slow cortical layer (L5/L6) late response |
| Engram heatmap | Long-term memory trace in synaptic weights |

---

## Running the Scripts

```bash
# Visualization (needs only checkpoint, runs fast)
python visualize_model.py

# Statistical profiling (streams 500 ImageNet samples, ~5-10 min on CPU)
python pattern_extractor.py
```

**Prerequisites:** `torch`, `spikingjelly`, `datasets` (HuggingFace), `torchvision`, `matplotlib`, `numpy`, `tqdm`  
**Checkpoint:** `./checkpoints/10-384-T4.pth.tar`  
**Outputs:** `./analysis_outputs/`
