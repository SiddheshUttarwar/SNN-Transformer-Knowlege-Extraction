# SpikeGate Framework

SpikeGate is an industry-standard, label-agnostic framework for evaluating and implementing dynamic hardware gating in Spiking Neural Network (SNN) Transformers.

It provides a comprehensive suite of tools to automatically profile, evaluate, and inject power-saving gating policies (like static pruning, temporal early-exits, and redundancy gating) directly into transformer attention heads without requiring retraining.

## Core Components

- **AutoProfiler** (`spikegate.profiler`): Automatically hooks into SNNs to measure Q/K/V sparsity per timestep, STDP temporal gaps, neuron health (dead/ghost neurons), and cross-head correlation. Generates optimal hardware gating policies.
- **DynamicGateController** (`spikegate.gating`): Central policy manager that dynamically injects gating behaviors during runtime based on generated profiles or manual overrides.
- **SpikingGatedAttention** (`spikegate.attention`): Wrap standard attention layers in this module to simulate hardware-level computation bypassing (saves KV projections on pruned heads or inactive timesteps).
- **GatingAblationStudy** (`spikegate.evaluator`): Measures compute savings and Baseline Fidelity (label-agnostic accuracy) for any gating policy. Also supports MASK-ONE-OUT head importance estimation.
- **ReportGenerator** (`spikegate.reporter`): Orchestrates large-scale multi-run evaluations (e.g., 10 runs x 500 images) and outputs publication-quality Markdown reports and matplotlib visualizations.

## Quickstart

```python
import torch
from spikegate import AutoProfiler, DynamicGateController, ReportGenerator

# 1. Load your model and data
model = ...
data_chunks = [...] # List of dataloaders for multi-run stability testing

# 2. Automatically generate a gating profile
profiler = AutoProfiler(model, T=4)
profile = profiler.calibrate(data_chunks[0], device='cuda', num_blocks=10, num_heads=6)

# 3. Load the policy controller
gate_ctrl = DynamicGateController("gating_profile.json")

# 4. Generate a comprehensive multi-run stability report
reporter = ReportGenerator(model, gate_ctrl, profile, num_blocks=10, num_heads=6)
report_path = reporter.run_full_pipeline(data_chunks, device='cuda')
```
