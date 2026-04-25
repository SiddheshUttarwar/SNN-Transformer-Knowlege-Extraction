# Spiking Neural Network (SNN) Explainability & Hardware Gating

![SNN Analysis Header](https://img.shields.io/badge/NeurIPS-Submission-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch) ![SpikingJelly](https://img.shields.io/badge/Framework-SpikingJelly-yellow)

This repository serves as the official supplementary code and data module for our research investigating the biological abstractions and inherent hardware optimization topologies within deep Spiking Neural Networks (SNNs)—specifically evaluating the **MaxFormer** Vision Transformer architecture.

## Overview
By mapping core Artificial Neural Network (ANN) pruning concepts alongside established cortical neuroscience frameworks (e.g., Sparse Predictive Coding, STDP, Hebbian LTP, and Gamma-Band Synchrony), we constructed a completely novel algorithmic extraction engine. This engine mathematically intercepts continuous Spiking Self-Attention (SSA) tensors during sustained ImageNet-1K inferences to deduce strictly enforceable Neuromorphic Hardware gating pathways.

Our codebase proves that specific localized Transformer nodes naturally adopt explicit neuromorphic constraints, exhibiting macroscopic behaviors such as perfect stimulus-agnostic hallucination ("Ghost Neurons") or dense temporal sparsity.

## Repository Structure

### 1. Neuromorphic Probing Scripts (`.py`)
We provide our isolated forward-hook evaluation architecture capable of parsing the discrete $T=4$ temporal spike integrations emitted linearly by `attn_lif` nodes without interrupting generalized gradient deployment.
* `visualize_model.py`: Extracts raw biological indicators (Sparsity, Engram Magnitudes, STDP phase timing, Attention Synchrony).
* `pattern_extractor.py`: Consolidates the aforementioned biology indicators into a Boolean logical truth table, outputting a highly targeted hardware-gating constraint dictionary (`gating_profile.json`).
* `[...]_ghost_neuron_map.py`: A sequence of analytical scripts iteratively narrowing down zero-variance, low-entropy localized feature-maps mathematically simulating spontaneous dopaminergic excitation (Baseline-Anchored Ghost Neurons).

### 2. Analytical Explanations (`Analysis_Explanations/`)
A dense repository containing heavily parameterized, formal academic Markdown documents designed specifically for theoretical peer review. Each of the 9 documents cleanly outlines the assumed scientific Hypothesis, formalizes the Algorithmic execution strategy sequentially, isolates the experimental Falsification/Validation limits, and explicitly pairs extracted hardware constraints against resulting visual bounds.

### 3. Empirical Artifacts (`analysis_outputs/`)
Contains structural proof of theory:
* `Images/`: Formally extracted 2D spatial distribution heatmaps capturing localized node behaviors systematically mapped sequentially.
* `JSONs/`: Strictly parameterized categorical dictionaries converting visual heuristics directly into programmatic logic gates dynamically capable of routing parallel architectural boundaries natively during continuous neuromorphic inference execution.

## Deployment & Verification
This repository was decoupled locally from standard ImageNet dataloaders to strictly protect evaluation frameworks without bloating package sizes. The extracted constraints mapping isolated spatial token arrays identically against formal visual bounds securely anchor our algorithmic claims validating hardware optimization frameworks systematically.

*Note: Developed and formulated for NeurIPS Submission. Contains targeted `MaxFormer` validation checkpoints optimized over sequential evaluation logic boundaries bounding specific architectural testing regimes.*
