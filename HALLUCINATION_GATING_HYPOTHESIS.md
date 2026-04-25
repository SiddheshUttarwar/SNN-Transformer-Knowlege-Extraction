# Hypothesis: Gating Hallucinated Spikes in Spiking Neural Networks

## 1. Core Hypothesis Points
1. **The Origins of Hallucination:** In deep Spiking Neural Networks (SNNs), specific LIF (Leaky Integrate-and-Fire) neurons fire spontaneously (hallucinate) even when the macro-network input is strictly zero. We hypothesize that this is caused purely by static pre-synaptic biases ($b$) from preceding Convolutional or LayerNorm layers pushing the resting membrane potential ($V$) above the firing threshold ($V_{th}$).
2. **Mutual Information Extraction:** We hypothesize that pure ghost neurons (those that reliably fire under zero-input) carry zero mutual information regarding the image stimulus. They function purely as static constant-current offsets.
3. **Painless Energy Reduction:** Because these specific neurons carry no stimulus-specific entropy, surgically suppressing or gating them will reduce the network's overall total spike count (which correlates directly to neuromorphic energy consumption) without degrading the network's classification accuracy.

## 2. Mathematical Implementation

To suppress these hallucinated spikes, we can implement two biologically-plausible, fast gating mechanisms derived from our `ghost_neurons.json` mapping.

### Mechanism A: Static Channel Ablation (Synaptic Pruning)
For a given attention head $h$ and channel $c$ internally identified as a ghost channel:
Let $S_{out}(t) \in \{0, 1\}^C$ be the binary spike output vector of the `attn_lif` node at time $t$.
We construct a static diagonal masking matrix $M \in \mathbb{R}^{C \times C}$ where:

$$ M_{i,i} = \begin{cases} 0 & \text{if channel } i \text{ is a ghost channel} \\ 1 & \text{otherwise} \end{cases} $$

The gated spike output becomes:
$$ \hat{S}_{out}(t) = M \cdot S_{out}(t) $$

### Mechanism B: Pre-Synaptic Bias Suppression (Glial Clearance Proxy)
Instead of masking the spike after it happens (which still burns energy calculating the spike), we prevent the membrane potential from reaching the threshold in the first place.
The membrane potential dynamics in SpikingJelly typically follow:

$$ V(t) = V(t-1) \cdot \tau + (W \cdot X(t) + b) $$

Where $b$ is the static bias parameter from the earlier norm/conv stage. 
Using our explicit ghost neuron map, we alter the PyTorch State Dictionary prior to inference loading:

$$ b_c \leftarrow 0 \quad \forall c \in \text{completely\_ghost\_channels} $$

This permanently forces the resting membrane potential $V(t)$ below $V_{th}$ when the stimulus $X(t)$ is null.

## 3. Success Criteria

The experimental validation of this hypothesis is considered **Successful** if the following conditions are met during inference over a benchmark dataset (e.g., Tiny-ImageNet / ImageNet-1K):
* **Energy Reduction:** The total macroscopic spike rate demonstrates a measurable reduction. Every prevented spike directly corresponds to a saved Accumulate (AC) operation in neuromorphic hardware.
* **Accuracy Preservation:** The Top-1 and Top-5 evaluation accuracy of the network remains structurally identical to the baseline accuracy (tolerating a negligible degradation margin of $\leq 0.1\%$).
* **Computational Overhead:** The implementation of either the static mask $M$ or the bias ablation $b_c=0$ incurs zero additional runtime latency overhead on GPU/Hardware platforms.

## 4. Failure Conditions

The hypothesis is considered **Falsified** (meaning the structural role of ghost neurons was misunderstood) if any of the following occur:
* **Catastrophic Accuracy Collapse:** The Top-1 accuracy drops disproportionately (e.g., $> 1.0\%$). This would definitively prove that the assumption about "zero mutual information" is false—implying the constant hallucinatory spiking actually encodes crucial steady-state contextual biases needed by subsequent transformer layers to evaluate self-attention properly.
* **Cascading Silence (Network Death):** Gating these specific ghost channels starves the downstream layers of expected input current, causing those deeper layers to fail to reach threshold entirely, resulting in completely zeroed predictions.
