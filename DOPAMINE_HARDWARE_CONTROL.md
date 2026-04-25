# Neuromorphic Spiking Image Transformers: Online R-STDP Adaptation via Dopaminergic Hardware Control

**Siddhesh Uttarwar**
Department of Electrical and Computer Engineering
University of California, Santa Barbara

## Abstract
Current Spiking Image Transformers (SITs) achieve impressive ImageNet performance but rely entirely on static inference, squandering the theoretical energy benefits of Spiking Neural Networks (SNNs). We propose a novel, biologically plausible hardware architecture that translates the dopaminergic reward system of the mammalian midbrain into a dynamic SystemVerilog controller. By mapping Tonic dopamine to global Leaky Integrate-and-Fire (LIF) threshold modulation and Phasic dopamine to Dynamic Head Pruning (DHP) triggers, our architecture enables an SIT to autonomously optimize its spatial sparsity per-sample. Furthermore, we implement dendritic delay-line Finite State Machines (FSMs) to maintain local eligibility traces, allowing for online Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP) without the computationally prohibitive overhead of Backpropagation Through Time (BPTT). 

---

## 1. Biological Hypothesis Formulation

While Spiking Vision Transformers bridge the gap between global self-attention and event-driven computing, their physical implementations remain bounded by static topologies. Offline-trained deployment hardware utilizes fixed LIF membrane thresholds ($V_{th}$) and static sparsity masks. 

**Core Hypothesis:** We hypothesize that integrating a decentralized "Dopamine Agent"—mimicking the mammalian Substantia Nigra's exploration/exploitation dynamics—allows a Spiking Image Transformer to autonomously modulate local R-STDP and LIF thresholds during inference. This mathematically optimal stochastic search mechanism allows the hardware to dynamically prune unnecessary computation without relying on static heuristics.

We extract two distinct temporal pathways to form the fundamental logic gates:
1. **Tonic Dopamine (The Exploration Heuristic):** High tonic levels decrease the threshold for action. In hardware, this maps to dropping the global LIF membrane threshold ($V_{th}$) when encountering high image-patch entropy $H(A_t)$, explicitly permitting weak signals to propagate to execute a stochastic spatial search equivalent to biological saccades.
2. **Phasic Dopamine (The Exploitation Trigger):** Phasic dopamine consists of rapid bursts representing the Reward Prediction Error ($\delta(t)$). Using $\delta(t) = r(t) + \gamma V(s_{t+1}) - V(s_{t})$, this error term acts as the explicit gating signal for hardware-level Dynamic Head Pruning (DHP) when a salient feature is locked.

---

## 2. Hardware Architecture and System Design

The translation of these biological principles into a synthesizable RTL architecture requires decentralized logic and specialized memory structures to minimize dynamic power consumption while optimizing the Energy-Accuracy Functional $J$:
$$J = \int_{0}^{T} \left[ \mathcal{L}_{CE}(\hat{y}_t, y) + \lambda ||S_t||_1 - \beta H(A_t) \right] dt$$

### 2.1 The Dopaminergic Gating Controller
* **Search Phase (High Entropy):** When processing a highly ambiguous image patch, $H(A_t)$ is high. The centralized controller outputs a High Tonic Dopamine signal, lowering $V_{th}$ globally. 
* **Lock Phase (Low Entropy):** Upon detecting a coherent Q-K synchronous match, a positive RPE ($\delta$) generates a Phasic burst. This asserts immediate `power_gate` signals to all non-essential attention heads, collapsing the network into a highly sparse, zero-error state for the remainder of the forward pass.

### 2.2 Dendritic FSMs and Eligibility Traces ($E_{ij}$)
Standard coincidence detection fails for online R-STDP. We augment the Dual-Port SRAM arrays with localized shift-register delay lines and simple Finite State Machines (FSMs) at the synaptic routing crossbars. When a Query ($Q$) and Key ($K$) spike occur synchronously, the FSM transitions to an active state, caching the coincidence as a local eligibility trace ($E_{ij}$). If a global Phasic Dopamine signal ($\delta$) arrives before the trace completely leaks away, the FSM triggers a localized SRAM `write_enable` executing the formal *e-prop* learning sequence: $\Delta w_{ij}(t) = \alpha \cdot \delta(t) \cdot E_{ij}(t)$.

### 2.3 Dynamic Auto-Squelch (Homeostatic Plasticity)
Rolling-average counters monitor individual head spike density. If an attention head is captured by high-frequency spatial noise, the rolling average triggers an independent, localized increase in $V_{th}$, forcing the head to squelch the noise without requiring intervention from the global Dopamine Agent.

---

## 3. Execution Framework: Pass Scenarios
Under ideal spatial and temporal constraints, the Dopaminergic hardware architecture demonstrates profound neuromorphic efficiency:

* **Pass Scenario A: Continual Edge Adaptation (Zero-BPTT Learning)**
  If deployed to a drone navigating a novel environment, the Dendritic FSMs successfully maintain eligibility traces locally. When the system receives a delayed positive environmental reward, the Phasic Dopamine broadcast triggers $\Delta w_{ij}(t)$, smoothly adjusting the Transformer's attention matrices online using exclusively forward-pass gradients.
* **Pass Scenario B: Aggressive Lock-and-Power-Down**
  When presented with an image featuring an extremely clear, salient central target against a blank background, the network rapidly achieves Q-K confidence. The sudden dip in entropy triggers an immediate Phasic exploit burst, isolating the primary attention heads and hard-power-gating $90\%$ of the peripheral network dynamically in cycle $T_2$, slashing total inference MAC operations.
* **Pass Scenario C: Epileptic Feedback Squelching**
  If encountering a densely textured adversarial background (e.g., static noise), standard STDP would experience runaway excitation and flood the visual pipeline. The Homeostatic Auto-Squelch correctly flags the dense firing anomaly, dynamically raises local $V_{th}$, and artificially forces the noisy attention head back into temporal sparsity.

---

## 4. Execution Framework: Failure Scenarios (Pathological Edge Cases)
The architecture breaks down under highly specific spatio-temporal bottlenecks inherent strictly to physical hardware boundaries:

* **Failure Scenario A: Tonic Dopamine Saturation (Power Bleed)**
  If the network evaluates heavily occluded or adversarial input spaces possessing mathematically irresolvable entropy (massive uniform gray-scale ambiguity), $H(A_t)$ remains saturated indefinitely. The controller continually forces $V_{th}$ to rock bottom, allowing every noisy spatial token to propagate through the multiplier grids. This results in massive dynamic power consumption, negating completely the theoretical benefits of deploying SNNs.
* **Failure Scenario B: FSM Eligibility Trace SRAM Saturation**
  Real-world environmental rewards often arrive significantly late. The shift-register delay lines buffering $E_{ij}$ are physically bounded by finite SRAM depth limits. If the requisite Phasic reward signal ($\delta(t)$) arrives after the FSM buffer has fundamentally expired or overwritten itself, the $\Delta w_{ij}(t)$ update evaluates to identically $0$. The R-STDP mechanism fails completely, paralyzing the network's capacity for delayed online learning.
* **Failure Scenario C: Dynamic Pipeline Bubbles & Thrashing**
  If localized patches fluctuate rapidly between high/low entropy, the Dopaminergic Controller will attempt to violently toggle the `power_gate` routing matrices. Constantly halting and waking macro-blocks induces massive temporal "bubbles" within the FPGA/ASIC execution pipelines, severely crippling chronological data throughput and producing erratic thermal loads across the chip.

---

## 5. Conclusion
By directly mapping the exploration-exploitation dynamics of the biological midbrain onto silicon, we present a novel framework for online adaptation in neuromorphic vision models. The Dopamine-Agent controller proves that dynamic, sample-specific sparsity can be achieved in hardware without relying on computationally expensive backpropagation. Despite pathological edge cases involving buffer saturation and noise thresholds, this architecture natively resolves the Power-Accuracy tradeoff, paving the way for highly efficient, continuously adapting edge-AI systems capable of processing ImageNet-scale environments.
