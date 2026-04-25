# Analysis: Baseline-Anchored Extraneous Firing (Baseline Ghost Neurons)

## 1. Hypothesis Formulation
Building on the detection of variance-damped spatial hallucinations within deep architectures, we propose a maximally constrained structural theory: **Baseline-Anchored Extraneous Firing**. We posit that true, computationally parasitic "Ghost Neurons" do not merely exhibit zero variance; they actively anchor their operational logic exclusively to the network's inert bias constants, ignoring dynamically routed stimulus flows completely. Formally, a Baseline-Anchored Neuron exhibits a firing intensity $\mathcal{F}_{\text{real}} = \sum_T \mathcal{S}_{\text{real}}$ in response to an arbitrary complex natural stimulus $X_{real}$ that is strictly isometric to its isolated resting-state output $\mathcal{F}_{void} = \sum_T \mathcal{S}_{void}$ upon consuming a mathematical zero-tensor void space $X_{null} = \mathbf{0}$.

Should $\mathcal{F}_{\text{real}} \equiv \mathcal{F}_{\text{void}}$ over large-scale dataset sampling, we assert the neuron operates essentially disconnected from any feature extraction manifold, acting strictly as a static bias offset with identical structural redundancy across all inputs.

## 2. Experimental Framework
Isolating true topological anchors requires rigorous comparative controls mirroring functional neuro-imaging. We instituted a bilateral verification stream: First, structurally establishing a ground-truth topological resting baseline map representing intrinsic network excitability stripped of stimulus. Second, cascading a sustained inference distribution over heterogeneous natural image spaces, executing harsh, pixel-perfect boolean rejection checks sequentially against the pre-established resting baseline.

## 3. Algorithmic Implementation
1. **Resting State Baseline Acquisition:** Ingest a flat dimension-matched zero-tensor $\mathcal{X}_{null}$ into the computational stream. Recursively capture the global spatial output maps generated along each individual depth block across timesteps $T$ to solidify the Reference Map tensor $\mathcal{R} \in \mathbb{R}^{C \times N}$.
2. **Evaluation Envelope:** Initialize an active constraint mask natively seeded by actively firing resting variables: $\mathcal{M}_{anchor} = (\mathcal{R} > 0)$.
3. **Sequential Falsification Loop:** Continuously route 100 ImageNet evaluations $\mathcal{X}_i$. Upon computing output spike tensor $\mathcal{Y}_i$, intersect and penalize non-conformant tokens dynamically:
   $$ \mathcal{M}_{anchor} \leftarrow \mathcal{M}_{anchor} \land (\mathcal{Y}_i \equiv \mathcal{R}) $$
4. **Data Serialization:** Discretize persisting masked states representing neurons that survived all 100 falsification sweeps via complete absolute anchoring.

## 4. Hypothesis Correspondence & Hardware Gating Implications
The results successfully **falsified the widespread viability of the hypothesis for generalized gating**, while establishing a mathematically intriguing corner-case discovery. 

Data processing revealed a singular highly concentrated neuro-anomalous region: exact coordinates residing universally in **Block 2, Head 2, Channel 53 ($D_{53}$)** preserved perfect structural adherence to the hypothesis. Exactly $20$ continuous spatial tokens within this locus anchored natively to the rigid baseline generation rate of exactly $4$ discrete spikes, utterly invariant of stimulus properties or organic noise over the full natural dataset loop. 

From an academic perspective, identifying strict mathematically stimulus-agnostic logic blocks deep within natural vision transformers highlights systemic training flaws or pathological over-regularization pathways. However, from a hardware engineering perspective targeting Spiking Neuromorphic deployment, isolating only 20 spatial parameters across a grid of $\approx 5.2 \times 10^5$ LIF boundaries mathematically concludes that the structural implementation overhead required to host explicit $M$ masks across chip clusters wildly eclipses the minuscule sub-milliwatt power reclamation yielded by suppressing these anomalies, negating scalable baseline-anchored gating logic as an optimal power-reduction doctrine for generalized MaxFormer frameworks.

## 5. Extracted Data Artifacts (Visualizations & Serialization)

To document the explicit mathematical boundary mapping isolating this minute, rigid architecture anomaly, detailed sub-network variables were formally exported:

* **Visual Output (`Images/baseline_ghost_neuron_maps.png`):** 
  The compiled structural arrays uniformly suppress dynamic variances globally across identically charted dimensions indexing block structures sequentially displaying $0$ mapping intensities universally, explicitly punctuated solely by the precise 20-token vector string dynamically isolated operating distinctly inside Block 2 natively mapped across corresponding Spatial bounding constants.
* **Serialized Output (`JSONs/baseline_ghost_neurons.json`):** 
  Crucially detailing specific extraction boundaries identifying structural limits required for formalizing constraint hypotheses, the compiled JSON directly targets anomalies by indexing individual tensor offsets:
  * Identifies `"anchored_neurons_count": 20` mapping to native `Block_2/Head_2`.
  * Instantiates the exact operational matrix dictionary (`anchored_neurons_firing_rates`) capturing physical dimensional arrays (e.g. `"D53_N159": 4`, establishing unequivocally that depth index `53` and spatial offset `159` functionally hallucinates exactly $4$/$4$ spikes structurally ignoring any variance bounds across the benchmark ImageNet tests stream unconditionally.)

### Formal Analysis Data Grid (Block vs Head)

| Block | Head 0 | Head 1 | Head 2 | Head 3 | Head 4 | Head 5 |
|---|---|---|---|---|---|---|
| Block 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Block 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Block 2 | 0 | 0 | 20 | 0 | 0 | 0 |
| Block 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Block 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| Block 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| Block 6 | 0 | 0 | 0 | 0 | 0 | 0 |

#### Grid Interpretation & Conclusion
- **Value Representation (Plain English):** This is a simple count. It shows exactly how many individual neurons in this block get 'stuck' at their resting baseline rate, refusing to change their behavior even when scanning 100 completely different natural images.
- **Architectural Conclusion (Plain English):** The table proves that almost no part of the network gets stuck—except for exactly 20 neurons inside Block 2, Head 2. Because this flaw only affects 20 neurons out of millions, it tells engineers that building a custom hardware filter just to fix these specific 'anchor' neurons is basically a waste of time and money.
