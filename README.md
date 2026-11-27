# Wi-Fi Person Re-Identification (Wi-PER81)

## Motivation & Prior Issues

The original repository baseline contained several issues that distorted evaluation metrics:

1. **Closed-Set Evaluation:** The test set used the same 81 identities present during training.
2. **Identity Parsing Bug:** Flawed string indexing grouped separate individuals into single labels during evaluation, generating false positives.
3. **mAP Calculation:** The evaluation script omitted Rank-1 hits from the Mean Average Precision calculation.
4. **Missing L2 Normalization:** Output feature embeddings lacked normalization, causing latent vectors of unseen IDs to collapse toward zero and triggering division-by-zero errors in cosine similarity calculations.

This repository enforces a strict open-set evaluation split, systematically ablates data representations (RGB heatmaps vs. raw `float32` tensors), and transitions from 2D Siamese CNNs to native Graph Neural Networks (GNNs) trained via metric learning, culminating in a multi-channel (Amplitude + Phase) graph architecture. Architectures are compared at equal temporal window (100 vs. 200 packets), in amplitude-only and amplitude + phase configurations, together with a study of phase sanitization and of branch separation (Two-Stream).

## Limitations of the Original Paper

In the original paper, although Wi-Fi signals intrinsically possess both amplitude and phase properties, the re-identification approach focused solely on signal magnitude:

- **Data Extraction:** The baseline system derived amplitude values by calculating them from the real and imaginary components of the Channel State Information (CSI).
- **Neural Network Input:** The foundational Siamese model relied exclusively on signal magnitude heatmaps rendered as 8-bit RGB PNG images from IQR-sanitized amplitude values.
- **Dataset Potential:** While the original neural networks exploited only amplitude heatmaps, the Wi-PER81 dataset includes raw complex CSI matrices, which mathematically contain the full phase information.

Our phase-integrated pipelines break this limitation by extracting raw phase alongside amplitudes. The experiments show (i) the bottleneck of RGB heatmap quantization (switching CNNs to raw `float32` tensors yields **+26.6% Rank-1** in amplitude-only, 57.6% → 84.2%), and (ii) that the phase is exploited much more consistently by Graph Attention Networks than by 2D CNNs, which are penalized by raw phase on short windows.

## Repository Structure

Each experimental pipeline is organized in its own folder containing standardized notebooks (`processing.ipynb`, `generator.ipynb`, `benchmark*.ipynb`). Local version suffixes (`_v1`, `_v2`, …) are used only when multiple variants exist within the same folder. Datasets and generated `output*/` directories are created locally by the notebooks and are not tracked.

```text
.
├── README.md
├── requirements.txt
├── original_paper.pdf                                # Original Wi-PER paper
├── update-tlmgr-latest.sh                            # TeX Live (tlmgr) update helper script
├── original_pipeline/                                # Original paper reproduction (closed-set)
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── benchmark_v1.ipynb                            # Original baseline (bugged mAP)
│   └── benchmark_v2.ipynb                            # Corrected mAP evaluation
├── openset_pipeline/                                 # Open-set split baselines (classification loss)
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── benchmark_v1.ipynb                            # 40/41 open-set split (unnormalized)
│   └── benchmark_v2.ipynb                            # Fixed ID parsing, mAP, and L2 norm
├── openset_tripletloss_pipeline/                     # Open-set CNN metric learning (RGB Amp-only)
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── generator_200pks.ipynb                        # 200-packet window generator
│   ├── benchmark_v1.ipynb                            # Initial Hard Triplet Loss CNN (20 epochs)
│   ├── benchmark_v2.ipynb                            # Intermediate tuning
│   ├── benchmark_v3.ipynb                            # Upgraded Siamese CNN (100 epochs)
│   └── benchmark_200pks.ipynb                        # Upgraded CNN, 200-packet window
├── openset_tripletloss_pipeline_w_phase/             # Open-set CNN metric learning (RGB Amp + Phase)
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── generator_v2.ipynb
│   ├── benchmark.ipynb                               # Light-Gated Two-Stream CNN (6-ch RGB)
│   └── benchmark_v2.ipynb
├── openset_raw_tripletloss_pipeline/                 # Open-set CNN on raw float32 tensors (Amp-only)
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── generator_v2.ipynb
│   ├── benchmark.ipynb                               # Raw 1-channel float32 CNN, 100 packets
│   ├── benchmark_200pks.ipynb                        # Raw CNN, 200 packets
│   ├── benchmark_attn.ipynb                          # CNN + Attention on subcarriers, 100 packets
│   ├── benchmark_attn_200pks.ipynb                   # CNN + Attention on subcarriers, 200 packets
│   └── benchmark_norm.ipynb                          # Input normalization variant
├── openset_raw_tripletloss_pipeline_w_phase/         # Open-set CNN on raw float32 tensors (Amp + Phase)
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── generator_v2.ipynb
│   ├── benchmark.ipynb                               # Raw 2-channel float32 CNN, 100 packets
│   ├── benchmark_200pks.ipynb                        # Raw CNN, 200 packets
│   ├── benchmark_attn.ipynb                          # CNN + Attention on subcarriers, 100 packets
│   └── benchmark_attn_200pks.ipynb                   # CNN + Attention on subcarriers, 200 packets
├── graph_representation_pipeline/                    # Native GNN pipeline (Amp-only) & ablations
│   ├── backup/                                       # Best GNN checkpoints + run notebooks
│   │   ├── r1_809_map_675/
│   │   ├── r1_824_map_699/
│   │   └── r1_834_mAP_703/
│   ├── graph_pipeline_report.pdf
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── generator_v2.ipynb
│   ├── benchmark_v1.ipynb ... benchmark_v9.ipynb     # Iterative GNN baselines & ablations
│   ├── benchmark_v10_200p.ipynb                      # 200-packet window run
│   └── benchmark_sanitization_{none,iqr,hampel}_{100,200}pks_v2.ipynb   # Sanitization ablation
├── graph_representation_pipeline_w_phase/            # Native GNN with Phase integration
│   ├── processing.ipynb                              # Extracts raw phase alongside amplitude
│   ├── generator.ipynb                               # Stacks synchronized Amp + Phase graphs
│   ├── benchmark.ipynb                               # 100-packet Multi-channel GNN
│   ├── benchmark_sanitization_{none,iqr,hampel}_100pks_v2.ipynb         # Sanitization ablation (100 pkt)
│   ├── benchmark_sanphase_100pks.ipynb               # Linear Phase Sanitization (100 pkt)
│   └── data_pipeline_updates/                        # 200-packet sliding window (stride=50)
│       ├── processing.ipynb
│       ├── generator.ipynb
│       ├── benchmark_v1.ipynb                        # 200-window Multi-channel GNN (raw phase)
│       ├── benchmark_v2.ipynb                        # Secondary 200-window variant
│       ├── benchmark_2stream.ipynb                   # GNN Two-Stream (separate Amp/Phase 1D CNN branches)
│       ├── benchmark_sanitization_{none,iqr,hampel}_200pks_v2.ipynb     # Sanitization ablation (200 pkt)
│       ├── benchmark_sanphase_200pks.ipynb           # Linear Phase Sanitization + global IQR clipping
│       └── benchmark_sanphase_hampel_200pks.ipynb    # Linear Phase Sanitization + temporal Hampel filter (Best)
├── graph_representation_pipeline_prometheus_lab_data/ # Cross-environment GNN experiments
│   ├── processing.ipynb
│   ├── generator_v1.ipynb                            # Amplitude-only graph generator
│   ├── generator_v2.ipynb                            # Amp + Phase graph generator
│   └── benchmark_v1.ipynb ... benchmark_v10.ipynb    # Iterative Prometheus Lab benchmarks
└── utils/                                            # Visualization scripts
    ├── graph-comparison-dynamic.py
    └── graph-comparison-fixed.py
```

## Pipeline Overview

- **`original_pipeline/`:** Reproduces the original paper setup (`benchmark_v1`: closed-set split, label slicing bugs, and flawed mAP logic; `benchmark_v2`: corrected mAP).
- **`openset_pipeline/`:**
  - `v1`: Introduces a strict 40-train / 41-test open-set identity split while retaining classification losses (`CrossEntropy` + `CosineEmbeddingLoss`) without normalization.
  - `v2`: Fixes ID parsing, corrects mAP calculation, and adds L2 feature normalization to prevent latent collapse.
- **`openset_tripletloss_pipeline/` (CNN RGB Amp-Only):** Replaces classification losses with Hard Triplet Loss and a `PKSampler` ($P=8, K=4$) over 3-channel RGB amplitude heatmaps, scaling from 20 epochs (`v1`) to an upgraded 100-epoch architecture (`v3`).
- **`openset_tripletloss_pipeline_w_phase/` (CNN RGB Amp + Phase):** Extends the RGB heatmap CNN to 6 channels (3 RGB Amplitude + 3 RGB Phase). Replaces naive early fusion, which suffers from colormap phase-wrapping noise, with a lightweight phase branch and per-feature Gated Residual Fusion trained from scratch.
- **`openset_raw_tripletloss_pipeline/` (CNN Raw `float32` Amp-Only):** Bypasses `plt.imshow()` PNG conversion and resizing completely, feeding raw 1-channel `float32` tensors directly into the 2D Siamese CNN. Includes a **CNN + Attention** variant over the subcarriers (`benchmark_attn*`), introduced to test whether the GNN advantage comes from the attention mechanism alone, and 100- vs. 200-packet windows.
- **`openset_raw_tripletloss_pipeline_w_phase/` (CNN Raw `float32` Amp + Phase):** Same setup with 2-channel `[2, T, 52]` tensors (Amplitude + raw Phase), with CNN, CNN + Attention, and 100- vs. 200-packet windows. Together with the amplitude-only folder, it completes the symmetric ablation matrix and isolates the impact of data representation from neural architecture.
- **`graph_representation_pipeline/` (GNN Amp-Only & Ablations):**
  - `v1`–`v5`: Transitions to a native `GATConv` architecture with L2 attention regularization, vectorized evaluation, and 100-epoch training (`v5` reaches 83.4% Rank-1; checkpoint in `backup/r1_834_mAP_703`).
  - `v6`–`v9`: Architectural ablations (attention heads, loss/LR tuning, temporal resolution, multi-scale 1D CNNs) confirming an amplitude-only GNN performance ceiling around ~83.4% Rank-1.
  - `v10_200p` and `benchmark_sanitization_*`: 200-packet window and amplitude sanitization ablations (none / IQR / Hampel).
- **`graph_representation_pipeline_w_phase/` (GNN Amp + Phase):**
  - `benchmark.ipynb`: Integrates raw phase with sanitized amplitude as a 2-channel node feature matrix (`[52, 2, 100]`) for the GNN.
  - `benchmark_sanphase_100pks`: Linear Phase Sanitization on the 100-packet window (IQR and Hampel variants).
  - `data_pipeline_updates/`: Expands the temporal window from 100 to 200 packets using a sliding window (`stride=50`), capturing full human macro-gait cycles and expanding training/query volume. Hosts the 200-packet raw-phase GNN (`v1`, `v2`), the **Two-Stream** GNN (`benchmark_2stream`) and the phase sanitization experiments (`benchmark_sanphase_200pks` with global IQR clipping, `benchmark_sanphase_hampel_200pks` with temporal Hampel filter).
- **`graph_representation_pipeline_prometheus_lab_data/`:** Cross-environment validation suite adapting the GNN pipeline to the custom Prometheus Lab CSI dataset (`generator_v1`/`v2` and iterative benchmarks `v1`–`v10`).

## GNN Architecture (`graph_representation_pipeline_w_phase`)

- **Graph Topology:** Subcarriers are modeled as 52 nodes in a fully connected graph (2,704 edges) to preserve non-adjacent multipath frequency correlations.
- **Multi-Channel Node Features:** Each node accepts a synchronized 2-channel sequence ($in\_channels=2$) combining IQR-sanitized Amplitude and Phase (raw or sanitized).
- **Positional Encoding:** An `nn.Embedding(52, 16)` layer injects physical subcarrier index order into the permutation-invariant GNN.
- **Dynamic Temporal 1D CNN:** Slides across the temporal sequence per node (100 or 200 packets) to extract motion signatures prior to graph attention routing, using `AdaptiveAvgPool1d(1)` to absorb varying window lengths.
- **Multi-Faceted Readout:** Concatenates `global_mean_pool` and `global_max_pool` embeddings followed by L2 normalization.
- **Vectorized Evaluation:** Pre-extracts gallery embeddings in batches to avoid GPU OOM and computes vectorized cosine similarity across all probe queries.
- **Phase Sanitization (optional):** Linear Phase Sanitization applied on the graphs, combined with either global IQR clipping or a temporal Hampel filter for impulsive spikes.
- **Two-Stream Variant (optional):** Separate 1D convolutional branches for amplitude and phase, fused at node level before graph attention (currently tested on raw phase only).

---

## Benchmark Results

All results are open-set (40 train / 41 test identities). Rank-1 and mAP in %.

### Overall progression

| Pipeline (Folder) | Version | Architecture & Input | Loss / Setup | Split | Epochs | Rank-1 (%) | mAP (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `original_pipeline` | `v1` | CNN — RGB PNG (Amp) | CrossEntropy + Cosine | Closed-Set | 20 | 70.3* | 70.6* |
| `openset_pipeline` | `v2` | CNN — RGB PNG (Amp) | CrossEntropy + Cosine + L2 | Open-Set | 20 | 70.7 | 55.1 |
| `openset_tripletloss_pipeline` | `v1` | CNN — RGB PNG (Amp) | Hard Triplet Loss | Open-Set | 20 | 47.8 | 29.9 |
| `openset_tripletloss_pipeline` | `v3` | Upgraded CNN — RGB PNG (Amp) | Hard Triplet Loss | Open-Set | 100 | 57.6 | 48.1 |
| `openset_tripletloss_pipeline_w_phase` | `—` | Light-Gated CNN — RGB PNG (Amp+Phase) | Hard Triplet Loss | Open-Set | 100 | 58.3 | 45.5 |
| `openset_raw_tripletloss_pipeline_w_phase` | `—` | Upgraded CNN — Raw `float32` (Amp+Phase, 100-pkt) | Hard Triplet Loss | Open-Set | 100 | 78.3 | 63.0 |
| `openset_raw_tripletloss_pipeline` | `—` | Upgraded CNN — Raw `float32` (Amp, 100-pkt) | Hard Triplet Loss | Open-Set | 100 | 84.2 | 72.6 |
| `graph_representation_pipeline` | `v5` | GNN (`GATConv`) — Raw Graph (Amp, 100-pkt) | Triplet + L2 Reg | Open-Set | 100 | 83.4 | 70.3 |
| `graph_representation_pipeline_w_phase` | `—` | GNN (`GATConv`) — Raw Graph (Amp+Phase, 100-pkt) | Triplet + L2 Reg | Open-Set | 100 | 84.9 | 71.1 |
| `graph_representation_pipeline_w_phase/data_pipeline_updates` | `v1` | GNN (`GATConv`) — Raw Graph (Amp+Phase, 200-pkt) | Triplet + L2 Reg | Open-Set | 100 | 85.8 | 74.2 |
| `graph_representation_pipeline_w_phase/data_pipeline_updates` | `sanphase_hampel_200pks` | **GNN (`GATConv`) — Raw Graph (Amp+Sanitized Phase w/ Hampel, 200-pkt)** | **Triplet + L2 Reg** | **Open-Set** | **100** | **86.4** | **76.9** |

**`original_pipeline` (`v1`) metrics are invalidated by closed-set data leakage and evaluation script bugs.*

### Architecture comparison at equal temporal window (Rank-1 % / mAP %)

Window = 100 vs. 200 packets (`stride=50`), amplitude-only vs. amplitude + raw phase. `—` = mAP not reported.

| Model | 100 pkt — Amp | 100 pkt — Amp+Phase | 200 pkt — Amp | 200 pkt — Amp+Phase |
| --- | --- | --- | --- | --- |
| CNN — Raw `float32` | 84.15 / 72.57 | 78.29 / 62.98 | 79.48 / — | 80.92 / — |
| CNN + Attention — Raw `float32` | 82.20 / — | 78.54 / — | 83.36 / 69.64 | 84.65 / 73.55 |
| GNN (`GATConv`) | 83.40 / 70.30 | 84.88 / 71.14 | 82.93 / 69.66 | **85.80 / 74.20** |

Effect of adding raw phase (Δ Rank-1):

| Model | 100 pkt | 200 pkt |
| --- | --- | --- |
| CNN | −5.86 | +1.44 |
| CNN + Attention | −3.66 | +1.29 |
| GNN | +1.48 | +2.87 |

### GNN phase variants (Rank-1 % / mAP %)

| Phase processing | 100 pkt | 200 pkt |
| --- | --- | --- |
| Raw phase, single-stream | 84.88 / 71.14 | 85.80 / 74.20 |
| Raw phase, Two-Stream | — | 83.36 / 70.87 |
| Linear Phase Sanitization + global IQR clipping | 83.17 / 70.06 | 84.36 / **77.79** |
| Linear Phase Sanitization + temporal Hampel filter | 82.20 / 70.53 | **86.37** / 76.90 |

## Key Findings

- **Attention vs. graph representation:** A CNN + Attention over subcarriers was added to check whether the GNN advantage stems only from the attention mechanism. On 100 packets the CNNs are penalized by raw phase (CNN −5.86%, CNN + Attention −3.66% Rank-1), whereas the GNN already benefits from it (+1.48%). On 200 packets all models benefit from phase, but the GNN gains the most (+2.87%, from 82.93% to 85.80% Rank-1, 74.20% mAP). This suggests that the GNN advantage is not due exclusively to attention, and that the graph representation provides an additional contribution. In amplitude-only, however, the raw CNN (84.15% Rank-1 at 100 packets) is competitive with the GNN (83.40%).
- **Two-Stream vs. Single-Stream (200 pkt, raw phase):** Separating the amplitude and phase 1D CNN branches speeds up early convergence (59.68% Rank-1 at epoch 1 vs. 40.89% single-stream) but plateaus at 83.36% Rank-1 / 70.87% mAP, below joint temporal fusion (85.80% / 74.20%). With raw phase, jointly extracting amplitude and phase information along the temporal axis appears more effective.
- **Phase sanitization:** With 100 packets, both IQR and Hampel variants speed up early learning but stay slightly below the raw-phase peak (84.88% / 71.14%). With 200 packets, global IQR clipping reaches 84.36% Rank-1 (77.79% mAP), while the temporal Hampel filter reaches the best result so far (86.37% Rank-1, 76.90% mAP). On longer windows, local filtering of impulsive spikes seems to preserve the temporal phase dynamics better than global clipping.
- **Contribution of phase depends on configuration:** Its benefit varies with architecture, window length and preprocessing, motivating adaptive fusion strategies (see below).

---

## Future Work & Ongoing Experiments

- **Two-Stream with Sanitized Phase:** Running GNN and raw CNN Two-Stream variants on Linear-sanitized phase (IQR / Hampel), since Two-Stream has so far been tested on raw phase only.
- **Gated Adaptive Fusion (Amp ↔ Phase):** Node-level and/or graph-level gated fusion that dynamically weights the contribution of amplitude and phase, so the network can learn to modulate phase information according to the observed signal and prevent noisy components from degrading the final representation.
- **Phase Unwrapping & CFO Removal:** Linear Phase Sanitization has been explored; the next step is mathematically unwrapped phase trajectories to remove Carrier Frequency Offset (CFO) and provide smooth kinematic curves.
- **Graph Edge Sparsity Prior:** Replacing the fully connected graph topology with a k-NN or localized subcarrier adjacency matrix to filter frequency noise.
- **Temporal Data Augmentation:** Implementing random temporal cropping, subcarrier masking, and Gaussian noise injection during training to reduce overfitting to static multipath fading.
- **Cross-Dataset Validation *(In Progress)*:** Finalizing the `graph_representation_pipeline_prometheus_lab_data` benchmarks to evaluate environmental generalization across distinct indoor spaces.