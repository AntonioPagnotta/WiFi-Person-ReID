# Wi-Fi Person Re-Identification (Wi-PER81)

## Motivation & Prior Issues

The original repository baseline contained several issues that distorted evaluation metrics:

1. **Closed-Set Evaluation:** The test set used the same 81 identities present during training.
2. **Identity Parsing Bug:** Flawed string indexing grouped separate individuals into single labels during evaluation, generating false positives.
3. **mAP Calculation:** The evaluation script omitted Rank-1 hits from the Mean Average Precision calculation.
4. **Missing L2 Normalization:** Output feature embeddings lacked normalization, causing latent vectors of unseen IDs to collapse toward zero and triggering division-by-zero errors in cosine similarity calculations.

This repository enforces a strict open-set evaluation split, systematically ablates data representations (RGB heatmaps vs. raw `float32` tensors), and transitions from 2D Siamese CNNs to native Graph Neural Networks (GNNs) trained via metric learning—culminating in a multi-channel (Amplitude + Phase) graph architecture.

## Limitations of the Original Paper

In the original paper, although Wi-Fi signals intrinsically possess both amplitude and phase properties, the re-identification approach focused solely on signal magnitude:

- **Data Extraction:** The baseline system derived amplitude values by calculating them from the real and imaginary components of the Channel State Information (CSI).
- **Neural Network Input:** The foundational Siamese model relied exclusively on signal magnitude heatmaps rendered as 8-bit RGB PNG images from IQR-sanitized amplitude values.
- **Dataset Potential:** While the original neural networks exploited only amplitude heatmaps, the Wi-PER81 dataset includes raw complex CSI matrices, which mathematically contain the full phase information.

Our phase-integrated pipelines break this limitation by extracting raw phase alongside sanitized amplitudes, demonstrating both the bottleneck of RGB heatmap quantization (+20% Rank-1 jump when switching CNNs to raw `float32` tensors) and the architectural advantage of Graph Attention Networks over 2D CNNs.

## Repository Structure

Each experimental pipeline is organized in its own folder containing standardized notebooks (`processing.ipynb`, `generator.ipynb`, `benchmark.ipynb`). Local version suffixes (`_v1`, `_v2`, …) are used only when multiple variants exist within the same folder:

```text
├── acquisizioni/                                    # Custom raw CSI acquisitions
├── assets/
├── data/
├── original_pipeline/                               # Original paper reproduction (closed-set)
│   ├── output_original_split/
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── benchmark_v1.ipynb                           # Original baseline (bugged mAP)
│   └── benchmark_v2.ipynb                           # Corrected mAP evaluation
├── openset_pipeline/                                # Open-set split baselines (classification loss)
│   ├── output_new_split/
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── benchmark_v1.ipynb                           # 40/41 open-set split (unnormalized)
│   └── benchmark_v2.ipynb                           # Fixed ID parsing, mAP, and L2 norm
├── openset_tripletloss_pipeline/                    # Open-set CNN metric learning (RGB Amp-only)
│   ├── output/
│   ├── output_v3_run/
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── benchmark_v1.ipynb                           # Initial Hard Triplet Loss CNN (20 epochs)
│   ├── benchmark_v2.ipynb                           # Intermediate tuning
│   └── benchmark_v3.ipynb                           # Upgraded Siamese CNN (100 epochs)
├── openset_tripletloss_pipeline_w_phase/            # Open-set CNN metric learning (RGB Amp + Phase)
│   ├── output/
│   ├── processing.ipynb
│   ├── generator.ipynb
│   └── benchmark.ipynb                              # Light-Gated Two-Stream CNN (6-ch RGB)
├── openset_raw_tripletloss_pipeline_w_phase/        # Open-set CNN on Raw float32 Tensors (Amp + Phase)
│   ├── output/
│   ├── processing.ipynb
│   ├── generator.ipynb
│   └── benchmark.ipynb                              # Raw 2-channel float32 CNN ([2, 100, 52])
├── graph_representation_pipeline/                   # Native GNN pipeline (Amp-only) & Ablations
│   ├── backup/
│   ├── output/
│   ├── graph_pipeline_report.pdf
│   ├── processing.ipynb
│   ├── generator.ipynb
│   ├── benchmark_v1.ipynb                           # Initial GNN baseline
│   ├── benchmark_v2.ipynb                           # 50-epoch run
│   ├── benchmark_v3.ipynb                           # 100-epoch run
│   ├── benchmark_v4.ipynb                           # L2 attention regularization (100 epochs)
│   ├── benchmark_v5.ipynb                           # Best Amp-only GNN checkpoint
│   ├── benchmark_v6.ipynb                           # Multi-attention heads ablation
│   ├── benchmark_v7.ipynb                           # Loss criterion & learning rate ablation
│   ├── benchmark_v8.ipynb                           # Extended temporal resolution ablation
│   └── benchmark_v9.ipynb                           # Multi-scale 1D CNN ablation
├── graph_representation_pipeline_w_phase/           # Native GNN with Phase Integration
│   ├── data_pipeline_updates/                       # 200-packet sliding window (stride=50)
│   │   ├── output/
│   │   ├── processing.ipynb
│   │   ├── generator.ipynb
│   │   ├── benchmark_v1.ipynb                       # 200-window Multi-channel GNN (Best)
│   │   └── benchmark_v2.ipynb                       # Secondary 200-window variant
│   ├── output/
│   ├── processing.ipynb                             # Extracts raw phase alongside amplitude
│   ├── generator.ipynb                              # Stacks synchronized Amp + Phase graphs
│   └── benchmark.ipynb                              # 100-packet Multi-channel GNN
├── graph_representation_pipeline_prometheus_lab_data/ # Cross-environment GNN experiments
│   ├── output/
│   ├── processing.ipynb
│   ├── generator_v1.ipynb                           # Amplitude-only graph generator
│   ├── generator_v2.ipynb                           # Amp + Phase graph generator
│   └── benchmark_v1.ipynb ... benchmark_v10.ipynb   # Iterative Prometheus Lab benchmarks
├── utils/                                           # Visualization scripts
├── README.md
└── requirements.txt

```

## Pipeline Overview

* **`original_pipeline/`:** Reproduces the original paper setup (`benchmark_v1`: closed-set split, label slicing bugs, and flawed mAP logic; `benchmark_v2`: corrected mAP).
* **`openset_pipeline/`:**
* `v1`: Introduces a strict 40-train / 41-test open-set identity split while retaining classification losses (`CrossEntropy` + `CosineEmbeddingLoss`) without normalization.
* `v2`: Fixes ID parsing, corrects mAP calculation, and adds L2 feature normalization to prevent latent collapse.


* **`openset_tripletloss_pipeline/` (CNN RGB Amp-Only):** Replaces classification losses with Hard Triplet Loss and a `PKSampler` ($P=8, K=4$) over 3-channel RGB amplitude heatmaps, scaling from 20 epochs (`v1`) to an upgraded 100-epoch architecture (`v3`).
* **`openset_tripletloss_pipeline_w_phase/` (CNN RGB Amp + Phase):** Extends the RGB heatmap CNN to 6 channels (3 RGB Amplitude + 3 RGB Phase). Replaces naive early fusion—which suffers from colormap phase-wrapping noise—with a lightweight phase branch and per-feature Gated Residual Fusion trained from scratch.
* **`openset_raw_tripletloss_pipeline_w_phase/` (CNN Raw `float32` Amp + Phase):** Bypasses `plt.imshow()` PNG conversion and resizing completely, feeding raw `[2, 100, 52]` `float32` tensors directly into the 2D Siamese CNN to isolate the impact of data representation from neural architecture.
* **`graph_representation_pipeline/` (GNN Amp-Only & Ablations):**
* `v1`–`v5`: Transitions to a native `GATConv` architecture with L2 attention regularization, vectorized evaluation, and 100-epoch training (`v5` reaches 83.4% Rank-1).
* `v6`–`v9`: Architectural ablations (attention heads, loss/LR tuning, temporal resolution, multi-scale 1D CNNs) confirming an amplitude-only performance ceiling around ~83.4% Rank-1.


* **`graph_representation_pipeline_w_phase/` (GNN Amp + Phase):**
* `benchmark.ipynb`: Integrates raw phase with sanitized amplitude as a 2-channel node feature matrix (`[52, 2, 100]`) for the GNN.
* `data_pipeline_updates/` (`v1`, `v2`): Expands the temporal window from 100 to 200 packets using a sliding window (`stride=50`), capturing full human macro-gait cycles and expanding training/query volume.


* **`graph_representation_pipeline_prometheus_lab_data/`:** Cross-environment validation suite adapting the GNN pipeline to the custom Prometheus Lab CSI dataset (`generator_v1`/`v2` and iterative benchmarks `v1`–`v10`).

## GNN Architecture (`graph_representation_pipeline_w_phase`)

* **Graph Topology:** Subcarriers are modeled as 52 nodes in a fully connected graph (2,704 edges) to preserve non-adjacent multipath frequency correlations.
* **Multi-Channel Node Features:** Each node accepts a synchronized 2-channel sequence ($in\_channels=2$) combining IQR-sanitized Amplitude and raw Phase.
* **Positional Encoding:** An `nn.Embedding(52, 16)` layer injects physical subcarrier index order into the permutation-invariant GNN.
* **Dynamic Temporal 1D CNN:** Slides across the temporal sequence per node (100 or 200 packets) to extract motion signatures prior to graph attention routing, using `AdaptiveAvgPool1d(1)` to absorb varying window lengths.
* **Multi-Faceted Readout:** Concatenates `global_mean_pool` and `global_max_pool` embeddings followed by L2 normalization.
* **Vectorized Evaluation:** Pre-extracts gallery embeddings in batches to avoid GPU OOM and computes vectorized cosine similarity across all probe queries.

---

## Benchmark Results

| Pipeline (Folder) | Version | Architecture & Input | Loss / Setup | Split | Epochs | Rank-1 (%) | mAP (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `original_pipeline` | `v1` | CNN — RGB PNG (Amp) | CrossEntropy + Cosine | Closed-Set | 20 | 70.3* | 70.6* |
| `openset_pipeline` | `v2` | CNN — RGB PNG (Amp) | CrossEntropy + Cosine + L2 | Open-Set | 20 | 70.7 | 55.1 |
| `openset_tripletloss_pipeline` | `v1` | CNN — RGB PNG (Amp) | Hard Triplet Loss | Open-Set | 20 | 47.8 | 29.9 |
| `openset_tripletloss_pipeline` | `v3` | Upgraded CNN — RGB PNG (Amp) | Hard Triplet Loss | Open-Set | 100 | 57.6 | 48.1 |
| `openset_tripletloss_pipeline_w_phase` | `—` | Light-Gated CNN — RGB PNG (Amp+Phase) | Hard Triplet Loss | Open-Set | 100 | 58.3 | 45.5 |
| `openset_raw_tripletloss_pipeline_w_phase` | `—` | Upgraded CNN — Raw `float32` (Amp+Phase) | Hard Triplet Loss | Open-Set | 100 | 78.3 | 63.0 |
| `graph_representation_pipeline` | `v5` | GNN (`GATConv`) — Raw Graph (Amp) | Triplet + L2 Reg | Open-Set | 100 | 83.4 | 70.3 |
| `graph_representation_pipeline_w_phase` | `—` | GNN (`GATConv`) — Raw Graph (Amp+Phase, 100-pkt) | Triplet + L2 Reg | Open-Set | 100 | 84.9 | 71.1 |
| `graph_representation_pipeline_w_phase/data_pipeline_updates` | `v1` | **GNN (`GATConv`) — Raw Graph (Amp+Phase, 200-pkt)** | **Triplet + L2 Reg** | **Open-Set** | **100** | **85.8** | **74.2** |

**`original_pipeline` (`v1`) metrics are invalidated by closed-set data leakage and evaluation script bugs.*

---

## Future Work & Ongoing Experiments

* **Raw `float32` Amplitude-Only CNN Ablation:** Completing the symmetric 3×2 ablation matrix by running a 1-channel raw `float32` CNN (Amplitude-only) alongside the 2-channel raw CNN (`78.3%`).
* **Phase Unwrapping & CFO Removal:** Upgrading from raw phase input to mathematically unwrapped phase trajectories to remove Carrier Frequency Offset (CFO) and provide smooth kinematic curves.
* **Graph Edge Sparsity Prior:** Replacing the fully connected graph topology with a k-NN or localized subcarrier adjacency matrix to filter frequency noise.
* **Temporal Data Augmentation:** Implementing random temporal cropping, subcarrier masking, and Gaussian noise injection during training to reduce overfitting to static multipath fading.
* **Cross-Dataset Validation *(In Progress)*:** Finalizing the `graph_representation_pipeline_prometheus_lab_data` benchmarks to evaluate environmental generalization across distinct indoor spaces.

```
