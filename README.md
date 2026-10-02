# Wi-Fi Person Re-Identification (Wi-PER81)

## Motivation & Prior Issues

The original repository baseline contained several issues that distorted evaluation metrics:

1. **Closed-Set Evaluation:** The test set used the same 81 identities present during training.
2. **Identity Parsing Bug:** Flawed string indexing grouped separate individuals into single labels during evaluation, generating false positives.
3. **mAP Calculation:** The evaluation script omitted Rank-1 hits from the Mean Average Precision calculation.
4. **Missing L2 Normalization:** Output feature embeddings lacked normalization, causing latent vectors of unseen IDs to collapse toward zero and triggering division-by-zero errors in cosine similarity calculations.

This repository enforces a strict open-set evaluation split and transitions from 2D heatmap CNNs to native Graph Neural Networks (GNNs) trained via metric learning.

---

## Repository Structure

The project is structured into four sequential experimental pipelines, each containing its respective data processing, dataset generation, and benchmarking notebooks:

```text
├── graph_representation_pipeline/           # V4/V4.1: Native GNN pipeline
│   ├── output/
│   ├── WiPER-Benchmarking_v4.1_100epochs.ipynb
│   ├── WiPER-Benchmarking_v4.ipynb
│   ├── WiPER-Benchmarking_v4_50epochs.ipynb
│   ├── WiPER-Benchmarking_v4_100epochs.ipynb
│   ├── WiPER-DataProcessing.ipynb
│   └── WiPER-DatasetGeneration_v3.ipynb
├── openset_pipeline/                        # V2/V2.5: Open-set split baselines
│   ├── output_new_split/
│   ├── WiPER-Benchmarking_v2.5.ipynb
│   ├── WiPER-Benchmarking_v2.ipynb
│   ├── WiPER-DataProcessing.ipynb
│   └── WiPER-DatasetGeneration_v2.ipynb
├── openset_tripletloss_pipeline/            # V3: Open-set metric learning baseline
│   ├── output_v3_run/
│   ├── WiPER-Benchmarking_v3.ipynb
│   ├── WiPER-DataProcessing.ipynb
│   └── WiPER-DatasetGeneration_v2.ipynb
├── original_pipeline/                       # V1: Original paper reproduction
│   ├── output_original_split/
│   ├── WiPER-Benchmarking_original.ipynb
│   ├── WiPER-Benchmarking_original_correct_map.ipynb
│   ├── WiPER-DataProcessing.ipynb
│   └── WiPER-DatasetGeneration_original.ipynb

```

---

## Pipeline Overview

* **`original_pipeline/` (V1):** Original paper setup (closed-set split, label slicing bugs, and flawed mAP logic).


* **`openset_pipeline/` (V2 / V2.5):**
* `v2`: Introduces a 50/50 open-set split (40 train IDs, 41 test IDs) while retaining classification losses (CrossEntropy + CosineEmbeddingLoss) without normalization.


* `v2.5`: Fixes ID parsing, corrects the mAP calculation, and introduces L2 feature normalization to prevent latent collapse.




* **`openset_tripletloss_pipeline/` (V3 - CNN Baseline):** Replaces classification losses with Hard Triplet Loss and uses a `PKSampler` (P=8, K=4) over 2D CNN heatmaps.


* **`graph_representation_pipeline/` (V4 / V4.1 - GNN Architecture):**
* `v4`: Transitions from image spectrograms to PyTorch Geometric, modeling CSI subcarriers directly as nodes in a fully connected graph.


* `v4.1`: Replaces attention entropy penalties with L2 attention regularization, introduces vectorized similarity evaluation, and scales training to 100 epochs.





---

## GNN Architecture (V4 / V4.1)

* **Graph Topology:** Subcarriers are modeled as 52 nodes in a fully connected graph (2,704 edges) to preserve non-adjacent multipath frequency correlations.
* **Positional Encoding:** An `nn.Embedding(52, 16)` layer injects physical subcarrier order into the permutation-invariant GNN.
* **Temporal 1D CNN:** Slides across the 100-packet sequence per node to capture gait signatures prior to graph routing.
* **Multi-Faceted Readout:** Concatenates `global_mean_pool` and `global_max_pool` embeddings.
* **Vectorized Evaluation:** Pre-extracts gallery feature representations in batches to avoid GPU OOM, running vectorized cosine similarity over probe queries with array-based CMC tracking.



---

## Benchmark Results

| Pipeline | Notebook | Loss / Setup | Split | Epochs | Rank-1 (%) | mAP (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **V1** | `WiPER-Benchmarking_original.ipynb`<br> | CrossEntropy + Cosine | Closed-Set | 20 | 70.3* | 70.6* |
| **V2.5** | `WiPER-Benchmarking_v2.5.ipynb`<br> | CrossEntropy + Cosine | Open-Set | 20 | 70.7 | 55.1 |
| **V3** | `WiPER-Benchmarking_v3.ipynb`<br> | Hard Triplet Loss | Open-Set | 20 | 47.8 | 29.9 |
| **V4** | `WiPER-Benchmarking_v4_50epochs.ipynb`<br> | Triplet + GAT | Open-Set | 50 | 75.1 | 62.8|
| **V4.1** | `WiPER-Benchmarking_v4.1_100epochs.ipynb`<br> | Triplet + L2 Reg | Open-Set | 100 | **82.7**<br> | **70.6**<br> |

**V1 metrics are invalidated by data leakage and evaluation script bugs.*

---

## Planned Experiments

* Tuning the number of attention heads in the GAT layers.
* Margin tuning for the Hard Triplet Loss objective.
