# Wi-Fi Person Re-Identification (on Wi-PER81)

## 📌 Context & Motivation

The original implementation presented several critical issues that artificially inflated the evaluation metrics and masked severe overfitting:
1. **Flawed Re-ID Protocol (Closed-Set):** The original evaluation protocol tested the model on the exact same 81 identities used during training.
2. **ID Slicing Bug:** A bug in the label parsing logic grouped distinct individuals under the same identity during testing, resulting in false positive matches.
3. **mAP Calculation Error:** The Mean Average Precision (mAP) function excluded Rank-1 (perfect matches) from its computation.
4. **Latent Space Collapse:** The absence of L2 normalization caused vectors of unseen identities to collapse to zero, leading to division-by-zero errors in cosine similarity calculations.

This repository addresses these flaws by implementing a robust Open-Set evaluation protocol and introducing a pure Metric Learning approach.

## 🚀 Repository Structure & Pipeline Versions

The repository documents the evolution of the pipeline across four distinct versions to highlight the impact of the bug fixes and architectural changes:

*   **`WiPER-Benchmarking_original.ipynb` (V1):** The original paper's implementation. Contains the closed-set protocol, ID slicing bugs, and flawed mAP calculation.
*   **`WiPER-Benchmarking_v2.ipynb` (V2):** Introduces a rigorous **Open-Set Split (50/50)**. 40 identities are used for training, and 41 completely unseen identities are used for testing. However, it still uses the original classification loss (CrossEntropy + CosineEmbeddingLoss) and lacks L2 normalization.
*   **`WiPER-Benchmarking_v2_5.ipynb` (V2.5):** The "Structurally Fixed" version. Maintains the Open-Set split and original losses, but fixes the ID slicing bug, corrects the mAP formula, and introduces **L2 Normalization** to prevent latent space collapse. *Note: This version clearly exposes the severe overfitting caused by using closed-set classification losses for Re-ID.*
*   **`WiPER-Benchmarking_v3.ipynb` (V3 - Current Baseline):** The **Pure Metric Learning** approach. It entirely drops the classification heads and CrossEntropy loss. It introduces a `PKSampler` (P=8, K=4) for balanced dynamic batching and optimizes the network using exclusively the **Hard Triplet Loss**. This version yields a stable, organic learning curve that truly generalizes to unseen identities.

## 📊 Evaluation & Metrics

Due to the heavy computational cost of the original 50-epoch training schedule (>28 hours), recent experiments (V2, V2.5, V3) were optimized to run for **20 epochs**. This provides a sufficient window to analyze loss convergence and metric trends while allowing for faster iteration.

### Results Comparison (20 Epochs)

| Pipeline Version | Training Strategy | Best Rank-1 | Best mAP | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **V1 (Original)** | CrossEntropy + Cosine (Closed-Set) | *70.3%* | *70.6%* | Metrics invalidated by structural bugs. |
| **V2.5 (Bug Fix)** | CrossEntropy + Cosine (Open-Set) | 70.7% | 55.1% | Achieves best results at Epoch 1, then suffers from severe overfitting and degrades. |
| **V3 (Metric Learning)** | **Hard Triplet Loss** (Open-Set) | **47.8%** | **29.9%** | Highly stable training. Loss decreases consistently, and performance improves organically over time. |

*Note: While absolute metric values are lower in V3 compared to the overfitted V2.5, V3 represents closer mathematically sound State-of-the-Art for generalizable Wi-Fi Re-ID on this dataset.*

## 🛠️ Data Pre-processing

The repository also includes the data generation scripts used to transition from raw CSI matrices to the visual heatmaps fed into the Siamese Network:
*   `WiPER-Benchmarking_original_2.ipynb` & `_3.ipynb`
*   `WiPER-Benchmarking_v3_2.ipynb`

These scripts handle the IQR sanitization of amplitude values, the strict separation of train/test identities, and the rendering of the final PNG heatmaps.