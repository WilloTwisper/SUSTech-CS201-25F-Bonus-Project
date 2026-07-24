# SUSTech CS201 - Discrete Mathematics 2025 Fall Bonus Project

## Topological Consistency Verification and Adaptive Reconstruction in Medical Image Segmentation

**Score: 4/5**

This project addresses a fundamental problem in medical image segmentation: deep learning models (nnUNet) treat pixels independently and frequently produce **fractured (disconnected) predictions** for thin anatomical structures. By modeling the image as a **Grid Graph** and applying **graph-theoretic invariants** (Betti numbers, Euler characteristic, connected components), we develop a hybrid discrete-continuous pipeline that restores topological correctness to nnUNet segmentation masks.

**Core Insight:** A valid anatomical segmentation must satisfy \(b_0(G) = 1\) (a single connected component). We evolve through five methods, from simple graph pruning to a dual-regression adaptive model that achieves **100% topological accuracy** and **+1.38% Dice improvement**.

---

## Pipeline Overview

```
nnUNet Predictions
       |
       v
evaluate_fragmentation.py  ──► Topological Audit (identifies most fractured class)
       |
       v
topo_analysis_v1.py        ──► Discrete Graph Pruning (largest component only)
topo_analysis_v2.py        ──► Morphological Closing + Pruning
topo_analysis_v3.py        ──► Polynomial Curve Fitting (fixed thickness)
topo_analysis_v4.py        ──► Global Extrapolation + Structure-Preserving Fusion
topo_analysis_v5.py        ──► [BEST] Dual-Regression Adaptive Model
       |
       v
main.tex                   ──► LaTeX Report
```

---

## Methodology Evolution

### Step 0: Topological Audit (`evaluate_fragmentation.py`)

Scans all 256 validation predictions, computes \(\mathbb{E}[b_0]\) (average connected components) per semantic class. **Class 6** identified as the most fragmented layer with \(\mathbb{E}[b_0] \approx 2.7\).

### Phase I: Discrete Graph Pruning (V1)

- **Algorithm:** Connected Component Labeling (CCL) + largest component retention
- **Result:** Achieves 100% topology but **decreases Dice** (-0.0019) by discarding valid anatomical data
- **Limitation:** Enforces \(b_0=1\) by deleting limbs rather than reconstructing

### Phase II: Local Morphological Closing (V2)

- **Algorithm:** Binary closing (disk radius=10) followed by V1 pruning
- **Limitation:** Gaps exceed 50px; larger kernels would merge distinct anatomical layers
- **Conclusion:** Local morphological operators insufficient for large-scale fractures

### Phase III: Global Polynomial Reconstruction (V3/V4)

- **Algorithm:** 2nd-degree polynomial fit \(y = ax^2 + bx + c\) to foreground pixels, rasterization, OR fusion
- **V3:** Fixed line thickness (4px)
- **V4:** Full-width extrapolation + thin bridge (3px)
- **Limitation:** Fixed-thickness bridge creates unnatural bottleneck

### Phase IV: Dual-Regression Adaptive Model (V5 - Best)

- **Trajectory Regression:** \(y = f(x)\) — 2nd-degree polynomial fit for the central path
- **Caliber Regression:** \(r = g(x)\) — 2nd-degree polynomial fit for local thickness (via `cv2.distanceTransform`)
- **Adaptive Rasterization:** Circles of radius \(r\) at each \((x, f(x))\), creating a **biologically natural varying-thickness bridge**
- **Fusion:** \(G_{final} = G_{original} \cup G_{bridge}\) + LCC pruning
- **Result: 100% topological accuracy, +0.0138 Dice improvement (+1.38%)**

---

## Results

| Method | Topo Accuracy | Mean Dice | \(\Delta\) Dice |
|--------|:------------:|:---------:|:--------------:|
| Baseline (nnUNet) | 67.58% | 0.2784 | — |
| V1 (Pruning) | 100.00% | 0.2765 | -0.0019 |
| V2 (Morphological) | 100.00% | 0.2765 | -0.0019 |
| V3 (Fixed Fit) | 100.00% | — | — |
| V4 (Extrapolation) | 100.00% | 0.2851 | +0.0067 |
| **V5 (Adaptive)** | **100.00%** | **0.2922** | **+0.0138** |

---

## Usage

### 1. Prepare Data

```bash
python prepare_data.py
```
Copies nnUNet predictions and ground-truth masks into `Project_Data/`.

### 2. Run Topological Audit

```bash
python evaluate_fragmentation.py
```
Identifies the most fragmented class (Class 6 recommended).

### 3. Apply Topological Correction

```bash
# Individual methods
python topo_analysis_v1.py
python topo_analysis_v2.py
python topo_analysis_v3.py
python topo_analysis_v4.py

# Best method
python topo_analysis_v5.py
```

### 4. Compile Report

```bash
xelatex main.tex
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, polynomial fitting |
| `opencv-python` | Image I/O, distance transform, drawing |
| `scikit-image` | Connected component labeling, Euler number |
| `matplotlib` | Result visualization |
| `tqdm` | Progress bars |

---

## Project Structure

```
.
��   prepare_data.py              # Data preparation (nnUNet -> local)
��   evaluate_fragmentation.py    # Topological audit (b0 per class)
��   topo_analysis_v1.py          # Largest component pruning
��   topo_analysis_v2.py          # Morphological closing + pruning
��   topo_analysis_v3.py          # Polynomial fitting (fixed thickness)
��   topo_analysis_v4.py          # Global extrapolation + thin bridge
��   topo_analysis_v5.py          # [BEST] Dual-regression adaptive model
��   main.tex                     # LaTeX report
��   Project_Data/
��   ��   Predictions/              # nnUNet prediction masks
��   ��   GroundTruth/              # Ground truth masks
��   ��   Report_Assets_V1/         # V1 visualizations
��   ��   Report_Assets_V2/         # V2 visualizations
��   ��   Report_Assets_V3/         # V3 visualizations
��   ��   Report_Assets_V4/         # V4 visualizations
��   ��   Report_Assets_V5/         # V5 visualizations
```

---

## Data Notice

The OCT dataset used in this project is from the **iMED research group**, authorized strictly for the purpose of this course project. Redistribution is prohibited.

---

## Author

刘家亮 (Jialiang Liu) — SID 12412719 — SUSTech CS201 Discrete Mathematics 2025 Fall