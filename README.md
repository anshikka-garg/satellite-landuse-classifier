# Satellite Image Land-Use Classifier & Temporal Change Detector

A computer vision system that classifies satellite tiles into 10 land-use
categories using transfer learning (ResNet-18), detects temporal land-cover
change via embedding-based cosine similarity, and presents both through an
interactive Streamlit dashboard with GradCAM explainability.

Built as a capstone project for the Celebal Technologies Data Science
Internship (Celebal Excellence Internship Program, May–July 2026).

## Project Status — All Core Modules Complete

| Module | Status |
|---|---|
| Data pipeline + spatial block split (`01`) | ✅ Done |
| Baseline CNN, naive vs. block split (`02`) | ✅ Done |
| Transfer learning, two-phase fine-tuning (`03`) | ✅ Done |
| Temporal change detection, ROC-based thresholding (`04`) | ✅ Done |
| Cross-dataset error analysis (`05`) | ✅ Done |
| Streamlit dashboard (`app/app.py`) | ✅ Done |
| Bonus A — GradCAM | ✅ Done |
| Bonus B — Multi-threshold toggle | ✅ Done |
| Bonus C — Embedding visualization (t-SNE/UMAP) | Not attempted |
| Bonus D — Class imbalance experiment | Not attempted |
| Final PDF report | ✅ Done |
| Demo video | ✅ Done |

## Results Summary

### Classification — Ablation Table (EuroSAT test set)

| Model | Split | Test Accuracy | Macro-F1 |
|---|---|---|---|
| BaselineCNN (from scratch) | Naive | 91.2% | 0.9068 |
| BaselineCNN (from scratch) | Block | 91.4% | 0.9100 |
| ResNet-18 (frozen backbone) | Naive | — | — |
| ResNet-18 (frozen backbone) | Block | — | — |
| ResNet-18 (two-phase fine-tuned) | Naive | — | — |
| ResNet-18 (two-phase fine-tuned) | Block | — | — |

*(Full numbers in `data/processed/ablation_complete.csv`)*

**Key finding — spatial leakage:** contrary to the initial hypothesis, the
naive random split did not produce meaningfully inflated metrics compared to
the block split (macro-F1 gap of only ~0.003). We attribute this to the
block size (40) being large enough that both splits already achieve
reasonable class-level separation, and to our block split being an
index-locality approximation rather than a true GPS-based spatial split
(EuroSAT's public RGB release carries no coordinates — see
`src/dataset.py` docstring for the full explanation). This is reported as a
genuine empirical finding rather than adjusted to match expectations.

### Change Detection

- **ROC AUC: 0.9418** — cosine similarity between ResNet-18 embeddings
  cleanly separates same-class ("unchanged") from different-class
  ("changed") tile pairs.
- **Optimal threshold (Youden's J statistic): 0.376**
- 5 sample 5×5 region-grid change maps generated and visualized
  (`reports/region_change_heatmap_*.png`)

### Cross-Dataset Generalization (UC Merced holdout)

Evaluated the fine-tuned model on UC Merced Land Use tiles (11 of 21 classes
mapped to EuroSAT's label space — see `UCM_TO_EUROSAT` in `src/dataset.py`).
Top-5 misclassifications and hypotheses documented in
`reports/top_5_errors.png` and `notebooks/05_evaluation.ipynb`.

## Datasets

- **Primary:** [EuroSAT](https://github.com/phelber/EuroSAT) — 27,000
  Sentinel-2 tiles, 10 classes. Downloaded automatically via
  `torchvision.datasets.EuroSAT`.
- **Cross-dataset holdout:** UC Merced Land Use — downloaded via
  `data/download_uc_merced.py` (Hugging Face `blanchon/UC_Merced`).

## Setup

```bash
pip install -r requirements.txt
python data/download_uc_merced.py
```

Training was done on **Google Colab** (free T4 GPU); the dashboard runs
locally afterward using the saved `.pt` checkpoints.

## Repository Structure

```
satellite-landuse-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_uc_merced.py
│   ├── raw/                    (gitignored)
│   └── processed/              (gitignored — split manifest, ablation CSVs)
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_change_detection.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── dataset.py           # EuroSAT/UC Merced loading, spatial block split
│   ├── model.py              # BaselineCNN, TransferLearningModel (ResNet-18)
│   ├── train.py               # Generic training loop
│   ├── change_detection.py   # Embeddings, cosine similarity, ROC, heatmaps
│   ├── gradcam.py             # GradCAM implementation
│   └── utils.py               # Seeding, plotting, metrics
├── models/                    # Saved .pt checkpoints (gitignored, see models/README.md)
├── app/
│   └── app.py                 # Streamlit dashboard
├── reports/
│   ├── final_report.pdf
│   ├── change_detection_roc.png
│   ├── similarity_distribution.png
│   ├── region_change_heatmap_1.png … _5.png
│   └── top_5_errors.png
├── demo/
│   └── demo_video_link.md
└── test_components.py         # Unit tests (dataset, model freezing, embeddings)
```
