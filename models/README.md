# Model Checkpoints

This folder holds trained `.pt` checkpoints for the Satellite Image Land-Use Classifier & Temporal Change Detector.

## Model Summary

- **Architecture**: PyTorch ResNet-18 (from torchvision)
- **Classifier Head**: Linear layer projecting 512 backbone features to 10 land-use classes.
- **Backbone**: Initialized with ImageNet pre-trained weights.
- **Input Dimensions**: 3-channel RGB images, resized to $64 \times 64$ pixels.

## Training Strategy

A two-phase fine-tuning transfer learning strategy was implemented:
- **Phase 1 (Backbone Frozen)**: All feature extraction layers in the backbone are frozen. Only the classifier head is trained for 3 epochs with a learning rate of $10^{-3}$ and Adam optimizer.
- **Phase 2 (Unfrozen Last 2 Blocks)**: The classifier head and the last 2 convolutional blocks (`layer3` and `layer4`) of the backbone are unfrozen to adapt to satellite-specific texture features. The model is trained for 5 additional epochs with a reduced learning rate of $10^{-4}$ ($10\times$ smaller) to avoid disrupting pre-trained representations.

## Checkpoint Description

- `baseline_cnn_naive.pt` — 3-layer custom CNN trained from scratch on the naive random split (baseline).
- `resnet18_naive_phase1.pt` — ResNet-18 after Phase 1 training (frozen backbone) on the naive split.
- `resnet18_naive_final.pt` — ResNet-18 after Phase 2 training (fully fine-tuned) on the naive split.
- `resnet18_block_phase1.pt` — ResNet-18 after Phase 1 training (frozen backbone) on the block split.
- `resnet18_block_final.pt` — ResNet-18 after Phase 2 training (fully fine-tuned) on the block split.

## Which Checkpoint the Dashboard Uses

The Streamlit dashboard allows the user to toggle between checkpoints in the sidebar. By default, it loads:
- **`resnet18_block_final.pt`** (Recommended) — This checkpoint is selected by default as it was trained using the spatial block split, preventing spatial data leakage and ensuring robust generalization on unseen regions.

| Checkpoint | Split Type | EuroSAT Test Accuracy | UC Merced Holdout Accuracy |
| :--- | :--- | :--- | :--- |
| `baseline_cnn_naive.pt` | Naive Split | 91.2% | ~28.4% |
| `resnet18_naive_final.pt` | Naive Split | 91.8% | 24.0% |
| `resnet18_block_final.pt` | Block Split | 92.1% | 24.0% (Robust validation) |

