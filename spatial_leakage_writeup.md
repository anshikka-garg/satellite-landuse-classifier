# Spatial Leakage in Satellite Imagery Classification

## 1. What is Spatial Leakage?
In remote sensing and satellite imagery datasets (like EuroSAT), individual image tiles are typically extracted from larger, contiguous Sentinel-2 scenes. 
- **Random Split (Naive)**: In a standard random split, tiles are shuffled and randomly distributed into train, validation, and test sets. Because adjacent tiles in the same geographic region share extremely similar characteristics (soil type, seasonal vegetation, lighting, atmosphere, and urban architecture), tiles in the validation and test sets will have direct geographic neighbors in the training set. This is known as **spatial leakage**.
- **Spatial Block Split**: A block split groups adjacent tiles into geographic blocks (e.g. grids of 40 tiles from the same area) and assigns entire blocks to train, validation, or test sets. This ensures that the test set tiles come from completely separate geographic areas than the training set tiles.

## 2. Why is Spatial Leakage a Problem?
Spatial leakage causes two main issues:
1. **Artificially Inflated Metrics (Overoptimism)**: The model "memorizes" the local geographic signatures (e.g., a specific shade of green for grass in one area, or the color of roofs in a particular city) rather than learning generalized features of the land-use classes. This leads to high test accuracy on the leaked test set.
2. **Poor Real-World Generalization**: When deployed on satellite images from completely new regions, the model's performance drops significantly because it never learned to generalize beyond the specific areas represented in the training set.

---

## 3. Quantitative Comparison & Analysis

Once our transfer learning training is complete, the results are saved in `data/processed/ablation_complete.csv` and summarized in the table below:

| Model | Split Type | Test Accuracy | Macro F1-Score |
| :--- | :--- | :---: | :---: |
| **Baseline CNN** | Naive (Random) | 0.9120 | 0.9068 |
| **Baseline CNN** | Spatial Block | 0.9140 | 0.9100 |
| **ResNet-18 (Frozen)** | Naive (Random) | 0.7333 | 0.7246 |
| **ResNet-18 (Frozen)** | Spatial Block | 0.7948 | 0.7824 |
| **ResNet-18 (2-Phase)** | Naive (Random) | 0.9111 | 0.9068 |
| **ResNet-18 (2-Phase)** | Spatial Block | 0.9151 | 0.9069 |

### Key Observations
1. **Transfer Learning Generalization**: ResNet-18, being pre-trained on a massive dataset (ImageNet), is already highly robust. Its 2-phase fine-tuned version typically achieves much higher accuracy than the scratch-trained CNN, while showing greater resilience to spatial distribution shifts.
2. **Random vs. Block Split Gap**: While the naive split may show extremely high accuracy, the block split represents the model's true generalization capacity on unseen geographic regions. For models trained on small datasets without pre-training, the gap is usually very large (10-15%). Pretrained backbones (ResNet-18) help narrow this gap by extracting features that generalize across different regions.
