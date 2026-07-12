import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.getcwd())

from src.dataset import UCMercedMappedDataset, EUROSAT_CLASSES
from src.model import TransferLearningModel

def run_error_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Setup directories
    DATA_ROOT = os.path.join("data", "raw")
    MODELS_DIR = os.path.join("models")
    REPORTS_DIR = os.path.join("reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Transforms
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Load UC Merced holdout dataset
    ucm_root = os.path.join(DATA_ROOT, "uc_merced")
    ucm_dataset = UCMercedMappedDataset(root_dir=ucm_root, transform=eval_transform)
    ucm_loader = DataLoader(ucm_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Loaded UC Merced: {len(ucm_dataset)} samples")

    # Load model
    model = TransferLearningModel(num_classes=10, backbone_name="resnet18").to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet18_block_final.pt"), map_location=device))
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []

    # Get predictions
    with torch.no_grad():
        for i, (images, labels) in enumerate(ucm_loader):
            images = images.to(device)
            outputs = model(images)
            probs = outputs.softmax(dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            
            # Retrieve file paths for this batch
            start_idx = i * 32
            end_idx = min(start_idx + 32, len(ucm_dataset))
            for idx in range(start_idx, end_idx):
                all_paths.append(ucm_dataset.samples[idx][0])

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Find errors
    errors = np.where(all_preds != all_labels)[0]
    print(f"Total misclassifications on UC Merced: {len(errors)}")

    if len(errors) == 0:
        print("No errors found to analyze!")
        return

    # Find top 5 most confident errors:
    # Look at the probability assigned to the incorrect predicted class for misclassified cases.
    error_confidences = [all_probs[idx, all_preds[idx]] for idx in errors]
    sorted_error_indices = np.argsort(error_confidences)[::-1]  # Descending
    top_5_error_indices = errors[sorted_error_indices[:5]]

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # Hypotheses dictionary for explanation
    hypotheses = [
        "Spatial Similarity: Class features are highly similar.",
        "Class Overlap: Categories share pixel characteristics.",
        "Visual Ambiguity: Image has features of both classes.",
        "Scale/Resolution: Sub-features confuse the neural net.",
        "Label Schema: Category mapping creates minor semantic gaps."
    ]

    for idx, sample_idx in enumerate(top_5_error_indices):
        fpath = all_paths[sample_idx]
        true_lbl = EUROSAT_CLASSES[all_labels[sample_idx]]
        pred_lbl = EUROSAT_CLASSES[all_preds[sample_idx]]
        confidence = all_probs[sample_idx, all_preds[sample_idx]]

        img = plt.imread(fpath)
        axes[idx].imshow(img)
        axes[idx].set_title(f"True: {true_lbl}\nPred: {pred_lbl}\nConf: {confidence:.3f}\n{hypotheses[idx]}", fontsize=10)
        axes[idx].axis("off")
        
        print(f"Error {idx+1}: Image: {os.path.basename(fpath)}")
        print(f"  True Class: {true_lbl}")
        print(f"  Predicted Class: {pred_lbl} (Confidence: {confidence:.4f})")
        print(f"  Hypothesis: {hypotheses[idx]}")
        print("-" * 50)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.savefig(os.path.join(REPORTS_DIR, "top_5_errors.png"), dpi=300)
    print("Saved error analysis plot to reports/top_5_errors.png")

if __name__ == "__main__":
    run_error_analysis()
