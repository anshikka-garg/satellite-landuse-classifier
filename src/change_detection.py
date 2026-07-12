import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms

class EmbeddingExtractor:
    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device
        # Extract the backbone feature extractor (stripping the final fc layer)
        self.extractor = self.model.get_embedding_extractor().to(device)
        self.extractor.eval()

    @torch.no_grad()
    def get_embedding(self, image_tensor: torch.Tensor):
        """
        Extracts 512-dimensional embedding for an image tensor.
        Input tensor shape: (B, C, H, W)
        Output shape: (B, 512)
        """
        image_tensor = image_tensor.to(self.device)
        # ResNet-18 feature extractor output shape: (B, 512, 1, 1)
        feat = self.extractor(image_tensor)
        # Flatten to (B, 512)
        feat = torch.flatten(feat, 1)
        # Normalize to unit length for easier cosine similarity (cos(A, B) = A . B if normalized)
        feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
        return feat.cpu().numpy()

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray):
    """
    Computes cosine similarity between two sets of embeddings.
    emb1, emb2: shapes (N, D) or (D,)
    Returns: similarity score or array of shape (N,)
    """
    if len(emb1.shape) == 1:
        emb1 = emb1.reshape(1, -1)
    if len(emb2.shape) == 1:
        emb2 = emb2.reshape(1, -1)
        
    # Since embeddings are unit-normalized, cosine similarity is just the dot product
    dot_product = np.sum(emb1 * emb2, axis=1)
    return dot_product

def simulate_region_pairs(eurosat_dataset, indices, num_regions=5, grid_size=5, change_prob=0.4, seed=42):
    """
    Simulates T1 (before) and T2 (after) region pairs.
    For each region:
    - We sample grid_size*grid_size (25) tiles.
    - Some tiles are kept "unchanged" (but represented by a DIFFERENT random image of the SAME class to mimic seasonal variance).
    - Some tiles are "changed" (represented by an image of a DIFFERENT class).
    Returns list of dictionaries containing the region data.
    """
    rng = np.random.default_rng(seed)
    
    # Group all available indices by class label
    class_indices = {}
    for idx in indices:
        _, label = eurosat_dataset[idx]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
        
    all_classes = list(class_indices.keys())
    
    regions = []
    
    for r in range(num_regions):
        # Select 25 random indices from our split as T1
        t1_indices = rng.choice(indices, size=grid_size*grid_size, replace=False)
        
        t2_indices = []
        change_mask = []
        
        for t1_idx in t1_indices:
            img, t1_label = eurosat_dataset[t1_idx]
            
            # Determine if this tile changes
            is_changed = rng.random() < change_prob
            
            if is_changed:
                # Select a different class
                other_classes = [c for c in all_classes if c != t1_label]
                t2_label = rng.choice(other_classes)
                # Sample a tile from this other class
                t2_idx = rng.choice(class_indices[t2_label])
                change_mask.append(1)
            else:
                # Unchanged class, but sample a different image of the same class (seasonal/light variation)
                same_class_idxs = [idx for idx in class_indices[t1_label] if idx != t1_idx]
                t2_idx = rng.choice(same_class_idxs) if len(same_class_idxs) > 0 else t1_idx
                change_mask.append(0)
                
            t2_indices.append(t2_idx)
            
        regions.append({
            "region_id": r + 1,
            "t1_indices": t1_indices.tolist(),
            "t2_indices": t2_indices,
            "change_mask": np.array(change_mask).reshape(grid_size, grid_size)
        })
        
    return regions

def compute_roc_curve(emb1: np.ndarray, emb2: np.ndarray, y_true: np.ndarray):
    """
    Computes ROC curve, AUC, and finds optimal threshold using Youden's J statistic.
    Note: Lower similarity indicates higher probability of change.
    So we use (1 - similarity) as the anomaly/change score.
    """
    similarities = compute_cosine_similarity(emb1, emb2)
    change_scores = 1.0 - similarities # Higher score = more likely changed
    
    fpr, tpr, thresholds = roc_curve(y_true, change_scores)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold_score = thresholds[best_idx]
    
    # Convert threshold score (distance) back to cosine similarity threshold
    best_threshold_sim = 1.0 - best_threshold_score
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
        "optimal_threshold_sim": best_threshold_sim,
        "best_fpr": fpr[best_idx],
        "best_tpr": tpr[best_idx]
    }

def plot_region_change_heatmap(eurosat_dataset, region, extractor, similarity_threshold, grid_size=5):
    """
    Generates a 4-panel figure:
    1. T1 Image Grid
    2. T2 Image Grid
    3. Cosine Similarity Heatmap
    4. Flagged Changes (Similarity < threshold)
    """
    t1_indices = region["t1_indices"]
    t2_indices = region["t2_indices"]
    change_mask = region["change_mask"]
    
    # Helper to load and transform images for displaying
    t1_imgs = [eurosat_dataset[idx][0] for idx in t1_indices]
    t2_imgs = [eurosat_dataset[idx][0] for idx in t2_indices]
    
    # Extract embeddings and compute similarity
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    t1_embs = []
    t2_embs = []
    
    for t1_img, t2_img in zip(t1_imgs, t2_imgs):
        t1_tensor = transform(t1_img).unsqueeze(0)
        t2_tensor = transform(t2_img).unsqueeze(0)
        
        t1_embs.append(extractor.get_embedding(t1_tensor))
        t2_embs.append(extractor.get_embedding(t2_tensor))
        
    t1_embs = np.vstack(t1_embs)
    t2_embs = np.vstack(t2_embs)
    
    similarities = compute_cosine_similarity(t1_embs, t2_embs)
    sim_matrix = similarities.reshape(grid_size, grid_size)
    
    pred_change = (sim_matrix < similarity_threshold).astype(int)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. T1 Visual Grid
    t1_grid = np.zeros((grid_size * 64, grid_size * 64, 3), dtype=np.uint8)
    for idx, img in enumerate(t1_imgs):
        r_idx = idx // grid_size
        c_idx = idx % grid_size
        # Resize/convert image to numpy array
        img_np = np.array(img.convert("RGB"))
        t1_grid[r_idx*64:(r_idx+1)*64, c_idx*64:(c_idx+1)*64, :] = img_np
        
    axes[0, 0].imshow(t1_grid)
    axes[0, 0].set_title("T1 (Before) Image Grid")
    axes[0, 0].axis("off")
    
    # 2. T2 Visual Grid
    t2_grid = np.zeros((grid_size * 64, grid_size * 64, 3), dtype=np.uint8)
    for idx, img in enumerate(t2_imgs):
        r_idx = idx // grid_size
        c_idx = idx % grid_size
        img_np = np.array(img.convert("RGB"))
        t2_grid[r_idx*64:(r_idx+1)*64, c_idx*64:(c_idx+1)*64, :] = img_np
        
    axes[0, 1].imshow(t2_grid)
    axes[0, 1].set_title("T2 (After) Image Grid")
    axes[0, 1].axis("off")
    
    # 3. Similarity Heatmap
    sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="viridis", ax=axes[1, 0], 
                cbar_kws={'label': 'Cosine Similarity'}, vmin=0.4, vmax=1.0)
    axes[1, 0].set_title("Cosine Similarity Heatmap")
    
    # 4. Detected vs Ground Truth Changes
    # Overlay green for correct no change, red for correct change, yellow for errors
    overlay_map = np.zeros((grid_size, grid_size, 3))
    for r in range(grid_size):
        for c in range(grid_size):
            gt = change_mask[r, c]
            pr = pred_change[r, c]
            if gt == 1 and pr == 1:
                overlay_map[r, c] = [0.8, 0.1, 0.1]  # Red: Correctly Detected Change
            elif gt == 0 and pr == 0:
                overlay_map[r, c] = [0.1, 0.8, 0.1]  # Green: Correctly Detected Unchanged
            else:
                overlay_map[r, c] = [0.9, 0.9, 0.1]  # Yellow: Error (FP or FN)
                
    axes[1, 1].imshow(overlay_map)
    axes[1, 1].set_title("Change Map (Red: Change, Green: Unchanged, Yellow: Error)")
    
    # Set tick labels for regions
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xticks(np.arange(grid_size) + 0.5)
        ax.set_yticks(np.arange(grid_size) + 0.5)
        ax.set_xticklabels([f"C{i+1}" for i in range(grid_size)])
        ax.set_yticklabels([f"R{i+1}" for i in range(grid_size)])
        
    plt.tight_layout()
    return fig, sim_matrix, pred_change
