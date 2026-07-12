"""
dataset.py
----------
Dataset loading + spatial block split for EuroSAT, plus UC Merced class mapping.
"""

from torchvision.datasets import EuroSAT
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

UCM_TO_EUROSAT = {
    "agricultural": "AnnualCrop",
    "forest": "Forest",
    "freeway": "Highway",
    "harbor": "River",
    "denseresidential": "Residential",
    "mediumresidential": "Residential",
    "sparseresidential": "Residential",
    "river": "River",
    "storagetanks": "Industrial",
    "buildings": "Industrial",
    "runway": "Highway",
}


def load_eurasat(root: str, download: bool = True):
    ds = EuroSAT(root=root, download=download, transform=None)
    return ds


def naive_random_split(n_samples: int, splits=(0.7, 0.15, 0.15), seed: int = 42):
    assert abs(sum(splits) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_train = int(splits[0] * n_samples)
    n_val = int(splits[1] * n_samples)
    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train:n_train + n_val].tolist(),
        "test": idx[n_train + n_val:].tolist(),
    }


def block_split(labels, splits=(0.7, 0.15, 0.15), block_size: int = 40, seed: int = 42):
    assert abs(sum(splits) - 1.0) < 1e-6
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for lab, idxs in by_class.items():
        blocks = [idxs[i:i + block_size] for i in range(0, len(idxs), block_size)]
        block_order = rng.permutation(len(blocks))

        n_blocks = len(blocks)
        n_train_blocks = int(round(splits[0] * n_blocks))
        n_val_blocks = int(round(splits[1] * n_blocks))

        train_blocks = block_order[:n_train_blocks]
        val_blocks = block_order[n_train_blocks:n_train_blocks + n_val_blocks]
        test_blocks = block_order[n_train_blocks + n_val_blocks:]

        for b in train_blocks:
            train_idx.extend(blocks[b])
        for b in val_blocks:
            val_idx.extend(blocks[b])
        for b in test_blocks:
            test_idx.extend(blocks[b])

    return {"train": train_idx, "val": val_idx, "test": test_idx}


class IndexSubsetDataset(Dataset):
    """
    Thin wrapper that applies a transform to a subset of an underlying
    dataset given by index list — used so train/val/test can each get their
    own transform (augmentation only on train) while sharing one underlying
    EuroSAT object instead of loading images from disk three times.
    """

    def __init__(self, base_dataset, indices, transform=None):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label


class UCMercedMappedDataset(Dataset):
    """
    Loads UC Merced images and maps their 21 categories to EuroSAT's 10 categories
    using the UCM_TO_EUROSAT mapping dictionary. Filters out classes not in the dictionary.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            if class_name in UCM_TO_EUROSAT:
                eurosat_class = UCM_TO_EUROSAT[class_name]
                label_idx = EUROSAT_CLASSES.index(eurosat_class)

                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        fpath = os.path.join(class_dir, fname)
                        self.samples.append((fpath, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label