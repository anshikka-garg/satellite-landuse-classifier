import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_class_distribution(labels, class_names, title="Class distribution"):
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(class_names, counts, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel("Number of tiles")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c),
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def plot_sample_grid(images, labels, class_names, n_per_class=5):
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class,
                              figsize=(n_per_class * 1.6, n_classes * 1.6))
    labels = np.array(labels)
    for row, cname in enumerate(class_names):
        idxs = np.where(labels == row)[0]
        chosen = np.random.choice(idxs, size=min(n_per_class, len(idxs)), replace=False)
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                img = images[chosen[col]]
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cname, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_loss_curves(history: dict, title="Training curves"):
    has_acc = "train_acc" in history
    n_plots = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    axes = np.atleast_1d(axes)

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if has_acc:
        axes[1].plot(history["train_acc"], label="train")
        axes[1].plot(history["val_acc"], label="val")
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, class_names, title="Confusion matrix", normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def compute_f1_report(y_true, y_pred, class_names):
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    per_class_dict = {c: float(f) for c, f in zip(class_names, per_class)}
    return per_class_dict, float(macro), report