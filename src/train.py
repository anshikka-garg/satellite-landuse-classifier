"""
train.py
--------
Generic training loop shared by BaselineCNN and TransferLearningModel.
Both 02_baseline_cnn.ipynb and 03_transfer_learning.ipynb call these same
functions, so their numbers are directly comparable — no subtle differences
in how loss/accuracy are computed between the two.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # switch to training mode (activates dropout, batchnorm updates)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()               # clear old gradients
        outputs = model(images)             # forward pass
        loss = criterion(outputs, labels)   # compute how wrong we are
        loss.backward()                     # compute gradients (backward pass)
        optimizer.step()                    # update weights using those gradients

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()  # tells PyTorch: no gradients needed here, saves memory + time
def evaluate(model, dataloader, criterion, device):
    model.eval()  # switch to evaluation mode (disables dropout, freezes batchnorm stats)

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def run_training(model, train_loader, val_loader, criterion, optimizer,
                  device, num_epochs, checkpoint_path=None, verbose=True):

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch}/{num_epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if checkpoint_path is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f"  -> New best val_acc={val_acc:.4f}, saved checkpoint to {checkpoint_path}")

    return history