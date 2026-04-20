import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ecg_pipeline.dataset import PTBXLDataset
from src.ecg_pipeline.model import ECGClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
PTBXL_PATH    = "data/ptbxl"
SAVE_PATH     = "outputs/models/ecg_model.pt"
EMBEDDING_DIM = 64
NUM_CLASSES   = 2
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for signals, labels in loader:
        signals, labels = signals.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += signals.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(DEVICE), labels.to(DEVICE)
            logits = model(signals)
            loss = criterion(logits, labels)

            total_loss += loss.item() * signals.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += signals.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


def train():
    print(f"Device: {DEVICE}")

    train_ds = PTBXLDataset(PTBXL_PATH, split="train")
    val_ds   = PTBXLDataset(PTBXL_PATH, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    model     = ECGClassifier(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
    print(f"Model saved to: {SAVE_PATH}")

    # Final test evaluation
    test_ds     = PTBXLDataset(PTBXL_PATH, split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model.load_state_dict(torch.load(SAVE_PATH))
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")


if __name__ == "__main__":
    train()