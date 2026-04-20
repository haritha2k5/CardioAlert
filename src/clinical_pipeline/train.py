import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.clinical_pipeline.dataset import load_uci_data
from src.clinical_pipeline.model import ClinicalClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
UCI_PATH      = "data/uci"
SAVE_PATH     = "outputs/models/clinical_model.pt"
EMBEDDING_DIM = 32
NUM_CLASSES   = 2
BATCH_SIZE    = 32
EPOCHS        = 50
LR            = 2e-3
LAMBDA_SPARSE = 1e-4          # TabNet sparsity regularization weight
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits, M_loss = model(X_batch)
        loss = criterion(logits, y_batch) - LAMBDA_SPARSE * M_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits, M_loss = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


def train():
    print(f"Device: {DEVICE}")

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     cat_idxs, cat_dims,
     scaler, feature_cols) = load_uci_data(UCI_PATH)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Features: {feature_cols}")
    print(f"Cat idxs: {cat_idxs} | Cat dims: {cat_dims}")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    model = ClinicalClassifier(
        input_dim=X_train.shape[1],
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
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
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": X_train.shape[1],
                "cat_idxs": cat_idxs,
                "cat_dims": cat_dims,
                "feature_cols": feature_cols,
                "scaler": scaler,
            }, SAVE_PATH)
            print(f"  * Saved best model (AUC: {best_auc:.4f})")

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")

    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")


if __name__ == "__main__":
    train()