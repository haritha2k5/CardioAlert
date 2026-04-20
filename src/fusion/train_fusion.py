import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ecg_pipeline.dataset import PTBXLDataset
from src.ecg_pipeline.model import ECGClassifier
from src.clinical_pipeline.dataset import load_uci_data
from src.clinical_pipeline.model import ClinicalClassifier
from src.fusion.fusion_model import FusionModel

# ── Config ─────────────────────────────────────────────────────────────────────
ECG_MODEL_PATH      = "outputs/models/ecg_model.pt"
CLINICAL_MODEL_PATH = "outputs/models/clinical_model.pt"
FUSION_SAVE_PATH    = "outputs/models/fusion_model.pt"

PTBXL_PATH    = "data/ptbxl"
UCI_PATH      = "data/uci"

ECG_EMB_DIM      = 64
CLINICAL_EMB_DIM = 32
NUM_FUSION_CLASSES = 3   # Low / Moderate / High

BATCH_SIZE = 32
EPOCHS     = 30
LR         = 5e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Risk label mapping ─────────────────────────────────────────────────────────
# ECG label:      0 = NORM, 1 = MI
# Clinical label: 0 = no disease, 1 = disease
# Fusion label:   0 = Low, 1 = Moderate, 2 = High
def assign_fusion_label(ecg_label: int, clinical_label: int) -> int:
    if ecg_label == 0 and clinical_label == 0:
        return 0   # Low
    elif ecg_label == 1 and clinical_label == 0:
        return 1   # Moderate
    elif ecg_label == 0 and clinical_label == 1:
        return 1   # Moderate
    else:
        return 2   # High (both signals indicate risk)


def extract_ecg_embeddings(ecg_model, ptbxl_path, split, device):
    """Extract embeddings and labels from PTB-XL using trained ECG encoder."""
    from torch.utils.data import DataLoader
    ds = PTBXLDataset(ptbxl_path, split=split)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    embeddings, labels = [], []
    ecg_model.eval()
    with torch.no_grad():
        for signals, lbls in loader:
            signals = signals.to(device)
            emb = ecg_model.get_embedding(signals)
            embeddings.append(emb.cpu())
            labels.append(lbls)

    return torch.cat(embeddings), torch.cat(labels)


def extract_clinical_embeddings(clinical_model, X, device):
    """Extract embeddings from clinical data using trained clinical encoder."""
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    clinical_model.eval()
    with torch.no_grad():
        emb = clinical_model.get_embedding(X_t)
    return emb.cpu()


def build_synthetic_pairs(ecg_embs, ecg_labels, clinical_embs, clinical_labels):
    """
    Synthetic pairing strategy:
    - Pair each ECG sample with a random clinical sample sharing the same label.
    - If no same-label match, pair with any clinical sample.
    - Fusion label is derived from combined ECG + clinical label.
    """
    ecg_labels    = ecg_labels.numpy()
    clinical_labels = clinical_labels.numpy()

    paired_ecg, paired_clinical, fusion_labels = [], [], []

    for i in range(len(ecg_embs)):
        ecg_lbl = ecg_labels[i]

        # Try to find a matching clinical sample by label
        match_idxs = np.where(clinical_labels == ecg_lbl)[0]
        if len(match_idxs) == 0:
            match_idxs = np.arange(len(clinical_labels))

        j = np.random.choice(match_idxs)
        clin_lbl = clinical_labels[j]
        fusion_lbl = assign_fusion_label(int(ecg_lbl), int(clin_lbl))

        paired_ecg.append(ecg_embs[i])
        paired_clinical.append(clinical_embs[j])
        fusion_labels.append(fusion_lbl)

    return (
        torch.stack(paired_ecg),
        torch.stack(paired_clinical),
        torch.tensor(fusion_labels, dtype=torch.long),
    )


def train_fusion():
    print(f"Device: {DEVICE}")

    # ── Load pretrained encoders ───────────────────────────────────────────────
    print("Loading pretrained ECG encoder...")
    ecg_model = ECGClassifier(embedding_dim=ECG_EMB_DIM, num_classes=2).to(DEVICE)
    ecg_model.load_state_dict(torch.load(ECG_MODEL_PATH, map_location=DEVICE))
    for param in ecg_model.parameters():
        param.requires_grad = False   # freeze encoder

    print("Loading pretrained clinical encoder...")
    clinical_ckpt = torch.load(CLINICAL_MODEL_PATH, map_location=DEVICE)
    clinical_model = ClinicalClassifier(
        input_dim=clinical_ckpt["input_dim"],
        cat_idxs=clinical_ckpt["cat_idxs"],
        cat_dims=clinical_ckpt["cat_dims"],
        embedding_dim=CLINICAL_EMB_DIM,
        num_classes=2,
    ).to(DEVICE)
    clinical_model.load_state_dict(clinical_ckpt["model_state"])
    for param in clinical_model.parameters():
        param.requires_grad = False   # freeze encoder

    # ── Extract embeddings ─────────────────────────────────────────────────────
    print("Extracting ECG embeddings (train)...")
    ecg_embs_train, ecg_labels_train = extract_ecg_embeddings(ecg_model, PTBXL_PATH, "train", DEVICE)

    print("Extracting ECG embeddings (val)...")
    ecg_embs_val, ecg_labels_val = extract_ecg_embeddings(ecg_model, PTBXL_PATH, "val", DEVICE)

    print("Extracting clinical embeddings...")
    (X_train, y_train, X_val, y_val, X_test, y_test,
     cat_idxs, cat_dims, scaler, feature_cols) = load_uci_data(UCI_PATH)

    clinical_embs_train = extract_clinical_embeddings(clinical_model, X_train, DEVICE)
    clinical_embs_val   = extract_clinical_embeddings(clinical_model, X_val,   DEVICE)

    clinical_labels_train = torch.tensor(y_train, dtype=torch.long)
    clinical_labels_val   = torch.tensor(y_val,   dtype=torch.long)

    # ── Synthetic pairing ──────────────────────────────────────────────────────
    print("Building synthetic paired dataset...")
    paired_ecg_train, paired_clin_train, fusion_labels_train = build_synthetic_pairs(
        ecg_embs_train, ecg_labels_train, clinical_embs_train, clinical_labels_train
    )
    paired_ecg_val, paired_clin_val, fusion_labels_val = build_synthetic_pairs(
        ecg_embs_val, ecg_labels_val, clinical_embs_val, clinical_labels_val
    )

    print(f"Fusion train samples: {len(fusion_labels_train)}")
    print(f"Label distribution: {torch.bincount(fusion_labels_train)}")

    train_ds = TensorDataset(paired_ecg_train, paired_clin_train, fusion_labels_train)
    val_ds   = TensorDataset(paired_ecg_val,   paired_clin_val,   fusion_labels_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Train fusion model ─────────────────────────────────────────────────────
    fusion_model = FusionModel(
        ecg_embedding_dim=ECG_EMB_DIM,
        clinical_embedding_dim=CLINICAL_EMB_DIM,
        num_classes=NUM_FUSION_CLASSES,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        fusion_model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for ecg_emb, clin_emb, labels in train_loader:
            ecg_emb  = ecg_emb.to(DEVICE)
            clin_emb = clin_emb.to(DEVICE)
            labels   = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = fusion_model(ecg_emb, clin_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * ecg_emb.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += ecg_emb.size(0)

        # Validate
        fusion_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for ecg_emb, clin_emb, labels in val_loader:
                ecg_emb  = ecg_emb.to(DEVICE)
                clin_emb = clin_emb.to(DEVICE)
                labels   = labels.to(DEVICE)
                logits = fusion_model(ecg_emb, clin_emb)
                loss = criterion(logits, labels)
                val_loss += loss.item() * ecg_emb.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += ecg_emb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_acc = train_correct / train_total
        val_acc   = val_correct / val_total
        scheduler.step(val_loss / val_total)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(FUSION_SAVE_PATH), exist_ok=True)
            torch.save(fusion_model.state_dict(), FUSION_SAVE_PATH)
            print(f"  * Saved best fusion model (Val Acc: {best_val_acc:.4f})")

    print(f"\nFusion training complete. Best Val Acc: {best_val_acc:.4f}")
    print("\nClassification Report (Val):")
    print(classification_report(all_labels, all_preds, labels=[0, 1, 2], target_names=["Low", "Moderate", "High"]))


if __name__ == "__main__":
    train_fusion()