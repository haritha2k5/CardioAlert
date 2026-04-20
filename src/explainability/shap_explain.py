import os
import sys
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ecg_pipeline.model import ECGClassifier
from src.clinical_pipeline.model import ClinicalClassifier
from src.clinical_pipeline.dataset import load_uci_data
from src.fusion.fusion_model import FusionModel

# ── Config ─────────────────────────────────────────────────────────────────────
ECG_MODEL_PATH      = "outputs/models/ecg_model.pt"
CLINICAL_MODEL_PATH = "outputs/models/clinical_model.pt"
FUSION_MODEL_PATH   = "outputs/models/fusion_model.pt"
UCI_PATH            = "data/uci"
PLOT_SAVE_DIR       = "outputs/plots"
DEVICE              = torch.device("cpu")   # SHAP works best on CPU

ECG_EMB_DIM      = 64
CLINICAL_EMB_DIM = 32
RISK_LABELS      = ["Low", "Moderate", "High"]


def load_models(clinical_ckpt_path):
    """Load all three trained models."""
    # ECG model
    ecg_model = ECGClassifier(embedding_dim=ECG_EMB_DIM, num_classes=2).to(DEVICE)
    ecg_model.load_state_dict(torch.load(ECG_MODEL_PATH, map_location=DEVICE))
    ecg_model.eval()

    # Clinical model
    ckpt = torch.load(clinical_ckpt_path, map_location=DEVICE)
    clinical_model = ClinicalClassifier(
        input_dim=ckpt["input_dim"],
        cat_idxs=ckpt["cat_idxs"],
        cat_dims=ckpt["cat_dims"],
        embedding_dim=CLINICAL_EMB_DIM,
        num_classes=2,
    ).to(DEVICE)
    clinical_model.load_state_dict(ckpt["model_state"])
    clinical_model.eval()

    # Fusion model
    fusion_model = FusionModel(
        ecg_embedding_dim=ECG_EMB_DIM,
        clinical_embedding_dim=CLINICAL_EMB_DIM,
        num_classes=3,
    ).to(DEVICE)
    fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
    fusion_model.eval()

    return ecg_model, clinical_model, fusion_model, ckpt


class FusedModelWrapper(torch.nn.Module):
    """
    Wraps ECG encoder + clinical encoder + fusion model into one callable.
    SHAP operates only on clinical features (tabular, interpretable).
    ECG embedding is fixed as context — clinical features are the explanation target.
    """

    def __init__(self, ecg_model, clinical_model, fusion_model, fixed_ecg_emb):
        super().__init__()
        self.ecg_model      = ecg_model
        self.clinical_model = clinical_model
        self.fusion_model   = fusion_model
        self.fixed_ecg_emb  = fixed_ecg_emb   # (1, ECG_EMB_DIM) — fixed context

    def forward(self, clinical_input):
        clinical_emb = self.clinical_model.get_embedding(clinical_input)
        ecg_emb = self.fixed_ecg_emb.expand(clinical_input.shape[0], -1)
        logits = self.fusion_model(ecg_emb, clinical_emb)
        probs = torch.softmax(logits, dim=1)
        return probs


def explain_clinical_features(
    ecg_signal: np.ndarray,
    clinical_features: np.ndarray,
    feature_names: list,
    background_clinical: np.ndarray,
    n_samples: int = 50,
):
    """
    Run SHAP DeepExplainer on clinical features given a fixed ECG embedding.

    Args:
        ecg_signal:          (1, 12, 1000) numpy array
        clinical_features:   (1, n_features) numpy array
        feature_names:       list of feature names
        background_clinical: (n_bg, n_features) background samples for SHAP
        n_samples:           number of background samples to use

    Returns:
        shap_values: SHAP values for each class
        prediction:  predicted risk class index
    """
    ecg_model, clinical_model, fusion_model, _ = load_models(CLINICAL_MODEL_PATH)

    # Get fixed ECG embedding
    ecg_t = torch.tensor(ecg_signal, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        fixed_ecg_emb = ecg_model.get_embedding(ecg_t)   # (1, 64)

    wrapped = FusedModelWrapper(ecg_model, clinical_model, fusion_model, fixed_ecg_emb)
    wrapped.eval()

    # Background data for SHAP (subset of training data)
    bg_t = torch.tensor(background_clinical[:n_samples], dtype=torch.float32).to(DEVICE)
    patient_t = torch.tensor(clinical_features, dtype=torch.float32).to(DEVICE)

    explainer = shap.DeepExplainer(wrapped, bg_t)
    shap_values = explainer.shap_values(patient_t, check_additivity=False)   # list of arrays, one per class

    # Get prediction
    with torch.no_grad():
        probs = wrapped(patient_t)
    pred_class = probs.argmax(dim=1).item()

    return shap_values, pred_class, probs.cpu().numpy()[0]


def plot_shap_bar(shap_values, feature_names, pred_class, probs, save_path=None):
    """Plot SHAP bar chart for the predicted class."""
    class_shap = shap_values[pred_class][0]   # (n_features,)

    sorted_idx = np.argsort(np.abs(class_shap))[::-1]
    sorted_vals = class_shap[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    colors = ["#d73027" if v > 0 else "#4575b4" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on model output)")
    ax.set_title(
        f"CardioAlert — Feature Importance\n"
        f"Predicted Risk: {RISK_LABELS[pred_class]} "
        f"({probs[pred_class]*100:.1f}% confidence)",
        fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP plot saved to: {save_path}")
    plt.show()
    return fig


def run_shap_on_test_sample():
    """
    Demo: Run SHAP explanation on a test sample from UCI dataset.
    Requires a dummy ECG signal shape (use zeros for demo without PTB-XL).
    """
    (X_train, y_train, X_val, y_val, X_test, y_test,
     cat_idxs, cat_dims, scaler, feature_cols) = load_uci_data(UCI_PATH)

    # Use first test patient
    patient_clinical = X_test[0:1]

    # Dummy ECG signal (replace with real wfdb signal in production)
    dummy_ecg = np.zeros((1, 12, 1000), dtype=np.float32)

    print(f"Patient clinical features: {dict(zip(feature_cols, patient_clinical[0]))}")

    shap_values, pred_class, probs = explain_clinical_features(
        ecg_signal=dummy_ecg,
        clinical_features=patient_clinical,
        feature_names=feature_cols,
        background_clinical=X_train,
    )

    print(f"\nPredicted Risk: {RISK_LABELS[pred_class]}")
    print(f"Probabilities — Low: {probs[0]:.3f} | Moderate: {probs[1]:.3f} | High: {probs[2]:.3f}")

    plot_shap_bar(
        shap_values,
        feature_cols,
        pred_class,
        probs,
        save_path=os.path.join(PLOT_SAVE_DIR, "shap_explanation.png"),
    )

    return shap_values, pred_class, probs


if __name__ == "__main__":
    run_shap_on_test_sample()