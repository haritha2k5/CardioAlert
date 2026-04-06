import os
import sys
import io
import tempfile
import numpy as np
import torch
import wfdb
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ecg_pipeline.model import ECGClassifier
from src.clinical_pipeline.model import ClinicalClassifier
from src.clinical_pipeline.dataset import load_uci_data
from src.fusion.fusion_model import FusionModel
from src.explainability.shap_explain import explain_clinical_features

# ── Config ─────────────────────────────────────────────────────────────────────
ECG_MODEL_PATH      = "outputs/models/ecg_model.pt"
CLINICAL_MODEL_PATH = "outputs/models/clinical_model.pt"
FUSION_MODEL_PATH   = "outputs/models/fusion_model.pt"
UCI_PATH            = "data/uci"

ECG_EMB_DIM      = 64
CLINICAL_EMB_DIM = 32
RISK_LABELS      = ["Low", "Moderate", "High"]
DEVICE           = torch.device("cpu")

app = FastAPI(
    title="CardioAlert API",
    description="Silent Heart Attack Detection and Risk Prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading (singleton on startup) ───────────────────────────────────────
ecg_model      = None
clinical_model = None
fusion_model   = None
background_X   = None
feature_cols   = None
scaler         = None


@app.on_event("startup")
def load_models():
    global ecg_model, clinical_model, fusion_model, background_X, feature_cols, scaler

    # ECG model
    ecg_model = ECGClassifier(embedding_dim=ECG_EMB_DIM, num_classes=2).to(DEVICE)
    ecg_model.load_state_dict(torch.load(ECG_MODEL_PATH, map_location=DEVICE))
    ecg_model.eval()

    # Clinical model
    ckpt = torch.load(CLINICAL_MODEL_PATH, map_location=DEVICE)
    clinical_model = ClinicalClassifier(
        input_dim=ckpt["input_dim"],
        cat_idxs=ckpt["cat_idxs"],
        cat_dims=ckpt["cat_dims"],
        embedding_dim=CLINICAL_EMB_DIM,
        num_classes=2,
    ).to(DEVICE)
    clinical_model.load_state_dict(ckpt["model_state"])
    clinical_model.eval()
    scaler       = ckpt["scaler"]
    feature_cols = ckpt["feature_cols"]

    # Fusion model
    fusion_model = FusionModel(
        ecg_embedding_dim=ECG_EMB_DIM,
        clinical_embedding_dim=CLINICAL_EMB_DIM,
        num_classes=3,
    ).to(DEVICE)
    fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
    fusion_model.eval()

    # Background data for SHAP
    (X_train, *_) = load_uci_data(UCI_PATH)
    background_X = X_train

    print("All models loaded successfully.")


# ── Request / Response schemas ─────────────────────────────────────────────────
class ClinicalInput(BaseModel):
    age:      float
    sex:      int
    cp:       int
    trestbps: float
    chol:     float
    fbs:      int
    restecg:  int
    thalach:  float
    exang:    int
    oldpeak:  float
    slope:    int
    ca:       int
    thal:     int


class PredictionResponse(BaseModel):
    risk_label:     str
    risk_index:     int
    probabilities:  dict
    shap_explanation: dict


# ── Helper ─────────────────────────────────────────────────────────────────────
def preprocess_clinical(clinical: ClinicalInput) -> np.ndarray:
    """Convert ClinicalInput to preprocessed numpy array matching training format."""
    numerical = [
        clinical.age, clinical.trestbps, clinical.chol,
        clinical.thalach, clinical.oldpeak,
    ]
    categorical = [
        clinical.sex, clinical.cp, clinical.fbs, clinical.restecg,
        clinical.exang, clinical.slope, clinical.ca, clinical.thal,
    ]
    features = np.array(numerical + categorical, dtype=np.float32).reshape(1, -1)
    features[0, :5] = scaler.transform(features[:, :5])[0]   # scale numerical only
    return features


def load_ecg_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Load ECG from uploaded .hea/.dat file bytes.
    Expects a wfdb-compatible record uploaded as a zip or direct header bytes.
    For simplicity: writes to temp file, reads with wfdb.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        hea_path = os.path.join(tmpdir, "record.hea")
        with open(hea_path, "wb") as f:
            f.write(file_bytes)
        try:
            record = wfdb.rdrecord(os.path.join(tmpdir, "record"))
            signal = record.p_signal.T.astype(np.float32)   # (12, 1000)
            signal = (signal - signal.mean(axis=1, keepdims=True)) / (
                signal.std(axis=1, keepdims=True) + 1e-8
            )
            return signal[np.newaxis, ...]   # (1, 12, 1000)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"ECG file read error: {str(e)}")


def get_ecg_embedding(ecg_signal: np.ndarray) -> torch.Tensor:
    ecg_t = torch.tensor(ecg_signal, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return ecg_model.get_embedding(ecg_t)


def get_clinical_embedding(clinical_array: np.ndarray) -> torch.Tensor:
    clin_t = torch.tensor(clinical_array, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return clinical_model.get_embedding(clin_t)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "CardioAlert API is running.", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    ecg_file: UploadFile = File(..., description="ECG .hea file (wfdb format)"),
    age:      float = Form(...),
    sex:      int   = Form(...),
    cp:       int   = Form(...),
    trestbps: float = Form(...),
    chol:     float = Form(...),
    fbs:      int   = Form(...),
    restecg:  int   = Form(...),
    thalach:  float = Form(...),
    exang:    int   = Form(...),
    oldpeak:  float = Form(...),
    slope:    int   = Form(...),
    ca:       int   = Form(...),
    thal:     int   = Form(...),
):
    clinical_input = ClinicalInput(
        age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
        oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
    )

    # Process ECG
    ecg_bytes  = await ecg_file.read()
    ecg_signal = load_ecg_from_bytes(ecg_bytes)
    ecg_emb    = get_ecg_embedding(ecg_signal)

    # Process clinical
    clinical_array = preprocess_clinical(clinical_input)
    clinical_emb   = get_clinical_embedding(clinical_array)

    # Fuse and predict
    with torch.no_grad():
        logits = fusion_model(ecg_emb, clinical_emb)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))

    # SHAP explanation (clinical features only)
    try:
        shap_values, _, _ = explain_clinical_features(
            ecg_signal=ecg_signal,
            clinical_features=clinical_array,
            feature_names=feature_cols,
            background_clinical=background_X,
            n_samples=30,
        )
        shap_for_pred = shap_values[pred_class][0]
        shap_dict = {
            feat: round(float(val), 4)
            for feat, val in zip(feature_cols, shap_for_pred)
        }
    except Exception:
        shap_dict = {}

    return PredictionResponse(
        risk_label=RISK_LABELS[pred_class],
        risk_index=pred_class,
        probabilities={
            "Low":      round(float(probs[0]), 4),
            "Moderate": round(float(probs[1]), 4),
            "High":     round(float(probs[2]), 4),
        },
        shap_explanation=shap_dict,
    )


@app.post("/predict-clinical-only", response_model=PredictionResponse)
async def predict_clinical_only(clinical: ClinicalInput):
    """
    Predict using clinical data only (no ECG file).
    ECG embedding is replaced with a zero vector.
    Useful for demo or when ECG is unavailable.
    """
    clinical_array = preprocess_clinical(clinical)
    clinical_emb   = get_clinical_embedding(clinical_array)

    # Zero ECG embedding
    ecg_emb = torch.zeros(1, ECG_EMB_DIM).to(DEVICE)

    with torch.no_grad():
        logits = fusion_model(ecg_emb, clinical_emb)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))

    return PredictionResponse(
        risk_label=RISK_LABELS[pred_class],
        risk_index=pred_class,
        probabilities={
            "Low":      round(float(probs[0]), 4),
            "Moderate": round(float(probs[1]), 4),
            "High":     round(float(probs[2]), 4),
        },
        shap_explanation={},
    )


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
