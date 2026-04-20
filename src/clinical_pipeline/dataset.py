import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERICAL_COLS   = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COL       = "num"   

CAT_MAPS = {
    "sex":     {"Male": 0, "Female": 1},
    "cp":      {"typical angina": 0, "atypical angina": 1,
                "non-anginal": 2, "asymptomatic": 3},
    "fbs":     {True: 1, False: 0, "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0, 1: 1, 0: 0},
    "restecg": {"normal": 0, "st-t abnormality": 1,
                "lv hypertrophy": 2},
    "exang":   {True: 1, False: 0, "TRUE": 1, "FALSE": 0,
                "True": 1, "False": 0, 1: 1, 0: 0},
    "slope":   {"upsloping": 0, "flat": 1, "downsloping": 2},
    "thal":    {"normal": 0, "fixed defect": 1, "reversable defect": 2},
}


def load_uci_data(uci_path: str):
    """
    Loads and preprocesses the UCI Heart Disease dataset (920-row version).
    Handles text categories, missing values, and binarizes the target.
    Returns train/val/test splits as numpy arrays ready for TabNet.
    """
    for fname in ["heart.csv", "heart_disease_uci.csv"]:
        csv_file = os.path.join(uci_path, fname)
        if os.path.exists(csv_file):
            break
    else:
        raise FileNotFoundError(
            f"No UCI CSV found in {uci_path}. "
            "Expected heart.csv or heart_disease_uci.csv"
        )

    df = pd.read_csv(csv_file)
    print(f"Loaded {csv_file} — shape: {df.shape}")

    drop_cols = ["id", "dataset", "ca"]   
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col, mapping in CAT_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = df.dropna(subset=NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL])

    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    print(f"After cleaning — shape: {df.shape}")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    scaler = StandardScaler()
    X_train[:, :5] = scaler.fit_transform(X_train[:, :5])
    X_val[:, :5]   = scaler.transform(X_val[:, :5])
    X_test[:, :5]  = scaler.transform(X_test[:, :5])

    cat_idxs = list(range(len(NUMERICAL_COLS), len(feature_cols)))
    cat_dims = [int(df[col].nunique()) + 1 for col in CATEGORICAL_COLS]

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        cat_idxs, cat_dims,
        scaler,
        feature_cols,
    )