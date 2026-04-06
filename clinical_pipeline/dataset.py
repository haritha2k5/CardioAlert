import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# UCI column names (dataset has no header row in some versions)
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_COLS   = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL       = "target"


def load_uci_data(uci_path: str):
    """
    Loads and preprocesses UCI Heart Disease dataset.
    Returns train/val/test splits as numpy arrays ready for TabNet.
    """
    csv_file = os.path.join(uci_path, "heart.csv")
    df = pd.read_csv(csv_file)

    # Handle datasets with or without header
    if df.columns[0] == "age":
        pass  # header already present
    else:
        df = pd.read_csv(csv_file, header=None, names=UCI_COLUMNS)

    # Drop rows with missing values (UCI uses ? for missing)
    df = df.replace("?", np.nan).dropna()

    # Ensure correct dtypes
    for col in CATEGORICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Binarize target: 0 = no disease, 1 = disease present
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)

    # Train / val / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42, stratify=y)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

    # Scale numerical columns only (first 5 columns)
    scaler = StandardScaler()
    X_train[:, :5] = scaler.fit_transform(X_train[:, :5])
    X_val[:, :5]   = scaler.transform(X_val[:, :5])
    X_test[:, :5]  = scaler.transform(X_test[:, :5])

    cat_idxs = list(range(len(NUMERICAL_COLS), len(feature_cols)))
    cat_dims = [
        int(df[col].nunique()) + 1
        for col in CATEGORICAL_COLS
    ]

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        cat_idxs, cat_dims,
        scaler,
        feature_cols,
    )
