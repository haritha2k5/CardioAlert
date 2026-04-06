import os
import ast
import wfdb
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


LABEL_MAP = {"NORM": 0, "MI": 1}


def load_ptbxl_metadata(ptbxl_path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ptbxl_path, "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    scp_df = pd.read_csv(os.path.join(ptbxl_path, "scp_statements.csv"), index_col=0)

    def get_label(scp_dict):
        for code, confidence in scp_dict.items():
            if confidence >= 50 and code in scp_df.index:
                diag_class = scp_df.loc[code, "diagnostic_class"]
                if diag_class in LABEL_MAP:
                    return diag_class
        return None

    df["label"] = df["scp_codes"].apply(get_label)
    df = df[df["label"].notna()].copy()
    df["label_idx"] = df["label"].map(LABEL_MAP)
    return df


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG recordings.
    Returns: signal tensor (12, 1000) and label index.
    Uses 100Hz recordings (filename_lr).
    """

    def __init__(self, ptbxl_path: str, split: str = "train", sampling_rate: int = 100):
        self.ptbxl_path = ptbxl_path
        self.sampling_rate = sampling_rate

        df = load_ptbxl_metadata(ptbxl_path)

        # Official PTB-XL split: strat_fold 1-8 = train, 9 = val, 10 = test
        if split == "train":
            self.df = df[df["strat_fold"] <= 8].reset_index(drop=True)
        elif split == "val":
            self.df = df[df["strat_fold"] == 9].reset_index(drop=True)
        elif split == "test":
            self.df = df[df["strat_fold"] == 10].reset_index(drop=True)
        else:
            raise ValueError(f"split must be train/val/test, got: {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = os.path.join(self.ptbxl_path, row["filename_lr"])
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # (1000, 12)
        signal = signal.T  # (12, 1000) — channels first for Conv1d
        signal = signal.astype(np.float32)

        # Normalize per lead
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

        label = int(row["label_idx"])
        return signal, label
