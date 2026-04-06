import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """
    Late fusion model that combines ECG and clinical embeddings.

    Input:
        ecg_embedding:      (batch, ecg_embedding_dim)      default 64
        clinical_embedding: (batch, clinical_embedding_dim) default 32

    Output:
        logits: (batch, 3)  — Low / Moderate / High risk
    """

    def __init__(
        self,
        ecg_embedding_dim: int = 64,
        clinical_embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        fused_dim = ecg_embedding_dim + clinical_embedding_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, num_classes),
        )

    def forward(self, ecg_emb, clinical_emb):
        fused = torch.cat([ecg_emb, clinical_emb], dim=1)
        logits = self.fusion_head(fused)
        return logits
