import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNet


class ClinicalEncoder(nn.Module):
    """
    TabNet-based encoder for clinical tabular data.
    Input:  (batch, n_features) float32
    Output: (batch, embedding_dim) torch tensor
    """

    def __init__(
        self,
        input_dim: int,
        cat_idxs: list,
        cat_dims: list,
        embedding_dim: int = 32,
        n_steps: int = 3,
        n_d: int = 16,
        n_a: int = 16,
        n_independent: int = 2,
        n_shared: int = 2,
        momentum: float = 0.02,
        epsilon: float = 1e-15,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # cat_emb_dim must be a list — one embedding size per categorical column
        cat_emb_dim = [1] * len(cat_dims)

        self.tabnet = TabNet(
            input_dim=input_dim,
            output_dim=n_d,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=1.3,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            momentum=momentum,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,   # ← list, not int
            group_attention_matrix=torch.eye(input_dim),
        )

        self.projection = nn.Sequential(
            nn.Linear(n_d, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        tabnet_out, M_loss = self.tabnet(x)
        embedding = self.projection(tabnet_out)
        return embedding, M_loss


class ClinicalClassifier(nn.Module):
    """
    Full classifier wrapping ClinicalEncoder with classification head.
    Used during standalone clinical model training.
    At fusion time, only ClinicalEncoder is used.
    """

    def __init__(
        self,
        input_dim: int,
        cat_idxs: list,
        cat_dims: list,
        embedding_dim: int = 32,
        num_classes: int = 2,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
    ):
        super().__init__()
        self.encoder = ClinicalEncoder(
            input_dim=input_dim,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            embedding_dim=embedding_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        embedding, M_loss = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, M_loss

    def get_embedding(self, x):
        embedding, _ = self.encoder(x)
        return embedding