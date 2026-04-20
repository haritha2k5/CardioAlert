import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x):
        return self.block(x)


class ECGEncoder(nn.Module):
    """
    1D CNN encoder for 12-lead ECG signals.
    Input shape:  (batch, 12, 1000)
    Output shape: (batch, embedding_dim)  — embedding, NOT classification logits
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            ConvBlock1D(12, 32, kernel_size=7),
            nn.MaxPool1d(kernel_size=2, stride=2),   # (batch, 32, 500)

            ConvBlock1D(32, 64, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, stride=2),   # (batch, 64, 250)

            ConvBlock1D(64, 128, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, stride=2),   # (batch, 128, 125)

            ConvBlock1D(128, 256, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2),   # (batch, 256, 62)

            ConvBlock1D(256, 128, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),                  # (batch, 128, 1)
        )

        self.projection = nn.Sequential(
            nn.Flatten(),                             # (batch, 128)
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.encoder(x)
        embedding = self.projection(features)
        return embedding


class ECGClassifier(nn.Module):
    """
    Full classifier wrapping ECGEncoder with a classification head.
    Used during standalone ECG model training on PTB-XL.
    At fusion time, only ECGEncoder is used.
    """

    def __init__(self, embedding_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.encoder = ECGEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits

    def get_embedding(self, x):
        return self.encoder(x)