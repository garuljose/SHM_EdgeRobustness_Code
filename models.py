import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Full-signal / naive-impute classifier backbone: standard 1D-CNN."""

    def __init__(self, in_ch=4, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):  # x: (B, C, T)
        z = self.net(x).squeeze(-1)
        return self.fc(z)


class ReconGRU(nn.Module):
    """Reconstruction-then-classify baseline, approximating the
    gated-dilated-convolution + GRU reconstruction approach (Tran The et al., 2026):
    a GRU encoder-decoder denoising autoencoder that reconstructs the clean
    signal from corrupted input; the reconstruction is then handed to a
    (separately trained) downstream classifier at inference time."""

    def __init__(self, in_ch=4, hidden=48):
        super().__init__()
        self.encoder = nn.GRU(in_ch, hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.out = nn.Linear(hidden, in_ch)

    def forward(self, x):  # x: (B, T, C)
        h, _ = self.encoder(x)
        d, _ = self.decoder(h)
        return self.out(d)


class CompactEncoder(nn.Module):
    """Lightweight shared encoder for the proposed model: fewer channels than
    BaselineCNN, used both for self-supervised masked-reconstruction pretraining
    and (via CompactClassifier) direct classification from corrupted input --
    no explicit full-signal reconstruction step is needed at inference."""

    def __init__(self, in_ch=4, feat=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, feat, kernel_size=5, padding=2), nn.ReLU(),
        )

    def forward(self, x):  # x: (B, C, T) -> (B, feat, T/2)
        return self.net(x)


class CompactDecoder(nn.Module):
    """Small decoder used only during self-supervised pretraining (discarded
    at inference / fine-tuning time -- this is what keeps the deployed model
    small relative to ReconGRU)."""

    def __init__(self, feat=16, in_ch=4):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.net = nn.Sequential(
            nn.Conv1d(feat, 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(8, in_ch, kernel_size=5, padding=2),
        )

    def forward(self, z):
        return self.net(self.up(z))


class CompactClassifier(nn.Module):
    """Proposed model: CompactEncoder + small classification head. This is the
    module actually deployed / quantized at inference time."""

    def __init__(self, encoder: CompactEncoder, feat=16, n_classes=2):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feat, n_classes)

    def forward(self, x):  # x: (B, C, T)
        z = self.encoder(x)
        z = self.pool(z).squeeze(-1)
        return self.fc(z)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
