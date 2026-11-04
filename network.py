import torch
from torch import nn


class MultiScaleBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations):
        super().__init__()
        width = channels // 3

        # narrow, medium and wide branches
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels, width, kernel_size=k, dilation=d, padding=d * (k - 1) // 2),
                nn.GroupNorm(4, width),
                nn.GELU(),
            )
            for k, d in zip(kernel_sizes, dilations)
        )
        # 1x1 conv that mixes the three branches
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.Dropout(0.05),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(x + self.fusion(out))  # residual connection


class MultiScaleResidualCNN(nn.Module):
    def __init__(self, channels=48, feature_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, padding=7),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            MultiScaleBlock(channels, kernel_sizes=(5, 7, 9), dilations=(1, 2, 4)),
            nn.AvgPool1d(2),
            MultiScaleBlock(channels, kernel_sizes=(3, 5, 7), dilations=(1, 3, 6)),
            nn.AvgPool1d(2),
            MultiScaleBlock(channels, kernel_sizes=(3, 5, 7), dilations=(1, 4, 8)),
            nn.AvgPool1d(2),
            # summarize the spectrum into 128 features
            nn.AdaptiveAvgPool1d(12),
            nn.Flatten(),
            nn.Linear(channels * 12, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(self.stem(x))).squeeze(1)
