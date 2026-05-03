import torch
import torch.nn as nn


class SimpleAST(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)
