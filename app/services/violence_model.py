"""Conv3D Violence Classifier — trained on Real Life Violence Situations Dataset."""

import torch.nn as nn


class ViolenceClassifier(nn.Module):
    """
    Conv3D model for binary violence classification (v2 weights: ``violence_model_v2.pt``).
    Input: (batch, 3, 16, 64, 64) — 16 frames at 64x64
    Output: (batch, 2) — [non-violent, violent] logits (class 1 = violence)
    """

    NUM_FRAMES = 16
    FRAME_SIZE = 64

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(256 * 1 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
