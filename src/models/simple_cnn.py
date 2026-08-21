"""
SimpleCNN baseline model for Tankri OCR.
Used as a lightweight custom baseline model.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Lightweight 3-layer Convolutional Neural Network baseline.
    """

    def __init__(self, num_classes: int, dropout: float = 0.0):
        """
        Initialize the CNN.

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        dropout : float
            Dropout rate for classification layer.
        """
        super().__init__()
        self.architecture = "SimpleCNN"
        self.pretrained = False
        self.dropout = dropout
        self.frozen_layers = "none"

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        if dropout > 0.0:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 32 * 32, 128),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 32 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        from src.utils.utils import get_layer_grad_status
        _, self.trainable_layers = get_layer_grad_status(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x
