import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

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

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 32 * 32,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x
    

class ResNet18Model(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = models.resnet18(weights=weights)

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze ONLY layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final classifier
        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)
