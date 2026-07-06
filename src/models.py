import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):

    def __init__(self, num_classes, dropout=0.0):
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
                nn.Linear(
                    128 * 32 * 32,
                    128
                ),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(
                    128,
                    num_classes
                ),
            )
        else:
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

        from src.utils import get_layer_grad_status
        _, self.trainable_layers = get_layer_grad_status(self)

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x
    

class ResNet18Model(nn.Module):

    def __init__(self, num_classes, pretrained=True, dropout=0.0, unfreeze_layer3=False, unfreeze_layer4=True):
        super().__init__()
        self.architecture = "ResNet18"
        self.pretrained = pretrained
        self.dropout = dropout
        self.unfreeze_layer3 = unfreeze_layer3
        self.unfreeze_layer4 = unfreeze_layer4

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = models.resnet18(weights=weights)

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer3 if specified
        if unfreeze_layer3:
            for param in self.model.layer3.parameters():
                param.requires_grad = True

        # Unfreeze layer4 if specified
        if unfreeze_layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True

        # Replace final classifier
        in_features = self.model.fc.in_features

        if dropout > 0.0:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.model.fc = nn.Linear(
                in_features,
                num_classes,
            )

        from src.utils import get_layer_grad_status
        self.frozen_layers, self.trainable_layers = get_layer_grad_status(self)

    def forward(self, x):
        return self.model(x)
