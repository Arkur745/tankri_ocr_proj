"""
ResNet18 Model definition for Tankri OCR.
Handles baseline architecture loading and progressive domain adaptation freezing.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Model(nn.Module):
    """
    ResNet18 wrapper class with custom classification/adaptation heads.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
        unfreeze_layer3: bool = False,
        unfreeze_layer4: bool = True
    ):
        """
        Initialize the ResNet18 wrapper.

        Parameters
        ----------
        num_classes : int
            Number of output classes (alphabet size).
        pretrained : bool
            Whether to load pretrained ImageNet weights.
        dropout : float
            Dropout rate for classification layer.
        unfreeze_layer3 : bool
            Whether layer3 parameters are trainable.
        unfreeze_layer4 : bool
            Whether layer4 parameters are trainable.
        """
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

        # Freeze everything by default
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

        from src.configs import config
        enable_da = getattr(config, "ENABLE_DOMAIN_ADAPTATION", False)
        add_head = getattr(config, "ADD_ADAPTATION_HEAD", False)
        head_dim = getattr(config, "ADAPTATION_HEAD_DIM", 256)

        if enable_da and add_head:
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(head_dim, num_classes)
            )
        else:
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

        from src.utils.utils import get_layer_grad_status
        self.frozen_layers, self.trainable_layers = get_layer_grad_status(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass."""
        return self.model(x)


def configure_trainable_layers(model: nn.Module):
    """
    Configure layer freezing/unfreezing for Progressive Domain Adaptation.
    Prints a clean parameters summary to stdout.

    Parameters
    ----------
    model : nn.Module
        ResNet18Model instance to configure.
    """
    from src.configs import config
    
    # Check if we are using the ResNet18 wrapper or a sub-model
    resnet = getattr(model, "model", model)
    
    freeze_backbone = getattr(config, "FREEZE_BACKBONE", True)
    train_layer3 = getattr(config, "TRAIN_LAYER3", False)
    train_layer4 = getattr(config, "TRAIN_LAYER4", True)
    train_classifier = getattr(config, "TRAIN_CLASSIFIER", True)
    
    # Apply baseline freezing
    # Set all parameters to requires_grad = True first, then selectively freeze
    for param in model.parameters():
        param.requires_grad = True
        
    if freeze_backbone:
        if hasattr(resnet, "conv1"):
            for param in resnet.conv1.parameters():
                param.requires_grad = False
        if hasattr(resnet, "bn1"):
            for param in resnet.bn1.parameters():
                param.requires_grad = False
        if hasattr(resnet, "layer1"):
            for param in resnet.layer1.parameters():
                param.requires_grad = False
        if hasattr(resnet, "layer2"):
            for param in resnet.layer2.parameters():
                param.requires_grad = False
                
    if hasattr(resnet, "layer3"):
        for param in resnet.layer3.parameters():
            param.requires_grad = train_layer3
            
    if hasattr(resnet, "layer4"):
        for param in resnet.layer4.parameters():
            param.requires_grad = train_layer4
            
    if hasattr(resnet, "fc"):
        for param in resnet.fc.parameters():
            param.requires_grad = train_classifier

    # Dynamically update model wrapper attributes so downstream utils get correct values
    from src.utils.utils import get_layer_grad_status
    model.frozen_layers, model.trainable_layers = get_layer_grad_status(model)

    # Print summary
    print("\n====================================")
    print("Trainable Parameters")
    print("====================================")
    
    def get_status_str(requires_grad_list):
        if not requires_grad_list:
            return "N/A"
        if all(requires_grad_list):
            return "Trainable"
        if all(not r for r in requires_grad_list):
            return "Frozen"
        return "Partial"
        
    def get_params_grad_status(module):
        if module is None:
            return []
        return [p.requires_grad for p in module.parameters()]
        
    conv1_status = get_status_str(get_params_grad_status(getattr(resnet, "conv1", None)))
    layer1_status = get_status_str(get_params_grad_status(getattr(resnet, "layer1", None)))
    layer2_status = get_status_str(get_params_grad_status(getattr(resnet, "layer2", None)))
    layer3_status = get_status_str(get_params_grad_status(getattr(resnet, "layer3", None)))
    layer4_status = get_status_str(get_params_grad_status(getattr(resnet, "layer4", None)))
    classifier_status = get_status_str(get_params_grad_status(getattr(resnet, "fc", None)))
    
    print(f"Conv1 : {conv1_status}")
    print(f"Layer1 : {layer1_status}")
    print(f"Layer2 : {layer2_status}")
    print(f"Layer3 : {layer3_status}")
    print(f"Layer4 : {layer4_status}")
    print(f"Classifier : {classifier_status}")
    print("")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters : {total_params}")
    print(f"Trainable Parameters : {trainable_params}")
    print("====================================\n")
