from pathlib import Path

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
    filepath,
    label_to_idx=None,
    idx_to_label=None,
    image_size=None,
    num_classes=None,
):
    """
    Save model checkpoint.
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "image_size": image_size,
        "num_classes": num_classes,
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath,
    model,
    optimizer=None,
    map_location=None,
):
    """
    Load model checkpoint.

    Returns
    -------
    checkpoint : dict
    """

    checkpoint = torch.load(
        filepath,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_layer_grad_status(model):
    """
    Determine the frozen and trainable layers/components of a PyTorch model dynamically.
    Works for any architecture (EfficientNet, ViT, CRNN, ResNet, etc.) by grouping parameters by their top-level modules.
    """
    status = {}
    for name, param in model.named_parameters():
        parts = name.split('.')
        component = parts[0]
        # Skip wrapper self.model prefix if exists (e.g. for ResNet18Model wrapper)
        if component == "model" and len(parts) > 1:
            component = parts[1]
            
        if component not in status:
            status[component] = []
        status[component].append(param.requires_grad)

    frozen = []
    trainable = []
    for component, grads in status.items():
        if all(grads):
            trainable.append(component)
        elif all(not g for g in grads):
            frozen.append(component)
        else:
            trainable.append(f"{component} (partial)")
            frozen.append(f"{component} (partial)")

    frozen_str = ", ".join(frozen) if frozen else "none"
    trainable_str = ", ".join(trainable) if trainable else "none"
    return frozen_str, trainable_str

