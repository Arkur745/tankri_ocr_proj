from pathlib import Path

from PIL import Image
import torch


def load_model(model, checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Supports both checkpoint dicts and plain state_dicts
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def predict_image(
    image_path,
    model,
    transform,
    idx_to_label,
    device,
):
    """
    Predict a single image.

    Returns
    -------
    predicted_label : str
    """

    image_path = Path(image_path)

    image = Image.open(image_path).convert("L")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        prediction = outputs.argmax(dim=1).item()

    return idx_to_label[prediction]


def predict_folder(
    folder_path,
    model,
    transform,
    idx_to_label,
    device,
):
    """
    Predict all images inside a folder.

    Returns
    -------
    dict
        {
            "1.png": "𑚀",
            "2.png": "𑚁",
            ...
        }
    """

    folder_path = Path(folder_path)

    predictions = {}

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
    }

    for image_path in sorted(folder_path.iterdir()):

        if image_path.suffix.lower() not in image_extensions:
            continue

        label = predict_image(
            image_path=image_path,
            model=model,
            transform=transform,
            idx_to_label=idx_to_label,
            device=device,
        )

        predictions[image_path.name] = label

    return predictions
