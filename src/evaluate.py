import torch


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate a trained model on a dataloader.

    Returns
    -------
    loss : float
    accuracy : float
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / len(dataloader)
    accuracy = correct / total

    return loss, accuracy


def predict(
    model,
    image,
    transform,
    device,
    idx_to_label,
):
    """
    Predict a single image.

    Parameters
    ----------
    image : PIL.Image

    Returns
    -------
    predicted_label : str
    """

    model.eval()

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        prediction = outputs.argmax(dim=1).item()

    return idx_to_label[prediction]

