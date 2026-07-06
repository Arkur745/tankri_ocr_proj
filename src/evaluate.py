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


def log_evaluation_artifacts(
    model,
    val_loader,
    device,
    save_dir,
    history,
    label_to_idx,
    idx_to_label,
    best_checkpoint_name="best_model.pth",
):
    """
    Generate and log all evaluation artifacts.
    This should be called from the evaluation pipeline (notebook or module).
    """
    import mlflow
    import json
    from pathlib import Path
    from src.experiment_logging import (
        save_classification_report,
        save_confusion_matrix,
        save_training_curves,
    )
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the saved best model checkpoint weights
    checkpoint_path = save_dir / best_checkpoint_name
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded best checkpoint from {checkpoint_path} for evaluation.")
    else:
        print(f"Warning: Best checkpoint not found at {checkpoint_path}. Using current model state.")
        
    # 2. Log model to MLflow (must correspond to best validation state)
    if mlflow.active_run():
        mlflow.pytorch.log_model(model, artifact_path="model")
        
    # 3. Collect true labels and predictions
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            
    # 4. Generate & Save Artifacts
    target_names = [idx_to_label[i] for i in range(len(label_to_idx))]
    
    # Classification Report (CSV)
    report_path = save_dir / "classification_report.csv"
    per_class_path = save_dir / "per_class_metrics.csv"
    save_classification_report(y_true, y_pred, target_names, report_path, per_class_path)
    
    # Confusion Matrix
    cm_path = save_dir / "confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, target_names, cm_path)
    
    # Training Curves
    curves_path = save_dir / "training_curves.png"
    if history is not None:
        save_training_curves(
            history["train_loss"],
            history["val_loss"],
            history["train_accuracy"],
            history["val_accuracy"],
            curves_path,
        )
        
    # 5. Save Label Mappings
    label_to_idx_path = save_dir / "label_to_idx.json"
    idx_to_label_path = save_dir / "idx_to_label.json"
    with open(label_to_idx_path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=4)
    with open(idx_to_label_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_label.items()}, f, ensure_ascii=False, indent=4)
        
    # 6. Log everything to MLflow
    if mlflow.active_run():
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(per_class_path))
        mlflow.log_artifact(str(cm_path))
        if history is not None:
            mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(label_to_idx_path))
        mlflow.log_artifact(str(idx_to_label_path))
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))
            
    print("Logged evaluation artifacts to MLflow.")


