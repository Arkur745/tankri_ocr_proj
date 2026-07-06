from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def save_classification_report(
    y_true,
    y_pred,
    target_names,
    report_path,
    per_class_path,
):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_str = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    if report_path.suffix == ".csv":
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(target_names))),
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(report_path, index=True)
    else:
        report_path.write_text(report_str, encoding="utf-8")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        zero_division=0,
    )

    per_class_df = pd.DataFrame({
        "class_index": list(range(len(target_names))),
        "class_name": target_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    })

    per_class_path = Path(per_class_path)
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    per_class_df.to_csv(per_class_path, index=False)

    return report_str, per_class_df


def save_confusion_matrix(y_true, y_pred, target_names, image_path):
    """Save a confusion matrix plot image."""
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, target_names, fontsize=8)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing.*")
        plt.savefig(str(image_path), bbox_inches="tight", dpi=200)
        
    plt.close()
    return image_path


def save_augmentation_config(transform, output_path):
    """Save a readable augmentation pipeline description to a text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_text = repr(transform)
    except Exception:
        config_text = str(transform)

    output_path.write_text(config_text, encoding="utf-8")
    return output_path


def save_training_curves(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    image_path,
):
    """Save training curves for loss and accuracy."""
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(str(image_path.with_name("training_loss.png")), bbox_inches="tight", dpi=200)
    plt.close()

    accuracy_path = image_path.with_name("training_accuracy.png")
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(str(accuracy_path), bbox_inches="tight", dpi=200)
    plt.close()

    combined_path = image_path
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(str(combined_path), bbox_inches="tight", dpi=200)
    plt.close()

    return combined_path
