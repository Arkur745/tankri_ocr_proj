from pathlib import Path

import mlflow
import torch
from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    scheduler=None,
    save_best=True,
    save_dir="models",
):
    """
    Train a PyTorch model.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0

    for epoch in range(epochs):

        # ==================================================
        # TRAIN
        # ==================================================

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            leave=False,
        )

        for images, labels in train_bar:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        # ==================================================
        # VALIDATION
        # ==================================================

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Val]",
            leave=False,
        )

        with torch.no_grad():

            for images, labels in val_bar:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        # ==================================================
        # Scheduler
        # ==================================================

        if scheduler is not None:
            scheduler.step()

        # ==================================================
        # History
        # ==================================================

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # ==================================================
        # MLflow
        # ==================================================

        if mlflow.active_run():

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "train_accuracy",
                train_accuracy,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

        # ==================================================
        # Save Best Model
        # ==================================================

        if save_best and val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_val_accuracy": best_val_accuracy,
            }

            torch.save(
                checkpoint,
                save_dir / "best_model.pth",
            )

        # ==================================================
        # Save Last Model
        # ==================================================

        last_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        torch.save(
            last_checkpoint,
            save_dir / "last_model.pth",
        )

        # ==================================================
        # Print
        # ==================================================

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    # ==================================================
    # Final MLflow Logging
    # ==================================================

    if mlflow.active_run():
        mlflow.log_metric(
            "best_val_accuracy",
            best_val_accuracy,
        )

        mlflow.log_artifact(
            str(save_dir / "best_model.pth")
        )

    return history
