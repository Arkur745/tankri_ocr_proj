from pathlib import Path
import json

import mlflow
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.utils import get_layer_grad_status


def save_config_snapshot(output_path):
    """Save a snapshot of the full experiment configuration from config.py to a JSON file."""
    import src.configs.config as config
    from pathlib import Path
    
    config_dict = {}
    for attr in dir(config):
        if attr.isupper():
            val = getattr(config, attr)
            if isinstance(val, Path):
                val = str(val)
            try:
                json.dumps(val)
                config_dict[attr] = val
            except TypeError:
                pass
                
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
    return output_path


def log_experiment_config():
    """Automatically log all experiment configuration values from config.py to MLflow."""
    import mlflow
    import src.configs.config as config
    from pathlib import Path
    
    if not mlflow.active_run():
        return
        
    for attr in dir(config):
        if attr.isupper():
            val = getattr(config, attr)
            if isinstance(val, Path):
                val = str(val)
            try:
                mlflow.log_param(attr.lower(), val)
            except Exception:
                pass


def log_training_params(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    scheduler=None,
    image_size=None,
    label_to_idx=None,
):
    """Log training configuration parameters dynamically to MLflow."""
    active_run = mlflow.active_run()
    if not active_run:
        return
        
    client = mlflow.tracking.MlflowClient()
    run_id = active_run.info.run_id
    logged_params = client.get_run(run_id).data.params
    
    def safe_log_param(key, value):
        key_check = key.lower()
        existing_val = None
        existing_key = None
        for k, v in logged_params.items():
            if k.lower() == key_check:
                existing_val = v
                existing_key = k
                break
                
        if existing_key is None:
            mlflow.log_param(key, value)
        elif str(existing_val) != str(value):
            print(f"Warning: Param '{existing_key}' already logged with value '{existing_val}'. "
                  f"Attempted to log new value '{value}'. Skipping to prevent MLflow conflict.")

    train_size = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
    val_size = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 0
    dataset_size = train_size + val_size
    
    num_classes = len(label_to_idx) if label_to_idx is not None else (
        getattr(train_loader.dataset, "num_classes", None) if hasattr(train_loader, "dataset") else None
    )
    if num_classes is None and hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        num_classes = model.fc.out_features
    elif num_classes is None and hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for layer in reversed(model.classifier):
            if isinstance(layer, nn.Linear):
                num_classes = layer.out_features
                break
                
    safe_log_param("dataset_size", dataset_size)
    safe_log_param("train_size", train_size)
    safe_log_param("val_size", val_size)
    safe_log_param("num_classes", num_classes)
    
    # Reusable model metadata
    architecture = getattr(model, "architecture", model.__class__.__name__)
    if architecture == "ResNet18Model":
        architecture = "ResNet18"
    safe_log_param("model_architecture", architecture)
    
    pretrained = getattr(model, "pretrained", False)
    safe_log_param("pretrained", pretrained)
    
    if hasattr(model, "frozen_layers"):
        frozen_layers = model.frozen_layers
    else:
        frozen_layers, _ = get_layer_grad_status(model)
        
    if hasattr(model, "trainable_layers"):
        trainable_layers = model.trainable_layers
    else:
        _, trainable_layers = get_layer_grad_status(model)
        
    safe_log_param("frozen_layers", frozen_layers)
    safe_log_param("trainable_layers", trainable_layers)
    
    optimizer_name = optimizer.__class__.__name__
    learning_rate = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0].get('weight_decay', 0.0)
    scheduler_name = scheduler.__class__.__name__ if scheduler is not None else "None"
    loss_function = criterion.__class__.__name__
    label_smoothing = getattr(criterion, 'label_smoothing', 0.0)
    
    safe_log_param("optimizer", optimizer_name)
    safe_log_param("learning_rate", learning_rate)
    safe_log_param("weight_decay", weight_decay)
    safe_log_param("scheduler", scheduler_name)
    safe_log_param("loss_function", loss_function)
    safe_log_param("label_smoothing", label_smoothing)
    
    batch_size = train_loader.batch_size if hasattr(train_loader, "batch_size") else None
    safe_log_param("batch_size", batch_size)
    safe_log_param("epochs", epochs)
    
    if image_size is None:
        try:
            if hasattr(train_loader.dataset, "image_size"):
                image_size = train_loader.dataset.image_size
            else:
                sample_batch = next(iter(train_loader))
                image_size = sample_batch[0].shape[-1]
        except Exception:
            image_size = 224
    safe_log_param("image_size", image_size)
    
    from src.configs.config import RANDOM_SEED
    safe_log_param("random_seed", RANDOM_SEED)
    
    # Log safe tags (git hash, project root) under active run
    from src.utils.mlflow_utils import get_git_commit_hash
    from src.utils.mlflow_init import get_project_root
    
    project_root = get_project_root()
    commit_hash = get_git_commit_hash(project_root)
    if commit_hash is not None:
        mlflow.set_tag("git_commit", commit_hash)
    mlflow.set_tag("project_root", str(project_root))



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
    label_to_idx=None,
    idx_to_label=None,
    image_size=None,
):
    """
    Train a PyTorch model.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from src.configs import config
    enable_da = getattr(config, "ENABLE_DOMAIN_ADAPTATION", False)
    
    # Define checkpoint filenames based on adaptation status
    best_ckpt_name = "best_domain_adapted_model.pth" if enable_da else "best_model.pth"
    last_ckpt_name = "last_domain_adapted_model.pth" if enable_da else "last_model.pth"
    
    if enable_da:
        print("\n" + "="*50)
        print("DOMAIN ADAPTATION MODE DETECTED")
        print("="*50)
        
        # 1. Load baseline checkpoint weights
        checkpoint_path = Path(getattr(config, "DOMAIN_ADAPTATION_CHECKPOINT", "models/best_model.pth"))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(config.PROJECT_ROOT) / checkpoint_path
            
        print(f"Loading baseline weights from: {checkpoint_path}")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model_state_dict"]
            
            # Safe loading with shape filtering
            model_state = model.state_dict()
            filtered_state = {}
            for k, v in state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        print(f"Skipping key {k} due to shape mismatch: checkpoint {v.shape} vs model {model_state[k].shape}")
                else:
                    print(f"Skipping key {k} (not in model architecture)")
            
            model.load_state_dict(filtered_state, strict=False)
            print("Successfully loaded baseline weights!")
        else:
            print(f"WARNING: Baseline checkpoint not found at {checkpoint_path}. Starting adaptation from scratch/pretrained.")
            
        # 2. Reconfigure trainable layers
        from src.models.resnet import configure_trainable_layers
        configure_trainable_layers(model)
        
        # 3. Re-bind optimizer with domain adaptation LR and only trainable parameters
        import torch.optim as optim
        da_lr = getattr(config, "DOMAIN_ADAPTATION_LR", 1e-5)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=da_lr)
        print(f"Re-bound Adam optimizer with learning_rate={da_lr} and {len(trainable_params)} parameter groups.")
        
        # 4. Re-bind CosineAnnealingLR scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        da_epochs = getattr(config, "DOMAIN_ADAPTATION_EPOCHS", epochs)
        epochs = da_epochs # override loop count
        scheduler = CosineAnnealingLR(optimizer, T_max=da_epochs)
        print(f"Re-bound CosineAnnealingLR scheduler with T_max={da_epochs} epochs.")
        print("="*50 + "\n")

    # Resolve metadata for self-contained checkpoints
    if label_to_idx is None:
        label_to_idx = getattr(train_loader.dataset, "label_to_idx", None)
    if idx_to_label is None and label_to_idx is not None:
        idx_to_label = {v: k for k, v in label_to_idx.items()}
    if image_size is None:
        try:
            from src.configs.config import IMAGE_SIZE as config_image_size
            image_size = config_image_size
        except ImportError:
            image_size = 224
    num_classes = len(label_to_idx) if label_to_idx is not None else None

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0
    best_epoch = 0

    if not mlflow.active_run():
        print(
            "WARNING: No active MLflow run detected. "
            "Metrics and artifacts will not be logged. "
            "Wrap training in `with mlflow.start_run(...)` before calling train()."
        )
    else:
        # Log config snapshot
        config_snapshot_path = save_dir / "config_snapshot.json"
        save_config_snapshot(config_snapshot_path)
        mlflow.log_artifact(str(config_snapshot_path))
        
        # Log experiment configuration values dynamically
        log_experiment_config()

        # Log training parameters dynamically
        log_training_params(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            scheduler=scheduler,
            image_size=image_size,
            label_to_idx=label_to_idx,
        )

        # Log augmentation config automatically
        transform = getattr(train_loader.dataset, "transform", None)
        if transform is not None:
            aug_path = save_dir / "augmentation_config.txt"
            from src.utils.experiment_logging import save_augmentation_config
            save_augmentation_config(transform, aug_path)
            mlflow.log_artifact(str(aug_path))


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

            from src.configs.config import MIXUP_ALPHA
            if MIXUP_ALPHA > 0.0:
                import numpy as np
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                batch_size = images.size()[0]
                index = torch.randperm(batch_size).to(device)
                
                mixed_images = lam * images + (1 - lam) * images[index]
                labels_a, labels_b = labels, labels[index]
                
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                train_correct += (lam * (predictions == labels_a).sum().item() + (1 - lam) * (predictions == labels_b).sum().item())
                train_total += labels.size(0)
            else:
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
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_val_accuracy": best_val_accuracy,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "image_size": image_size,
                "num_classes": num_classes,
            }

            torch.save(
                checkpoint,
                save_dir / best_ckpt_name,
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
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "image_size": image_size,
            "num_classes": num_classes,
        }

        torch.save(
            last_checkpoint,
            save_dir / last_ckpt_name,
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

        mlflow.log_metric(
            "best_epoch",
            best_epoch,
        )

        mlflow.set_tag("best_epoch", str(best_epoch))
        mlflow.set_tag("best_val_accuracy", str(best_val_accuracy))

        if label_to_idx is not None and idx_to_label is not None:
            label_to_idx_path = save_dir / "label_to_idx.json"
            idx_to_label_path = save_dir / "idx_to_label.json"
            with open(label_to_idx_path, "w", encoding="utf-8") as f:
                json.dump(label_to_idx, f, ensure_ascii=False, indent=4)
            with open(idx_to_label_path, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in idx_to_label.items()}, f, ensure_ascii=False, indent=4)

    return history

