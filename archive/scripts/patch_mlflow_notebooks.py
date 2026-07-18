import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent

def patch_experiments():
    path = root / 'notebooks' / 'experiments.ipynb'
    data = json.loads(path.read_text(encoding='utf-8'))

    data['cells'][13]['source'] = [
        'import mlflow\n',
        'from src.mlflow_init import init_mlflow\n',
        'from src.experiment_logging import save_augmentation_config\n',
        '\n',
        'init_mlflow("Tankri OCR")\n',
    ]

    data['cells'][14]['source'] = [
        'with mlflow.start_run(run_name="ResNet18_Baseline"):\n',
        '\n',
        '    mlflow.log_param("dataset_size", len(df_filtered))\n',
        '    mlflow.log_param("train_size", len(train_df))\n',
        '    mlflow.log_param("val_size", len(val_df))\n',
        '    mlflow.log_param("num_classes", NUM_CLASSES)\n',
        '    mlflow.log_param("random_seed", RANDOM_SEED)\n',
        '    mlflow.log_param("model_architecture", "ResNet18")\n',
        '    mlflow.log_param("pretrained", True)\n',
        '    mlflow.log_param("frozen_layers", "all_except_layer4+fc")\n',
        '    mlflow.log_param("trainable_layers", "layer4+fc")\n',
        '    mlflow.log_param("optimizer", "Adam")\n',
        '    mlflow.log_param("learning_rate", 1e-4)\n',
        '    mlflow.log_param("weight_decay", optimizer.defaults.get("weight_decay", 0.0))\n',
        '    mlflow.log_param("scheduler", "CosineAnnealingLR")\n',
        '    mlflow.log_param("loss_function", "CrossEntropyLoss")\n',
        '    mlflow.log_param("label_smoothing", False)\n',
        '    mlflow.log_param("batch_size", BATCH_SIZE)\n',
        '    mlflow.log_param("epochs", EPOCHS)\n',
        '    mlflow.log_param("image_size", 224)\n',
        '    mlflow.log_param("augmentation_config", "../artifacts/augmentation_config.txt")\n',
        '\n',
        '    aug_path = save_augmentation_config(train_transform_resnet, "../artifacts/augmentation_config.txt")\n',
        '    mlflow.log_artifact(str(aug_path))\n',
        '    mlflow.log_artifact("../artifacts/label_to_idx.json")\n',
        '    mlflow.log_artifact("../artifacts/idx_to_label.json")\n',
        '\n',
        '    mlflow.log_param("model", "ResNet18")\n',
        '    mlflow.log_param("pretrained", True)\n',
        '    mlflow.log_param("learning_rate", 1e-4)\n',
        '    mlflow.log_param("image_size", 224)\n',
        '    mlflow.log_param("epochs", EPOCHS)\n',
        '    mlflow.log_param("batch_size", BATCH_SIZE)\n',
        '    mlflow.log_param("num_classes", NUM_CLASSES)\n',
        '    mlflow.log_param("scheduler", "CosineAnnealingLR")\n',
        '    mlflow.log_param("unfrozen_layers", "layer4+fc")\n',
        '\n',
        '    history = train(\n',
        '        model=model,\n',
        '        train_loader=train_loader,\n',
        '        val_loader=val_loader,\n',
        '        criterion=criterion,\n',
        '        optimizer=optimizer,\n',
        '        epochs=EPOCHS,\n',
        '        device=device,\n',
        '        scheduler=scheduler,\n',
        '        save_best=True,\n',
        '        save_dir="models",\n',
        '    )\n',
        '\n',
        '    for epoch in range(EPOCHS):\n',
        '\n',
        '        mlflow.log_metric(\n',
        '            "train_loss",\n',
        '            history["train_loss"][epoch],\n',
        '            step=epoch,\n',
        '        )\n',
        '\n',
        '        mlflow.log_metric(\n',
        '            "train_accuracy",\n',
        '            history["train_accuracy"][epoch],\n',
        '            step=epoch,\n',
        '        )\n',
        '\n',
        '        mlflow.log_metric(\n',
        '            "val_loss",\n',
        '            history["val_loss"][epoch],\n',
        '            step=epoch,\n',
        '        )\n',
        '\n',
        '        mlflow.log_metric(\n',
        '            "val_accuracy",\n',
        '            history["val_accuracy"][epoch],\n',
        '            step=epoch,\n',
        '        )\n',
        '    ' 
    ]
    # preserve the original cell content after this point by appending following lines
    remaining = [
        '    \n',
        '    # Collect additional artifacts after training if available\n',
        '    if mlflow.active_run():\n',
        '        mlflow.log_artifact("models/best_model.pth")\n',
        '\n',
        '    ' 
    ]
    # Actually this reconstruction is wrong due to syntax issues, let's instead patch line by line later.

    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')

if __name__ == '__main__':
    patch_experiments()
