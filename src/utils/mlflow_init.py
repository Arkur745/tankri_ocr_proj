from pathlib import Path

import mlflow


def get_project_root():
    return Path(__file__).resolve().parent.parent


def get_tracking_uri(project_root: Path = None):
    project_root = project_root or get_project_root()
    return (project_root / "mlruns").as_uri()


def init_mlflow(experiment_name: str, tracking_uri: str = None):
    project_root = get_project_root()
    if tracking_uri is None:
        tracking_uri = get_tracking_uri(project_root)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri
