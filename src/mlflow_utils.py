import subprocess
from pathlib import Path

import mlflow


def get_git_commit_hash(project_root: Path = None):
    """Return the current git commit hash if available."""
    project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit
    except Exception:
        return None


def init_mlflow(experiment_name: str, tracking_uri: str = None, project_root: Path = None):
    """Initialize MLflow tracking for the project."""
    project_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    if tracking_uri is None:
        tracking_uri = str(project_root / "mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    commit_hash = get_git_commit_hash(project_root)
    if commit_hash is not None:
        mlflow.set_tag("git_commit", commit_hash)

    mlflow.set_tag("project_root", str(project_root))
    mlflow.set_tag("tracking_uri", tracking_uri)

    return tracking_uri


def serialize_transform(transform):
    """Serialize a torchvision transform pipeline into readable text."""
    try:
        return repr(transform)
    except Exception:
        try:
            return str(transform)
        except Exception:
            return "<unable to serialize transform>"


def save_augmentation_config(transform, output_path):
    """Save an augmentation pipeline description to a text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_transform(transform), encoding="utf-8")
    return output_path
