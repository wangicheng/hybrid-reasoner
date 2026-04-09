from pathlib import Path
from typing import Optional


DEFAULT_ANNOTATIONS_DIR = Path("data/experiments/annotations")
DEFAULT_ANNOTATIONS_FILENAME = "annotated.csv"


def resolve_pools_dir(base_dir: Path) -> Path:
    if base_dir.name == "pools":
        return base_dir

    if list(base_dir.glob("*_blind.csv")) or list(base_dir.glob("*_truth.json")):
        return base_dir

    return base_dir / "pools"


def resolve_annotations_dir(annotations_dir: Optional[str] = None) -> Path:
    if annotations_dir:
        return Path(annotations_dir)
    return DEFAULT_ANNOTATIONS_DIR


def resolve_annotation_output_path(
    experiment_name: str,
    annotations_dir: Optional[str] = None,
) -> Path:
    _ = experiment_name
    return resolve_annotations_dir(annotations_dir) / DEFAULT_ANNOTATIONS_FILENAME


def resolve_annotation_input_path(
    experiment_name: str,
    pools_dir: Path,
    annotations_dir: Optional[str] = None,
) -> Path:
    shared_dir = resolve_annotations_dir(annotations_dir)
    shared_path = resolve_annotation_output_path(
        experiment_name=experiment_name,
        annotations_dir=annotations_dir,
    )
    if shared_path.exists():
        return shared_path

    legacy_shared_path = shared_dir / f"{experiment_name}_annotated.csv"
    if legacy_shared_path.exists():
        return legacy_shared_path

    legacy_path = pools_dir / f"{experiment_name}_annotated.csv"
    return legacy_path
