from pathlib import Path
from typing import Optional


DEFAULT_ANNOTATIONS_DIR = Path("data/experiments/annotations")
DEFAULT_ANNOTATIONS_FILENAME = "annotated.csv"


def resolve_annotations_dir(annotations_dir: Optional[str] = None) -> Path:
    if annotations_dir:
        return Path(annotations_dir)
    return DEFAULT_ANNOTATIONS_DIR


def resolve_annotation_path(
    annotations_dir: Optional[str] = None,
) -> Path:
    return resolve_annotations_dir(annotations_dir) / DEFAULT_ANNOTATIONS_FILENAME
