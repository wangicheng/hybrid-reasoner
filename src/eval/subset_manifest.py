import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


def build_subset_manifest(
    *,
    book_ids: Sequence[str],
    base_seed: int,
    repeats: int,
) -> Dict[str, Any]:
    normalized_book_ids = [
        str(book_id).strip()
        for book_id in book_ids
        if str(book_id).strip()
    ]

    manifest_repeats: List[Dict[str, Any]] = []
    for repeat_index in range(1, max(1, repeats) + 1):
        rng = random.Random(base_seed + repeat_index - 1)
        permutation = list(normalized_book_ids)
        rng.shuffle(permutation)
        manifest_repeats.append(
            {
                "repeat_index": repeat_index,
                "repeat_seed": base_seed + repeat_index - 1,
                "permutation_book_ids": permutation,
            }
        )

    return {
        "base_seed": base_seed,
        "catalog_size": len(normalized_book_ids),
        "repeats": manifest_repeats,
    }


def save_subset_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_subset_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Subset manifest must be an object: {path}")
    return manifest


def get_repeat_entry(manifest: Dict[str, Any], repeat_index: int) -> Dict[str, Any]:
    for repeat_entry in manifest.get("repeats", []):
        if int(repeat_entry.get("repeat_index", 0) or 0) == repeat_index:
            return repeat_entry
    raise KeyError(f"Repeat {repeat_index} not found in subset manifest")


def resolve_subset_book_ids(
    manifest: Dict[str, Any],
    *,
    repeat_index: int,
    subset_size: int,
) -> List[str]:
    repeat_entry = get_repeat_entry(manifest, repeat_index)
    permutation = [
        str(book_id).strip()
        for book_id in repeat_entry.get("permutation_book_ids", [])
        if str(book_id).strip()
    ]
    return permutation[: max(0, subset_size)]
