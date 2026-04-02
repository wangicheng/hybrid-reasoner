import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


DEFAULT_TAG_DESCRIPTIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "tag_descriptions.json"


@lru_cache(maxsize=1)
def load_tag_descriptions(path: Optional[str] = None) -> Dict[str, str]:
    """
    Load tag description metadata from JSON.

    The JSON file is expected to be a mapping from tag name to a short
    Traditional Chinese description.
    """
    desc_path = Path(path) if path else DEFAULT_TAG_DESCRIPTIONS_PATH
    try:
        with desc_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with desc_path.open("r", encoding="utf-16") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Tag description file '{desc_path}' not found.") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tag descriptions from '{desc_path}': {exc}"
        ) from exc

    if not isinstance(data, dict) or not data:
        raise RuntimeError(
            f"Tag description file '{desc_path}' is empty or has an unexpected format."
        )

    normalized: Dict[str, str] = {}
    for key, value in data.items():
        tag = str(key).strip()
        desc = str(value).strip()
        if tag and desc:
            normalized[tag] = desc
    if not normalized:
        raise RuntimeError(
            f"Tag description file '{desc_path}' does not contain usable descriptions."
        )
    return normalized


def build_tag_context_text(
    tag_order: Iterable[str],
    descriptions: Mapping[str, str],
) -> str:
    """
    Build a stable multiline prompt block from a tag order and description map.
    """
    lines = []
    for tag in tag_order:
        desc = descriptions.get(tag, "").strip()
        if not desc:
            desc = f"{tag} 類型相關作品。"
        lines.append(f"- {tag}：{desc}")
    return "\n".join(lines)
