import json
from pathlib import Path
from typing import Any


def save_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Save metadata to a JSON file.

    Args:
        metadata (dict[str, Any]): Metadata to save.
        path (Path): Path to the output JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_metadata(path: Path) -> dict[str, Any]:
    """Load metadata from a JSON file.

    Args:
        path (Path): Path to the input JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(metadata, dict):
        raise ValueError(f"Loaded metadata is not a dictionary: {type(metadata)}")

    return metadata
