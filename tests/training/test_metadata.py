from pathlib import Path

import pytest

from house_prediction.training.metadata import load_metadata, save_metadata


def test_metadata_round_trip_equality(tmp_path: Path) -> None:
    metadata = {
        "model": "ridge",
        "alpha": 1.0,
        "metrics": {
            "mae": 10000.0,
            "rmse": 15000.0,
        },
        "features": ["bedrooms", "bathrooms", "sqft_living"],
    }

    path = tmp_path / "metadata.json"

    save_metadata(metadata, path)
    loaded_metadata = load_metadata(path)

    assert loaded_metadata == metadata


def test_nested_directories_are_created(tmp_path: Path) -> None:
    metadata = {"model": "ridge"}

    path = tmp_path / "artifacts" / "models" / "ridge" / "metadata.json"

    assert not path.parent.exists()

    save_metadata(metadata, path)

    assert path.parent.exists()
    assert path.exists()


def test_missing_metadata_file_raises_file_not_found_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing" / "metadata.json"

    with pytest.raises(FileNotFoundError):
        load_metadata(path)


def test_invalid_json_root_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"

    path.write_text('["ridge", 1.0]')

    with pytest.raises(ValueError):
        load_metadata(path)
