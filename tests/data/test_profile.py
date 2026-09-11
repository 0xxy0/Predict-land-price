import json
from pathlib import Path

import pandas as pd

from house_prediction.data.profile import (
    load_frame,
    profile_frame,
    write_json_report,
    write_markdown_report,
)
from house_prediction.data.schema import REQUIRED_COLUMNS


def make_profile_frame() -> pd.DataFrame:
    row: dict[str, object] = {
        "date": "2014-05-02 00:00:00",
        "price": 450000.0,
        "bedrooms": 3,
        "bathrooms": 1.5,
        "sqft_living": 1340,
        "sqft_lot": 7912,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_above": 1340,
        "sqft_basement": 0,
        "yr_built": 1955,
        "yr_renovated": 0,
        "street": "Example Street",
        "city": "Seattle",
        "statezip": "WA 98101",
        "country": "USA",
    }

    return pd.DataFrame([row])


def test_profile_reports_shape() -> None:
    frame = make_profile_frame()

    report = profile_frame(frame)

    assert report["row_count"] == 1
    assert report["column_count"] == len(REQUIRED_COLUMNS)
    assert report["duplicate_rows"] == 0


def test_profile_reports_missing_counts() -> None:
    frame = make_profile_frame()
    frame.loc[0, "price"] = None

    report = profile_frame(frame)

    assert report["missing_values"]["price"] == 1


def test_profile_reports_duplicates() -> None:
    frame = pd.concat(
        [make_profile_frame(), make_profile_frame()],
        ignore_index=True,
    )

    report = profile_frame(frame)

    assert report["row_count"] == 2
    # duplicated() counts the second occurrence as a duplicate not the first
    assert report["duplicate_rows"] == 1


def test_profile_includes_validation_issues() -> None:
    frame = make_profile_frame()
    frame.loc[0, "price"] = 0

    report = profile_frame(frame)

    issue_rules = {issue["rule"] for issue in report["validation_issues"]}

    assert "out_of_range" in issue_rules


def test_profile_does_not_mutate_input() -> None:
    frame = make_profile_frame()
    original = frame.copy(deep=True)

    profile_frame(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_json_report_is_written(tmp_path: Path) -> None:
    frame = make_profile_frame()
    report = profile_frame(frame)
    output_path = tmp_path / "reports" / "profile_report.json"
    write_json_report(report, output_path)
    assert output_path.exists()

    # Read the JSON file and verify its contents
    with output_path.open("r", encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert loaded_report["row_count"] == 1
    assert loaded_report["column_count"] == len(REQUIRED_COLUMNS)


def test_load_frame_reads_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    make_profile_frame().to_csv(input_path, index=False)

    frame = load_frame(input_path)

    assert len(frame) == 1
    assert set(frame.columns) == set(sorted(REQUIRED_COLUMNS))


def test_markdown_report_is_written(tmp_path: Path) -> None:
    frame = make_profile_frame()
    report = profile_frame(frame)
    output_path = tmp_path / "reports" / "profile_report.md"

    write_markdown_report(report, output_path)

    markdown = output_path.read_text(encoding="utf-8")

    assert "# Data Profile" in markdown
    assert "## Overview" in markdown
    assert "## Columns" in markdown
    assert "price" in markdown
    assert "## Validation Issues" in markdown


def test_markdown_report_contains_validation_issue(
    tmp_path: Path,
) -> None:
    frame = make_profile_frame()
    frame.loc[0, "price"] = 0
    report = profile_frame(frame)
    output_path = tmp_path / "profile_report.md"

    write_markdown_report(report, output_path)

    markdown = output_path.read_text(encoding="utf-8")

    assert "out_of_range" in markdown
    assert "error" in markdown
    assert "price" in markdown
