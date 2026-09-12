import json
from pathlib import Path
from typing import Any

import pandas as pd

from house_prediction.data.schema import validate_frame


def profile_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Profile a DataFrame and return a dictionary of statistics."""
    issues = validate_frame(frame)

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(count) for column, count in frame.isnull().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_counts": {column: int(unique) for column, unique in frame.nunique(dropna=False).items()},
        "validation_issues": [
            {
                "rule": issue.rule,
                "column": issue.column,
                "severity": issue.severity.value,
                "count": int(issue.count),
                "details": issue.details,
            }
            for issue in issues
        ],
    }


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    """Write the report dictionary to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, sort_keys=True, ensure_ascii=False)


def load_frame(input_path: Path) -> pd.DataFrame:
    """Load a CSV file for profiling."""
    return pd.read_csv(input_path)


def write_markdown_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a human-readable profiling report as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Profile",
        "",
        "## Overview",
        "",
        f"- Rows: {report['row_count']}",
        f"- Columns: {report['column_count']}",
        f"- Duplicate rows: {report['duplicate_rows']}",
        "",
        "## Columns",
        "",
        "| Column | Data Type | Missing | Unique |",
        "|---|---|---:|---:|",
    ]

    for column in report["columns"]:
        lines.append(f"| {column} | {report['dtypes'][column]} | {report['missing_values'][column]} | {report['unique_counts'][column]} |")

    lines.extend(
        [
            "",
            "## Validation Issues",
            "",
        ]
    )

    if report["validation_issues"]:
        lines.extend(
            [
                "| Rule | Column | Severity | Count |",
                "|---|---|---|---:|",
            ]
        )

        for issue in report["validation_issues"]:
            lines.append(f"| {issue['rule']} | {issue['column'] or ''} | {issue['severity']} | {issue['count']} |")
    else:
        lines.append("No validation issues found.")

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Main function to profile a CSV file and write reports."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile a CSV file and generate reports.",
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to the output directory for the JSON report.",
        required=True,
    )

    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Path to the output directory for the Markdown report.",
        required=True,
    )

    args = parser.parse_args()

    frame = load_frame(args.input_path)
    report = profile_frame(frame)

    write_json_report(report, args.json_output)
    write_markdown_report(report, args.markdown_output)


if __name__ == "__main__":
    main()
