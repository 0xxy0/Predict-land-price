import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from house_prediction.data.schema import (
    DATETIME_COLUMNS,
    MINIMUM_RANGES,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    STATEZIP_PATTERN,
    VALID_CODE_DOMAINS,
)


@dataclass(frozen=True)
class CleaningAudit:
    """Results of a deterministic data cleaning run."""

    input_rows: int
    output_rows: int
    removed_rows: int
    removed_by_rule: dict[str, int]
    changed_by_rule: dict[str, int]
    warnings: list[str]


def _record_removed(
    frame: pd.DataFrame,
    mask: pd.Series,
    rule: str,
    removed_by_rule: dict[str, int],
    warnings: list[str],
) -> pd.DataFrame:
    count = int(mask.sum())
    if count:
        removed_by_rule[rule] = count
        warnings.append(f"{count} rows removed by {rule}.")
        return frame.loc[~mask].copy()
    return frame


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningAudit]:
    """Clean a raw housing frame and return it with an audit report."""
    frame = df.copy(deep=True)
    input_rows = len(frame)
    removed_by_rule: dict[str, int] = {}
    changed_by_rule: dict[str, int] = {}
    warnings: list[str] = []

    columns = frame.columns.tolist()
    normalized_columns = [str(column).strip().lower().replace(" ", "_") for column in columns]
    if columns != normalized_columns:
        frame.columns = normalized_columns
        changed_by_rule["normalize_column_names"] = len(columns)

    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    for column in ("street", "city", "country"):
        original = frame[column].astype("string")
        normalized = original.str.strip()
        changed = int((normalized != original).fillna(False).sum())
        if changed:
            frame[column] = normalized
            changed_by_rule[f"normalize_{column}"] = changed

    original_statezip = frame["statezip"].astype("string")
    normalized_statezip = original_statezip.str.strip().str.upper()
    statezip_changed = int((normalized_statezip != original_statezip).fillna(False).sum())
    if statezip_changed:
        frame["statezip"] = normalized_statezip
        changed_by_rule["normalize_statezip"] = statezip_changed

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in DATETIME_COLUMNS:
        parsed_dates = pd.to_datetime(frame[column], errors="coerce", format="mixed")
        invalid_date_mask = frame[column].notna() & parsed_dates.isna()
        frame = _record_removed(
            frame,
            invalid_date_mask,
            "remove_invalid_dates",
            removed_by_rule,
            warnings,
        )
        frame[column] = parsed_dates.loc[frame.index]

    missing_mask = frame[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    frame = _record_removed(
        frame,
        missing_mask,
        "remove_missing_required_values",
        removed_by_rule,
        warnings,
    )

    for column, allowed_values in VALID_CODE_DOMAINS.items():
        frame = _record_removed(
            frame,
            ~frame[column].isin(allowed_values),
            f"remove_invalid_{column}",
            removed_by_rule,
            warnings,
        )

    for column, minimum in MINIMUM_RANGES.items():
        if column in {"price", "sqft_living", "sqft_lot"}:
            invalid_mask = frame[column] <= minimum
        else:
            invalid_mask = frame[column] < minimum
        frame = _record_removed(
            frame,
            invalid_mask,
            "remove_zero_price" if column == "price" else f"remove_invalid_{column}",
            removed_by_rule,
            warnings,
        )

    frame = _record_removed(
        frame,
        frame.duplicated(),
        "remove_duplicates",
        removed_by_rule,
        warnings,
    )

    sale_year = frame["date"].dt.year
    invalid_year_mask = (frame["yr_built"] > sale_year) | ((frame["yr_renovated"] != 0) & ((frame["yr_renovated"] < frame["yr_built"]) | (frame["yr_renovated"] > sale_year)))
    frame = _record_removed(
        frame,
        invalid_year_mask,
        "remove_invalid_year_relationship",
        removed_by_rule,
        warnings,
    )

    square_footage_mismatch = frame["sqft_above"] + frame["sqft_basement"] != frame["sqft_living"]
    frame = _record_removed(
        frame,
        square_footage_mismatch,
        "remove_square_footage_mismatches",
        removed_by_rule,
        warnings,
    )

    valid_statezip = frame["statezip"].str.match(STATEZIP_PATTERN, na=False)
    frame = _record_removed(
        frame,
        ~valid_statezip,
        "remove_invalid_statezip",
        removed_by_rule,
        warnings,
    )

    split_statezip = frame["statezip"].str.extract(r"^(?P<state>[A-Z]{2}) (?P<zip_code>\d{5})$")
    if len(frame) and ("state" not in frame.columns or "zip_code" not in frame.columns):
        frame = frame.assign(
            state=split_statezip["state"].astype("string"),
            zip_code=split_statezip["zip_code"].astype("string"),
        )
        changed_by_rule["split_statezip"] = len(frame)

    frame = frame.reset_index(drop=True)
    output_rows = len(frame)
    audit_report = CleaningAudit(
        input_rows=input_rows,
        output_rows=output_rows,
        removed_rows=input_rows - output_rows,
        removed_by_rule=removed_by_rule,
        changed_by_rule=changed_by_rule,
        warnings=warnings,
    )
    return frame, audit_report


def write_cleaned_frame(frame: pd.DataFrame, output_path: Path) -> None:
    """Write cleaned data to a new CSV path, creating its parent directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_audit_report(audit: CleaningAudit, output_path: Path) -> None:
    """Write the cleaning audit as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(audit), indent=4, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    """Clean a CSV file and write cleaned data plus an audit report."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean a housing CSV file.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_path)
    cleaned_frame, audit = clean_frame(frame)
    write_cleaned_frame(cleaned_frame, args.output)
    write_audit_report(audit, args.audit_output)


if __name__ == "__main__":
    main()
