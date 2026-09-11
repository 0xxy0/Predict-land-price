import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    rule: str
    column: str | None
    severity: Severity
    count: int
    details: dict[str, Any]


REQUIRED_COLUMNS = frozenset(
    {
        "date",
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "street",
        "city",
        "statezip",
        "country",
    }
)

NUMERIC_COLUMNS = frozenset(
    {
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
    }
)

TEXT_COLUMNS = frozenset(
    {
        "street",
        "city",
        "statezip",
        "country",
    }
)

TARGET_COLUMN = "price"

MISSING_VALUE_POLICY = {
    TARGET_COLUMN: Severity.ERROR,
}

VALID_CODE_DOMAINS = {
    "waterfront": frozenset({0, 1}),
    "view": frozenset({0, 1, 2, 3, 4}),
    "condition": frozenset({1, 2, 3, 4, 5}),
}

DATETIME_COLUMNS = frozenset({"date"})

MINIMUM_RANGES = {
    "price": 0,
    "bedrooms": 0,
    "bathrooms": 0,
    "sqft_living": 1,
    "sqft_lot": 1,
    "floors": 0.5,
    "sqft_above": 0,
    "sqft_basement": 0,
}

SQUARE_FOOTAGE_COLUMNS = frozenset(
    {
        "sqft_living",
        "sqft_above",
        "sqft_basement",
    }
)

STATEZIP_PATTERN = re.compile(r"^[A-Z]{2}\s\d{5}$")


def validate_frame(frame: pd.DataFrame) -> list[ValidationIssue]:
    """Returns a list of validation issues found in the given DataFrame."""
    actual_columns = set(frame.columns)
    issues: list[ValidationIssue] = []

    missing_columns = sorted(REQUIRED_COLUMNS - actual_columns)
    if missing_columns:
        issues.append(
            ValidationIssue(
                rule="missing_required_columns",
                column=None,
                severity=Severity.ERROR,
                count=len(missing_columns),
                details={"columns": missing_columns},
            )
        )

    unexpected_columns = sorted(actual_columns - REQUIRED_COLUMNS)
    if unexpected_columns:
        issues.append(
            ValidationIssue(
                rule="unexpected_columns",
                column=None,
                severity=Severity.WARNING,
                count=len(unexpected_columns),
                details={"columns": unexpected_columns},
            )
        )

    for column in sorted(NUMERIC_COLUMNS & actual_columns):
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        invalid_mask = frame[column].notna() & numeric_values.isna()

        if invalid_mask.any():
            issues.append(
                ValidationIssue(
                    rule="non_numeric_values",
                    column=column,
                    severity=Severity.ERROR,
                    count=int(invalid_mask.sum()),
                    details={
                        "values": frame.loc[invalid_mask, column].astype(str).tolist()
                    },
                )
            )

    for column in sorted(REQUIRED_COLUMNS & actual_columns):
        missing_mask = frame[column].isna()

        if missing_mask.any():
            severity = MISSING_VALUE_POLICY.get(
                column,
                Severity.WARNING,
            )

            issues.append(
                ValidationIssue(
                    rule="missing_values",
                    column=column,
                    severity=severity,
                    count=int(missing_mask.sum()),
                    details={"missing_rows": frame.index[missing_mask].tolist()},
                )
            )

    for column, allowed_values in VALID_CODE_DOMAINS.items():
        if column not in actual_columns:
            continue

        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        invalid_mask = (
            frame[column].notna()
            & numeric_values.notna()
            & ~numeric_values.isin(allowed_values)
        )

        if invalid_mask.any():
            issues.append(
                ValidationIssue(
                    rule="invalid_category",
                    column=column,
                    severity=Severity.ERROR,
                    count=int(invalid_mask.sum()),
                    details={
                        "values": frame.loc[invalid_mask, column].tolist(),
                        "allowed_values": sorted(allowed_values),
                    },
                )
            )

    for column in sorted(DATETIME_COLUMNS & actual_columns):
        datetime_values = pd.to_datetime(frame[column], errors="coerce")
        invalid_mask = frame[column].notna() & datetime_values.isna()

        if invalid_mask.any():
            issues.append(
                ValidationIssue(
                    rule="invalid_dates",
                    column=column,
                    severity=Severity.ERROR,
                    count=int(invalid_mask.sum()),
                    details={
                        "values": frame.loc[invalid_mask, column].astype(str).tolist()
                    },
                )
            )

    for column, minimum in MINIMUM_RANGES.items():
        if column not in actual_columns:
            continue

        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        if column in {"price", "sqft_living", "sqft_lot"}:
            invalid_mask = (
                frame[column].notna()
                & numeric_values.notna()
                & (numeric_values <= minimum)
            )
        else:
            invalid_mask = (
                frame[column].notna()
                & numeric_values.notna()
                & (numeric_values < minimum)
            )

        if invalid_mask.any():
            issues.append(
                ValidationIssue(
                    rule="out_of_range",
                    column=column,
                    severity=Severity.ERROR,
                    count=int(invalid_mask.sum()),
                    details={
                        "minimum": minimum,
                        "values": frame.loc[invalid_mask, column].tolist(),
                    },
                )
            )

    if SQUARE_FOOTAGE_COLUMNS <= actual_columns:
        square_footage = frame[list(SQUARE_FOOTAGE_COLUMNS)].apply(
            pd.to_numeric,
            errors="coerce",
        )

        complete_mask = square_footage.notna().all(axis=1)
        inconsistent_mask = complete_mask & (
            square_footage["sqft_above"] + square_footage["sqft_basement"]
            != square_footage["sqft_living"]
        )

        if inconsistent_mask.any():
            issues.append(
                ValidationIssue(
                    rule="inconsistent_square_footage",
                    column=None,
                    severity=Severity.WARNING,
                    count=int(inconsistent_mask.sum()),
                    details={
                        "columns": sorted(SQUARE_FOOTAGE_COLUMNS),
                        "rows": frame.index[inconsistent_mask].tolist(),
                    },
                )
            )

    if "statezip" in actual_columns:
        statezip_values = frame["statezip"].astype("string")
        invalid_mask = frame["statezip"].notna() & ~statezip_values.str.match(
            STATEZIP_PATTERN, na=False
        )

        if invalid_mask.any():
            issues.append(
                ValidationIssue(
                    rule="invalid_statezip",
                    column="statezip",
                    severity=Severity.ERROR,
                    count=int(invalid_mask.sum()),
                    details={
                        "values": frame.loc[invalid_mask, "statezip"]
                        .astype(str)
                        .tolist()
                    },
                )
            )

    if {"date", "yr_built", "yr_renovated"} <= actual_columns:
        sale_dates = pd.to_datetime(frame["date"], errors="coerce")
        built_years = pd.to_numeric(frame["yr_built"], errors="coerce")
        renovated_years = pd.to_numeric(
            frame["yr_renovated"],
            errors="coerce",
        )

        sale_years = sale_dates.dt.year

        comparable_mask = (
            sale_years.notna() & built_years.notna() & renovated_years.notna()
        )

        invalid_year_mask = comparable_mask & (
            (built_years > sale_years)
            | (
                (renovated_years != 0)
                & ((renovated_years < built_years) | (renovated_years > sale_years))
            )
        )

        if invalid_year_mask.any():
            issues.append(
                ValidationIssue(
                    rule="invalid_year_relationship",
                    column=None,
                    severity=Severity.ERROR,
                    count=int(invalid_year_mask.sum()),
                    details={
                        "columns": ["date", "yr_built", "yr_renovated"],
                        "rows": frame.index[invalid_year_mask].tolist(),
                    },
                )
            )

    duplicate_mask = frame.duplicated(keep=False)

    if duplicate_mask.any():
        issues.append(
            ValidationIssue(
                rule="duplicate_rows",
                column=None,
                severity=Severity.WARNING,
                count=int(duplicate_mask.sum()),
                details={
                    "rows": frame.index[duplicate_mask].tolist(),
                },
            )
        )

    return issues
