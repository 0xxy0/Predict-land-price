from pathlib import Path

import pandas as pd

from house_prediction.data.schema import (
    DATETIME_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TEXT_COLUMNS,
    Severity,
    validate_frame,
)

EXPECTED_COLUMNS = {
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


def make_valid_row() -> dict[str, object]:
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

    return row


def test_required_columns_match_raw_dataset_contract() -> None:
    assert REQUIRED_COLUMNS == EXPECTED_COLUMNS


def test_column_groups_do_not_contain_unknown_columns() -> None:
    grouped_columns = DATETIME_COLUMNS | NUMERIC_COLUMNS | TEXT_COLUMNS

    assert grouped_columns <= REQUIRED_COLUMNS


def test_missing_required_column_is_reported() -> None:

    frame = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS - {"price"}))
    issues = validate_frame(frame)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.rule == "missing_required_columns"
    assert issue.column is None
    assert issue.severity == Severity.ERROR
    assert issue.count == len(REQUIRED_COLUMNS) - len(frame.columns)
    assert issue.details["columns"] == ["price"]


def test_complete_columns_have_no_missing_column_issue() -> None:

    frame = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))

    issues = validate_frame(frame)

    assert issues == []


def test_extra_column_is_reported_as_warning() -> None:
    import pandas as pd

    frame = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS | {"source_file"}))

    issues = validate_frame(frame)

    assert len(issues) == 1
    issue = issues[0]
    assert issue.rule == "unexpected_columns"
    assert issue.column is None
    assert issue.severity == Severity.WARNING
    assert issue.count == 1
    assert issue.details["columns"] == ["source_file"]


def test_missing_and_extra_columns_produce_two_issues() -> None:
    frame = pd.DataFrame(columns=sorted((REQUIRED_COLUMNS - {"price"}) | {"source_file"}))

    issues = validate_frame(frame)

    assert len(issues) == 2
    assert issues[0].rule == "missing_required_columns"
    assert issues[0].severity == Severity.ERROR
    assert issues[1].rule == "unexpected_columns"
    assert issues[1].severity == Severity.WARNING


def test_non_numeric_value_is_reported() -> None:
    row = make_valid_row()
    row["price"] = "not-a-number"

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    numeric_issue = next(issue for issue in issues if issue.rule == "non_numeric_values")

    assert numeric_issue.column == "price"
    assert numeric_issue.severity == Severity.ERROR
    assert numeric_issue.count == 1
    assert numeric_issue.details["values"] == ["not-a-number"]


def test_numeric_string_is_accepted() -> None:
    row = make_valid_row()
    row["price"] = "450000.0"

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    assert not any(issue.rule == "non_numeric_values" for issue in issues)


def test_missing_value_is_not_a_numeric_type_issue() -> None:
    row = make_valid_row()
    row["price"] = None

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    assert not any(issue.rule == "non_numeric_values" for issue in issues)


def test_numeric_validation_does_not_mutate_input() -> None:
    row = make_valid_row()
    row["price"] = "450000.0"

    frame = pd.DataFrame([row])
    original_price = frame["price"].copy()

    validate_frame(frame)

    pd.testing.assert_series_equal(frame["price"], original_price)


def test_missing_target_is_reported_as_error() -> None:
    row = make_valid_row()
    row["price"] = None

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    missing_issue = next(issue for issue in issues if issue.rule == "missing_values")

    assert missing_issue.column == "price"
    assert missing_issue.severity == Severity.ERROR
    assert missing_issue.count == 1


def test_missing_feature_is_reported_as_warning() -> None:
    row = make_valid_row()
    row["bedrooms"] = None

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    missing_issue = next(issue for issue in issues if issue.rule == "missing_values" and issue.column == "bedrooms")

    assert missing_issue.severity == Severity.WARNING
    assert missing_issue.count == 1


def test_missing_value_count_is_per_column() -> None:
    first_row = make_valid_row()
    second_row = make_valid_row()

    first_row["bedrooms"] = None
    second_row["bedrooms"] = None

    frame = pd.DataFrame([first_row, second_row])

    issues = validate_frame(frame)

    missing_issue = next(issue for issue in issues if issue.rule == "missing_values" and issue.column == "bedrooms")

    assert missing_issue.count == 2


def test_missing_value_validation_does_not_mutate_input() -> None:
    row = make_valid_row()
    row["price"] = None

    frame = pd.DataFrame([row])
    original_frame = frame.copy(deep=True)

    validate_frame(frame)

    pd.testing.assert_frame_equal(frame, original_frame)


def test_invalid_categorical_code_is_reported() -> None:
    row = make_valid_row()
    row["waterfront"] = 2

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    domain_issue = next(issue for issue in issues if issue.rule == "invalid_category" and issue.column == "waterfront")

    assert domain_issue.severity == Severity.ERROR
    assert domain_issue.count == 1
    assert domain_issue.details["values"] == [2]


def test_valid_categorical_codes_are_accepted() -> None:
    row = make_valid_row()
    row["waterfront"] = 1
    row["view"] = 4
    row["condition"] = 5

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    assert not any(issue.rule == "invalid_category" for issue in issues)


def test_multiple_invalid_categories_are_reported_per_column() -> None:
    row = make_valid_row()
    row["waterfront"] = 2
    row["view"] = 8

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    category_issues = [issue for issue in issues if issue.rule == "invalid_category"]

    assert len(category_issues) == 2
    assert {issue.column for issue in category_issues} == {
        "waterfront",
        "view",
    }


def test_invalid_date_is_reported() -> None:
    row = make_valid_row()
    row["date"] = "not-a-date"

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    date_issue = next(issue for issue in issues if issue.rule == "invalid_dates")

    assert date_issue.column == "date"
    assert date_issue.severity == Severity.ERROR
    assert date_issue.count == 1
    assert date_issue.details["values"] == ["not-a-date"]


def test_valid_date_is_accepted() -> None:
    row = make_valid_row()
    row["date"] = "2014-05-02 00:00:00"

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    assert not any(issue.rule == "invalid_dates" for issue in issues)


def test_date_validation_does_not_mutate_input() -> None:
    row = make_valid_row()
    frame = pd.DataFrame([row])
    original_date = frame["date"].copy()

    validate_frame(frame)

    pd.testing.assert_series_equal(frame["date"], original_date)


def test_non_positive_price_is_reported() -> None:
    row = make_valid_row()
    row["price"] = 0

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    range_issue = next(issue for issue in issues if issue.rule == "out_of_range" and issue.column == "price")

    assert range_issue.severity == Severity.ERROR
    assert range_issue.count == 1


def test_zero_basement_area_is_valid() -> None:
    row = make_valid_row()
    row["sqft_basement"] = 0

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    assert not any(issue.rule == "out_of_range" and issue.column == "sqft_basement" for issue in issues)


def test_negative_living_area_is_reported() -> None:
    row = make_valid_row()
    row["sqft_living"] = -1

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    range_issue = next(issue for issue in issues if issue.rule == "out_of_range" and issue.column == "sqft_living")

    assert range_issue.count == 1


def test_multiple_out_of_range_rows_are_counted() -> None:
    first_row = make_valid_row()
    second_row = make_valid_row()

    first_row["price"] = 0
    second_row["price"] = -10

    frame = pd.DataFrame([first_row, second_row])

    issues = validate_frame(frame)

    range_issue = next(issue for issue in issues if issue.rule == "out_of_range" and issue.column == "price")

    assert range_issue.count == 2


def test_square_footage_relationship_is_valid() -> None:
    row = make_valid_row()
    row["sqft_living"] = 2000
    row["sqft_above"] = 1500
    row["sqft_basement"] = 500

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    assert not any(issue.rule == "inconsistent_square_footage" for issue in issues)


def test_square_footage_relationship_violation_is_reported() -> None:
    row = make_valid_row()
    row["sqft_living"] = 2000
    row["sqft_above"] = 1500
    row["sqft_basement"] = 400

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    relationship_issue = next(issue for issue in issues if issue.rule == "inconsistent_square_footage")

    assert relationship_issue.column is None
    assert relationship_issue.severity == Severity.WARNING
    assert relationship_issue.count == 1
    assert set(relationship_issue.details["columns"]) == {
        "sqft_living",
        "sqft_above",
        "sqft_basement",
    }


def test_invalid_statezip_is_reported() -> None:
    row = make_valid_row()
    row["statezip"] = "invalid-zip"

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    statezip_issue = next(issue for issue in issues if issue.rule == "invalid_statezip")

    assert statezip_issue.column == "statezip"
    assert statezip_issue.severity == Severity.ERROR
    assert statezip_issue.count == 1


def test_renovation_before_construction_is_reported() -> None:
    row = make_valid_row()
    row["yr_built"] = 2000
    row["yr_renovated"] = 1990

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    year_issue = next(issue for issue in issues if issue.rule == "invalid_year_relationship")

    assert year_issue.severity == Severity.ERROR
    assert year_issue.count == 1


def test_zero_renovation_year_is_valid_sentinel() -> None:
    row = make_valid_row()
    row["yr_renovated"] = 0

    frame = pd.DataFrame([row])

    issues = validate_frame(frame)

    assert not any(issue.rule == "invalid_year_relationship" for issue in issues)


def test_duplicate_rows_are_reported() -> None:
    row = make_valid_row()
    frame = pd.DataFrame([row, row])

    issues = validate_frame(frame)

    duplicate_issue = next(issue for issue in issues if issue.rule == "duplicate_rows")

    assert duplicate_issue.severity == Severity.WARNING
    assert duplicate_issue.count == 2


def test_raw_dataset_has_expected_schema_and_known_findings() -> None:
    path = Path("data/raw/KC_housing_data.csv")
    frame = pd.read_csv(path)

    issues = validate_frame(frame)

    assert not any(issue.rule == "missing_required_columns" for issue in issues)
    assert not any(issue.rule == "unexpected_columns" for issue in issues)
    assert not any(issue.rule == "invalid_dates" for issue in issues)
    assert not any(issue.rule == "invalid_statezip" for issue in issues)

    price_issue = next(issue for issue in issues if issue.rule == "out_of_range" and issue.column == "price")
    assert price_issue.count == 49

    year_issue = next(issue for issue in issues if issue.rule == "invalid_year_relationship")
    assert year_issue.count == 195
