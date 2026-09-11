from typing import Any

import pandas as pd

from house_prediction.data.cleaner import clean_frame


def make_clean_frame() -> pd.DataFrame:
    # Use lowercase strings to match normalized formats natively
    row: dict[str, Any] = {
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


def test_clean_frame_removes_invalid_price_rows() -> None:
    frame = make_clean_frame()
    frame.loc[0, "price"] = -100000

    _, report = clean_frame(frame)

    assert report.removed_rows == 1
    assert report.removed_by_rule["remove_zero_price"] == 1


def test_clean_frame_removes_invalid_date_rows() -> None:
    frame = make_clean_frame()
    frame.loc[0, "date"] = "invalid_date"

    _, report = clean_frame(frame)

    assert report.removed_rows == 1
    assert report.removed_by_rule["remove_invalid_dates"] == 1


def test_clean_frame_removes_invalid_statezip_rows() -> None:
    frame = make_clean_frame()
    frame.loc[0, "statezip"] = "INVALID"

    _, report = clean_frame(frame)

    assert report.removed_rows == 1
    assert report.removed_by_rule["remove_invalid_statezip"] == 1


def test_clean_frame_removes_square_footage_mismatch_rows() -> None:
    frame = make_clean_frame()
    frame.loc[0, "sqft_above"] = 1000
    frame.loc[0, "sqft_basement"] = 400
    frame.loc[0, "sqft_living"] = 1500

    _, report = clean_frame(frame)

    assert report.removed_rows == 1
    assert report.removed_by_rule["remove_square_footage_mismatches"] == 1


def test_clean_frame_swaps_year_built_and_renovated() -> None:
    frame = make_clean_frame()
    frame.loc[0, "yr_built"] = 2000
    frame.loc[0, "yr_renovated"] = 1990

    cleaned_frame, report = clean_frame(frame)

    assert report.removed_by_rule["remove_invalid_year_relationship"] == 1
    assert cleaned_frame.empty


def test_clean_frame_removes_duplicates() -> None:
    frame = make_clean_frame()
    frame = pd.concat([frame, frame], ignore_index=True)

    _, report = clean_frame(frame)

    assert report.removed_rows == 1
    assert report.removed_by_rule["remove_duplicates"] == 1


def test_clean_frame_no_changes() -> None:
    frame = make_clean_frame()

    cleaned_frame, report = clean_frame(frame)

    assert report.removed_rows == 0
    assert report.changed_by_rule["split_statezip"] == 1
    assert cleaned_frame.loc[0, "state"] == "WA"
    assert cleaned_frame.loc[0, "zip_code"] == "98101"
    assert pd.api.types.is_datetime64_any_dtype(cleaned_frame["date"])


def test_clean_frame_multiple_issues() -> None:
    frame = make_clean_frame()
    frame.loc[0, "price"] = -100000
    frame.loc[0, "date"] = "invalid_date"
    frame.loc[0, "statezip"] = "INVALID"
    frame.loc[0, "sqft_above"] = 1000
    frame.loc[0, "sqft_basement"] = 400
    frame.loc[0, "sqft_living"] = 1500
    frame.loc[0, "yr_built"] = 2000
    frame.loc[0, "yr_renovated"] = 1990

    _, report = clean_frame(frame)

    # When multiple dropping operations target the exact same single row,
    # it can only be removed once from the absolute index.
    assert report.removed_rows == 1


def test_clean_frame_with_no_rows() -> None:
    frame = pd.DataFrame(
        columns=[
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
        ]
    )

    cleaned_frame, report = clean_frame(frame)

    assert report.removed_rows == 0
    assert report.changed_by_rule == {}
    assert cleaned_frame.empty


def test_clean_frame_is_idempotent() -> None:
    frame = make_clean_frame()

    first_frame, first_report = clean_frame(frame)
    second_frame, second_report = clean_frame(first_frame)

    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert second_report.removed_rows == 0
    assert second_report.removed_by_rule == {}
    assert second_report.changed_by_rule == {}
    assert first_report.output_rows == second_report.input_rows


def test_clean_frame_removes_invalid_categories() -> None:
    frame = make_clean_frame()
    frame.loc[0, "condition"] = 6

    _, report = clean_frame(frame)

    assert report.removed_by_rule["remove_invalid_condition"] == 1


def test_clean_frame_removes_missing_required_values() -> None:
    frame = make_clean_frame()
    frame.loc[0, "bedrooms"] = None

    _, report = clean_frame(frame)

    assert report.removed_by_rule["remove_missing_required_values"] == 1


def test_clean_frame_with_all_invalid_rows() -> None:
    frame = make_clean_frame()
    frame.loc[0, "price"] = -100000
    frame.loc[0, "date"] = "invalid_date"
    frame.loc[0, "statezip"] = "INVALID"
    frame.loc[0, "sqft_above"] = 1000
    frame.loc[0, "sqft_basement"] = 400
    frame.loc[0, "sqft_living"] = 1500
    frame.loc[0, "yr_built"] = 2000
    frame.loc[0, "yr_renovated"] = 1990

    cleaned_frame, report = clean_frame(frame)

    assert report.removed_rows == 1
    assert cleaned_frame.empty


def test_clean_frame_with_mixed_valid_and_invalid_rows() -> None:
    frame = make_clean_frame()
    frame.loc[0, "price"] = -100000
    frame.loc[0, "date"] = "invalid_date"
    frame.loc[0, "statezip"] = "INVALID"
    frame.loc[0, "sqft_above"] = 1000
    frame.loc[0, "sqft_basement"] = 400
    frame.loc[0, "sqft_living"] = 1500
    frame.loc[0, "yr_built"] = 2000
    frame.loc[0, "yr_renovated"] = 1990

    valid_row = {
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
    frame = pd.concat([frame, pd.DataFrame([valid_row])], ignore_index=True)

    cleaned_frame, report = clean_frame(frame)

    assert report.removed_rows == 1
    assert cleaned_frame.shape[0] == 1
