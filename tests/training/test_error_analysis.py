import numpy as np
import pandas as pd
import pytest

from house_prediction.training.error_analysis import build_error_report

frame = pd.DataFrame(
    {
        "date": ["2014-05-02", "2014-06-03"],
        "price": [200_000.0, 1_500_000.0],
    }
)
predictions = np.array([180_000.0, 1_000_000.0])


def test_build_error_report() -> None:
    report = build_error_report(frame, predictions)

    assert report["prediction"].tolist() == [180_000.0, 1_000_000.0]
    assert report["error"].tolist() == [20_000.0, 500_000.0]
    assert report["absolute_error"].tolist() == [20_000.0, 500_000.0]
    assert report["sale_year"].tolist() == [2014, 2014]
    assert report["price_band"].notna().all()

    with pytest.raises(ValueError):
        build_error_report(frame, np.array([100_000.0]))


def test_input_immutability() -> None:
    original_frame = frame.copy()
    build_error_report(frame, predictions)
    pd.testing.assert_frame_equal(frame, original_frame)
