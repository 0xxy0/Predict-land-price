import logging

from house_prediction.logging_config import configure_logging


def test_configure_logging() -> None:
    configure_logging()
    logger = logging.getLogger("test")
    assert logger is not None
