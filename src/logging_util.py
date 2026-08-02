"""Shared stdout logger for the audit modules."""
import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a stdout logger; safe to call repeatedly (no duplicate handlers)."""
    import warnings
    warnings.filterwarnings("ignore")
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
