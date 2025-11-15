"""
Simple logger wrapper using Python logging.
"""
import logging
import sys


def get_logger(name: str, level=logging.INFO, log_file: str = None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s:%(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
