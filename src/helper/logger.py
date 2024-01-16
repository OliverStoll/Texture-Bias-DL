import logging
import sys

format_str = "[%(asctime)s]  %(levelname)s | %(name)s   -   %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
