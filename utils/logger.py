import os
import logging


FORMAT = logging.Formatter('%(asctime)s, %(message)s')


def setup_logger(name, log_dir, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(log_dir, log_file))
    handler.setFormatter(FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
