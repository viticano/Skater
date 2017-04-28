"""Funcs for logging"""
import logging


def build_logger(log_level, logger_name):
    logger = logging.Logger(logger_name)
    logger.setLevel(log_level)
    msg_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(msg_formatter)
    stream_handler.setFormatter(msg_formatter)
    logger.addHandler(stream_handler)
    return logger
