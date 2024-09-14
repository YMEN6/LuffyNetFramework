# -*- coding:utf8 -*-
import os
import logging

from datetime import datetime


def setup_logger(root_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    today = datetime.today().strftime("%Y-%m-%d_%H_%M_%S.log")
    file_name = os.path.join(root_dir, today)

    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


# log_dir = os.path.join(
#     os.path.split(os.path.abspath("."))[0], "logs"
# )
# NAS_LOGGER = setup_logger(log_dir)

NAS_LOGGER = None
