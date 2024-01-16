import os
import logging

# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, "data")
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Path to folder where output is stored
PATH_OUTPUT_DIR = os.path.join(CONFIG_DIR, "output")
os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

# Logging path
PATH_LOGS_DIR = os.path.join(PATH_OUTPUT_DIR, "logs")
os.makedirs(PATH_LOGS_DIR, exist_ok=True)

# Logging level
LOGGER_NAME = "agml_cyf"
LOG_FILE = "agml_cyf.log"
LOG_LEVEL = logging.DEBUG
