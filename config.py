import os
import logging
from datetime import datetime


# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, "data")
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Key used for the location index
KEY_LOC = "loc_id"
# Key used for the year index
KEY_YEAR = "year"
# Key used for yield targets
KEY_TARGET = "yield"

# Logging path
PATH_LOGS_DIR = os.path.join(CONFIG_DIR, "logs")
os.makedirs(PATH_LOGS_DIR, exist_ok=True)

# Logging level
LOGGER_NAME = "agml_cybench"
LOG_FILE = datetime.now().strftime(LOGGER_NAME + "_%H_%M_%d_%m_%Y.log")
LOG_LEVEL = logging.DEBUG
