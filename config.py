import os
import logging
from datetime import datetime

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
LOG_FILE = datetime.now().strftime("agml_cyf_%H_%M_%d_%m_%Y.log")
LOG_LEVEL = logging.DEBUG

# Comet API key
api_key_path = os.path.join(CONFIG_DIR, "comet_api_key.txt")
comet_api_key = None
if os.path.isfile(api_key_path):
    with open(api_key_path) as f:
        comet_api_key = f.readline()
