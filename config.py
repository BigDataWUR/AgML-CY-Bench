import os
import logging
import logging.config
from datetime import datetime
import yaml

# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, "data")
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Path to folder where benchmark results
PATH_RESULTS_DIR = os.path.join(CONFIG_DIR, "output", "runs")
os.makedirs(PATH_RESULTS_DIR, exist_ok=True)

# Key used for the location index
KEY_LOC = "loc_id"
# Key used for the year index
KEY_YEAR = "year"
# Key used for yield targets
KEY_TARGET = "yield"
# Key used for dates matching observations
KEY_DATES = "dates"

# Soil indicators
# for sample data
SOIL_INDICATORS = ["sm_fc", "sm_wp"]
# TODO: uncomment for cybench data
# SOIL_INDICATORS = ["awc", "drainage_class", "bulk_density"]

# Weather indicators
WEATHER_INDICATORS = ["tmin", "tmax", "tavg", "prec"]

# Base temperature for corn and wheat for growing degree days wheat:0 corn:10
GDD_BASE_TEMP = {"wheat": 0, "corn": 10}

# Remote sensing indicators.
# Keep them separate because they have different temporal resolution
RS_FAPAR = "fapar"
RS_NDVI = "ndvi"

# Logging
PATH_LOGS_DIR = os.path.join(CONFIG_DIR, "output", "logs")
os.makedirs(PATH_LOGS_DIR, exist_ok=True)

LOG_FILE = datetime.now().strftime("agml_cybench_%H_%M_%d_%m_%Y.log")
LOG_LEVEL = logging.DEBUG

# Based on examples from
# https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "filename": os.path.join(PATH_LOGS_DIR, LOG_FILE),
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8",
        }
    },
    "loggers": {
        "": {"handlers": ["file_handler"], "level": LOG_LEVEL, "propagate": True}
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
