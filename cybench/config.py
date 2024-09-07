import os
import logging
import logging.config
from datetime import datetime


# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, "data")
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Path to folder where output is stored
PATH_OUTPUT_DIR = os.path.join(CONFIG_DIR, "output")
os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

# Path to folder where benchmark results
PATH_RESULTS_DIR = os.path.join(PATH_OUTPUT_DIR, "runs")
os.makedirs(PATH_RESULTS_DIR, exist_ok=True)


DATASETS = {
    "maize": [
        "AO",
        "AR",
        "AT",
        "BE",
        "BF",
        "BG",
        "BR",
        "CN",
        "CZ",
        "DE",
        "DK",
        "EE",
        "EL",
        "ES",
        "ET",
        "FI",
        "FR",
        "HR",
        "HU",
        "IE",
        "IN",
        "IT",
        "LS",
        "LT",
        "LV",
        "MG",
        "ML",
        "MW",
        "MX",
        "MZ",
        "NE",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SK",
        "SN",
        "TD",
        "US",
        "ZA",
        "ZM",
    ],
    "wheat": [
        "AR",
        "AT",
        "AU",
        "BE",
        "BG",
        "BR",
        "CN",
        "CZ",
        "DE",
        "DK",
        "EE",
        "EL",
        "ES",
        "FI",
        "FR",
        "HR",
        "HU",
        "IE",
        "IN",
        "IT",
        "LT",
        "LV",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SK",
        "US",
    ],
}

# key used for 2-letter country code
KEY_COUNTRY = "country_code"
# Key used for the location index
KEY_LOC = "adm_id"
# Key used for the year index
KEY_YEAR = "year"
# Key used for yield targets
KEY_TARGET = "yield"
# Key used for dates matching observations
KEY_DATES = "dates"

# Soil properties
SOIL_PROPERTIES = ["awc", "bulk_density"]  # "drainage_class", "bulk_density"]

# Static predictors. Add more when available
STATIC_PREDICTORS = SOIL_PROPERTIES

# Weather indicators
METEO_INDICATORS = ["tmin", "tmax", "tavg", "prec", "cwb", "rad"]

# Remote sensing indicators.
# Keep them separate because they have different temporal resolution
RS_FPAR = "fpar"
RS_NDVI = "ndvi"

# Soil moisture indicators: surface moisture, root zone moisture
SOIL_MOISTURE_INDICATORS = ["ssm"]  # , "rsm"]

# Time series predictors
TIME_SERIES_PREDICTORS = (
    METEO_INDICATORS + [RS_FPAR, RS_NDVI] + SOIL_MOISTURE_INDICATORS
)

# All predictors. Add more when available
ALL_PREDICTORS = STATIC_PREDICTORS + TIME_SERIES_PREDICTORS

# Crop calendar entries: start of season, end of season
CROP_CALENDAR_ENTRIES = ["sos", "eos"]

# Feature design
# Base temperature for corn and wheat for growing degree days wheat:0 maize:10.
# From @poudelpratishtha.
GDD_BASE_TEMP = {
    "maize": 10,
    "wheat": 0,
}

GDD_UPPER_LIMIT = {
    "maize": 35,
    "wheat": None,
}


# Lead time for forecasting
# Choices: "middle-of-season", "quarter-of-season",
# "n-day(s)" where n is an integer
FORECAST_LEAD_TIME = "middle-of-season"

# Buffer period before the start of season
SPINUP_DAYS = 90

# Logging
PATH_LOGS_DIR = os.path.join(PATH_OUTPUT_DIR, "logs")
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
