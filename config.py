import os

# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, "data")
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Path to folder where output is stored
PATH_OUTPUT_DIR = os.path.join(CONFIG_DIR, "output")
os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)
