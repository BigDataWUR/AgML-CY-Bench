import os


# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, 'data')
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Key used for the location index
KEY_LOC = 'loc_id'
# Key used for the year index
KEY_YEAR = 'year'
# Key used for yield targets
KEY_TARGET = 'yield'



