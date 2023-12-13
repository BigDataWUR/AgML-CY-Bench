import os

import pandas as pd

from config import PATH_DATA_DIR

"""
    
    Code for reading data from disk into a configured pd.DataFrame

"""


# For now include:  TODO -- remove
#    - METEO_COUNTY_US
#    - REMOTE SENSING
#    - SOIL COUNTY
#    - YIELD COUNTY

# Source directory for US county data
DATA_PATH = os.path.join(PATH_DATA_DIR, 'county-data_US', 'county-data')

FILENAME_YIELD = 'YIELD_COUNTY_US.csv'
FILENAME_REMOTE_SENSING = 'REMOTE_SENSING_COUNTY_US.csv'
FILENAME_SOIL = 'SOIL_COUNTY_US.csv'
FILENAME_METEO = 'METEO_COUNTY_US.csv'


# TODO -- short descriptions of individual columns
# TODO -- check if data is available. If not -> download

def get_yield_data() -> pd.DataFrame:
    """
    Get a DataFrame containing yield data

    The DataFrame is indexed by (COUNTY_ID, FYEAR) and has one column named YIELD

    Sample:

                      YIELD
    COUNTY_ID  FYEAR
    AL_AUTAUGA 2018   165.6
    AL_DALLAS  2018   137.3
    AL_ELMORE  2018   170.5
    AL_PERRY   2018   145.2
    AL_BALDWIN 2018   152.0
    ...                 ...
    AZ_GRAHAM  1995   159.4
    AZ_YUMA    1995   176.9
    AZ_COCHISE 1994   173.3
    AZ_GRAHAM  1994   170.4
    AZ_YUMA    1994   157.5

    """
    path = os.path.join(DATA_PATH, FILENAME_YIELD)
    df = pd.read_csv(path, index_col=['COUNTY_ID', 'FYEAR'])
    return df


def get_soil_data() -> pd.DataFrame:
    """
        Get a DataFrame containing soil data

        The DataFrame is indexed by COUNTY_ID and has columns: SM_WHC  SM_DEPTH

        Sample:

                               SM_WHC  SM_DEPTH
            COUNTY_ID
            IN_GRANT        13.002000       100
            KS_CRAWFORD     14.994000       100
            KY_HICKMAN      13.784495       100
            KY_MEADE        11.406873       100
            MO_WAYNE        11.559165       100
            ...                   ...       ...
            IN_SPENCER      12.032872       100
            KS_LEAVENWORTH  14.327066       100
            MI_BARRY        13.002000       100
            WI_EAU_CLAIRE   11.760716       100
            IN_CRAWFORD     11.071188       100

    """
    path = os.path.join(DATA_PATH, FILENAME_SOIL)
    df = pd.read_csv(path, index_col=['COUNTY_ID'])
    return df


def get_meteo_data() -> pd.DataFrame:
    """
    Get a DataFrame containing meteo data

    The DataFrame is indexed by (COUNTY_ID, FYEAR, DEKAD) and has columns:
    ['TMAX', 'TMIN', 'TAVG', 'VPRES', 'WSPD', 'PREC', 'ET0', 'RAD']

    Sample:
                                  TMAX      TMIN  ...        ET0            RAD
    COUNTY_ID    FYEAR DEKAD                      ...
    AL_LAWRENCE  2000  1      19.83040  -2.33547  ...  10.277900   83932.601562
                       2      17.91720  -1.50550  ...  13.988700   93131.398438
                       3       6.19371  -7.63232  ...  10.985100  105886.000000
                       4      17.30620  -4.09625  ...  15.645700  147779.000000
                       5      20.93210  -1.02242  ...  16.986799  119626.000000
    ...                            ...       ...  ...        ...            ...
    WI_WINNEBAGO 2018  32      4.15934  -8.34600  ...   9.893360   82754.703125
                       33      8.64746  -7.62442  ...   8.364780   53410.101562
                       34      2.21179 -16.40110  ...   3.790080   57822.601562
                       35      5.25425  -9.52872  ...   4.716210   55321.300781
                       36      5.83723  -9.20447  ...   8.013530   40785.699219

    """
    path = os.path.join(DATA_PATH, FILENAME_METEO)
    df = pd.read_csv(path, index_col=['COUNTY_ID', 'FYEAR', 'DEKAD'])
    return df


def get_remote_sensing_data() -> pd.DataFrame:
    """
    Get a DataFrame containing remote sensing data

    The DataFrame is indexed by (COUNTY_ID, FYEAR, DEKAD) and has one column: FAPAR

                                 FAPAR
    COUNTY_ID    FYEAR DEKAD
    AL_LAWRENCE  2003  1      0.269896
                       2      0.246880
                       3      0.238996
                       4      0.242146
                       5      0.252464
    ...                            ...
    WI_WINNEBAGO 2007  32     0.267533
                       33     0.227278
                       34     0.192729
                       35     0.168795
                       36     0.164726

    """
    path = os.path.join(DATA_PATH, FILENAME_REMOTE_SENSING)
    df = pd.read_csv(path, index_col=['COUNTY_ID', 'FYEAR', 'DEKAD'])
    return df


if __name__ == '__main__':

    print(get_yield_data())
    print(get_soil_data())
    print(get_meteo_data())
    print(get_remote_sensing_data())
