import pandas as pd
import numpy as np

import datetime

from config import PATH_DATA_DIR

def getDekad(date_str, date_format):
    dt = datetime.datetime.strptime(date_str, date_format)
    dekad = 3
    if (dt.day <= 10):
        dekad = 1
    elif (dt.day <= 20):
        dekad = 2

    dekad += (dt.month - 1) * 3
    return dekad

cn_code = "FR"
nuts_level = 3
rename_cols = {
    "NUTS_ID" : "REGION",
    "Time" : "DATE",
    "Temperature_Air_2m_Max_24h" : "TMAX",
    "Temperature_Air_2m_Mean_24h" : "TAVG",
    "Temperature_Air_2m_Min_24h" : "TMIN",
    "Precipitation_Flux" : "PREC",
    "Solar_Radiation_Flux" : "RAD",
    "Wind_Speed_10m_Mean" : "WSPD",
    "FPAR_WINTER_WHEAT" : "FPAR"
}

date_format = "%Y-%m-%d"

sel_cols = ["REGION", "DATE", "YEAR", "DEKAD", "TMAX", "TMIN", "TAVG", "RAD", "PREC", "WSPD", "FPAR"]
predictors = pd.read_csv(PATH_DATA_DIR + "/Eu_AgERA5_FPAR_WINTER_WHEAT_2000_2023.csv", header=0)
cn_predictors = predictors[(predictors["CNTR_CODE"] == cn_code) &
                           (predictors["LEVL_CODE"] == nuts_level)]
cn_predictors = cn_predictors.rename(columns=rename_cols)
cn_predictors["YEAR"] = cn_predictors["DATE"].str[:4]
cn_predictors["DEKAD"] = cn_predictors.apply(lambda row: getDekad(row["DATE"], date_format), axis=1)
cn_predictors = cn_predictors[sel_cols]
print(cn_predictors.sort_values(by=["REGION", "YEAR", "DEKAD"]).head(10))