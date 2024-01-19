import pandas as pd
import os

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
    "NUTS_ID" : "NUTS3_ID",
    "Time" : "DATE",
    "Temperature_Air_2m_Max_24h" : "TMAX",
    "Temperature_Air_2m_Mean_24h" : "TAVG",
    "Temperature_Air_2m_Min_24h" : "TMIN",
    "Precipitation_Flux" : "PREC",
    "Solar_Radiation_Flux" : "RAD",
    "Wind_Speed_10m_Mean" : "WSPD",
    "fpar" : "FPAR"
}

# date_format = "%Y-%m-%d"
# kelvin_zero = 273

# sel_cols = ["CROP_NAME", "NUTS3_ID", "DATE", "FYEAR", "DEKAD", "TMAX", "TMIN", "TAVG", "RAD", "PREC", "WSPD", "FPAR"]
# predictors_csv = os.path.join(PATH_DATA_DIR, "EU_AgERA5_Match_FPAR_winter_wheat_2000_2022.csv")
# predictors = pd.read_csv(predictors_csv, header=0)
# predictors["CROP_NAME"] = "Soft wheat"
# # print(list(predictors.columns))
# cn_predictors = predictors[(predictors["CNTR_CODE"] == cn_code) &
#                            (predictors["LEVL_CODE"] == nuts_level)]
# cn_predictors = cn_predictors.rename(columns=rename_cols)
# cn_predictors["FYEAR"] = cn_predictors["DATE"].str[:4]
# cn_predictors["DEKAD"] = cn_predictors.apply(lambda row: getDekad(row["DATE"], date_format), axis=1)
# for c in ["TMAX", "TMIN", "TAVG"]:
#     cn_predictors[c] = cn_predictors[c] - kelvin_zero
# cn_predictors = cn_predictors[sel_cols]
# print(cn_predictors.sort_values(by=["NUTS3_ID", "FYEAR", "DEKAD"]).head(10))

# remote_sensing_df = cn_predictors[["NUTS3_ID", "FYEAR", "DEKAD", "FPAR"]]
# meteo_df = cn_predictors[["NUTS3_ID", "FYEAR", "DEKAD", "TMAX", "TMIN", "TAVG", "PREC", "RAD"]]

# filename_suffix = "_NUTS" + str(nuts_level) + "_" + cn_code + ".csv"
# remote_sensing_csv = os.path.join(PATH_DATA_DIR, "REMOTE_SENSING" + filename_suffix)
# remote_sensing_df.to_csv(remote_sensing_csv, index=False)
# meteo_csv = os.path.join(PATH_DATA_DIR, "METEO" + filename_suffix)
# meteo_df.to_csv(meteo_csv, index=False)

# # Yield data
# yield_eu_csv = os.path.join(PATH_DATA_DIR, "YIELD_EU.csv")
# yield_eu_df = pd.read_csv(yield_eu_csv, header=0)
# yield_cn_df = yield_eu_df[yield_eu_df["REGION"].str[:2] == cn_code]
# yield_cn_df = yield_cn_df.rename(columns={"REGION" : "NUTS3_ID", "YEAR" : "FYEAR"})
# yield_cn_df = yield_cn_df[["CROP_NAME", "NUTS3_ID", "FYEAR", "YIELD"]]
# print(yield_cn_df.head(5))
# yield_csv = os.path.join(PATH_DATA_DIR, "YIELD" + filename_suffix)
# yield_cn_df.to_csv(yield_csv, index=False)

# soil data
soil_csv = os.path.join(PATH_DATA_DIR, "data_FR", "SOIL_NUTS3_FR.csv")
soil_df = pd.read_csv(soil_csv, header=0)
soil_df = soil_df.rename(columns={"IDREGION" : "NUTS3_ID"})
print(soil_df.head())
soil_df.to_csv(soil_csv, index=False)

meteo_daily_csv = os.path.join(PATH_DATA_DIR, "data_FR", "METEO_DAILY_NUTS3_FR.csv")
meteo_daily_df = pd.read_csv(meteo_daily_csv, header=0)

date_format = "%Y%m%d"
meteo_daily_df["DATE"] = meteo_daily_df["DATE"].astype(str)
meteo_daily_df["FYEAR"] = meteo_daily_df["DATE"].str[:4]
meteo_daily_df["DEKAD"] = meteo_daily_df.apply(lambda row: getDekad(row["DATE"], date_format), axis=1)
print(meteo_daily_df.head(5))
et0_daily_df = meteo_daily_df[["IDREGION", "FYEAR", "DEKAD", "ET0"]]
et0_dekadal_df = et0_daily_df.groupby(["IDREGION", "FYEAR", "DEKAD"]).agg(ET0=("ET0", "mean")).reset_index()
et0_dekadal_df = et0_dekadal_df.rename(columns={"IDREGION" : "NUTS3_ID"})
et0_dekadal_df["FYEAR"] = et0_dekadal_df["FYEAR"].astype("int64")

print(et0_dekadal_df.dtypes)
meteo_dekadal_csv = os.path.join(PATH_DATA_DIR, "data_FR", "METEO_NUTS3_FR.csv")
meteo_dekadal_df = pd.read_csv(meteo_dekadal_csv, header=0)
meteo_dekadal_df = meteo_dekadal_df.merge(et0_dekadal_df, on=["NUTS3_ID", "FYEAR", "DEKAD"])
meteo_dekadal_df.to_csv(meteo_dekadal_csv, index=False)