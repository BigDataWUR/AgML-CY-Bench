import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import date


def doy_to_date(doy, year):
    # Based on
    # https://www.geeksforgeeks.org/python-convert-day-number-to-date-in-particular-year/

    # Pad on the left with 0's.
    # See https://docs.python.org/3/library/stdtypes.html. Look for str.rjust().
    if (isinstance(doy, int)):
        doy = str(doy)
    if (isinstance(year, int)):
        year = str(year)

    doy.rjust(3 + len(doy), '0')
    date_str = datetime.strptime(year + "-" + doy, "%Y-%j").strftime("%Y%m%d")

    return date_str

def date_to_dekad(date_str):
    month = int(date_str[4:6])
    day_of_month = int(date_str[6:])
    dekad = (month - 1) * 3
    if (day_of_month <= 10):
        dekad += 1
    elif (day_of_month <= 20):
        dekad += 2
    else:
        dekad += 3 

    return dekad

def dekad_to_date(dekad, year):
    if (dekad % 3) == 1:
        month = int(dekad/3) + 1
        day_of_month = "01"
    elif (dekad % 3) == 2:
        month = int(dekad/3) + 1
        day_of_month = "11"
    else:
        month = int(dekad/3)
        day_of_month = "21"

    if (month < 10):
        return str(year) + "0" + str(month) + day_of_month
    else:
        return str(year) + str(month) + day_of_month
    
def update_harvest_year(harvest_date, year, new_year):
    if (year != new_year):
        return harvest_date.replace(str(year), str(new_year))

    return harvest_date

def update_date(harvest_date, diff_with_harvest):
    # NOTE add the difference between YYYY1231 and harvest_date
    # Then add diff_with_harvest
    end_of_year = date(harvest_date.year, 12, 31)
    end_of_year_delta = (end_of_year - harvest_date).days
    days_delta = end_of_year_delta + diff_with_harvest
    return (harvest_date + timedelta(days=days_delta))

def merge_with_crop_calendar(df, crop_cal_df):
    df = df.merge(crop_cal_df, on=["loc_id"])
    df["planting_date"] = df.apply(lambda r: doy_to_date(r["planting_doy"], r["year"]),
                                   axis=1)
    df["harvest_date"] = df.apply(lambda r: doy_to_date(r["maturity_doy"], r["year"]),
                                  axis=1)
    df["date"] = df.apply(lambda r: dekad_to_date(r["dekad"], r["year"]),
                          axis=1)

    return df

def rotate_data_by_crop_calendar(df, crop_cal_df):
    crop_cal_cols = ["loc_id", "planting_doy", "maturity_doy"]
    crop_cal_df = crop_cal_df.astype({"planting_doy" : int, "maturity_doy" : int})
    df = merge_with_crop_calendar(df, crop_cal_df[crop_cal_cols])

    df["new_year"] = np.where(df["date"] > df["harvest_date"],
                              df["year"] + 1,
                              df["year"])
    df = df.astype({"harvest_date" : str})
    df["harvest_date"] = df.apply(lambda r: update_harvest_year(r["harvest_date"],
                                                                r["year"],
                                                                r["new_year"]),
                                  axis=1)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["harvest_date"] = pd.to_datetime(df["harvest_date"], format="%Y%m%d")
    df["harvest_diff"] = (df["date"] - df["harvest_date"]).dt.days
    # NOTE pd.to_datetime() creates a Timestamp object
    df = df.rename(columns={"date" : "original_date"})
    df["date"] = df.apply(lambda r: update_date(r["harvest_date"].date(),
                                                r["harvest_diff"]),
                              axis=1)
    return df

rs_df = pd.read_csv("data/data_US/county_data/REMOTE_SENSING_COUNTY_US.csv", header=0)
crop_cal_df = pd.read_csv("data/data_US/CROP_CALENDAR_COUNTY_US.csv", header=0)
rs_df = rotate_data_by_crop_calendar(rs_df, crop_cal_df)
rs_df = rs_df[["loc_id", "original_date", "new_year", "date", "fapar"]]
print(rs_df.head(40))