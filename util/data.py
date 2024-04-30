import pandas as pd
from datetime import datetime


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

def data_to_pandas(data_items):
    data = []
    data_cols = None
    for item in data_items:
        if data_cols is None:
            data_cols = list(item.keys())

        data.append([item[c] for c in data_cols])

    return pd.DataFrame(data, columns=data_cols)

crop_cal_df = pd.read_csv("data/data_US/CROP_CALENDAR_COUNTY_US.csv", header=0)
crop_cal_df = crop_cal_df.astype({"planting_doy" : int, "maturity_doy" : int})
# crop_cal_df["planting_date"] = crop_cal_df.apply(lambda r: doy_to_date(r["planting_doy"], "2018"), axis=1)
# crop_cal_df["harvest_date"] = crop_cal_df.apply(lambda r: doy_to_date(r["maturity_doy"], "2018"), axis=1)
print(crop_cal_df.head(5))

rs_df = pd.read_csv("data/data_US/county_data/REMOTE_SENSING_COUNTY_US.csv", header=0)
sel_cols = ["loc_id", "planting_doy", "maturity_doy"]
rs_df = rs_df.merge(crop_cal_df[sel_cols], on=["loc_id"])
rs_df["planting_date"] = rs_df.apply(lambda r: doy_to_date(r["planting_doy"], r["year"]), axis=1)
rs_df["harvest_date"] = rs_df.apply(lambda r: doy_to_date(r["maturity_doy"], r["year"]), axis=1)
rs_df["date"] = rs_df.apply(lambda r: dekad_to_date(r["dekad"], r["year"]), axis=1)
print(rs_df.head(10))
