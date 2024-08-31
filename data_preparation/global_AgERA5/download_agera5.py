import os
import cdsapi
import concurrent.futures

# Most of the code here comes from Abdelrahman's library.
# https://github.com/AbdelrahmanAmr3/earthstat

AgERA5_params = {
    "Maximum_Temperature": ("2m_temperature", "24_hour_maximum"),
    "Minimum_Temperature": ("2m_temperature", "24_hour_minimum"),
    "Mean_Temperature": ("2m_temperature", "24_hour_mean"),
    "Solar_Radiation_Flux": ("solar_radiation_flux", None),
    "Precipitation_Flux": ("precipitation_flux", None),
    # NOTE: Not used. Uncomment if required.
    # 'Wind_Speed': {'10m_wind_speed': '24_hour_mean'},
    # 'Vapour_Pressure': {'vapour_pressure': '24_hour_mean'},
}


# @author: Abdelrahman Saleh
def get_agera5_params(sel_param, year, area=None):
    if sel_param in AgERA5_params:
        var, stat = AgERA5_params[sel_param]

        # Construct the return dictionary
        retrieve_params = {
            "version": "1_1",
            "format": "zip",
            "variable": var,
            "year": str(year),
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
        }

        if stat is not None:
            retrieve_params["statistic"] = stat

        if area is not None:
            retrieve_params["area"] = area

        return retrieve_params
    else:
        raise Exception(f"parameter '{sel_param}' is not supported")


def download_agera5_year(cds, year, sel_param):
    download_path = os.path.join("/path/to/downloads", sel_param)
    os.makedirs(download_path, exist_ok=True)
    retrieve_params = get_agera5_params(sel_param, year)
    cds.retrieve(
        "sis-agrometeorological-indicators",
        retrieve_params,
        f"{download_path}/{sel_param}_{year}.zip",
    )


def download_agera5(cds, num_requests, start_year, end_year):
    # NOTE: $HOME/.cdsapirc needs to contain API key
    # Example contents of .cdsapirc:
    # url: https://cds.climate.copernicus.eu/api/v2
    # key: {cdsapi.uid}:{cdsapi.api_key}
    if (not os.path.exists(os.path.join(os.environ["HOME"], ".cdsapirc"))):
        raise Exception(".cdsapirc not found in $HOME")

    # Another NOTE: CDS api seems to be changing.
    # From Sept 2024, this script may not work.
    # Check https://confluence.ecmwf.int/x/uINmFw
    assert num_requests <= 16, Exception("Suggested number of requests is <= 16")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        tasks = [
            executor.submit(download_agera5_year, cds, year, parameter)
            for year in range(start_year, end_year + 1)
            for parameter in AgERA5_params
        ]

        for future in concurrent.futures.as_completed(tasks):
            future.result()


# NOTE:
# 1. Make sure you have enough disk space when you run the full script.
# 2. Some zip files seem to have errors. You may need to download them again.
cds = cdsapi.Client(progress=False)
download_agera5(cds, 16, 2001, 2023)
