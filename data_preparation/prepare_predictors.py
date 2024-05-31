# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:32:18 2024

@author: Guanyuan Shuai
"""
import os, sys,time
import geopandas as gpd
import pandas as pd
import numpy as np

import rasterio as rio
from rasterio.warp import transform_geom
from rasterio.crs import CRS
from shapely.geometry import shape
from rasterio.mask import mask
from datetime import datetime
import matplotlib.pyplot as plt
from rasterio.plot import show
import multiprocessing
import datetime
import warnings
from shapefile_operation import get_shapes, get_admin_id
from shapefile_operation import EU_COUNTRIES, FEWSNET_COUNTRIES
from file_operations import get_time_series_files
from process_file import process_file
from itertools import repeat
warnings.filterwarnings("ignore")


def prepare_predictors(prep_root, crop, country, admin_level,
                       shapefile, predictors, crop_mask_file):
    shapefile_path = os.path.join(prep_root, "shapefiles", shapefile)
    crop_mask_path = os.path.join(prep_root, "crop_masks", crop_mask_file)
    data_dir = os.path.join(prep_root, "Predictors")
    out_dir = os.path.join(prep_root, "output")
    geo_df = get_shapes(shapefile_path,
                        country=country,
                        admin_level=admin_level)

    geo_df = get_admin_id(geo_df, country)
    geo_df = geo_df[["adm_id", "geometry"]]
    print(geo_df.head(5))

    geometries = {
        adm_id : geo_df[geo_df["adm_id"] == adm_id]["geometry"].values[0] for adm_id in geo_df["adm_id"].unique()
    }

    #################Loop over each crop, year, and variable##########################
    ##########setup crop mask file###################
    for pred_source in predictors.keys():
        indicators = predictors[pred_source]
        for ind in indicators:
            is_time_series = indicators[ind][0] == "time series"
            is_continuous = indicators[ind][1] == "continuous"

            if (country in EU_COUNTRIES):
                country_dir = "EU_" + pred_source
            elif (country in FEWSNET_COUNTRIES):
                country_dir = "FEWSNET_" + pred_source
            else:
                country_dir = country + "_" + pred_source

            if (os.path.isdir(os.path.join(data_dir, pred_source, country_dir))):
                indicator_dir = os.path.join(data_dir, pred_source, country_dir, ind)
            else:
                indicator_dir = os.path.join(data_dir, pred_source, ind)

            # Time series data
            if is_time_series:
                for yr in range(start_year, end_year + 1):
                    print("Start working on", crop, country, ind, yr)
                    # unzip AgERA5 and rename
                    # if (pred_source == "AgERA5"):
                    #     unzipAgERA5andRename(data_dir, country_dir, pred_source, ind, yr)

                    files = get_time_series_files(indicator_dir,
                                                  exclude_filenames=[".ipynb_checkpoints"],
                                                  exclude_extensions=[".zip", ".ipynb"],
                                                  year=yr)

                    print('There are ' + str(len(files)) + ' files!')
                    start_time = time.time()
                    cpus = multiprocessing.cpu_count()
                    print(cpus)

                    files = sorted([os.path.join(indicator_dir, f) for f in files])
                    with multiprocessing.Pool(cpus) as pool:
                        # NOTE: multiprocessing using a target function with multiple arguments.
                        # Based on the answer to
                        # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
                        results = pool.starmap(process_file, zip(files,
                                                                repeat(indicator_dir),
                                                                repeat(out_dir),
                                                                repeat(crop), repeat(country),
                                                                repeat(ind), repeat(geometries),
                                                                repeat(crop_mask_path),
                                                                repeat(is_time_series),
                                                                repeat(is_continuous)))

                    m, s = divmod((time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    print("Time used: %02d:%02d:%02d" % (h, m, s))

                # delete AgERA5 files
                # if (pred_source == "AgERA5"):
                #     deleteAgERA5Files(indicator_dir)
            # Static data
            else:
                print("Start working on", crop, country, ind)
                files = os.listdir(indicator_dir)
                # should be one raster file
                assert (len(files) == 1)

                start_time = time.time()
                process_file(os.path.join(indicator_dir, files[0]),
                            indicator_dir,
                            out_dir, crop, country, ind,
                            geometries, crop_mask_path,
                            is_time_series, is_continuous)

                m, s = divmod((time.time() - start_time), 60)
                h, m = divmod(m, 60)

                print('Done for', crop, country, ind)
                print("Time used: %02d:%02d:%02d" % (h, m, s))

##################
# Directory paths
##################

prep_root = r'drive/MyDrive/Postdoc Research/AgMIP ML/AgML/AgML Community/AgML activities/Subnational crop yield forecasting/Predictor data preparation'

#####################
# setup study period
#####################
# NOTES:
# FPAR data starts from 2001. There may be some data from 2000, but it's not complete.
# GLDAS data starts from 2003. 2003 is not complete.
start_year = 2001
end_year = 2023

########
# crop
########
crops = ["wheat", "maize"]

############
# shapefile
############
selected_countries = {
    "wheat" : {
        # countries in Europe
        "NL" : {
            "admin_level" : 2,
            "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        },
        # "FR" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "DE" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "ES" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "IT" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "PL" : {
        #     "admin_level" : 2,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # # US
        # "US" : {
        #     "admin_level" : None, # Shapes at county level
        #     "shapefile" : "cb_2018_us_county_500k.zip"
        # }
    },
    "maize" : {
        # # countries in Europe
        # "NL" : {
        #     "admin_level" : 2,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "FR" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "DE" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "ES" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "PL" : {
        #     "admin_level" : 2,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "HU" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "IT" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # "RO" : {
        #     "admin_level" : 3,
        #     "shapefile" : "NUTS_RG_03M_2016_4326.zip"
        # },
        # # fewsnet countries
        # # ["AO", "BF", "ET", "LS", "MG", "MW", "MZ", "NE", "SN", "TD", "ZA", "ZM"]
        # "AO" : {
        #     "admin_level" : None, # Shapes at expected admin level
        #     "shapefile" : "FEWSNET_Africa.zip"
        # },
        # "BF" : {
        #     "admin_level" : None, # Shapes at expected admin level
        #     "shapefile" : "FEWSNET_Africa.zip"
        # },
        # "ET" : {
        #     "admin_level" : None, # Shapes at expected admin level
        #     "shapefile" : "FEWSNET_Africa.zip"
        # },
        # "LS" : {
        #     "admin_level" : None, # Shapes at expected admin level
        #     "shapefile" : "FEWSNET_Africa.zip"
        # },
        # "MG" : {
        #     "admin_level" : None, # Shapes at expected admin level
        #     "shapefile" : "FEWSNET_Africa.zip"
        # },
        # # US
        # "US" : {
        #     "admin_level" : None, # Shapes at county level
        #     "shapefile" : "cb_2018_us_county_500k.zip"
        # }
    }
}

#####################################################
# predictors
# NOTE: key should match
#       - variable name in raster
#       - directory name containing raster files
#####################################################

predictors = {
    "AgERA5" : {
        "Precipitation_Flux": ["time series", "continuous"],
        "Maximum_Temperature": ["time series", "continuous"],
        "Minimum_Temperature": ["time series", "continuous"],
        "Mean_Temperature": ["time series", "continuous"],
        "Solar_Radiation_Flux" : ["time series", "continuous"],
    },
    "MOD09CMG" : {
        "ndvi": ["time series", "continuous"],
    },
    "JRC_FPAR500m" : {
        "fpar": ["time series", "continuous"],
    },
    "WISE_Soil" : {
        "AWC" : ["static", "continuous"],
        "drainage_class": ["static", "categorical"],
        "bulk_density" : ["static", "continuous"]
    },
    "FAO_AQUASTAT" : {
        "ET0": ["time series", "continuous"],
    },
    "GLDAS" : {
        "rootzone_moisture": ["time series", "continuous"],
        "surface_moisture": ["time series", "continuous"],
    }
}

for crop in crops:
    #############
    # crop mask
    #############
    if (crop == "maize"):
        crop_mask_file = "crop_mask_maize_WC.tif"
    elif (crop == "wheat"):
        crop_mask_file = "crop_mask_winter_spring_cereals_WC.tif"
    else:
        crop_mask_file = "crop_mask_generic_asap.tif"

    countries = selected_countries[crop]
    for cn in countries:
        print("Working on", crop, cn)
        shapefile = countries[cn]["shapefile"]
        admin_level = countries[cn]["admin_level"]
        prepare_predictors(prep_root, crop, cn, admin_level,
                           shapefile, predictors, crop_mask_file)