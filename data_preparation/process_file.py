"""
@author: Guanyuan Shuai
"""
import os
import pandas as pd
from stats import geom_extract

def process_file(file, in_dir, out_dir,
                 crop, country, indicator_name,
                 geometries, crop_mask_path,
                 is_time_series=True, is_continuous=True):
    """
    Process one indicator raster file. Handles .nc or .tif files.

    :param file: path to raster file.
    :param in_dir: path to directory containing indicator raster file
    :param out_dir: path to write csv output
    :param indicator_name: indicator name
    :param geometry: geometry
    :param crop_mask_path: path to a raster file with crop mask area fraction weights
    :param is_time_series: flag to indicate whether data is static or time series
    :param is_continuous: flag to indicator whether data is categorical or continuous
    """
    basename = os.path.basename(file)
    fname, ext = os.path.splitext(basename)
    output_path = os.path.join(out_dir, crop, country, indicator_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if (is_continuous):
        aggr = "mean"
    else:
        aggr = "mode"

    if (is_time_series):
        col_names = ["crop_name", "adm_id", "date", indicator_name]
        date_str = fname[-8:]
        out_csv = "_".join([indicator_name, date_str, crop, country]) + ".csv"
    else:
        col_names = ["crop_name", "adm_id", indicator_name]
        out_csv = "_".join([indicator_name, crop, country]) + ".csv"

    df = pd.DataFrame(columns=col_names)

    ############################################
    # get predictor value for each admin region
    ############################################
    for adm_id, geometry in geometries.items():
        stats = geom_extract(geometry,
                             in_dir,
                             file,
                             indicator_name,
                             stats_out=[aggr],
                             crop_mask_file=crop_mask_path,
                             crop_mask_thresh=0,
                             thresh_type= 'Fixed')
        if (stats is not None) and (len(stats) > 0):
            mean_var = stats['stats'][aggr]
            if (is_time_series):
                adm_region_data = [crop, adm_id, date_str, mean_var]
            else:
                adm_region_data = [crop, adm_id, mean_var]

            df.loc[len(df.index)] = adm_region_data

    print(df.head())
    df.to_csv(os.path.join(output_path, out_csv), index=False)