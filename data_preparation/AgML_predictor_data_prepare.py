import os, sys,time
import numpy as np
import rasterio
import rasterio.features
import pandas as pd
import geopandas as gpd
from rasterio.crs import CRS
from itertools import repeat
from multiprocessing import Pool
import logging
from zipfile import ZipFile
from rasterio.warp import transform_geom
from shapely.geometry import shape
from rasterio.mask import mask
from datetime import datetime
import multiprocessing
import datetime
import warnings
warnings.filterwarnings("ignore")


log = logging.getLogger(__name__)


SUPPRESS_ERRORS = True
TIFF_DIRECTORY_NAME = "Tiff"
EU_COUNTRY_CODE_KEY = "CNTR_CODE"
EU_ADMIN_LEVEL_KEY = "LEVL_CODE"

# Country codes to admin level
# Austria (AT), Belgium (BE), Bulgaria (BG), Czech Republic (CZ), Germany (DE), Denmark (DK), Estonia (EE), Greece (EL),
# Spain (ES), Finland (FI), France (FR), Croatia (HR), Hungary (HU), Ireland (IE), Italy (IT), Lithuania (LT),
# Latvia (LV), The Netherlands (NL), Poland (PL), Portugal (PT), Romania (RO), Sweden (SE), Slovakia (SK)
EU_COUNTRIES = {
    "AT" : 2,
    "BE" : 2,
    "BG" : 2,
    "CZ" : 3,
    "DE" : 3,
    "DK" : 3,
    "EE" : 3,
    "EL" : 3,
    "ES" : 3,
    "FI" : 3,
    "FR" : 3,
    "HR" : 2,
    "HU" : 3,
    "IE" : 2,
    "IT" : 3,
    "LT" : 3,
    "LV" : 3,
    "NL" : 2,
    "PL" : 2,
    "PT" : 2,
    "RO" : 3,
    "SE" : 3,
    "SK" : 3
}

# Angola (AO), Burkina Faso (BF), Ethiopia (ET), Lesotho (LS), Madagascar (MG), Malawi (MW),
# Mozambique (MZ), Niger (NE), Senegal (SN), Chad (TD), South Africa (ZA), Zambia (ZM)
FEWSNET_COUNTRIES = ["AO", "BF", "ET", "LS", "MG", "MW", "MZ", "NE", "SN", "TD", "ZA", "ZM"]
FEWSNET_ADMIN_ID_KEY = "adm_id"


"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""

class UnableToExtractStats(Exception):
    pass

"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def read_masked(ds, mask, indexes=None, window=None,
                use_pixels='CENTER', out_mask_path=None,
                *args, **kwargs):
    
    """
    Reads the data fom the raster file and returns it as a numpy masked array.
    The geometry is used to define the mask.
    It returns only a subset of the data.
        It reads only the raster window containing the geometry. This optimizes the performance by reducing disk reads.
    The "use_pixels" parameter can be used to define how to handle the border pixels.
        Some require shapely to be available on the system
    :param ds: Raster file path or an instance of a rasterio.DatasetReader (already opened raster file).
    :param mask: geojson geometry used to mask the array, mask is a read mask (only data covered by the geometry is used)
    :param indexes: (list of ints or a single int, optional) – If indexes is a list, the result is a 3D array, but is a 2D array if it is a band index number.
    :param window: Optional window param to extract a partial dataset.
    :param use_pixels: Parameter that defines what pixels to use when masking with input geometry.
        CONTAINED - Only use pixels that are fully contained within the request geometry (requires shapely)
        ALL - Use all pixels touched by request geometry
        CENTER - Use all pixels whose center is within the request geometry
    :param out_mask_path: path on disk where to save the mask as geoTiff, skips if set to None
    :return: numpy array, optionally saves the mask as geoTiff to specified location
    """
    # open file
    dataset = ds if isinstance(ds, rasterio.DatasetReader) else rasterio.open(ds)

    # determine masking
    if use_pixels.upper() == 'CONTAINED':
        mask_invert = True
        mask_all_touched = True
    elif use_pixels.upper() == 'ALL':
        mask_invert = False
        mask_all_touched = True
    elif use_pixels.upper() == 'CENTER':
        mask_invert = False
        mask_all_touched = False

    # get read window
    if window:
        _window = window
    else:
        _window = rasterio.features.geometry_window(dataset, mask)

    # read the source dataset
    source = dataset.read(indexes, window=_window, *args, **kwargs)
    # return empty source if the window missed the raster
    if 0 in source.shape:
        return source

    # create invert mask for CONTAINED (requires Shapely)
    if mask_invert:
        import shapely.geometry
        # create geometry difference with intersect geom
        dataset_window_bounds = rasterio.windows.bounds(_window, dataset.transform)
        mask_shapely_geom = shapely.geometry.box(*list(dataset_window_bounds)).difference(shapely.geometry.shape(mask))
        mask = None if mask_shapely_geom.is_empty else shapely.geometry.mapping(mask_shapely_geom)

    # create transform and shape
    out_shape = source.shape[-2:]
    if window:
        # define transform with custom window and out shape
        out_transform = rasterio.transform.from_bounds(*dataset.window_bounds(window), *reversed(out_shape))
    else:
        out_transform = dataset.window_transform(_window)

    # create the mask array matching the raster windowed reading
    input_geom_mask = rasterio.features.geometry_mask(mask,
                                                      transform=out_transform, invert=mask_invert,
                                                      out_shape=out_shape,
                                                      all_touched=mask_all_touched)
    # write the mask output
    if out_mask_path:
        with rasterio.open(out_mask_path, 'w', driver='Gtiff', height=input_geom_mask.shape[0],
                           width=input_geom_mask.shape[1], count=1, dtype=np.uint8, crs=dataset.crs,
                           transform=out_transform) as tmp_dataset:
            tmp_dataset.write(input_geom_mask.astype(np.uint8), 1)

    # mask data arrays
    source = np.ma.array(source, mask=input_geom_mask)

    return source

"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def round_window(window):
    """
    Outputs a copy of the window rounded to the nearest whole pixel.
    Rounds the absolute size of the window extent which makes sure that all pixel cneters covered by the input window
    are included.
    :param window: Input window
    :return: rasterio.windows.Window
    """
    _col_off = round(window.col_off)
    _row_off = round(window.row_off)
    _width = round(window.col_off + window.width) - _col_off
    _height = round(window.row_off + window.height) - _row_off
    return rasterio.windows.Window(_col_off, _row_off, _width, _height)

"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def get_common_bounds_and_shape(geom, ds_list):
    """
    Returns a unified read window, bounds and resolution, for a given geometry on all raster datasets in a list.
    Used when combining datasets with heterogeneous resolution to read data with the extent (bounds) of the lowest
    resolution dataset and to use the resolution (shape) of the highest resolution dataset.
    Useful when trying to get unified arrays to perform calculation from heterogeneous datasets.
    The read window and resolution is defined with a following process.
        1. we find the lowest and highest resolution datasets by comparing their geometry window sizes (all pixels
           touched by geom).
        2. we get the geometry window bounds of the lowest resolution dataset and align them to the highest
           resolution dataset.
            - lowest resolution dataset window should be the maximum extent, extents covering all other dataset
              windows.
            - We align the bounds to the highest resolution dataset by defining the bounds window on the hres
              dataset and rounding them to the closest whole pixel
        3. we define unified read window and out_shape
            - We use the aligned bounds to define a read window for all datasets
            - We use the resolution of the hres rounded bounds window shape as the read out_shape
    :param geom: GeoJSON-like feature (implements __geo_interface__) – feature collection, or geometry.
    :param ds_list: List of raster file paths or rasterio datasets
    :return: Pair of tuples, shape and bounds - ((x_min, y_min, x_max, y_max), (rows, columns))
    """
    max_res_ds = None
    max_res_window_size = None
    min_res_window_size = None
    max_extent_window_bounds = None

    for ds in ds_list:
        dataset = ds if isinstance(ds, rasterio.DatasetReader) else rasterio.open(ds)
        # dataset = ds.to_rio()
        ds_window = rasterio.features.geometry_window(dataset, geom)

        if not max_res_ds:  # just assign values to all vars if first file in the loop
            max_res_ds = dataset
            max_res_window_size = rasterio.windows.shape(ds_window)
            min_res_window_size = rasterio.windows.shape(ds_window)
            max_extent_window_bounds = dataset.window_bounds(ds_window)
        elif rasterio.windows.shape(ds_window) > max_res_window_size:
            max_res_ds = dataset
            max_res_window_size = rasterio.windows.shape(ds_window)
        elif rasterio.windows.shape(ds_window) < min_res_window_size:
            max_extent_window_bounds = dataset.window_bounds(ds_window)
            min_res_window_size = rasterio.windows.shape(ds_window)

    # new bounds fitted to the highest resolution file
    out_bounds = max_res_ds.window_bounds(round_window(max_res_ds.window(*max_extent_window_bounds)))
    out_shape = rasterio.windows.shape(round_window(max_res_ds.window(*out_bounds)))

    return out_bounds, out_shape

"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def arr_stats(arr, weights=None,
              output=('min', 'max', 'sum', 'mean', 'count', 'mode')):
    """
    Extracts statistics from input array (arr).
    It uses weights array as weights if provided.
    List of statistics to extract is defined in the output parameter as a list or a comma separated string.
    :param arr: Input array
    :param weights: Array providing weights for each pixel value. Used to calculate stats.
    :param output: List of values to extract, can be a list or comma separated string.
        Possible values are:
            - min
            - max
            - sum - sum of all un masked arr pixels, no weight applied
            - mean - average value, if weights are provided calculates a weighted average
            - std - standard deviation, if weights are provided uses weighted calculation
            - median - median values, WARNING doesn't use weights
            - count - number of pixels used in a group, can be different to total number if a mask is applied
            - weight_sum - sum of weights
            - mode - majority value, used for category variable.
    :return: dict with calculated stats values
    """
    # prepare output and make sure it is a list
    _output = output if type(output) in [list, tuple] else output.split()
    _known_outputs = ('min', 'max', 'sum', 'mean', 'std', 'median', 'count', 'mode', 'weight_sum')  # todo handle more extractions
    if not any(elem in _output for elem in _known_outputs):
        raise Exception('Output not defined properly, should define at least one known output %s' % str(_known_outputs))

    out_vals = dict()
    # make sure array is a masked array
    _arr = np.ma.array(arr)
    if weights is not None:
        _weights = np.ma.masked_array(weights, mask = np.ma.getmask(arr))

    if any(elem in _output for elem in ('std', 'min', 'max', 'sum', 'median', 'mode')):
        arr_compressed = _arr.compressed()
        if weights is not None:
            weights_compressed = _weights.compressed()

    if 'mean' in _output:
        if weights is not None:
            ind = np.isnan(_arr) | np.isnan(weights) | (_arr <= -9999)
            out_vals['mean'] = np.ma.average(_arr[~ind], weights=weights[~ind])
        else:
            ind = np.isnan(_arr)
            out_vals['mean'] = np.ma.mean(_arr[~ind])

    if 'std' in _output:
        arr_size = np.size(arr_compressed)
        if weights is not None:
            # combine mask from the arr and create a compressed arr
            weights_compressed = np.ma.array(weights, mask=_arr.mask).compressed()
            ind = np.isnan(arr_compressed) | np.isnan(weights_compressed) | (arr_compressed <= -9999)

            if arr_size == 1 or np.sum(weights_compressed[~ind] > 0) == 1:
                out_vals['std'] = np.int8(0)
            elif arr_size > 0 and np.sum(weights_compressed[~ind]) > 0:
                out_vals['std'] = np.sqrt(
                    np.cov(arr_compressed[~ind], aweights=weights_compressed[~ind], ddof=0))
            else:
                out_vals['std'] = None
        else:
            if arr_size == 1:
                out_vals['std'] = np.int8(0)
            elif arr_size > 0:
                out_vals['std'] = np.sqrt(np.cov(arr_compressed, ddof=0))
            else:
                out_vals['std'] = None

    if 'min' in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals['min'] = arr_compressed[~ind].min()

    if 'max' in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals['max'] = arr_compressed[~ind].max()

    if 'sum' in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals['sum'] = arr_compressed[~ind].sum()

    if 'median' in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals['median'] = np.ma.median(arr_compressed[~ind])

    if 'mode' in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        if weights is not None:
            out_vals['mode'] = np.argmax(np.bincount(arr_compressed[~ind], weights=weights_compressed[~ind]))
        else:
            out_vals['mode'] = np.argmax(np.bincount(arr_compressed[~ind]))

    if 'count' in _output:
        out_vals['count'] = int((~_arr.mask).sum())

    if 'weight_sum' in _output and weights is not None:
        ind = np.isnan(weights) | (_arr <= -9999)
        weights_compressed = np.ma.array(weights[~ind], mask=_arr.mask[~ind]).compressed()
        out_vals['weight_sum'] = weights_compressed.sum()

    # convert to regular py types from np types which can cause problems down the line like JSON serialisation
    out_vals = {k: v.item() for k, v in out_vals.items()}

    return out_vals


"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def arr_classes_count(arr, cls_def, weights=None, border_include='min'):
    """
    Counts the number of array values in a class (bin) defined by min and max value.
    :param arr: Input array
    :param cls_def: list(dict) - List of dictionaries with Class definitions. A class is defined by its min and max value.
        [{'min': val1, 'max': val2}, ...]
    :param weights: Array with weights to apply when counting pixels. If not defined pixels are counted as 1
    :param border_include: str [min|max|both|None] - Parameter defining how to handle the border values.
        Options min|max|both meaning to use "min", "max" or "both" values as a part of the class.
    :return: list(dict) - input Classes definition expanded with the pixel count for that class.
    """
    _weights = weights if weights is not None else 1
    cls_out = []

    for cls in cls_def:
        if border_include is None:
            cls['val_count'] = np.sum(np.logical_and(arr > cls['min'], arr < cls['max']) * _weights)
        elif border_include.lower() == 'min':
            cls['val_count'] = np.sum(np.logical_and(arr >= cls['min'], arr < cls['max']) * _weights)
        elif border_include.lower() == 'max':
            cls['val_count'] = np.sum(np.logical_and(arr > cls['min'], arr <= cls['max']) * _weights)
        elif border_include.lower() == 'both':
            cls['val_count'] = np.sum(np.logical_and(arr >= cls['min'], arr <= cls['max']) * _weights)
        else:
            raise ValueError('Parameter "border_include" not defined properly. Allowed values are "min", "max", "both" or None')
        cls_out.append(cls)

    return cls_out


"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def arr_unpack(arr, scale=1, offset=0, nodata=None):
    """
    Converts the values in the array to native format by applying scale,
    offset and nodata (val or func).
    :param arr: array to be converted
    :param scale: conversion to native format scale factor
    :param offset: conversion to native format offset factor
    :param nodata: no data value or function to mask the dataset
    :return: np.array or np.ma.array converted (unpacked) to native values
    """
    # adjust for nodata
    if nodata:
        arr = apply_nodata(arr, nodata)
    # convert to native values
    if scale and scale != 1:
        arr = arr * scale
    if offset and offset != 0:
        arr = arr + offset

    return arr


"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def apply_nodata(arr, nodata):
    """
    Masks the array with nodata definition.
    If arr is masked array the two masks are combined.
    :param arr: array to be masked
    :param nodata: no data value, function or array to mask the dataset
    :return: np.ma.array masked with nodata
    """
    if isinstance(nodata, np.ndarray):
        return np.ma.array(arr, mask=nodata)
    elif callable(nodata):
        return np.ma.array(arr, mask=nodata(arr))
    elif type(nodata) in [list, tuple]:
        return np.ma.array(arr, mask=np.isin(arr, nodata))
    else:
        return np.ma.array(arr, mask=arr == nodata)


"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
def geom_extract(geometry, indicator, stats_out=('mean', 'std', 'min', 'max', 'sum', 'counts', 'mode'), afi=None,
                 classification=None, afi_thresh = None, thresh_type = None):
    """
    Extracts the indicator statistics on input geometry using the AFI as weights.

    Global variable SUPPRESS_ERRORS controls if a custom error (UnableToExtractStats) should be raised when it's not
    possible to extract stats with given parameters. By default it is set to suppress errors and only report a warning.
    This setup is for the use case when the function is called directly and can handle an empty output.
    The opposite case, when the errors are raised is used when this function is called in a multiprocessing pool and
    it's necessary to link a proper error message with a geometry/unit identifier.

    Handles heterogeneous datasets by using the tbx_util.raster.get_common_bounds_and_shape function.

    :param geometry: GeoJSON-like feature (implements __geo_interface__) – feature collection, or geometry.
    :param indicator: path to raster file or an already opened dataset (rasterio.DatasetReader) on which statistics are extracted
    :param stats_out: definition of statistics to extract, the list is directly forwarded to function
        asap_toolbox.util.raster.arr_stats.
        Additionally, accepts "counts" keyword that calculates following values:
            - total - overall unit grid coverage
            - valid_data - indicator without nodata
            - valid_data_after_masking - indicator used for calculation
            - weight_sum - total mask sum
            - weight_sum_used - mask sum after masking of dataset nodata is applied
    :param afi: path to Area Fraction index or weights - path to raster file or an already opened dataset (rasterio.DatasetReader)
    :param afi_thresh: threshold to mask out the afi data
    :param classification: If defined, calculates the pixel/weight sums of each class defined.
        Defined as JSON dictionary with borders as list of min, max value pairs and border behaviour definition:
            {
                borders: ((min1, max1), (min2, max2), ..., (min_n, max_n)),
                border_include: [min|max|both|None]
            }
    :return: dict with extracted stats divided in 3 groups:
        - stats - dict with calculated stats values (mean, std, min, max)
        - counts - dict with calculated count values (total; valid_data; valid_data_after_masking; weight_sum; weight_sum_used)
        - classification - dict with border definitions and values
        {
            stats: {mean: val, std: min: val, max: val, ...}
            counts: {total: val, valid_data: valid_data_after_masking: val, weight_sum: val, ...}
            classification: {
                borders: ((min1, max1), (min2, max2), ..., (min_n, max_n)),
                border_include: val,
                values: (val1, val2, val3,...)
            }
        }
        raises UnableToExtractStats error if geom outside raster, if the geometry didn't catch any pixels
    """
    output = dict()
    # make sure inputs are opened
    indicator_ds = indicator if isinstance(indicator, rasterio.DatasetReader) else rasterio.open(indicator)
    rasters_list = [indicator_ds]
    if afi:
        afi_ds = afi if isinstance(afi, rasterio.DatasetReader) else rasterio.open(afi)
        rasters_list.append(afi_ds)

    # get unified read window, bounds and resolution if heterogeneous resolutions
    try:
        read_bounds, read_shape = get_common_bounds_and_shape([geometry], rasters_list)
    except rasterio.errors.WindowError:
        e_msg = 'Geometry has no intersection with the indicator'
        if SUPPRESS_ERRORS:
            # log.warning('Skipping extraction! ' + e_msg)
            return
        else:
            raise UnableToExtractStats(e_msg)

    # fetch indicator array
    indicator_arr = read_masked(ds=indicator_ds, mask=[geometry], window=indicator_ds.window(*read_bounds), 
                                       indexes=None, use_pixels='CENTER', out_shape=read_shape)
    geom_mask = indicator_arr.mask
    # skip extraction if no pixels caught by geom
    if np.all(geom_mask):
        e_msg = 'No pixels caught by geometry'
        if SUPPRESS_ERRORS:
            # log.warning('Skipping extraction! ' + e_msg)
            return
        else:
            raise UnableToExtractStats(e_msg)
    # convert pixel values if ENVI file
    if indicator_ds.nodatavals:
        _dtype_conversion = dict(nodata=indicator_ds.nodatavals)
    if _dtype_conversion:
        indicator_arr = arr_unpack(indicator_arr, **_dtype_conversion)
    valid_data_mask = indicator_arr.mask

    # fetch mask array
    if afi:
        afi_arr = read_masked(ds=afi_ds, mask=[geometry], indexes=None, 
                                     window=afi_ds.window(*read_bounds), use_pixels='CENTER', out_shape=read_shape)
        
        if afi_thresh is not None:
            if thresh_type == 'Fixed':
                afi_arr[~np.isnan(afi_arr) & (afi_arr <= afi_thresh) & ~afi_arr.mask] = 0
                
            elif thresh_type == 'Percentile':
                
                m_afi_arr = afi_arr[~np.isnan(afi_arr) & (afi_arr > 0) & ~afi_arr.mask]
                
                if len(m_afi_arr) > 0:                
                    thresh_PT = np.percentile(m_afi_arr, afi_thresh)
                
                    afi_arr[~np.isnan(afi_arr) & (afi_arr <= thresh_PT) & ~afi_arr.mask] = 0
            
            afi_arr = np.ma.array(afi_arr, mask=(afi_arr.mask + (afi_arr == 0))) 
                
        # convert pixel values if ENVI file
        if afi_ds.nodatavals:
            _dtype_conversion = dict(nodata=afi_ds.nodatavals)
        if _dtype_conversion:
            afi_arr = arr_unpack(afi_arr, **_dtype_conversion)
        # apply the afi mask nodata mask to the dataset
        indicator_arr = np.ma.array(indicator_arr, mask=(afi_arr.mask + (afi_arr == 0)))

    # check if any data left after applying all the masks
    if np.sum(~indicator_arr.mask) == 0:
        e_msg = 'No data left after applying all the masks, mask sum == 0'
        if SUPPRESS_ERRORS:
            # log.warning('Skipping extraction! ' + e_msg)
            return output
        else:
            raise UnableToExtractStats(e_msg)

    # extractions
    if any(val in ('min', 'max', 'mean', 'sum' 'std', 'mode') for val in stats_out):
        
        output['stats'] = arr_stats(indicator_arr, afi_arr if afi else None, stats_out)

    if 'counts' in stats_out:
        output['counts'] = dict()
        # total - overall unit grid coverage
        output['counts']['total'] = int((~geom_mask).sum())
        # valid_data - indicator without nodata
        output['counts']['valid_data'] = int(np.sum(~valid_data_mask))
        if afi:
            output['counts']['valid_data_after_masking'] = int(np.sum(~indicator_arr.mask))
            # weight_sum - total mask sum
            output['counts']['weight_sum'] = afi_arr.sum()
            if type(output['counts']['weight_sum']) == np.uint64:
                output['counts']['weight_sum'] = int(output['counts']['weight_sum'])
            # weight_sum_used - mask sum after masking of dataset nodata is applied
            afi_arr_compressed = np.ma.array(afi_arr, mask=indicator_arr.mask).compressed()
            output['counts']['weight_sum_used'] = afi_arr_compressed.sum()
            if type(output['counts']['weight_sum_used']) == np.uint64:
                output['counts']['weight_sum_used'] = int(output['counts']['weight_sum_used'])

    if classification:
        cls_def = [{'min': _min, 'max': _max} for _min, _max in classification['borders']]
        classification_out = classification.copy()
        classification_out['border_include'] = classification.get('border_include', 'min')
        class_res = arr_classes_count(indicator_arr, cls_def=cls_def, weights=afi_arr if afi else None,
                                             border_include=classification_out['border_include'])
        classification_out['values'] = [i['val_count'] for i in class_res]
        output['classification'] = classification_out

    return output


"""
@author: Abdelrahman
""" 
def netCDFToTiff(netcdf_file, in_dir, default_crs='EPSG:4326'):
    """
    Converts a NetCDF file to TIFF format using a specified or default CRS and applies LZW compression.
    Assumes NetCDF has geospatial data.

    Args:
        netcdf_file (str): Path to the NetCDF file to be converted.
        default_crs (str): Default Coordinate Reference System in EPSG code. Defaults to 'EPSG:4326'.

    Returns:
        name of the .tif output file
    """
    assert (netcdf_file.endswith(".nc"))
    output_dir = os.path.join(in_dir, TIFF_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(netcdf_file).replace(".nc", ".tif"))

    with rasterio.open(netcdf_file) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
        if crs is None:
            crs = CRS.from_string(default_crs)
        kwargs = src.profile.copy()
        kwargs.update(
            driver='GTiff',
            height=src.height,
            width=src.width,
            count=data.shape[0],
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            compress="lzw"  # compression
        )

        with rasterio.open(output_file, 'w', **kwargs) as dst:
            dst.write(data)

    return output_file

"""
@author: Dilli R. Paudel
"""
def unzipAgERA5andRename(data_dir, country_dir, predictor_source, indicator, year):
    """
    AgERA5 data downloaded by Abdelrahman is stored as .zip files in Google Drive.
    """
    dir_path = os.path.join(data_dir, predictor_source, country_dir, indicator)
    filename = "_".join([country_dir, str(year), indicator]) + ".zip"
    file_path = os.path.join(dir_path, filename)
    files = [ f for f in os.listdir(dir_path) if f.endswith(".nc") and (str(year) in f)]
    # if files do not exist
    if (not files):
        with ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dir_path)

        # rename files
        files = [ f for f in os.listdir(dir_path) if f.endswith(".nc") and (str(year) in f)]
        for f in files:
            old_file = os.path.join(dir_path, f)
            try:
                agera5_date_index = f.index("AgERA5_" + str(year))
                date_str = f[agera5_date_index + 7: agera5_date_index+15]
                _, extension = os.path.splitext(f)
                new_name = "_".join([predictor_source, indicator, date_str]) + extension
                new_file = os.path.join(dir_path,  new_name)
                os.rename(old_file, new_file)

            except ValueError:
                continue

"""
@author: Dilli R. Paudel
"""
def deleteAgERA5Files(indicator_dir):
    """
    Delete extracted .nc files or converted .tif files.
    NOTE: Must be called after processing corresponding files.
    """
    files = [ f for f in os.listdir(indicator_dir) if f.endswith(".nc")]
    for f in files:
        os.remove(os.path.join(indicator_dir, f))

    # remove tif files
    tiff_dir = os.path.join(indicator_dir, TIFF_DIRECTORY_NAME)
    if (os.path.isdir(tiff_dir)):
        files = [ f for f in os.listdir(tiff_dir) if f.endswith(".tif")]
        for f in files:
            os.remove(os.path.join(tiff_dir, f))


"""
@author: Dilli R. Paudel
"""
def get_time_series_files(data_path,
                          exclude_filenames=[],
                          exclude_extensions=[],
                          year=2000):
    files = []
    for f in os.listdir(data_path):
        # exclude sub-directories
        if (os.path.isdir(os.path.join(data_path, f))):
            continue

        fname, ext = os.path.splitext(f)
        if ((fname in exclude_filenames) or
            (ext in exclude_extensions)):
            continue

        # we expect the last part of filename to be YYYYMMDD
        date_str = fname[-8:]
        if (int(date_str[:4]) == year):
            files.append(f)

    return files


"""
@author: Dilli R. Paudel
"""

def get_shapes(file_path, country="US", admin_level=None):
    geo_df = gpd.read_file(file_path)
    if (country in EU_COUNTRIES):
        geo_df = geo_df[geo_df[EU_COUNTRY_CODE_KEY] == country]
    elif (country in FEWSNET_COUNTRIES):
        geo_df = geo_df[geo_df[FEWSNET_ADMIN_ID_KEY].str[:2] == country]


    if (admin_level is not None):
        assert (country in EU_COUNTRIES)
        geo_df = geo_df[geo_df[EU_ADMIN_LEVEL_KEY] == admin_level]

    return geo_df


"""
@author: Dilli R. Paudel
"""

def get_admin_id(df, country):
    # European countries
    if (country in EU_COUNTRIES):
        df["adm_id"] = df["NUTS_ID"]
    elif (country == "US"):
        df["adm_id"] = "US-" + df["STATEFP"] + "-" + df["COUNTYFP"]
    elif (country == "CN"):
        df["adm_id"] = df["ADM1_PCODE"].str.replace('CN', 'CN-')
    # MEXICO:adm_id was created for Mexico using the number ID (1-32) for states coming from the original data
    # and prefix "MX" was added so we have "MX01" or "MX32". For more information check Mexico data card.
    elif (country == "MX"):
         df["adm_id"] = df["adm_id"]

    # elif (country == "IN"):
    #     df["adm_id"] = df["adm_id"]  #adm_id already exists
    # elif (country == "ML"):
    #     df["adm_id"] = df["adm_id"] # already exists
    # NOTE for FEWSNET, adm_id already exists
    # elif (country in FEWSNET_COUNTRIES):
        # FEWS NET's "FNID" is converted to "adm_id" during the data preparation
        # df["adm_id"] = df["adm_id"]

    return df


"""
@author: Guanyuan Shuai
"""

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
    


"""
@author: Guanyuan Shuai
"""
    
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
