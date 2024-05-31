# Author: Joint Research Center

import numpy as np
import rasterio
import xarray
from itertools import repeat
from multiprocessing import Pool
import logging

# from ..glimpse import envi


log = logging.getLogger(__name__)


SUPPRESS_ERRORS = True


class UnableToExtractStats(Exception):
    pass


def geom_extract(geometry,
                 indicator_dir,
                 indicator_file,
                 indicator,
                 stats_out=[],
                 crop_mask_file=None,
                 classification=None,
                 crop_mask_thresh = None,
                 thresh_type = None):
    """
    Extracts the indicator statistics on input geometry using the crop mask (area fractions) as weights.

    Global variable SUPPRESS_ERRORS controls if a custom error (UnableToExtractStats) should be raised when it's not
    possible to extract stats with given parameters. By default it is set to suppress errors and only report a warning.
    This setup is for the use case when the function is called directly and can handle an empty output.
    The opposite case, when the errors are raised is used when this function is called in a multiprocessing pool and
    it's necessary to link a proper error message with a geometry/unit identifier.

    Handles heterogeneous datasets by using the tbx_util.raster.get_common_bounds_and_shape function.

    :param geometry: geometry
    :param indicator_dir: path to directory containing indicator raster files
    :param indicator_file: path to raster file
    :param indicator: indicator name
    :param stats_out: definition of statistics to extract, the list is directly forwarded to function
        asap_toolbox.util.raster.arr_stats.
        Additionally, accepts "counts" keyword that calculates following values:
            - total - overall unit grid coverage
            - valid_data - indicator without nodata
            - valid_data_after_masking - indicator used for calculation
            - weight_sum - total mask sum
            - weight_sum_used - mask sum after masking of dataset nodata is applied
    :param crop_mask_file: path to a raster file with crop mask area fraction weights
    :param crop_mask_thresh: threshold to select the crop_mask data
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
    if (indicator_file.endswith(".tif")):
        indicator_ds = rasterio.open(indicator_file)
    elif (indicator_file.endswith(".nc")):
        tif_file = netCDFToTiff(indicator_file, indicator_dir)
        indicator_ds = rasterio.open(tif_file)
    else:
        e_msg = 'File format must be .tif or .nc'
        if SUPPRESS_ERRORS:
            # log.warning('Skipping extraction! ' + e_msg)
            return
        else:
            raise UnableToExtractStats(e_msg)

    rasters_list = [indicator_ds]

    if crop_mask_file:
        crop_mask_ds = rasterio.open(crop_mask_file)
        rasters_list.append(crop_mask_ds)

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
    # if indicator_ds.driver == 'ENVI':
    #     _dtype_conversion = envi.get_dtype_conversion(indicator_ds.name)
    if indicator_ds.nodatavals:
        _dtype_conversion = dict(nodata=indicator_ds.nodatavals)

    if _dtype_conversion:
        indicator_arr = arr_unpack(indicator_arr, **_dtype_conversion)

    valid_data_mask = indicator_arr.mask

    # fetch mask array
    if crop_mask_file:
        crop_mask_arr = read_masked(ds=crop_mask_ds, mask=[geometry], indexes=None,
                              window=crop_mask_ds.window(*read_bounds), use_pixels='CENTER', out_shape=read_shape)

        if crop_mask_thresh is not None:
            if thresh_type == 'Fixed':
                crop_mask_arr[~np.isnan(crop_mask_arr) & (crop_mask_arr <= crop_mask_thresh) & ~crop_mask_arr.mask] = 0

            elif thresh_type == 'Percentile':

                m_crop_mask_arr = crop_mask_arr[~np.isnan(crop_mask_arr) & (crop_mask_arr > 0) & ~crop_mask_arr.mask]

                if len(m_crop_mask_arr) > 0:
                    thresh_PT = np.percentile(m_crop_mask_arr, crop_mask_thresh)

                    crop_mask_arr[~np.isnan(crop_mask_arr) & (crop_mask_arr <= thresh_PT) & ~crop_mask_arr.mask] = 0

            crop_mask_arr = np.ma.array(crop_mask_arr, mask=(crop_mask_arr.mask + (crop_mask_arr == 0)))

        # convert pixel values if ENVI file
        # if crop_mask_ds.driver == 'ENVI':
        #     _dtype_conversion = envi.get_dtype_conversion(crop_mask_ds.name)

        if crop_mask_ds.nodatavals:
            _dtype_conversion = dict(nodata=crop_mask_ds.nodatavals)
        if _dtype_conversion:
            crop_mask_arr = arr_unpack(crop_mask_arr, **_dtype_conversion)

        # apply the crop_mask nodata mask to the dataset
        indicator_arr = np.ma.array(indicator_arr, mask=(crop_mask_arr.mask + (crop_mask_arr == 0)))

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
        output['stats'] = arr_stats(indicator_arr, crop_mask_arr if crop_mask_file else None, stats_out)

    if 'counts' in stats_out:
        output['counts'] = dict()
        # total - overall unit grid coverage
        output['counts']['total'] = int((~geom_mask).sum())
        # valid_data - indicator without nodata
        output['counts']['valid_data'] = int(np.sum(~valid_data_mask))
        if crop_mask_file:
            output['counts']['valid_data_after_masking'] = int(np.sum(~indicator_arr.mask))
            # weight_sum - total mask sum
            output['counts']['weight_sum'] = crop_mask_arr.sum()
            if type(output['counts']['weight_sum']) == np.uint64:
                output['counts']['weight_sum'] = int(output['counts']['weight_sum'])
            # weight_sum_used - mask sum after masking of dataset nodata is applied
            crop_mask_arr_compressed = np.ma.array(crop_mask_arr, mask=indicator_arr.mask).compressed()
            output['counts']['weight_sum_used'] = crop_mask_arr_compressed.sum()
            if type(output['counts']['weight_sum_used']) == np.uint64:
                output['counts']['weight_sum_used'] = int(output['counts']['weight_sum_used'])

    if classification:
        cls_def = [{'min': _min, 'max': _max} for _min, _max in classification['borders']]
        classification_out = classification.copy()
        classification_out['border_include'] = classification.get('border_include', 'min')
        class_res = arr_classes_count(indicator_arr, cls_def=cls_def, weights=crop_mask_arr if crop_mask_file else None,
                                      border_include=classification_out['border_include'])
        classification_out['values'] = [i['val_count'] for i in class_res]
        output['classification'] = classification_out

    return output