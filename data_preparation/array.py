# File array.py
# Author: Joint Research Center

import numpy as np
from itertools import repeat


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

    if any(elem in _output for elem in ('std', 'min', 'max', 'sum', 'median', 'mode')):
        arr_compressed = _arr.compressed()

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
            ind = np.isnan(arr_compressed) | np.isnan(weights_compressed) | (_arr <= -9999)

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
        out_vals['mode'] = np.argmax(np.bincount(arr_compressed[~ind]))

    if 'count' in _output:
        out_vals['count'] = int((~_arr.mask).sum())

    if 'weight_sum' in _output and weights is not None:
        ind = np.isnan(weights_compressed) | (arr_compressed <= -9999)
        weights_compressed = np.ma.array(weights[~ind], mask=_arr.mask[~ind]).compressed()
        out_vals['weight_sum'] = weights_compressed.sum()

    # convert to regular py types from np types which can cause problems down the line like JSON serialisation
    out_vals = {k: v.item() for k, v in out_vals.items()}

    return out_vals


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
