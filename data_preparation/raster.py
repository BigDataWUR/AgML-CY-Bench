# File: raster.py
# Author: Joint Research Center

import numpy as np
import rasterio
import rasterio.features


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