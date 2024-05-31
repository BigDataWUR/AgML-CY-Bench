# Author: Abdelrahman

import os
import rasterio
from rasterio.crs import CRS

TIFF_DIRECTORY_NAME = "Tiff"

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