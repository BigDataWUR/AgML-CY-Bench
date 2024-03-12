import rasterio
from rasterio.mask import mask
from rasterio.errors import CRSError
from rasterio import warp
from rasterio.enums import Resampling
from shapely.geometry import mapping
from pyproj import CRS
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re
import os
import glob


def loadData(directory_path):
    paths = glob.glob(directory_path+'/*.tif')
    return paths


def savedFilePath(file_path):

    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    return file_dir, file_name


def print_issue(issue_type, details, suggestion):
    print(f"ISSUE: {issue_type}")
    print("###"*22)
    for detail in details:
        print(detail)
    print("###"*22)
    print(f"SUGGESTION: {suggestion}\n")


def print_no_issue(issue_type, detail):
    print(f"NO ISSUE: {issue_type}: {detail}")


def check_resolutions(crop_mask, raster_data):
    if crop_mask.res != raster_data.res:
        print_issue("The pixel sizes of the crop mask and raster data are not identical",
                    [f"Crop Mask Resolution: {crop_mask.res}",
                        f"Raster Resolution   : {raster_data.res}"],
                    "Consider resampling the crop mask to match the pixel size of the raster data.")
    else:
        print_no_issue(
            "The pixel sizes of the crop mask and raster data are identical", crop_mask.res)


def check_projection_compatibility(crs_name1, crs_name2, data1_name, data2_name):
    if crs_name1 != crs_name2:
        print_issue(f"The projections of the {data1_name} and {data2_name} are not identical",
                    [f"{data1_name} CRS: {crs_name1}",
                        f"{data2_name} CRS: {crs_name2}"],
                    f"Consider reprojecting the {data1_name} to match the projection of the {data2_name}.")
    else:
        print_no_issue(
            f"The projections of the {data1_name} and {data2_name} are identical", crs_name1)


def checkDataCompatibility(raster_data_path, crop_mask_path, shapefile_path):
    try:
        with rasterio.open(crop_mask_path) as crop_mask, rasterio.open(raster_data_path) as raster_data:
            check_resolutions(crop_mask, raster_data)
            crop_mask_crs_name = CRS(crop_mask.crs).name
            raster_data_crs_name = CRS(raster_data.crs).name
            check_projection_compatibility(
                crop_mask_crs_name, raster_data_crs_name, "crop mask", "raster data")

        shapefile = gpd.read_file(shapefile_path)
        shapefile_crs_name = CRS(shapefile.crs).name
        check_projection_compatibility(
            raster_data_crs_name, shapefile_crs_name, "raster data", "shapefile")

        if crop_mask.res == raster_data.res and crop_mask_crs_name == raster_data_crs_name == shapefile_crs_name:
            print(
                "All data (crop mask, raster data, and shapefile) are compatible for preprocessing.")
        else:
            print("Some compatibility issues were detected. Please review the issues and apply the suggested solutions.")

    except CRSError as e:
        print(f"Error reading CRS data: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def checkDataCompatibilitys(raster_data_path, crop_mask_path, shapefile_path):
    actions = {'resample_crop_mask': False,
               'reproject_shapefile': False, 'is_compatible': True}

    try:
        with rasterio.open(crop_mask_path) as crop_mask, rasterio.open(raster_data_path) as raster_data:
            check_resolutions(crop_mask, raster_data)
            if crop_mask.res != raster_data.res:
                actions['resample_crop_mask'] = True
                actions['is_compatible'] = False

            crop_mask_crs_name = CRS(crop_mask.crs).name
            raster_data_crs_name = CRS(raster_data.crs).name
            check_projection_compatibility(
                crop_mask_crs_name, raster_data_crs_name, "crop mask", "raster data")
            if crop_mask_crs_name != raster_data_crs_name:
                actions['is_compatible'] = False

        shapefile = gpd.read_file(shapefile_path)
        shapefile_crs_name = CRS(shapefile.crs).name
        check_projection_compatibility(
            raster_data_crs_name, shapefile_crs_name, "raster data", "shapefile")
        if raster_data_crs_name != shapefile_crs_name:
            actions['reproject_shapefile'] = True
            actions['is_compatible'] = False

        if actions['is_compatible']:
            print(
                "All data (crop mask, raster data, and shapefile) are compatible for preprocessing.")
        else:
            print("Some compatibility issues were detected. Adjustments may be required.")

        return actions
    except CRSError as e:
        print(f"Error reading CRS data: {e}")
        return actions
    except Exception as e:
        print(f"An error occurred: {e}")
        return actions


def reprojectShapefileToRaster(raster_data_path, shapefile_path):
    try:
        file_dir, file_name = savedFilePath(shapefile_path)
        with rasterio.open(raster_data_path) as src:
            raster_crs = src.crs

        shapefile = gpd.read_file(shapefile_path)

        shapefile_reprojected = shapefile.to_crs(raster_crs)
        output_path = f'{file_dir}/reprojected_{file_name}'
        shapefile_reprojected.to_file(output_path)

        print(
            f"Shapefile reprojected to match raster CRS and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def filterShapefileForCountry(shapefile_path, countries, country_column_name):

    file_dir, file_name = savedFilePath(shapefile_path)
    gdf = gpd.read_file(shapefile_path)
    filtered_gdf = gdf[gdf[country_column_name].isin(countries)]
    output_path = f'{file_dir}/filtered_{file_name}'
    filtered_gdf.to_file(output_path)

    return output_path


def rescaleResampleCropMask(crop_mask_path, raster_data_path, scale_factor=(0, 100)):

    file_dir, file_name = savedFilePath(crop_mask_path)

    with rasterio.open(raster_data_path) as target_raster:
        target_transform = target_raster.transform

    with rasterio.open(crop_mask_path) as crop_mask:
        crop_mask_data = crop_mask.read(1)
        old_min, old_max = crop_mask_data.min(), crop_mask_data.max()
        new_min, new_max = scale_factor

        rescaled_data = ((crop_mask_data - old_min) /
                         (old_max - old_min)) * (new_max - new_min) + new_min

        out_meta = crop_mask.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": target_raster.height,
            "width": target_raster.width,
            "transform": target_transform,
            "crs": target_raster.crs,
            "compress": "DEFLATE",  # Specify compression scheme here
            "predictor": "2",  # Good for continuous data
            "zlevel": 1  # Compression level, 9 is the highest
        })

    resampled_data = np.empty(
        shape=(target_raster.height, target_raster.width), dtype=out_meta['dtype'])
    warp.reproject(
        source=rescaled_data,
        destination=resampled_data,
        src_transform=crop_mask.transform,
        src_crs=crop_mask.crs,
        dst_transform=target_transform,
        dst_crs=target_raster.crs,
        resampling=Resampling.bilinear
    )

    output_path = f'{file_dir}/rescaled_resampled_{file_name}'

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(resampled_data, 1)

    return output_path


def processCompatibilityIssues(actions, crop_mask_path, predictor_data_path, shapefile_path, rescale_factor):

    updated_paths = {
        'crop_mask': crop_mask_path,
        'shapefile': shapefile_path
    }

    if not actions['is_compatible']:
        if actions['resample_crop_mask']:
            print("Resampling crop mask...")
            updated_paths['crop_mask'] = rescaleResampleCropMask(crop_mask_path,
                                                                 predictor_data_path,
                                                                 scale_factor=rescale_factor)

        if actions['reproject_shapefile']:
            print("Reprojecting shapefile...")
            updated_paths['shapefile'] = reprojectShapefileToRaster(shapefile_path,
                                                                    predictor_data_path)

    else:
        print("No compatibility issues detected. Proceeding without resampling or reprojection.")

    return updated_paths


def clipRasterWithShapefile(raster_path, shapefile_path, invalid_values=None):

    file_dir, file_name = savedFilePath(raster_path)

    global output_clip

    output_clip = 'clipped '+file_dir

    os.makedirs(output_clip, exist_ok=True)

    # Load the shapefile
    shapefile = gpd.read_file(shapefile_path)

    with rasterio.open(raster_path) as src:
        # Ensure the shapefile is in the same CRS as the raster
        shapefile = shapefile.to_crs(src.crs)

        # Clip the raster with the shapefile
        geoms = [mapping(shape) for shape in shapefile.geometry]
        out_image, out_transform = mask(src, geoms, crop=True)

        # Filter out invalid values
        for invalid_value in invalid_values:
            out_image = np.where(out_image == invalid_value, np.nan, out_image)

        out_meta = src.meta.copy()

    # Update metadata to reflect the number of layers, new transform, and new dimensions
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        # Ensure dtype is float to accommodate NaN values
        "dtype": 'float32'
    })

    output_path = os.path.join(
        output_clip, f"clipped_{file_name}")  # Modify as needed
    # Write the clipped and filtered raster to file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def clipMultipleRasters(raster_paths, shapefile_path, invalid_values=None):

    for raster_path in tqdm(raster_paths, total=len(raster_paths), desc="Clipping Rasters"):
        clipRasterWithShapefile(raster_path, shapefile_path, invalid_values)

    print(f"Raster clipped successfully and saved to {output_clip}")

    return loadData(output_clip)


def extract_date_from_filename(filename):
    pattern = r'\d{8}'
    match = re.search(pattern, filename)
    date_str = match.group()
    return date_str


def aggregate_rasters_conditional(predictor_paths, shapefile_path, output_csv_path, crop_mask_path=None, use_crop_mask=False, predictor_name="Value"):
    data_list = []
    shape_file = gpd.read_file(shapefile_path)

    if use_crop_mask:
        if crop_mask_path is None:
            raise ValueError(
                "Crop mask path must be provided if use_crop_mask is True.")
        mask_src = rasterio.open(crop_mask_path)
        # Dynamically read the no data value from the crop mask metadata
        no_data_value = mask_src.nodata

    for raster in tqdm(predictor_paths, desc="Processing rasters", unit="raster"):
        file_name = os.path.basename(raster)
        date_str = extract_date_from_filename(file_name)

        with rasterio.open(raster) as src:
            for _, row in shape_file.iterrows():
                geom = row.geometry
                out_image, _ = mask(src, [geom], crop=True)

                if use_crop_mask:
                    mask_image, _ = mask(mask_src, [geom], crop=True)
                    if no_data_value is not None:
                        # Mask out no data values in the crop mask
                        valid_mask = (mask_image != no_data_value)
                    else:
                        # Assume all data is valid if no_data_value is not specified
                        valid_mask = np.ones_like(mask_image, dtype=bool)

                    valid_data = out_image[valid_mask]
                    valid_weights = mask_image[valid_mask]

                    if valid_weights.sum() > 0:  # Check to avoid division by zero
                        weighted_mean = np.nansum(
                            valid_data * valid_weights) / np.nansum(valid_weights)
                        # Round to 3 decimal places
                        value = np.round(weighted_mean, 3)
                    else:
                        value = np.nan  # No valid data points

                else:
                    # Round to 3 decimal places
                    value = np.round(np.nanmean(out_image), 3)

                new_row = {col: row[col]
                           for col in shape_file.columns if col != 'geometry'}
                new_row.update({
                    'date': date_str,
                    predictor_name: value
                })

                data_list.append(new_row)

    if use_crop_mask:
        mask_src.close()

    df = pd.DataFrame(data_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")
