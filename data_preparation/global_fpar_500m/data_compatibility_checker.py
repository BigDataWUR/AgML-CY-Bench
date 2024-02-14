import rasterio
import geopandas as gpd
from pyproj import CRS
from rasterio.errors import CRSError

def check_data_compatibility(crop_mask_path, raster_data_path, shapefile_path):
    try:
        # Open the crop mask and raster data using rasterio
        with rasterio.open(crop_mask_path) as crop_mask, rasterio.open(raster_data_path) as raster_data:
            # Check if the pixel sizes are identical for raster data
            if crop_mask.res != raster_data.res:
                print("Warning: The pixel sizes of the crop mask and raster data are not identical.")
                print("###"*22)
                print(f"Crop Mask Resolution: {crop_mask.res}") 
                print(f"Raster Resolution   : {raster_data.res}")
                print("###"*22)
                print('\n')
            else:
                print("The pixel sizes of the crop mask and raster data are identical.")
                print("###"*22)
                print(f"Crop Mask Resolution: {crop_mask.res}") 
                print(f"Raster Resolution   : {raster_data.res}")
                print("###"*22)
                print('\n')

            # Use pyproj CRS to get a human-readable CRS name
            crop_mask_crs_name = CRS(crop_mask.crs).name
            raster_data_crs_name = CRS(raster_data.crs).name

            # Check if the projections are identical for raster data
            if crop_mask.crs != raster_data.crs:
                print("Warning: The projections of the crop mask and raster data are not identical.")
                print("###"*22)
                print(f"Crop Mask CRS: {crop_mask_crs_name}") 
                print(f"Raster CRS  : {raster_data_crs_name}")
                print("###"*22)
                print('\n')
            else:
                print("The projections of the crop mask and raster data are identical.")
                print("###"*22)
                print(f"Crop Mask CRS: {crop_mask_crs_name}") 
                print(f"Raster CRS   : {raster_data_crs_name}")
                print("###"*22)
                print('\n')

        # Load the shapefile using geopandas
        shapefile = gpd.read_file(shapefile_path)
        shapefile_crs_name = CRS(shapefile.crs).name

        # Check if the shapefile projection matches the raster data
        with rasterio.open(raster_data_path) as raster_data:
            if CRS(shapefile.crs).name != CRS(raster_data.crs).name:
                print("Warning: The projection of the shapefile does not match the raster data.")
                print("###"*22)
                print(f"Shapefile CRS: {shapefile_crs_name}")
                print(f"Raster CRS   : {raster_data_crs_name}")
                print("###"*22)
                print('\n')
            else:
                print("The projection of the shapefile matches the raster data.")
                print("###"*22)
                print(f"Shapefile CRS: {shapefile_crs_name}")
                print(f"Raster CRS   : {raster_data_crs_name}")
                print("###"*22)
                print('\n')
        
        # Final check for preprocessing safety
        if len(set([CRS(crop_mask.crs).name, CRS(raster_data.crs).name, CRS(shapefile.crs).name])) == 1:
            print("All data (crop mask, raster data, and shapefile) are compatible for preprocessing.")
        else:
            print("Please correct the above issues before preprocessing.")


    except CRSError as e:
        print(f"Error reading CRS data: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")