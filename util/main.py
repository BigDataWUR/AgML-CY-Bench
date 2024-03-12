from geoutils import *


def main(
        data_dir,                  # directory path of predictor data files (ex: FPAR)
        crop_mask,                 # file path of the crop mask (GEOGLAM, WorldCereal etc.)
        shapefile,                 # file path of the shapefile
        rescale_factor,            # rescale the crop mask to (0,100) for example GEOGLAM original scale is (0 - 10,000)
        selected_countries,        # list of country names in the shapefile
        country_column_name,       # name of the column that contain the country names
        predictor_invalid_values,  # list of invalid values (flags) in predictor data, ex: FPAR data contains flags( 255=not processed, 254=water, 251=other land)
        output_csv_path,           # path of the output csv file (extracted statistical data)
        use_crop_mask,             # option to turn off using crop mask while aggregating data
        data_source):              # predictor name

    """
    Automates the workflow for extracting and aggregating statistical data from geospatial sources, 
    specifically utilizing shapefiles and crop masks. The process unfolds as follows:

    1. Data Loading: 
         -Loads predictor raster datasets, crop mask, and shapefile.

    2. Compatibility Check: Evaluates the compatibility among datasets by examining pixel sizes and projections:
         - It compares the predictor rasters and crop mask pixel size and projection.
         - checks the projection consistency of the shapefile with predictor

    3. Fix Compatibility Issues:  ### IF THE DATA IS NOT COMPATIBLE ###
         - Resampling + Rescalling: resamples the crop mask to match the pixel size of the predictor raster's, and rescale the crop mask to (0,100)
         - Reprojecting: reprojecting the mask and shapefile to match the geographic projection of the raster data. 

    4. Filter Shapefile:
         - filters the shapefile to only include the specified countries (ROI).

    5. Data Clipping: (save computational time and work with less RAM)
         - Clips the predictor data to the boundaries defined in the filtered shapefile.

    6. Data Aggregatio - Function of aggregation data work as follows:

         1. define list to build a dataframe by storing the aggregated data and save it as csv
         2. loading the filtered shapefile
         3. loading crop mask and identify no data values of crop mask
              - Loading crop mask outside the loops saves computational time

         4. loop through each each predictor data path (predictor_paths)
              - extract the date from the filename
              - loading the predictor data 
              5. loop through each gemotry object in the loaded filtered shapefile
                   - extract the geometry of object
                   - clip the predictor data to the geometry
                   - clip the crop mask to the geometry
                   - replace the no data values in crop mask defined from masks's meta data with nan
                   - mask the clipped predicotr data with updated crop mask (without no data values)
                   - calculate weighted mean (data: predictor, weights: crop mask) and round to 3 decimal places
              6. concatenate the aggregated data to the dataframe:
                   - create dict from contain the keys and valus of each row in shapefile
                   - add to dict the aggregated data (weighted mean)
                   - append the dict to the created list
         7. convert the list contains dicts to dataframe 
         8. save the dataframe as csv
         NOTE: if crop mask option is closed, the aggregation will be the mean of each cliped predictor
    """

    # create a list of predictor paths
    predictor_paths = loadData(data_dir)

    global predictor_instance
    # get a sample for checking the compatibility
    predictor_instance = predictor_paths[0]
    # run the compatibility check
    compatibility_result = checkDataCompatibility(predictor_instance,
                                                  crop_mask, shapefile)
    # fix compatibility issues and get dict contain updated paths with resampled and reprojected crop mask and shapefile
    updated_pathes = processCompatibilityIssues(compatibility_result,
                                                crop_mask,
                                                predictor_instance,
                                                shapefile,
                                                rescale_factor)
    # unpack dict to get the updated crop mask and shapefile
    updated_crop_mask_path, updated_shapefile_path = updated_pathes.values()
    # filter shapefile to ROI
    filtered_shapefile = filterShapefileForCountry(updated_shapefile_path,
                                                   selected_countries,
                                                   country_column_name=country_column_name)
    # clip the predictor data to the ROI
    clipped_director = clipMultipleRasters(predictor_paths,
                                           filtered_shapefile,
                                           invalid_values=predictor_invalid_values)
    # aggregate the predictor dataset
    aggregate_rasters_conditional(clipped_director,
                                  filtered_shapefile,
                                  output_csv_path,
                                  updated_crop_mask_path,
                                  use_crop_mask=use_crop_mask,
                                  predictor_name=data_source)
