from geoutils import *

def main(data_dir, 
         crop_mask, 
         shapefile,
         rescale_factor,
         selected_countries,
         country_column_name,
         crop_mask_invalid_values,
         output_csv_path,
         use_crop_mask,
         data_source):

    predictor_paths = loadData(data_dir)

    global predictor_instance

    predictor_instance = predictor_paths[0]

    compatibility_result = checkDataCompatibility(predictor_instance,
                                                   crop_mask,
                                                   shapefile)

    updated_pathes= processCompatibilityIssues(compatibility_result,
                                                crop_mask,
                                                predictor_instance,
                                                shapefile,
                                                rescale_factor)

    updated_crop_mask_path, updated_shapefile_path = updated_pathes.values()
    
    filtered_shapefile = filterShapefileForCountry(updated_shapefile_path,
                                             selected_countries,
                                             country_column_name=country_column_name)

    clipped_director = clipMultipleRasters(predictor_paths,
                            filtered_shapefile,
                            invalid_values=crop_mask_invalid_values)

    aggregate_rasters_conditional(clipped_director,
                                  filtered_shapefile,
                                  output_csv_path,
                                  updated_crop_mask_path,
                                  use_crop_mask=use_crop_mask,
                                  predictor_name=data_source)