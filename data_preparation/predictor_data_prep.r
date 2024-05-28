if(!"terra" %in% installed.packages()){install.packages("terra")}
if(!"reshape2" %in% installed.packages()){install.packages("reshape2")}
library(terra)
library(reshape2)
library(stringr)

valid_fpar_values <- function(fpar_rast) {
  # check data_preparation/global_fpar_500m
  # FPAR values: 0 to 100 (%). Valid range is 0-100.
  # Flags: 251 is "other land", 254 is "water", 255 is "not processed".
  fpar_rast[fpar_rast > 100] <- NA
  fpar_rast[fpar_rast <= 0] <- NA

  return(fpar_rast)
}

valid_ndvi_values <- function(ndvi_rast) {
  # check data_preparation/global_MOD09CMG
  # Value range: 50 to 250 for NDVI.
  ndvi_rast[ndvi_rast < 50] <- NA
  ndvi_rast[ndvi_rast > 250] <- NA

  return(ndvi_rast)
}

transform_ndvi_values <- function(ndvi_rast) {
  # check data_preparation/global_MOD09CMG
  #  To scale, apply the formula: (x - 50)/200
  ndvi_rast <- (ndvi_rast - 50)/200
  return(ndvi_rast)
}

valid_et0_values <- function(et0_rast) {
  # check data_preparation/global_ETo_FAO
  et0_rast[et0_rast == -9999] <- NA
  return(et0_rast)
}

temperature_kelvin_to_celsius <- function(temp_rast) {
  # temp_C = temp_K - 273.15
  temp_rast <- temp_rast - 273.15
  return(temp_rast)
}

# TODO: Add functions to filter valid values for other indicators
# TODO: If required, add functions to transform indicator values

filter_and_transform <- function(ind_rast, indicator) {
  if (indicator == "fpar") {
    ind_rast <- valid_fpar_values(ind_rast)
  } else if (indicator == "ndvi") {
    ind_rast <- valid_ndvi_values(ind_rast)
    ind_rast <- transform_ndvi_values(ind_rast)
  } else if (indicator == "ET0") {
    ind_rast <- valid_et0_values(ind_rast)
  } else {
    # TODO: there is no valid data range given for GLDAS.
    return(ind_rast)
  }
  # TODO: add code for other predictors
  # dict of predictors from python code
  # predictors = {
  #     "AgERA5" : {
  #         "Precipitation_Flux": ["time series", "continuous"],
  #         "Maximum_Temperature": ["time series", "continuous"], # requires K to C conversion
  #         "Minimum_Temperature": ["time series", "continuous"], # requires K to C conversion
  #         "Mean_Temperature": ["time series", "continuous"], # requires K to C conversion
  #         "Solar_Radiation_Flux" : ["time series", "continuous"],
  #     },
  #     "MOD09CMG" : {
  #         "ndvi": ["time series", "continuous"],
  #     },
  #     "JRC_FPAR500m" : {
  #         "fpar": ["time series", "continuous"],
  #     },
  #     "WISE_Soil" : {
  #         "AWC" : ["static", "continuous"],
  #         "drainage_class": ["static", "categorical"],
  #         "bulk_density" : ["static", "continuous"]
  #      },
  #      "GLDAS" : {
  #         "rootzone_moisture": ["time series", "continuous"],
  #         "surface_moisture": ["time series", "continuous"],
  #       }
  return(ind_rast)
}

crop_and_aggregate <- function(ind_rast, sel_shapes, crop_mask, is_categorical) {
  ind_rast <- crop(ind_rast, sel_shapes)
  crop_mask <- crop(crop_mask, sel_shapes)

  weighted_inds <- ind_rast * crop_mask 
  # for categorical data, extract the mode
  # TODO: check the behavior of modal
  if (is_categorical) {
    result <- extract(weighted_inds, sel_shapes, fun=modal, na.rm=TRUE, ID=FALSE)
    result$adm_id <- sel_shapes$adm_id
  } else {
    result <- extract(weighted_inds, sel_shapes, fun=sum, na.rm=TRUE, ID=FALSE)
    result$adm_id <- sel_shapes$adm_id

    result_w <- extract(crop_mask, sel_shapes, fun=sum, na.rm=TRUE, ID=FALSE)
    result_w$adm_id <- sel_shapes$adm_id

    # Divide the result by weights
    result[-ncol(result)] <- result[-ncol(result)]/result_w[-ncol(result_w)]
    result <- melt(result)
  }

  return(result)
}

aggregation_postprocess <- function(result, crop, indicator, is_time_series) {
  colnames(result) <- c("adm_id", "ind_file", indicator)
  if (is_time_series) {
    result$date <- str_sub(result$ind_file, -8,-1)
  }

  result$country <- substr(result$adm_id, 1, 2)
  result$crop_name <- crop
  if (is_time_series) {
    result <- result[, c("crop_name", "country", "adm_id",
                         "date", indicator)]
  } else {
    result <- result[, c("crop_name", "country", "adm_id", indicator)]
  }
  return(result)
}

save_csv <- function(result, crop, region, indicator) {
  if (!dir.exists(file.path(AGML_ROOT, "R-output",
                            crop, region, indicator))) {
    dir.create(file.path(AGML_ROOT, "R-output",
  		                   crop, region, indicator),
               recursive=TRUE)
  }
  write.csv(result,
            file.path(AGML_ROOT, "R-output",
     	                crop, region, indicator,
                      paste0(indicator, "_", region, ".csv")),
            row.names=FALSE)

}

resample_crop_mask <- function(crop_mask, ind_rast, resampled_crop_mask_file) {
  crop_mask <- resample(crop_mask, ind_rast, method="bilinear")
  writeRaster(x=crop_mask, filename=resampled_crop_mask_file)

  return(crop_mask)
}

valid_crop_mask_values <- function(crop_mask) {
  # valid crop mask values
  crop_mask[crop_mask > 100] <- NA
  crop_mask[crop_mask <= 0] <- NA
  return(crop_mask)
}

process_indicator <- function(crop, region, sel_shapes,
                              start_year, end_year,
                              crop_mask_file,
			                        indicator_source,
			                        indicator,
			                        filename_pattern,
			                        is_time_series,
			                        is_categorical) {
  resampled_crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                                        paste("crop_mask", crop, indicator,
                                              "res.tif", sep="_"))
  resampled <- FALSE
  if (file.exists(resampled_crop_mask_file)) {
    crop_mask <- rast(resampled_crop_mask_file)
    resampled <- TRUE
  } else {
    crop_mask <- rast(crop_mask_file)
  }

  # static data
  if (!is_time_series) {
    ind_rast <- rast(paste0(file.path(PREDICTORS_DATA_PATH,
				                    indicator_source, indicator),
			                      filename_pattern + ".tif"))
    if (!resampled) {
       crop_mask <- resample_crop_mask(crop_mask, ind_rast,
                                       resampled_crop_mask_file)
    }

    crop_mask <- valid_crop_mask_values(crop_mask)
    ind_rast <- filter_and_transform(ind_rast, indicator)
    # TODO: filter invalid values
    result <- crop_and_aggregate(ind_rast, sel_shapes, crop_mask, is_categorical)
    result <- aggregation_postprocess(result, crop, indicator, is_time_series)
    save_csv(result, crop, region, indicator)
  } else {
    region_results <- NULL
    for (yr in start_year:end_year) {
      # NOTE: doing a loop per month to handle daily data.
      # for fpar and ndvi, we could skip this loop and process one year at a time. 
      for (month_str in c("01", "02", "03", "04", "05", "06",
                          "07", "08", "09", "10", "11", "12")) {

        file_list <- list.files(path=file.path(PREDICTORS_DATA_PATH,
                                               indicator_source, indicator),
                                pattern=glob2rx(paste0(filename_pattern,
			  	  		                                       as.character(yr),
						                                           month_str,
						                                           "*")),
			        full.names=TRUE)
        if (length(file_list) == 0){
	        next
	      }

        rast_stack <- rast(file_list)
        if (!resampled) {
          crop_mask <- resample_crop_mask(crop_mask, rast_stack[[1]],
                                          resampled_crop_mask_file)
        }

        crop_mask <- valid_crop_mask_values(crop_mask)
        rast_stack <- filter_and_transform(rast_stack, indicator)

        # replicate crop mask to match indicator layers
        crop_masks <- rep(crop_mask, nlyr(rast_stack))
        # make sure pixels with NA values match
        crop_masks[is.na(rast_stack)] <- NA
        rast_stack[is.na(crop_masks)] <- NA

        result <- crop_and_aggregate(rast_stack, sel_shapes, crop_masks,
                                     is_categorical)
        result <- aggregation_postprocess(result, crop, indicator,
                                          is_time_series)
        if (is.null(region_results)) {
          region_results <- result
        } else {
          region_results <- rbind(region_results, result)
        }
	print(head(result))
      }
      # yearly results
      print(head(region_results))
    }
    save_csv(region_results, crop, region, indicator)
  }
}

get_shapes <- function(region) {
  sel_shapes <- NULL
  if ((region == "EU") |
      region %in% names(EU_countries)) {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_EU",
                                 "NUTS_RG_03M_2016_4326.shp"))
    if (region %in% names(EU_countries)) {
      sel_shapes <- sel_shapes[(sel_shapes$CNTR_CODE == region) &
                               (sel_shapes$LEVL_CODE == EU_countries[[region]])]
    }
    sel_shapes$adm_id <- sel_shapes$NUTS_ID
  } else if (region == "US") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_US",
                                 "cb_2018_us_county_500k.shp"))
    sel_shapes$adm_id <- paste0("US-", sel_shapes$STATEFP, "-",
                                sel_shapes$COUNTYFP)
  # } else if (region == "AR") {
  #   sel_shapes <- vect(AGML_ROOT, "shapefiles", "shapefiles_AR", "provincias.shp"))
  #   sel_shapes$adm_id <- paste0("AR", str_pad(sel_shapes$idDepartmento, 3, "0"))
  } else if (region == "AU") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_AU",
                                 "ABARES_regions_boundaries.shp"))
    sel_shapes$adm_id <- paste0("AU-", sel_shapes$AAGIS)
  # } else if (region == "BR") {
  } else if (region == "CN") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_CN",
                                 "chn_admbnda_adm1_ocha_2020.shp"))
    sel_shapes$adm_id <- paste0("CN-", substr(sel_shapes$ADM1_PCODE, 3, 5))
  # FEWSNET: Already has adm_id
  } else if ((region == "FEWSNET") |
             (region %in% FEWSNET_countries)) {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_FEWSNET",
                       "adm_shapefile_AgML_v0.1.shp"))
    if (region %in% FEWSNET_countries) {
      sel_shapes <- sel_shapes[substr(sel_shapes$adm_id, 1, 2) == region]
    }
  } else if (region == "IN") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_IN",
                       "India_585districts_adm2.shp"))
    sel_shapes <- project(sel_shapes, "EPSG:4326")
  # } else if (region == "ML") {
  # } else if (region == "MX") {
  #   sel_shapes <- project(sel_shapes, "EPSG:4326")
  }

  return(sel_shapes)
}


process_indicators <- function(crop, region, start_year, end_year, crop_mask_file) {
  # TODO: add indicator or predictor to list, also add source below
  indicators <- c("fpar",
                  "ndvi",
                  "ET0",
                  "AWC",
                  "drainage_class",
                  "bulk_density")
  # indicator source, also directory name
  indicator_sources <- c("JRC_FPAR500m",
                         "MOD09CMG",
                         "FAO_AQUASTAT",
                         "WISE_Soil",
                         "WISE_Soil",
                         "WISE_Soil")
  # this is the part before the date
  filename_prefixes <- c("fpar_",
                         "MOD09CMG_ndvi_",
                         "available_water_capacity",
                         "drainage_class",
                         "bulk_density")

  is_time_series <- c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE)
  is_categorical <- c(FALSE, FALSE, FALSE, FALSE, TRUE, FALSE)

  print(region)
  sel_shapes <- get_shapes(region)
  for(i in 1:length(indicators)) {
    ind <- indicators[[i]]
    filename_pattern <- filename_prefixes[[i]]
    ind_source <- indicator_sources[[i]]
    is_ts <- is_time_series[[i]]
    is_cat <- is_categorical[[i]]
    print(paste(ind, ind_source, filename_pattern, is_ts, is_cat))
    process_indicator(crop, region, sel_shapes, start_year, end_year,
                      crop_mask_file, ind_source, ind, filename_pattern,
                      is_ts, is_cat)
  }
}

crops <- c("maize", "wheat")
start_year <- 2000
end_year <- 2023
AGML_ROOT <- "/path/to/agml"
PREDICTORS_DATA_PATH <- file.path(AGML_ROOT, "predictors")

# countries_EU = { "AT" : 2, "BE" : 2, "BG" : 2, "CZ" : 3,
#                  "DE" : 3, "DK" : 3, "EE" : 3, "EL" : 3, "ES" : 3,
#                  "FI" : 3, "FR" : 3, "HR" : 2, "HU" : 3,
#                  "IE" : 2, "IT" : 3,
#                  "LT" : 3, "LV" : 3, "NL" : 2, "PL" : 2, "PT" : 2,
#                  "RO" : 3, "SE" : 3, "SK" : 3
#                }

EU_countries <- c(2, 2, 2, 3, 3, 3,
                  3, 3, 3, 3, 3, 2,
                  3, 2, 3, 3, 3, 2,
                  2, 2, 3, 3, 3)
names(EU_countries) <- c("AT", "BE", "BG", "CZ", "DE", "DK",
                         "EE", "EL", "ES", "FI", "FR", "HR",
                         "HU", "IE", "IT", "LT", "LV", "NL",
                         "PL", "PT", "RO", "SE", "SK")
FEWSNET_countries <- c("AO", "BF", "ET", "LS", "MG", "MW", "MZ",
                       "NE", "SN", "TD", "ZA", "ZM")
other_countries <- c("AR", "AU", "BR", "CN", "IN", "ML", "MX", "US")

for (crop in crops) {
  crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                              "crop_mask_generic_asap.tif")
  if (crop == "maize") {
    crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                                "crop_mask_maize_WC.tif")
  } else if (crop == "wheat") {
    crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                                "crop_mask_winter_spring_cereals_WC.tif")
  }

  process_indicators(crop, "EU", start_year, end_year, crop_mask_file)
  process_indicators(crop, "FEWSNET", start_year, end_year, crop_mask_file)
  for (cn in other_countries) {
    process_indicators(crop, cn, start_year, end_year, crop_mask_file)
  }
}
