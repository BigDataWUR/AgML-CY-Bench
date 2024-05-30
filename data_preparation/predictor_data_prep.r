if(!"terra" %in% installed.packages()){install.packages("terra")}
if(!"reshape2" %in% installed.packages()){install.packages("reshape2")}
library(terra)
library(reshape2)
library(stringr)

#############
# Version 1 #
#############
# NOTE: This version makes directly extracts and aggregates
# from raster stack. Preferred based on suggestion from
# our R expert.

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

# TODO: add indicator or predictor to list, also add source below
indicators <- c("fpar",
                "ndvi",
                "ET0",
                "surface_moisture",
                "rootzone_moisture",
                "AWC",
                "drainage_class")
# indicator source, also directory name
indicator_sources <- c("JRC_FPAR500m", # "fpar"
                       "MOD09CMG", # "ndvi"
                       "FAO_AQUASTAT", # "ET0"
                       "GLDAS", # "surface_moisture"
                       "GLDAS", # "rootzone_moisture"
                       "WISE_Soil", # "AWC"
                       "WISE_Soil") # "drainage_class"
# this is the part before the date
filename_prefixes <- c("fpar_", # "fpar"
                       "MOD09CMG_ndvi_", # "ndvi"
                       "AGERA5_ET0_", # "ET0"
                       "GLDAS_surface_moisture_A", # "surface_moisture"
                       "GLDAS_rootzone_moisture_A", # "rootzone_moisture"
                       "available_water_capacity", # "AWC"
                       "drainage_class") # "drainage_class"

is_time_series <- c(TRUE, # "fpar"
                    TRUE, # "ndvi"
                    TRUE, # "ET0"
                    TRUE, # "surface_moisture"
                    TRUE, # "rootzone_moisture"
                    FALSE, # "AWC"
                    FALSE) # "drainage_class"

is_categorical <- c(FALSE, # "fpar"
                    FALSE, # "ndvi"
                    FALSE, # "surface_moisture"
                    FALSE, # "rootzone_moisture"
                    FALSE, # "ET0"
                    FALSE, # "AWC"
                    TRUE) # "drainage_class"

process_indicators <- function(crop, region, start_year, end_year, crop_mask_file) {
  print(region)

  ###############################
  # Select shapes or boundaries #
  ###############################
  sel_shapes <- NULL
  if ((region == "EU") |
      region %in% names(EU_countries)) {
    eu_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                "shapefiles_EU",
                                "NUTS_RG_03M_2016_4326.shp"))
    for (cn in names(EU_countries)) {
      cn_shapes <- eu_shapes[(eu_shapes$CNTR_CODE == cn) &
                             (eu_shapes$LEVL_CODE == EU_countries[[cn]])]
      if (is.null(sel_shapes)){
        sel_shapes <- cn_shapes
      }else {
        sel_shapes <- rbind(sel_shapes, cn_shapes)
      }
    }
    sel_shapes$adm_id <- sel_shapes$NUTS_ID
  } else if (region %in% names(EU_countries)) {
    eu_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                "shapefiles_EU",
                                "NUTS_RG_03M_2016_4326.shp"))
    sel_shapes <- eu_shapes[(eu_shapes$CNTR_CODE == region) &
                            (eu_shapes$LEVL_CODE == EU_countries[[region]])]
    sel_shapes$adm_id <- sel_shapes$NUTS_ID
  } else if (region == "US") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_US",
                                 "cb_2018_us_county_500k.shp"))
    sel_shapes$adm_id <- paste("US", sel_shapes$STATEFP, sel_shapes$COUNTYFP, sep="-")
  } else if (region == "AR") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_AR",
                       "arg_admbnda_adm2_unhcr2017.shp"))
    # sel_shapes$adm_id <- paste0("AR", str_pad(sel_shapes$idDepartmento, 3, "0"))
  } else if (region == "AU") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_AU",
                                 "ABARES_regions_boundaries.shp"))
    sel_shapes$adm_id <- paste("AU", sel_shapes$AAGIS, sep="-")
  } else if (region == "BR") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_BR",
                       "bra_admbnda_adm2_ibge_2020.shp"))
  } else if (region == "CN") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_CN",
                                 "chn_admbnda_adm1_ocha_2020.shp"))
    sel_shapes$adm_id <- paste("CN", substr(sel_shapes$ADM1_PCODE, 3, 5), sep="-")
  # FEWSNET: Already has adm_id
  } else if ((region == "FEWSNET") |
             (region %in% FEWSNET_countries)) {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_FEWSNET",
                       "adm_shapefile_AgML_v0.1.shp"))
  } else if (region %in% FEWSNET_countries) {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_FEWSNET",
                       "adm_shapefile_AgML_v0.1.shp"))
    sel_shapes <- sel_shapes[substr(sel_shapes$adm_id, 1, 2) == region]
  } else if (region == "IN") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_IN",
                       "India_585districts_adm2.shp"))
    sel_shapes <- project(sel_shapes, "EPSG:4326")
  # TODO: 
  # } else if (region == "ML") {
  } else if (region == "MX") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_MX",
                       "00ent_edited.shp"))
    sel_shapes <- project(sel_shapes, "EPSG:4326")
  }

  ###############################
  # Process each indicator      #
  ###############################
  for(i in 1:length(indicators)) {
    indicator <- indicators[[i]]
    filename_pattern <- filename_prefixes[[i]]
    indicator_source <- indicator_sources[[i]]
    is_ts <- is_time_series[[i]]
    is_cat <- is_categorical[[i]]
    print(paste(indicator, indicator_source, filename_pattern, is_ts, is_cat))

    resampled_crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                                          paste("crop_mask", crop, indicator, "res.tif", sep="_"))
    resampled <- FALSE
    if (file.exists(resampled_crop_mask_file)) {
      crop_mask <- rast(resampled_crop_mask_file)
      resampled <- TRUE
    } else {
      crop_mask <- rast(crop_mask_file)
    }
    print(resampled_crop_mask_file)
    print(resampled)

    ###############
    # static data #
    ###############
    if (!is_ts) {
      ind_rast <- rast(file.path(PREDICTORS_DATA_PATH,
                                 indicator_source, indicator,
                                 filename_pattern + ".tif"))
      # resample crop mask to indicator extent and resolution
      if (!resampled) {
        crop_mask <- resample(crop_mask, ind_rast, method="bilinear")
        writeRaster(x=crop_mask, filename=resampled_crop_mask_file)
      }

      # filter invalid values
      crop_mask[crop_mask > 100] <- NA
      crop_mask[crop_mask < 0] <- NA

      # TODO: filter invalid values

      # Extract indicator values for boundaries
      ind_vals <- extract(ind_rast, sel_shapes, xy=TRUE)
      ind_vals$adm_id <- sel_shapes$adm_id[ind_vals$ID]
      ind_vals$ID <- NULL

      # extract crop mask values
      crop_mask_vals <- extract(crop_mask, sel_shapes, xy=TRUE)
      crop_mask_vals$adm_id <- sel_shapes$adm_id[crop_mask_vals$ID]
      crop_mask_vals$ID <- NULL
      names(crop_mask_vals) <- c("crop_area_fraction", "x", "y", "adm_id")

      ind_df <- as.data.frame(cbind(ind_vals, crop_mask_vals$crop_area_fraction))
      ind_df <- na.exclude(ind_df)
      names(ind_df)[names(ind_df) == "crop_mask_vals$crop_area_fraction"] <- "crop_area_fraction"
      colnames(ind_df) <- c("indicator", "x",	"y", "adm_id",	"crop_area_fraction")

      # for now, "drainage_class" is the only categorical data
      if (is_cat){
        # aggregate area fraction by category
        ind_df <- aggregate(list(sum_weight=ind_df$crop_area_fraction),
                            by=list(adm_id=ind_df$adm_id,
                                    indicator=ind_df$indicator),
                            FUN=sum)
        # order by sum_weight
        head(ind_df)
        ind_df <- ind_df[order(ind_df$adm_id, -ind_df$sum_weight),]
        # keep only the first item per adm_id
        ind_df <- ind_df[ !duplicated(ind_df$adm_id), ]
        ind_df$sum_weight <- NULL
      } else {
        # aggregate
        ind_df$weighted_ind <- ind_df$indicator * ind_df$crop_area_fraction
        ind_df <- aggregate(list(sum_ind=ind_df$weighted_ind,
                                 sum_weight=ind_df$crop_area_fraction),
                            by=list(adm_id=ind_df$adm_id), FUN=sum)
        ind_df$indicator <- ind_df$sum_ind/ind_df$sum_weight
        ind_df <- ind_df[, c("adm_id", "indicator")]
      }

      ind_df$crop_name <- crop
      ind_df <- ind_df[, c("crop_name", "adm_id", "indicator")]
      colnames(ind_df) <- c("crop_name", "adm_id", indicator)

      if (!dir.exists(file.path(AGML_ROOT, "R-output",
                                crop, region, indicator))) {
        dir.create(file.path(AGML_ROOT, "R-output",
                             crop, region, indicator),
                  recursive=TRUE)
      }
      print(head(ind_df))
      write.csv(ind_df,
                file.path(AGML_ROOT, "R-output",
                          crop, region, indicator,
                          paste0(indicator, "_", region, ".csv")),
                row.names=FALSE)
    } else {
      ####################
      # Time series data #
      ####################
      region_results <- NULL
      for (yr in start_year:end_year) {
        # NOTE: doing a loop per month to handle daily data.
        # for fpar and ndvi, we could skip this loop and process one year at a time. 
        file_list <- list.files(path=file.path(PREDICTORS_DATA_PATH,
                                               indicator_source, indicator),
                                pattern=glob2rx(paste0(filename_pattern, as.character(yr), "*")),
                                full.names=TRUE)
        num_year_files <- length(file_list)
        if (num_year_files == 0) {
          next
        }

        max_stack_size <- 50
        for (i in seq(1, num_year_files, by=max_stack_size)) {
          if ((i+max_stack_size-1) < num_year_files) {
            file_seq <- seq(i, i+max_stack_size-1)
          } else {
            file_seq <- seq(i, num_year_files)
          }

          sel_files <- file_list[file_seq]
          actual_stack_size <- length(sel_files)
          rast_stack <- rast(sel_files)
          # resample crop mask to indicator extent and resolution
          if (!resampled) {
            if (actual_stack_size == 1) {
              crop_mask <- resample(crop_mask, rast_stack, method="bilinear")
            } else {
              crop_mask <- resample(crop_mask, rast_stack[[1]], method="bilinear")
            }
            writeRaster(x=crop_mask, filename=resampled_crop_mask_file)
          }

          # filter invalid values
          # Setting NA values to 0 is fine for weights.
          crop_mask[crop_mask > 100] <- 0
          crop_mask[crop_mask < 0] <- 0

          # TODO: filter invalid values
          # filter and transform indicators
          if (indicator == "fpar") {
            # check data_preparation/global_fpar_500m
            # FPAR values: 0 to 100 (%). Valid range is 0-100.
            # Flags: 251 is "other land", 254 is "water", 255 is "not processed".
            rast_stack[rast_stack > 100] <- NA
            rast_stack[rast_stack < 0] <- NA
          } else if (indicator == "ndvi") {
            # check data_preparation/global_MOD09CMG
            # Value range: 50 to 250 for NDVI.
            rast_stack[rast_stack < 50] <- NA
            rast_stack[rast_stack > 250] <- NA
            #  To scale, apply the formula: (x - 50)/200
            rast_stack <- (rast_stack - 50)/200
          } else if ((indicator == "ET0") | (indicator_source == "GLDAS")) {
            # check data_preparation/global_ETo_FAO
            rast_stack[rast_stack == -9999] <- NA
          } else if (grepl("Temperature", indicator, fixed=TRUE)) {
            # temp_C = temp_K - 273.15
            rast_stack <- rast_stack - 273.15
          }

          # Crop rasters to shapes
          rast_stack = crop(rast_stack, sel_shapes)
          crop_mask = crop(crop_mask, sel_shapes)

          # replicate crop mask to match number of indicator layers
          # we set weights for NA values to 0
          crop_masks = rep(crop_mask, nlyr(rast_stack))
          crop_masks[is.na(rast_stack)] = 0

          rast_stack = crop_masks * rast_stack

          result = extract(rast_stack, sel_shapes, fun=sum, na.rm=TRUE, ID=FALSE)
          result$adm_id = sel_shapes$adm_id

          result_w = extract(crop_masks, sel_shapes, fun=sum, na.rm=TRUE, ID=FALSE)
          result_w$adm_id = sel_shapes$adm_id

          # Divide the result by weights. Added a tiny number to avoid division by Zero.
          result[-ncol(result)] = result[-ncol(result)]/(result_w[-ncol(result_w)] + 1e-6)
          result <- melt(result)
          
          colnames(result) <- c("adm_id", "ind_file", indicator)
          result$date <- str_sub(result$ind_file, -8,-1)
          result$crop_name <- crop
          result <- result[, c("crop_name", "adm_id",
                               "date", indicator)]
          print(result)
          if (is.null(region_results)) {
            region_results <- result
          } else {
            region_results <- rbind(region_results, result)
          }
        }
        print(head(region_results))
      }
      if (!dir.exists(file.path(AGML_ROOT, "R-output",
                                crop, region, indicator))) {
        dir.create(file.path(AGML_ROOT, "R-output",
                             crop, region, indicator),
                  recursive=TRUE)
      }
      write.csv(region_results,
                file.path(AGML_ROOT, "R-output",
                          crop, region, indicator,
                          paste0(indicator, "_", region, ".csv")),
                row.names=FALSE)
    }
  }
}


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
