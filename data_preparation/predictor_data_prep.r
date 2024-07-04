if(!"terra" %in% installed.packages()){install.packages("terra")}
if(!"reshape2" %in% installed.packages()){install.packages("reshape2")}
if(!"abjutils" %in% installed.packages()){install.packages("abjutils")}
library(terra)
library(reshape2)
library(stringr)
library(abjutils)


crops <- c("maize", "wheat")
start_year <- 2000
end_year <- 2023
AGML_ROOT <- "/path/to/agml"
PREDICTORS_DATA_PATH <- file.path(AGML_ROOT, "predictors")
OUTPUT_PATH <- file.path(AGML_ROOT, "R-output")

# Country codes and NUTS Level for yield statistics
EU_countries <- list("AT" = 2, "BE" = 2, "BG" = 2, "CZ" = 3,
                     "DE" = 3, "DK" = 3, "EE" = 3, "EL" = 3, "ES" = 3,
                     "FI" = 3, "FR" = 3, "HR" = 2, "HU" = 3,
                     "IE" = 2, "IT" = 3, "LT" = 3, "LV" = 3,
                     "NL" = 2, "PL" = 2, "PT" = 2, "RO" = 3,
                     "SE" = 3, "SK" = 3)

FEWSNET_countries <- c("AO", "BF", "ET", "LS", "MG", "MW", "MZ",
                       "NE", "SN", "TD", "ZA", "ZM")

indicators <- list(
  "fpar" = list("source" = "JRC_FPAR500m",
                "filename_pattern" = "fpar_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "ndvi" = list("source" = "MOD09CMG",
                "filename_pattern" = "MOD09CMG_ndvi_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "et0" = list("source" = "FAO_AQUASTAT",
                "filename_pattern" = "AGERA5_ET0_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "ssm" = list("source" = "GLDAS",
                "filename_pattern" = "GLDAS_surface_moisture_A",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "rsm" = list("source" = "GLDAS",
                "filename_pattern" = "GLDAS_rootzone_moisture_A",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "prec" = list("source" = "AgERA5",
                "filename_pattern" = "AgERA5_Precipitation_Flux_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "tmax" = list("source" = "AgERA5",
                "filename_pattern" = "AgERA5_Maximum_Temperature_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "tmin" = list("source" = "AgERA5",
                "filename_pattern" = "AgERA5_Minimum_Temperature_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "tavg" = list("source" = "AgERA5",
                "filename_pattern" = "AgERA5_Mean_Temperature_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "rad" = list("source" = "AgERA5",
                "filename_pattern" = "AgERA5_Solar_Radiation_Flux_",
                "is_time_series" = TRUE,
                "is_categorical" = FALSE),
  "awc" = list("source" = "WISE_Soil",
                "filename_pattern" = "awc",
                "is_time_series" = FALSE,
                "is_categorical" = FALSE),
  "bulk_density" = list("source" = "WISE_Soil",
                        "filename_pattern" = "bulk_density",
                        "is_time_series" = FALSE,
                        "is_categorical" = FALSE),
  "drainage_class" = list("source" = "WISE_Soil",
                          "filename_pattern" = "drainage_class",
                          "is_time_series" = FALSE,
                          "is_categorical" = TRUE),
  "st_clay" = list("source" = "WISE_Soil",
                    "filename_pattern" = "st_clay",
                    "is_time_series" = FALSE,
                    "is_categorical" = FALSE),
  "st_sand" = list("source" = "WISE_Soil",
                    "filename_pattern" = "st_sand",
                    "is_time_series" = FALSE,
                    "is_categorical" = FALSE),
  "st_silt" = list("source" = "WISE_Soil",
                    "filename_pattern" = "st_silt",
                    "is_time_series" = FALSE,
                    "is_categorical" = FALSE),
  # NOTE: "sos" and "eos" are crop specific.
  # Therefore, filename_pattern will need crop name as well.
  # Check code where these are processed.             
  "sos" = list("source" = "ESA_WC_Crop_Calendars",
                "filename_pattern" = "sos",
                "is_time_series" = FALSE,
                "is_categorical" = FALSE),
  "eos" = list("source" = "ESA_WC_Crop_Calendars",
                "filename_pattern" = "eos",
                "is_time_series" = FALSE,
                "is_categorical" = FALSE))


process_indicators <- function(crop, region, start_year, end_year, crop_mask_file) {
  # print(region)

  ###############################
  # Select shapes or boundaries #
  ###############################
  sel_shapes <- NULL
  if (region == "EU") {
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
    sel_shapes$adm_id <- sel_shapes$ADM2_PCODE
  } else if (region == "AU") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_AU",
                                 "ABARES_regions_boundaries.shp"))
    sel_shapes$adm_id <- paste("AU", sel_shapes$AAGIS, sep="-")
  } else if (region == "BR") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_BR",
                       "bra_admbnda_adm2_ibge_2020.shp"))
    sel_shapes$adm_id <- sel_shapes$ADM2_PCODE
  } else if (region == "CN") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_CN",
                                 "chn_admbnda_adm1_ocha_2020.shp"))
    sel_shapes$adm_id <- sel_shapes$ADM1_PCODE
  # FEWSNET countries: Already have adm_id
  } else if (region == "FEWSNET") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_FEWSNET",
                       "adm_shapefile_AgML_v0.1.shp"))
  } else if (region %in% FEWSNET_countries) {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_FEWSNET",
                       "adm_shapefile_AgML_v0.1.shp"))
    sel_shapes <- sel_shapes[substr(sel_shapes$adm_id, 1, 2) == region]
  # IN: Already has adm_id
  } else if (region == "IN") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_IN",
                       "India_585districts_adm2.shp"))
    sel_shapes <- project(sel_shapes, "EPSG:4326")
  # ML: Already has adm_id
  } else if (region == "ML") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                                 "shapefiles_ML",
                                 "cmdt_boundary.shp"))
  # MX: Already has adm_id
  } else if (region == "MX") {
    sel_shapes <- vect(file.path(AGML_ROOT, "shapefiles",
                       "shapefiles_MX",
                       "00ent_edited.shp"))
    sel_shapes <- project(sel_shapes, "EPSG:4326")
  }

  ###############################
  # Process each indicator      #
  ###############################
  for (indicator in names(indicators)) {
    filename_pattern <- indicators[[indicator]][["filename_pattern"]]
    indicator_source <- indicators[[indicator]][["source"]]
    is_time_series <- indicators[[indicator]][["is_time_series"]]
    is_categorical <- indicators[[indicator]][["is_categorical"]]
    # print(paste(indicator, indicator_source, filename_pattern
    #             is_time_series, is_categorical))

    resampled_crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                                          paste("crop_mask", crop, indicator, "res.tif", sep="_"))

    crop_mask_cropped <- FALSE
    resampled <- FALSE
    # NOTE: AgERA5 data is cropped to specific regions.
    # So we don't save resampled crop mask.
    if ((indicator_source != "AgERA5") &
        file.exists(resampled_crop_mask_file)) {
      crop_mask <- rast(resampled_crop_mask_file)
      resampled <- TRUE
    } else {
      crop_mask <- rast(crop_mask_file)
    }
    # print(resampled_crop_mask_file)
    # print(resampled)

    result <- NULL
    ###############
    # static data #
    ###############
    if (!is_time_series) {
      # NOTE for crop calendars, sos and eos are crop specific.
      if (indicator_source == "ESA_WC_Crop_Calendars") {
        ind_rast <- rast(file.path(PREDICTORS_DATA_PATH,
                                   indicator_source,
                                   paste0(crop, "_", filename_pattern, ".tif")))
      } else {
        ind_rast <- rast(file.path(PREDICTORS_DATA_PATH,
                                   indicator_source,
                                   paste0(filename_pattern, ".tif")))
      }
      
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

      result <- as.data.frame(cbind(ind_vals, crop_mask_vals$crop_area_fraction))
      result <- na.exclude(result)
      names(result)[names(result) == "crop_mask_vals$crop_area_fraction"] <- "crop_area_fraction"
      colnames(result) <- c("indicator", "x",	"y", "adm_id",	"crop_area_fraction")

      # for now, "drainage_class" is the only categorical data
      if (is_categorical){
        # aggregate area fraction by category
        result <- aggregate(list(sum_weight=result$crop_area_fraction),
                            by=list(adm_id=result$adm_id,
                                    indicator=result$indicator),
                            FUN=sum)
        # order by sum_weight
        # print(head(result))
        result <- result[order(result$adm_id, -result$sum_weight),]
        # keep only the first item per adm_id
        result <- result[ !duplicated(result$adm_id), ]
        result$sum_weight <- NULL
      } else {
        # aggregate
        result$weighted_ind <- result$indicator * result$crop_area_fraction
        result <- aggregate(list(sum_ind=result$weighted_ind,
                                 sum_weight=result$crop_area_fraction),
                            by=list(adm_id=result$adm_id), FUN=sum)
        result$indicator <- result$sum_ind/result$sum_weight
        result <- result[, c("adm_id", "indicator")]
      }

      result$crop_name <- crop
      result <- result[, c("crop_name", "adm_id", "indicator")]
      colnames(result) <- c("crop_name", "adm_id", indicator)
    } else {
      ####################
      # Time series data #
      ####################
      for (yr in start_year:end_year) {
        # NOTE: AgERA5 data is split by region.
        # So include region-specific directory in path.
        if (indicator_source == "AgERA5") {
          indicator_path = file.path(PREDICTORS_DATA_PATH, indicator_source,
                                     paste(region, indicator_source, sep="_"), indicator)
        } else {
          indicator_path = file.path(PREDICTORS_DATA_PATH, indicator_source, indicator)
        }

        file_list <- list.files(path=indicator_path,
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

          # Handle NetCDF files.
          # AgERA5 files (not ET0 files) are NetCDF or .nc files.
          is_netcdf_file <- endsWith(sel_files[[1]], ".nc")
          if (is_netcdf_file) {
            rast_stack <- rast(sel_files, 1)
            names(rast_stack) <- file_sans_ext(basename(sources(rast_stack)))
          } else {
            rast_stack <- rast(sel_files)
          }

          # resample crop mask to indicator extent and resolution
          if (!resampled) {
            if (actual_stack_size == 1) {
              crop_mask <- resample(crop_mask, rast_stack, method="bilinear")
            } else {
              crop_mask <- resample(crop_mask, rast_stack[[1]], method="bilinear")
            }
            resampled <- TRUE
            if (indicator_source != "AgERA5") {
              writeRaster(x=crop_mask, filename=resampled_crop_mask_file, overwrite=TRUE)
            }
          }

          # NOTE: crop and filter once per time series indicator.
          # Don't need to do this per year or per indicator stack.
          if (!crop_mask_cropped) {
            crop_mask = crop(crop_mask, sel_shapes)
            # filter invalid values
            # Setting NA values to 0 is fine for weights.
            crop_mask[crop_mask > 100] <- 0
            crop_mask[crop_mask < 0] <- 0
            crop_mask_cropped <- TRUE
          }

          # Crop rasters to shapes
          rast_stack = crop(rast_stack, sel_shapes)

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
            # check data_preparation/global_ET0_FAO
            rast_stack[rast_stack == -9999] <- NA
          } else if (grepl("Temperature", indicator, fixed=TRUE)) {
            # temp_C = temp_K - 273.15
            rast_stack <- rast_stack - 273.15
          }

          # replicate crop mask to match number of indicator layers
          # we set weights for NA values to 0
          crop_masks = rep(crop_mask, nlyr(rast_stack))
          crop_masks[is.na(rast_stack)] = 0

          # NOTE the order of multiplication matters
          rast_stack = rast_stack * crop_masks

          aggregates = extract(rast_stack, sel_shapes, fun=sum, na.rm=TRUE, ID=FALSE)
          aggregates$adm_id = sel_shapes$adm_id

          sum_wts = extract(crop_masks, sel_shapes, fun=sum, na.rm=TRUE, ID=FALSE)
          sum_wts$adm_id = sel_shapes$adm_id

          # Divide the aggregates by weights.
          # NOTE: Added a tiny number to avoid division by Zero.
          aggregates[-ncol(aggregates)] = aggregates[-ncol(aggregates)]/(sum_wts[-ncol(sum_wts)] + 1e-6)
          aggregates <- melt(aggregates)
          
          colnames(aggregates) <- c("adm_id", "ind_file", indicator)
          aggregates$date <- str_sub(aggregates$ind_file, -8,-1)
          aggregates$crop_name <- crop
          aggregates <- aggregates[, c("crop_name", "adm_id",
                               "date", indicator)]
          # print(head(aggregates))
          if (is.null(result)) {
            result <- aggregates
          } else {
            result <- rbind(result, aggregates)
          }
        }
        # print(head(result))
      }
    }

    if (!dir.exists(file.path(OUTPUT_PATH, crop, region))) {
        dir.create(file.path(OUTPUT_PATH, crop, region),
                   recursive=TRUE)
    }
    # print(head(result))
    write.csv(result,
              file.path(OUTPUT_PATH, crop, region,
                        paste0(indicator, "_", region, ".csv")),
              row.names=FALSE)
  }
}


for (crop in crops) {
  crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                              "crop_mask_generic_asap.tif")
  if (crop == "maize") {
    crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                                "crop_mask_maize_WC.tif")
    other_countries <- c("AR", "BR", "CN", "IN", "ML", "MX", "US") 
  } else if (crop == "wheat") {
    crop_mask_file <- file.path(AGML_ROOT, "crop_masks",
                                "crop_mask_winter_spring_cereals_WC.tif")
    other_countries <- c("AR", "AU", "BR", "CN", "IN", "US")
  }

  process_indicators(crop, "EU", start_year, end_year, crop_mask_file)
  process_indicators(crop, "FEWSNET", start_year, end_year, crop_mask_file)
  for (cn in other_countries) {
    process_indicators(crop, cn, start_year, end_year, crop_mask_file)
  }
}
