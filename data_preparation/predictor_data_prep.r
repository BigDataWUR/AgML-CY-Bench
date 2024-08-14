if(!"terra" %in% installed.packages()){install.packages("terra")}
if(!"reshape2" %in% installed.packages()){install.packages("reshape2")}
if(!"abjutils" %in% installed.packages()){install.packages("abjutils")}
if(!"pbapply" %in% installed.packages()){install.packages("pbapply", repos='http://cran.us.r-project.org')}
if(!"optparse" %in% installed.packages()){install.packages("optparse", repos='http://cran.us.r-project.org')}
if(!"data.table" %in% installed.packages()){install.packages("data.table", repos='http://cran.us.r-project.org')}
library(terra)
library(reshape2)
library(stringr)
library(abjutils)
library(pbapply)
library(optparse)
library(data.table)


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

get_shapes <- function(region) {
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
      if (is.null(sel_shapes)) {
        sel_shapes <- cn_shapes
      } else {
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
  }
  sel_shapes <- project(sel_shapes, "EPSG:4326")
  return(sel_shapes)
}

process_ts_raster <- function(indicator_file, indicator,
                              crop_mask_file, region_boundaries) {
  ##################################
  # Process one time series raster #
  ##################################
  crop_mask = rast(crop_mask_file)

  # Handle NetCDF files.
  # AgERA5 files (not ET0 files) are NetCDF or .nc files.
  is_netcdf_file <- endsWith(indicator_file, ".nc")
  if (is_netcdf_file) {
    ind_rast <- rast(indicator_file, subds=1, win=ext(region_boundaries))
  } else {
    ind_rast <- rast(indicator_file, win=ext(region_boundaries))
  }

  # filter and transform indicators
  if (indicator == "fpar") {
    # check data_preparation/global_fpar_500m
    # FPAR values: 0 to 100 (%). Valid range is 0-100.
    # Flags: 251 is "other land", 254 is "water", 255 is "not processed".
    ind_rast[ind_rast > 100] <- NA
    ind_rast[ind_rast < 0] <- NA
  } else if (indicator == "ndvi") {
    # check data_preparation/global_MOD09CMG
    # Value range: 50 to 250 for NDVI.
    ind_rast[ind_rast < 50] <- NA
    ind_rast[ind_rast > 250] <- NA
    #  To scale, apply the formula: (x - 50)/200
    ind_rast <- (ind_rast - 50)/200
  } else if (indicator %in% c("ET0", "surface_moisture", "rootzone_moisture")) {
    # check data_preparation/global_ET0_FAO
    ind_rast[ind_rast == -9999] <- NA
  } else if (grepl("Temperature", indicator, fixed=TRUE)) {
    # temp_C = temp_K - 273.15
    ind_rast <- ind_rast - 273.15
  }

  # replicate crop mask to match number of indicator layers
  # we set weights for NA values to 0
  crop_mask[is.na(ind_rast)] <- 0

  # NOTE the order of multiplication matters
  ind_rast <- ind_rast * crop_mask

  aggregates <- extract(ind_rast, region_boundaries, fun=sum, na.rm=TRUE, ID=FALSE)
  aggregates$adm_id <- region_boundaries$adm_id

  sum_wts <- extract(crop_mask, region_boundaries, fun=sum, na.rm=TRUE, ID=FALSE)
  sum_wts$adm_id <- region_boundaries$adm_id

  # Divide aggregates by weights. Added a tiny number to avoid division by Zero.
  aggregates[-ncol(aggregates)] <- aggregates[-ncol(aggregates)]/(sum_wts[-ncol(sum_wts)] + 1e-6)
  aggregates <- melt(aggregates)
          
  colnames(aggregates) <- c("adm_id", "ind_file", indicator)
  aggregates$date <- str_sub(aggregates$ind_file, -8,-1)
  aggregates$crop_name <- crop
  aggregates <- aggregates[, c("crop_name", "adm_id", "date", indicator)]
  return(aggregates)
}

process_ts_year <- function(year, file_path, filename_pattern,
                            indicator, crop_mask_file, region_boundaries) {
  ############################################
  # Process time series rasters for one year #
  ############################################
  file_list <- list.files(path=file_path,
                          pattern=glob2rx(paste0(filename_pattern, as.character(year), "*")),
                          full.names=TRUE)
  if (length(file_list) == 0) {
    return (NULL)
  }

  dfs <- pblapply(file_list, process_ts_raster, indicator=indicator,
                  crop_mask_file=crop_mask_file,
                  region_boundaries=region_boundaries)

  result <- rbindlist(dfs)
  return(result)
}

process_indicators <- function(crop, region,
                               sel_indicators,
                               start_year, end_year,
                               crop_mask_file, num_cpus) {
  for (indicator in sel_indicators) {
    filename_pattern <- indicators[[indicator]][["filename_pattern"]]
    indicator_source <- indicators[[indicator]][["source"]]
    is_time_series <- indicators[[indicator]][["is_time_series"]]
    is_categorical <- indicators[[indicator]][["is_categorical"]]

    # print(paste(indicator, indicator_source, filename_pattern, is_time_series, is_categorical))
    region_boundaries <- get_shapes(region)
    crop_mask <- rast(crop_mask_file)
    crop_mask[crop_mask > 100] <- NA
    crop_mask[crop_mask < 0] <- NA
    result <- NULL

    if (is_time_series) {
      ####################
      # Time series data #
      ####################

      # NOTE: AgERA5 data is split by region.
      # So include region-specific directory in path.
      if (indicator_source == "AgERA5") {
        indicator_path = file.path(PREDICTORS_DATA_PATH, indicator_source,
                                   paste(region, indicator_source, sep="_"), indicator)
      } else {
        indicator_path = file.path(PREDICTORS_DATA_PATH, indicator_source, indicator)
      }

      found_raster_file <- FALSE
      while(!found_raster_file) {
        file_list <- list.files(path=indicator_path,
                                pattern=glob2rx(paste0(filename_pattern, as.character(start_year), "*")),
                                full.names=TRUE)
        if (length(file_list) == 0) {
          warning(paste("Files not found for", start_year))
          start_year <- start_year + 1 
        } else {
          found_raster_file <- TRUE
        }
      }

      # Resample crop mask to indicator rastor resolution.
      ind_rast <- rast(file_list[[1]])
      crop_mask <- resample(crop_mask, ind_rast, method="bilinear")
      crop_mask <- crop(crop_mask, region_boundaries)
      # NOTE: We are overwriting in every iteration (per indicator).
      # This is fine because resampling is needed per indicator.
      resampled_filename <- file.path(AGML_ROOT, "crop_masks",
                                      "resampled_crop_mask.tif")
      writeRaster(crop_mask, resampled_filename, overwrite=TRUE)

      rm(list=c("ind_rast", "crop_mask"))
      dfs <- pblapply(start_year:end_year, process_ts_year,
                      file_path=indicator_path,
                      filename_pattern=filename_pattern,
                      indicator=indicator, 
                      crop_mask_file=resampled_filename,
                      region_boundaries=region_boundaries, 
                      cl=num_cpus)
 
      result <- rbindlist(dfs)
      rm(dfs)
      gc()
    } else {
      ###############
      # static data #
      ###############

      # NOTE for crop calendars, sos and eos are crop specific.
      if (indicator_source == "ESA_WC_Crop_Calendars") {
        ind_filename <- file.path(PREDICTORS_DATA_PATH,
                                  indicator_source,
                                  paste0(crop, "_", filename_pattern, ".tif"))
      } else {
        ind_filename <- file.path(PREDICTORS_DATA_PATH,
                                  indicator_source,
                                  paste0(filename_pattern, ".tif"))
      }
 
      # Resample crop mask to the resolution of indicator.
      ind_rast <- rast(ind_filename)
      crop_mask <- resample(crop_mask, ind_rast, method="bilinear")

      # Crop rasters
      crop_mask <- crop(crop_mask, region_boundaries)
      ind_rast <- crop(ind_rast, region_boundaries)

      # TODO: filter invalid indicator values

      # Extract indicator values for boundaries
      ind_vals <- extract(ind_rast, region_boundaries, xy=TRUE)
      ind_vals$adm_id <- region_boundaries$adm_id[ind_vals$ID]
      ind_vals$ID <- NULL

      # extract crop mask values
      crop_mask_vals <- extract(crop_mask, region_boundaries, xy=TRUE)
      crop_mask_vals$adm_id <- region_boundaries$adm_id[crop_mask_vals$ID]
      crop_mask_vals$ID <- NULL
      names(crop_mask_vals) <- c("crop_area_fraction", "x", "y", "adm_id")

      result <- as.data.frame(cbind(ind_vals, crop_area_fraction=crop_mask_vals$crop_area_fraction))
      result <- na.exclude(result)
      colnames(result) <- c("indicator", "x",	"y", "adm_id",	"crop_area_fraction")

      if (is_categorical) {
        # aggregate crop area fraction by category
        result <- aggregate(list(sum_weight=result$crop_area_fraction),
                            by=list(adm_id=result$adm_id,
                                    indicator=result$indicator),
                            FUN=sum)
        # order by sum_weight, largest first
        result <- result[order(result$adm_id, -result$sum_weight),]        
        # keep only the indicator category with the largest sum_weight
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

      # clean up memory
      rm(list=c("ind_rast", "ind_vals", "crop_mask_vals"))
      gc()
    }

    if (!dir.exists(file.path(OUTPUT_PATH, crop, region, indicator))) {
        dir.create(file.path(OUTPUT_PATH, crop, region, indicator),
                   recursive=TRUE)
    }
    # print(head(result))
    write.csv(result,
              file.path(OUTPUT_PATH, crop, region, indicator,
                        paste0(indicator, "_", region, ".csv")),
              row.names=FALSE) 
  }
}

option_list <- list(
  make_option(c("-c", "--crop"), type="character", default=NULL, 
              help="crop name", metavar="character"),
  make_option(c("-r", "--region"), type="character", default=NULL, 
              help="country or region", metavar="character"),
  make_option(c("-i", "--indicator"), type="character", default=NULL, 
              help="indicator", metavar="character"),
  make_option(c("-p", "--cpus"), type="integer", default=1, 
              help="number of cpus", metavar="number")
)
 
opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

num_cpus <- 16
if (!is.null(opt$cpus)) {
  num_cpus <- opt$cpus
}

if (!is.null(opt$crop)) {
  crops <- c(opt$crop)
}

sel_indicators <- names(indicators)
if (!is.null(opt$indicator)) {
  stopifnot(opt$indicator %in% names(indicators))
  sel_indicators <- c(opt$indicator)
}

regions <- c("EU", "FEWSNET")
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

  regions <- c(regions, other_countries)
  if (!is.null(opt$region)) {
    regions <- c(opt$region);
  }

  for (reg in regions) {
    process_indicators(crop, reg, sel_indicators,
                       start_year, end_year,
                       crop_mask_file, num_cpus)
  }
}
