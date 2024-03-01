if (!require("pacman")) install.packages("pacman")
pacman::p_load(crossmap, ggplot2, sf, terra, tidyverse)

## The .nc4 files were downloaded from: https://zenodo.org/records/5062513

data_path <- "data_preparation/crop_calendar_US/" #Path to the data directory

crops <- c("mai", "wwh") # Add more crops here
trts <- c("ir", "rf")

ggcmi_data <- xmap_vec(list(crops, trts), ~terra::rast(paste0(data_path, .x, "_", .y, "_ggcmi_crop_calendar_phase3_v1.01.nc4")))

## Get US shapefile to aggregate the global ggcmi data to US county levels
US_counties <- tigris::counties(cb = TRUE, resolution = "500k") |> 
  st_as_sf()

# Crop the shapefile to the continental US
continental_US <- US_counties |> 
  mutate(STATEFP = as.numeric(STATEFP)) |> 
  filter(STATEFP <= 56 & STATEFP != 02 & STATEFP != 15) |> 
  select(STATEFP, COUNTYFP, geometry)

# Plot the map to check the shapefile
ggplot(continental_US) + geom_sf()

# Testing with one NetCDF data at a time before setting up the functions for the list
corn_ir <- rast("data_preparation/crop_calendar_US/mai_ir_ggcmi_crop_calendar_phase3_v1.01.nc4")

# Reproject to match crs
con_US <- st_transform(con_US, st_crs(corn_ir)) 

# Crop the ggcmi raster with the US shapefile
corn_ir_crop <- crop(corn_ir, con_US)

# Extract data from the raster
corn_ir_values <- terra::extract(corn_ir_crop, con_US)

# Get mean values for each county
mean_days <- corn_ir_values |> 
  group_by(ID) |>
  summarise(mean_pday = mean(pday, na.rm = TRUE),
            mean_mday = mean(mday, na.rm = TRUE),
            mean_gslen = mean(gslen, na.rm = TRUE))

# Combine to the continental US shapefile
US_sf <- con_US |>
  (\(x) mutate(x, ID = seq_len(nrow(x))))() |> 
  left_join(mean_days, by = "ID")


## Next step: Create a function that operates on the list "ggcmi_data" do the same for all the rasters 
## Join with State and County names by FIPS codes (can use yield data for this)