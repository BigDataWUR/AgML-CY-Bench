if (!require("pacman")) install.packages("pacman")
pacman::p_load(crossmap, ggplot2, sf, terra, tidyverse, tigris)

## The .nc4 files were downloaded from: https://zenodo.org/records/5062513

data_path <- "data_preparation/crop_calendar_US/" #Path to the data directory

crops <- c("mai", "wwh") # Add more crops here
trts <- c("ir", "rf") # Add more treatments here

ggcmi_data <- xmap_vec(list(crops, trts), ~terra::rast(paste0(data_path, .x, "_", .y, "_ggcmi_crop_calendar_phase3_v1.01.nc4")))
names(ggcmi_data) <- xmap_vec(list(crops, trts), ~paste0(.x, "_", .y))

## Get US shapefile to aggregate the global ggcmi data to US county levels
US_counties <- tigris::counties(cb = TRUE, resolution = "500k") |> 
  st_as_sf()

# Crop the shapefile to continental US
continental_US <- US_counties |> 
  mutate(STATEFP = as.numeric(STATEFP), State = str_to_title(STATE_NAME), County = str_to_title(NAME)) |> 
  filter(STATEFP <= 56 & STATEFP != 02 & STATEFP != 15) |> 
  select(STATEFP, COUNTYFP, State, County, geometry)

# Plot to check the shapefile
ggplot(continental_US) + geom_sf()

# Reproject to match crs
continental_US <- st_transform(continental_US, crs(ggcmi_data[[1]]))

# Crop to US shapefile
ggcmi_cropped <- ggcmi_data |> 
  map(~terra::crop(.x, continental_US))

# Extract data
ggcmi_values <- ggcmi_cropped |> 
  map(~terra::extract(.x, continental_US))

# Aggregate by county
# Using mean right now but median might make more sense
mean_days <- function(all_values){ 
  mean_values <- all_values |> 
    group_by(ID) |>
    summarise(mean_pday = mean(planting_day, na.rm = TRUE),
              mean_mday = mean(maturity_day, na.rm = TRUE),
              mean_gslen = mean(growing_season_length, na.rm = TRUE))
  
  join_with_admin_ID <- continental_US |>
    (\(x) mutate(x, ID = seq_len(nrow(x))))() |> 
    left_join(mean_values, by = "ID")
  
  return(join_with_admin_ID)
}

crop_cal <- ggcmi_values |> map(~mean_days(.x))

crop_cal_df <- crop_cal |> 
  map2(names(crop_cal), ~mutate(.x, crop_trt = .y)) |> 
  bind_rows() |> 
  separate(crop_trt, c("Crop", "Trt"), sep = "_")

write_csv(crop_cal_df, "data_preparation/crop_calendar_US/crop_calendar_US.csv")
