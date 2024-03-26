if (!require("pacman")) install.packages("pacman")
pacman::p_load(crossmap, ggplot2, sf, terra, tidyverse, tigris)

## The .nc4 files were downloaded from: https://zenodo.org/records/5062513

data_path <- "data_preparation/global_crop_calendar/US_crop_calendar/" #Path to the data directory

crops <- c("mai", "wwh") # Add more crops here
trts <- c("ir", "rf")

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

# Aggregate by county using median

median_days <- function(all_values){ 
  median_values <- all_values |> 
    group_by(ID) |>
    summarise(med_pday = median(planting_day, na.rm = TRUE),
              med_mday = median(maturity_day, na.rm = TRUE),
              med_gslen = median(growing_season_length, na.rm = TRUE))
  
  join_with_admin_ID <- continental_US |>
    (\(x) mutate(x, ID = seq_len(nrow(x))))() |> 
    left_join(median_values, by = "ID")
  
  return(join_with_admin_ID)
}

crop_cal <- ggcmi_values |> map(~median_days(.x))

crop_cal_df <- crop_cal |> 
  map2(names(crop_cal), ~mutate(.x, crop_trt = .y)) |> 
  bind_rows() |> 
  separate(crop_trt, c("Crop", "Trt"), sep = "_")

crop_cal_ir <- crop_cal_df |> filter(Trt == "ir")
crop_cal_rf <- crop_cal_df |> filter(Trt == "rf")

# Both corn and winter wheat rainfed
write_csv(crop_cal_rf, "data_preparation/global_crop_calendar/US_crop_calendar/US_crop_calendar_rf.csv")

#################### Exploring irrigated vs. rainfed ########################

# ir_rf <- crop_cal_df |> 
#   pivot_wider(names_from = "Trt", values_from = c("med_pday", "med_mday", "med_gslen")) |> 
#   mutate(pday_diff = med_pday_ir - med_pday_rf,
#          mday_diff = med_mday_ir - med_mday_rf,
#          gslen_diff = med_gslen_ir - med_gslen_rf)
# 
# n_10 <- ir_rf |> filter(pday_diff > 10 | pday_diff < -10)
# n_20 <- ir_rf |> filter(pday_diff > 20 | pday_diff < -20)
# 
# nrow(n_10)/nrow(ir_rf) * 100 # 21% of the counties have a difference of more than 10 days in planting day between irrigated and rainfed
# nrow(n_20)/nrow(ir_rf) * 100 # 11% of the counties have a difference of more than 20 days in planting day between irrigated and rainfed
