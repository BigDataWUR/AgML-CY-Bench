if (!require("pacman")) install.packages("pacman")
pacman::p_load(ncdf4, tidyverse, tigris, sf)

# Uncomment the following line to download the shapefiles.
#download.file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip", destfile = "data_preparation/shapefiles_US/cb_2018_us_county_500k.zip", method = "wget", extra = "-r -p --random-wait")

# Uncomment the following line to unzip the shapefiles
# unzip("data_preparation/shapefiles_US/cb_2018_us_county_500k.zip", exdir = "data_preparation/shapefiles_US")

# Load the shapefile
counties <- st_read("data_preparation/shapefiles_US/cb_2018_us_county_500k.shp")

# Plot the shapefile
# ggplot(counties) + geom_sf()

#################### Using tigris package ########################

# We only need state names because counties doesn't have them.
states_sf <- tigris::states(cb=T) |> 
  select(STATEFP,NAME) |> 
  rename(State = NAME)

# Counties
county_sf <- tigris::counties(cb=T) |> 
  select(STATEFP, COUNTYFP, NAME) |> 
  left_join({states_sf |> st_drop_geometry()}) |> 
  rename(County = "NAME")

# Plot to test
# ggplot(county_sf) +
#   geom_sf()
# 
# ggplot(states_sf) +
#   geom_sf()

# Write to a shapefile
#st_write(county_sf, "datasets/US_counties.shp", append=FALSE)

