library(sf)
library(tidyverse)
library(tigris)

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
st_write(county_sf, "datasets/US_counties.shp", append=FALSE)

