library(tidyverse)
library(sf)

loc = '/datasets/work/af-sandpit/work/Jonathan/AgML/'
#AgML-crop-yield-forecasting/data_preparation/'

# loading data from gov website ; download on 05/03/2024
df = read.csv(paste0(loc,"fdp-beta-regional-historical.csv")) #file origin https://www.agriculture.gov.au/sites/default/files/documents/fdp-beta-regional-historical.csv
head(df)

# filtering wheat production and area swon
wheat = df %>% filter(Variable == 'Wheat produced (t)' | Variable == 'Wheat area sown (ha)')
wheat = wheat %>% select(-RSE)

# pivoting table
temp_df = wheat %>% pivot_wider(names_from = Variable , values_from = Value)
names(temp_df) = c('year','NAME','production','planted_area')
head(temp_df)

# grabbing the AAGIS ID from the shapefile
shape = read_sf(paste0(loc,'ABARES_regions_boundaries.shp'))
shape['adm_id'] = paste0('AU-',shape$AAGIS) # creates adm_id as AgML request
write_sf(shape, paste0(loc,'ABARES_regions_boundaries.shp')) # saves with adm_id

shape_df =  shape %>% select(adm_id,NAME) %>% st_drop_geometry()
wheat_df = merge(temp_df, shape_df, by='NAME', all.x = T)

names(wheat_df) = c('NAME','harvest_year','production','planted_area','adm_id') # naming convention for AgML

#dropping name column
wheat_df$NAME = NULL

# add columns to conform to AgML standards
wheat_df$crop_name = 'WHEAT'
wheat_df$country_code = 'AU'
wheat_df$season_name = NA
wheat_df$planting_year = NA
wheat_df$planting_date = NA
wheat_df$harvest_date = NA
wheat_df$yield = temp_df$production / temp_df$planted_area
wheat_df$harvest_area= NA

head(wheat_df)

# save .csv as AgML
write.csv(wheat_df,paste0(loc,'wheat_Australia.csv'))

# no need ----
#ggplot(wheat_df)+
#    geom_line(aes(x=harvest_year,y=yield))+
#    facet_wrap(~adm_id)+
#    theme_bw()

# Creating joined shape with statistics
#wheat_df = read.csv(paste0(loc,'crop_statistics_AU/','wheat_Australia.csv'))
#head(wheat_df)

#shape = read_sf(paste0(loc,'shapefiles_AU/','ABARES_regions_boundaries.shp'))
#names(shape) = c('ste','adm_id','oid_','name','geometry')
#head(shape)

#new_shape = merge(wheat_df, shape, by='adm_id', all.x = T)
#head(new_shape)

#write_sf(new_shape, paste0(loc,'shapefiles_AU/','ABARES_regions_with_stats.shp'))
