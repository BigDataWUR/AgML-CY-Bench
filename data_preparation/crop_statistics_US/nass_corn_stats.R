library(rnassqs)
#library(tidyUSDA)
library(tidyverse)

#api_key = readLines("nass_api.txt", warn = F)
nassqs_auth(key = api_key)

## Using package rnassqs

nass_crops <- "CORN"
years <- 2000:2022

nass_data_items <- c("CORN, GRAIN - ACRES HARVESTED",
                     "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
                     "CORN, GRAIN - PRODUCTION, MEASURED IN BU")

get_params <- function(data_item){
  params <- list(
  source_desc = "SURVEY",
  commodity_desc = nass_crops,
  domaincat_desc="NOT SPECIFIED",
  agg_level_desc = "COUNTY",
  year = years,
  short_desc = data_item
)
}

# Testing the function
# test <- get_params(nass_data_items[1])
# nassqs_record_count(test) # Counting records because NASS only allows 50k records at a time.
# test2 <- nassqs(test)

get_items <- map(nass_data_items, get_params)

raw_data <- map_dfr(get_items, nassqs)

select_stats <- raw_data |> 
  select('year', 'commodity_desc', 'county_code', 'county_name', 'state_name', 'statisticcat_desc', 'unit_desc', 'Value', 'CV (%)', 'short_desc') |> 
  filter(county_name != "OTHER (COMBINED) COUNTIES")

if (!file.exists(nass_stats_file)) {
  write.csv(select_stats, file=nass_stats_file, row.names=FALSE)
} else {
  write.table(select_stats, file=nass_stats_file, sep = ",", 
              append=TRUE, quote=FALSE,
              col.names=FALSE, row.names=FALSE)
}



