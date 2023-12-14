##################################################################################
# R script to download USDA NASS data                                            #
# See here for a guide:                                                          #
# https://cran.r-project.org/web/packages/tidyUSDA/vignettes/using_tidyusda.html #
##################################################################################

if(!"tidyUSDA" %in% installed.packages()){install.packages("tidyUSDA")}
library(tidyUSDA)

api_key <- "923E639C-69E9-320C-90AA-75721B9B96AE"

years <- seq(2000, 2022)
nass_crops <- "CORN"
csv_filenames <- c("CROP_AREA_COUNTY_US.csv",
                   "PRODUCTION_COUNTY_US.csv",
                   "YIELD_COUNTY_US.csv")
nass_categories <- c("AREA HARVESTED",
                     "PRODUCTION",
                     "YIELD")
nass_data_items <- c("CORN, GRAIN - ACRES HARVESTED",
                     "CORN, GRAIN - YIELD MEASURED IN BU/ACRE",
                     "CORN, GRAIN - PRODUCTION MEASURED IN BU")

for (i in seq_along(nass_categories)){
  category <- nass_categories[i]
  data_item <- nass_data_items[i]
  nass_stats_file <- csv_filenames[i]
  for (yr in years) {
    nass_stats <- tidyUSDA::getQuickstat(
      program ="SURVEY",
      sector="CROPS",
      group="FIELD CROPS",
      commodity=nass_crop,
      category=nass_category,
      data_item = nass_data_item,
      domain="TOTAL",
      geographic_level = "COUNTY",
      state=NULL,
      key = api_key,
      year = as.character(yr))

    nass_stats <- nass_stats[,c('county_code', 'county_name', 'state_name', 'commodity_desc', 'year', 'Value', 'CV (%)')]
    if (!file.exists(nass_stats_file)) {
      write.csv(nass_stats, file=nass_stats_file, row.names=FALSE)
    } else {
      write.table(nass_stats, file=nass_stats_file, sep = ",", 
                  append=TRUE, quote=FALSE,
                  col.names=FALSE, row.names=FALSE)
    }
  }
}