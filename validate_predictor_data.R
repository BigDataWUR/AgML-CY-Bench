library(sf)
library(terra)
library(tidyverse)

## The tif files were obtained from Google Drive.
## Change the path to file location.
awc <- rast("../AgML_data/awc.tif")
maize_mask <- rast("../AgML_data/crop_mask_maize_WC.tif")
fpar <- rast("../AgML_data/fpar_20190621.tif")
ndvi <- rast("../AgML_data/MOD09CMG_ndvi_20190101.tif")

## EU Shapefile obtained from Google Drive
eu_shapes <- vect("../AgML_data/NUTS_RG_03M_2016_4326/NUTS_RG_03M_2016_4326.shp")
#eu_shapes <- vect("Shapefiles/shapefiles_EU/NUTS_RG_01M_2016_4326_LEVL_2.shp")
nl_shapes <- eu_shapes[eu_shapes$CNTR_CODE == "NL"]

maize_mask_nl <- crop(maize_mask, nl_shapes)

awc_nl <- crop(awc, nl_shapes)
maize_mask_awc <- resample(maize_mask_nl, awc_nl, method="bilinear")
awc_nl[maize_mask_awc < 0 | maize_mask_awc > 100] <- NA
awc_nl <- mask(awc_nl, nl_shapes)

fpar_nl <- crop(fpar, nl_shapes)
maize_mask_fpar <- resample(maize_mask_nl, fpar_nl, method="bilinear")
fpar_nl[maize_mask_fpar < 0 | maize_mask_fpar > 100] <- NA
fpar_nl <- mask(fpar_nl, nl_shapes)

ndvi_nl <- crop(ndvi, nl_shapes)
maize_mask_ndvi <- resample(maize_mask_nl, ndvi_nl, method="bilinear")
ndvi_nl[maize_mask_ndvi < 0 | maize_mask_ndvi > 100] <- NA
ndvi_nl <- mask(ndvi_nl, nl_shapes)

## Data from cybench
soil_data <- read_csv("../AgML_data/soil_maize_NL.csv")
fpar_data <- read_csv("../AgML_data/cybench-data/maize/NL/fpar_maize_NL.csv") |> 
  filter(date == 20190621)
ndvi_data <- read_csv("../AgML_data/cybench-data/maize/NL/ndvi_maize_NL.csv") |> 
  filter(date == 20190101)

nl_shapes <- merge(nl_shapes, soil_data, by.x="NUTS_ID", by.y="adm_id")
nl_shapes <- merge(nl_shapes, fpar_data, by.x="NUTS_ID", by.y="adm_id")
nl_shapes <- merge(nl_shapes, ndvi_data, by.x="NUTS_ID", by.y="adm_id")

par(mfrow = c(1, 2)) # reset plotting window

# plot_awc
min_awc <- min(minmax(awc_nl)[1], min(nl_shapes$awc_nl))
max_awc <- max(minmax(awc_nl)[2], max(nl_shapes$awc_nl))
plot(nl_shapes, "awc", col=hcl.colors(100, palette="RdYlBu"), type="continuous", range=c(min_awc, max_awc))
plot(awc_nl, col=hcl.colors(100, palette="RdYlBu"), type="continuous", range=c(min_awc, max_awc))
plot(nl_shapes, add=TRUE)

# plot_fpar
min_fpar <- min(minmax(fpar_nl)[1], min(nl_shapes$fpar_nl))
max_fpar <- max(minmax(fpar_nl)[2], max(nl_shapes$fpar_nl))
plot(nl_shapes, "fpar", col=hcl.colors(100, palette="RdYlBu"), type="continuous", range=c(min_fpar, max_fpar))
plot(fpar_nl, col=hcl.colors(100, palette="RdYlBu"), type="continuous", range=c(min_fpar, max_fpar))
plot(nl_shapes, add=TRUE)

# plot_ndvi
min_ndvi <- min(minmax(ndvi_nl)[1], min(nl_shapes$ndvi_nl))
max_ndvi <- max(minmax(ndvi_nl)[2], max(nl_shapes$ndvi_nl))
plot(nl_shapes, "ndvi", col=hcl.colors(100, palette="RdYlBu"), type="continuous", range=c(min_ndvi, max_ndvi))
plot(ndvi_nl, col=hcl.colors(100, palette="RdYlBu"), type="continuous", range=c(min_ndvi, max_ndvi))
plot(nl_shapes, add=TRUE)
