# Data sources selection

## Weather variables and moisture indicators
We selected data from the [AgERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators?tab=overview) project, which provides daily weather variables at a 0.1° spatial resolution, directly relevant to crop yields, including temperature, precipitation, evapotranspiration, and solar radiation flux. AgERA5 offers agrometeorological indicators from 1979 to the present, derived from ERA5 reanalysis and tailored for agricultural studies. Its key advantage is the high-quality, operational input for numerous variables essential for crop yield forecasting, and it is freely available and well-documented on the Copernicus Climate Data Store [CDS](https://cds.climate.copernicus.eu/#!/home). 

Other datasets have limitations such as fewer variables, lower spatial resolution, or shorter temporal coverage. For instance, [MSWEP](https://www.gloh2o.org/mswep/) is a high-quality dataset but only includes precipitation. [CPC](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html) relies on station observations but has coarser spatial resolution. [W5E5](https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/114/) is bias-adjusted based on ERA5 but lacks real-time temporal coverage. Therefore, we chose AgERA5 due to its balance of quality and operational suitability.

The only relevant weather variable not covered by AgERA5 is soil moisture. For this, we use the ESA Climate Change Initiative for Soil Moisture dataset, available from 1978 to the present, also on the CDS. This dataset is the current state-of-the-art for satellite-based soil moisture records. Other options, like [GLADS](https://ldas.gsfc.nasa.gov/gldas/model-output) and [GLEAM](https://www.gleam.eu/), do not offer real-time temporal coverage.

### Contact
Raed Hamed

## Remote sensing indicators
Choices, selection and justification

### Contact
Ritvik Sahajpal, Michele Meroni

## Soil
We selected data from the [WISE project][global_soil_WISE/README.md] for soil properties. Another choice is [SoilGrids](https://www.isric.org/explore/soilgrids).

WISE soil data is a better choice than SoilGrids for the following reasons:
* agronomic interpretation is easier for WISE compared to SoilGrids.
* SoilGrids did not provide soil rootable depth and water holding capacity (this may have changed recently)
* WISE is based on soil maps whose properties are estimated by expert knowledge collected over the years. For example, in the Netherlands, we see that the performance of [BOFEK](https://doi.org/10.1016/j.geoderma.2022.116123) (similar to WISE) is still considerably better than SoilGrids.

### Contact
Allard de Wit

## Crop masks
Choices, selection and justification

### Contact
Ritvik Sahajpal, Michele Meroni

## Crop calendars
Choices, selection and justification

### Contact
Pratishtha Poudel, Michele Meroni
