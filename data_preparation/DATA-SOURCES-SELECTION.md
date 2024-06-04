# Data sources selection

## Weather variables and moisture indicators

| Source | Considered variables | Temporal frequency | Spatial resolution | Caveat| Operational | Open and Free? |
|------|------|------|------|------|------|:---:|
| [ERA5]( https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) | 2m temperature, Precipitation flux, Solar radiation flux, 10m wind speed, 2m relative humidity | Hourly | 0.25° | Not bias-corrected | Y | Y |
| [ERA5-land]( https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) | 2m temperature, Precipitation flux, Solar radiation flux, 10m wind speed, 2m relative humidity | Hourly | 0.1° | Not bias-corrected | Y | Y |
| [W5E5](https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/114/)| 2m temperature, Precipitation flux, Solar radiation flux, 10m wind speed, 2m relative humidity | Daily | 0.5° | low spatial resolution, does not offer real-time temporal coverage | N | Y |
| [AgERA5]( https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators?tab=overview) | 2m temperature, Precipitation flux, Solar radiation flux, 10m wind speed, 2m relative humidity | Daily | 0.1° | Not bias-corrected | Y | Y |
| [FAO, AgERA5 based]( https://data.apps.fao.org/static/data/index.html?prefix=static%2Fdata%2Fc3s%2FAGERA5_ET0) | Reference evapotranspiration| Daily | 0.1° | Not bias-corrected | Y | Y |
| [CPC](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html)| Precipitation and temperature| Daily | 0.5° | limited available variables, low spatial resolution | Y | Y |
| [GLEAM](https://www.gleam.eu/)| Evapotranspiration and soil moisture| Daily | 0.25° | does not offer real-time temporal coverage | N | Y |
| [MSWEP](https://www.gloh2o.org/mswep/)| Precipitation| 3-hourly | 0.1° | Single variable product | Y | Y |
| [GLADS](https://ldas.gsfc.nasa.gov/gldas/model-output) | Root zone soil moisture| Daily | 0.25° | latency of 2 to 6 month to obtain the main product | Y | Y |
| [ESA CCI SM]( https://cds.climate.copernicus.eu/portfolio/dataset/satellite-soil-moisture)| surface soil moisture| Daily | 0.25° | Single variable product | Y | Y |

We selected data from the [AgERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/sis-agrometeorological-indicators?tab=overview) project, which provides daily weather variables at a 0.1° spatial resolution, directly relevant to crop yields, including temperature, precipitation, and solar radiation flux. Furthermore, we selected reference evapotranspiration from the FAO dataset, which relies on the FAO Penman - Monteith method and takes input variables from the AgERA5 dataset. AgERA5 offers agrometeorological indicators from 1979 to the present, derived from ERA5 reanalysis and tailored for agricultural studies. Its key advantage is the high-quality, operational input for numerous variables essential for crop yield forecasting, and it is freely available and well-documented on the Copernicus Climate Data Store [CDS](https://cds.climate.copernicus.eu/#!/home). 

Other datasets have limitations such as fewer variables, lower spatial resolution, or shorter temporal coverage. For instance, [MSWEP](https://www.gloh2o.org/mswep/) is a high-quality dataset but only includes precipitation. [CPC](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html) relies on station observations but has coarser spatial resolution. [W5E5](https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/114/) is bias-adjusted based on ERA5 but lacks real-time temporal coverage. Therefore, we chose AgERA5 due to its balance of quality and operational suitability.

The only relevant weather variable not covered by AgERA5 is soil moisture. For this, we use the [GLADS](https://ldas.gsfc.nasa.gov/gldas/model-output) dataset, which is available from 2003 to present, and can be freely downloaded at https://disc.gsfc.nasa.gov/datasets?keywords=GLDAS. This dataset represents gridded and global soil moisture data developed by integrating satellite- and ground-based observational data products, using advanced land surface modeling and data assimilation techniques. Another option is to consider the [ESA CCI SM]( https://cds.climate.copernicus.eu/portfolio/dataset/satellite-soil-moisture) dataset for soil moisture although the latter only provides surface estimates. Finally, the [GLEAM](https://www.gleam.eu/) dataset does provide root zone soil moisture, but the data is currently available only up to December 2022.

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

| Source | Area/crops | Phases reported | Caveat | Open and Free? |
|------|------|------|------|:---:|
| [FAO-GIEWS](https://www.fao.org/giews/countrybrief/country.jsp?lang=en&code=DZA) | Country-level, main crops (wheat and maize covered) | Sowing-growing-harvest | It covers food insecure countries (e.g. no US or EU) | Y |
| [FAO-Crop Calendars](https://cropcalendar.apps.fao.org/#/home) | Many crops, sub-national, sometimes agro-eco zones | Sowing, harvesting | no detailed description, not all countries, sometimes seems not correct | Y |
| [GEOGLAM](https://cropmonitor.org/index.php/eodatatools/baseline-data/) |  | Planting through Early Vegetative*<br>Vegetative through Reproductive<br>Ripening through Harvest<br>Harvest (End of Season)<br>Out of Season | Not global cover (on public website) | Y |
| [WorldCereal](https://github.com/ucg-uv/research_products/tree/main/cropcalendars) - Selected for benchmark because rainfed and irrigated are aggregated making it consistent with the yield data | Merging of different sources, raster maps for wheat and maize (res 0.5°). Maize calendars available only for the first maize season. | SOS, EOS [temporal resolution: day] | Aggregation to admin level needed | Y |
| GGCMI: [Data](https://zenodo.org/records/5062513), [Paper](https://www.nature.com/articles/s43016-021-00400-y), [Supplementary material](https://static-content.springer.com/esm/art%3A10.1038%2Fs43016-021-00400-y/MediaObjects/43016_2021_400_MOESM1_ESM.pdf), [ISIMIP link](https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/115/)  | Gridded 0.5. Differentiate between irrigated and rainfed, winter and spring wheat. Only one season for maize. | planting and maturity dates, growing season length [temporal resolution: day] | Aggregation to admin level needed | Y |
| MIRCA: [Data](https://zenodo.org/records/7422506), [Documentation](https://www.uni-frankfurt.de/45218023/MIRCA) | 5 arc-minutes x 5 arc-minutes = ~ 9.2 km x 9.2 km at the equator<br>all major food crops including regionally important ones. Differentiate between irrigated and rainfed. Crops can have multiple growing seasons | the month in which the growing period starts and the month in which the growing period ends. [temporal resolution: month] | Based on the period 1998-2002<br>For 402 spatial units (e.g. California is one unit) | Y |
| SAGE: [Data](https://sage.nelson.wisc.edu/data-and-models/datasets/crop-calendar-dataset/netcdf-5-min/), [Paper](https://onlinelibrary.wiley.com/doi/10.1111/j.1466-8238.2010.00551.x) | Gridded global crop calendars plus other crop climate parameters, for 19 crops. Resolution for both: 5 minute & 0.5 Degree. | Planting & harvesting dates, growing season length, Summaries of other crop-climate characteristics | Global cover except USA & Canada | Y |
| [USDA](https://ipad.fas.usda.gov/ogamaps/cropcalendar.aspx) | Qualitative global crop calendars. Few windows per country, mostly determined by the number of rainfall seasons. | Planting and harvesting. | Global cover but the planting and harvesting dates have to be manually approximated. Useful for general information purposes only. | Y |

### Contact
Pratishtha Poudel, Michele Meroni
