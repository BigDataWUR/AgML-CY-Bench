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

The only relevant weather variable not covered by AgERA5 is soil moisture. For this, we use the [GLADS](https://ldas.gsfc.nasa.gov/gldas/model-output) dataset, which is available from 2003 to present, and can be freely downloaded [here](https://disc.gsfc.nasa.gov/datasets?keywords=GLDAS). This dataset represents gridded and global soil moisture data developed by integrating satellite- and ground-based observational data products, using advanced land surface modeling and data assimilation techniques. Another option is to consider the [ESA CCI SM]( https://cds.climate.copernicus.eu/portfolio/dataset/satellite-soil-moisture) dataset for soil moisture although the latter only provides surface estimates. Finally, the [GLEAM](https://www.gleam.eu/) dataset does provide root zone soil moisture, but the data is currently available only up to December 2022.

### Contact
Raed Hamed

## Remote sensing indicators
Various remote sensing indicators of biomass status and health exists at the global scale. These include various vegetation indexes (VIs; e.g. NDVI, EVI, etc.) and various biophysical indicators (e.g. FPAR, LAI). One of the mandatory requirement for sub-national yield forecasting is the availability of a long term time series of such indicators (i.e. > 20 years) and a short revisit time of the satellite to guarantee the existence of cloud free imagery. These requirements pragmatically restrict the possible choices to the MODIS coarse spatial resolution mission (and VIIRS for the continuity).
Although MODIS data can be directly downloaded from NASA LPDACC, the direct use of row VIs and biophysical variables is hampered by the presence of low quality of observations (mainly due to cloud presence) that need to be screened out, gap filled and corrected. This further processing involving the use of quality flags provided with the data and of a temporal smoothing procedure is typically time consuming and not trivial when applied to near real-time data, this latter of utmost importance for operational yield forecasting.
In view of an operational deployment of the sub-national yield forecasting, we selected two analysis ready operational products representing crop biomass and health: FPAR and NDVI. FPAR (Fraction of Adsorbed Photosynthetically Active Radiation) dekadal (10-day) and 500 m spatial resolution gap-filled and smoothed MODIS and VIIRS FPAR data is sourced from the European Commission's Joint Research Centre (JRC), https://agricultural-production-hotspots.ec.europa.eu/data/indicators_fpar/. The quality of the near real-time products of such processing line we assessed in Meroni et al. (2019, https://doi.org/10.1016/j.rse.2018.11.041) for the results on NDVI and in Seguini et al. (under preparation) for FPAR.


NDVI..


### Contact
FPAR: Michele Meroni
NDVI: Ritvik Sahajpal


## Soil
We selected data from the [WISE project](global_soil_WISE/README.md) for soil properties. Another choice is [SoilGrids](https://www.isric.org/explore/soilgrids).

WISE soil data is a better choice than SoilGrids for the following reasons:
* agronomic interpretation is easier for WISE compared to SoilGrids.
* SoilGrids did not provide soil rootable depth and water holding capacity (this may have changed recently)
* WISE is based on soil maps whose properties are estimated by expert knowledge collected over the years. For example, in the Netherlands, we see that the performance of [BOFEK](https://doi.org/10.1016/j.geoderma.2022.116123) (similar to WISE) is still considerably better than SoilGrids.

### Contact
Allard de Wit

## Crop masks
With the aim of making available global maize and winter-spring cereals map we sourced the only crop-type specific maps currently available: maize and winter-spring cereals high spatial resolution (10 m) maps from ESA WorldCereal (https://esa-worldcereal.org/en). Area Fraction Images fully spatially compatible with the JRC FPAR 500 m were derived from the WorldCereal maps.

### Contact
Francesco Collivignarelli, Michele Meroni, Ritvik Sahajpal

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

## Crop yield DE
We selected data from [Duden et al. (2023)](crop_statistics_DE/README.md) for crop (Grain Maize and Wheat) yield. Another choice is [EU-EUROSTAT](crop_statistics_EU). crop_statistics_DE is a better choice because of the below mentioned reasons:
- Duden et al. (2023) data source for both crops have a longer and harmonized time series (spanning from 1979-2021) available as opposed to dataset from crop_statistics_EU (which is available from 1999-2020).
- For Grain Maize, Duden et al. (2023) has data at NUTS level 3 (administrative units = 138), compared to the one from EU-EUROSTAT which is at NUTS level 1.
- Data for DE from EU-EUROSTAT also does not meet some consistency checks done by Ronchetti et al. (2024), e.g. yield = production/area.

### Contact
Rahel Laudien

### References
Duden, C., Nacke, C. & Offermann, F. Crop yields and area in Germany from 1979 to 2021 at a harmonized district-level. OpenAgrar https://doi.org/10.3220/DATA20231117103252-0 (2023).

Ronchetti, G., Nisini Scacchiafichi, L., Seguini, L., Cerrani, I., and van der Velde, M.: Harmonized European Union subnational crop statistics can reveal climate impacts and crop cultivation shifts, Earth Syst. Sci. Data, 16, 1623–1649, https://doi.org/10.5194/essd-16-1623-2024, 2024.

## Crop yield ML
TODO: Justification for ML data from ICRISAT vs FEWSNET.

### Contact
Janet Mutuku, Celeste Tchampi, PCS Traore 
