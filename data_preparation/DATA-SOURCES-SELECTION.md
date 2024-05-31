# Data sources selection

## Weather variables and moisture indicators
Choices, selection and justification

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
