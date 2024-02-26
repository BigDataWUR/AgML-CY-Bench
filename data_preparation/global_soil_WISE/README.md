# WISE derived soil properties on a 30 by 30 arc-seconds global grid

## Short description
The World Inventory of Soil Emission Potentials (WISE) project has developed homogenized sets of soil property estimates for the world. The WISE derived soil properties on a 30 by 30 arc-seconds global grid (WISE30sec) is a harmonized dataset of derived soil properties for the world and includes a soil-geographical and a soil attribute component. The GIS dataset was created using the soil map unit delineations of the broad scale Harmonised World Soil Database, version 1.21, with minor corrections, overlaid by a climate zones map (Köppen-Geiger) as co-variate, and soil property estimates derived from analyses of the ISRIC-WISE soil profile database for the respective mapped ‘soil/climate’ combinations.

Soil property estimates are presented for fixed depth intervals of 20 cm up to a depth of 100 cm, respectively of 50 cm between 100 cm to 200 cm (or less when appropriate) for so-called ‘synthetic’ profiles’ (as defined by their ‘soil/climate’ class). The respective soil property estimates were derived from statistical analyses of data for some 21,000 soil profiles managed in a working copy of the ISRIC-WISE database; this was done using an elaborate scheme of taxonomy-based transfer rules complemented with expert-rules that consider the ‘in-pedon’ consistency of the predictions. The type of rules used was flagged to provide an indication of the possible confidence (i.e. lineage) in the derived data.

## Link
**Overview**: https://data.isric.org/geonetwork/srv/eng/catalog.search#/metadata/dc7b283a-8f19-45e1-aaed-e9bd515119bc

**Data files**: https://files.isric.org/public/wise/wise_30sec_v1.zip

**Soil Organic Carbon GIS files**: https://files.isric.org/public/wise/wise30sec_soc_gis_files.zip

[Batjes (2016)](https://doi.org/10.1016/j.geoderma.2016.01.034) describes the procedure followed to prepare the data.

## Publisher
[ISRIC](https://www.isric.org/)

## Dataset owner
[ISRIC](https://www.isric.org/)

## Data card author
Dilli R. Paudel

## Dataset overview
The dataset considers 20 soil properties that are commonly required for global agro-ecological zoning, land evaluation, crop growth simulation, modelling of soil gaseous emissions, and analyses of global environmental change. It presents ‘best’ estimates for:
| Property | Meaning | Unit | Notes |
|:---------|:--------|:-----|:------|
| ALSAT | Aluminium saturation (as % of ECEC) | - | Calculated from other measured soil properties |
| BSAT  | Base saturation (as % of CECsoil)   | - | Calculated from other measured soil properties |
| BULK  | Bulk density | $kg$ $dm^{-3}$ |
| CECC  | Cation exchange capacity of clay size fraction ($CEC_{clay}$) | $cmol_c$ $kg^{-1}$ | $CEC_{clay}$ is calculated from $CEC_{soil}$ by correcting for contribution of organic matter |
| CECS  | Cation exchange capacity ($CEC_{soil}$) | $cmol_c$ $kg^{-1}$ | |
| CFRAG | Coarse fragments (> 2 mm; volume %) | - | |
| CLPC  | Clay (mass %) | - | |
| CNrt  | $C/N$ ratio | - | Calculated from other measured soil properties |
| DRAIN | Soil drainage class (observed, according to FAO (2006) | - | |
| ECEC  | Effective cation exchange capacity | $cmol_c$ $kg^{-1}$ | ECEC is defined as exchangeable ($Ca^{++}+Mg^{++}+K^++Na^+$) plus exchangeable ($H^++Al^{+++}$) (van Reeuwijk 2002). |
| ELCO  | Electrical conductivity | $dS$ $m^{-1}$ | |
| ESP  | Exchangeable sodium percentage | - | Calculated from other measured soil properties. |
| GYPS | Gypsum content | $g$ $kg^{-1}$ | |
| ORGC | Organic carbon | $g$ $kg^{-1}$ | |
| PHAQ | pH measured in water | - | |
| SDTO | Sand (mass %) | - | |
| STPC | Silt (mass %) | - | |
| TAWC | Available water capacity (from -33 to -1500 kPa) | $cm$ $m^{-1}$ | Calculated from other measured soil properties. Soil water potential limits for AWC conform to USDA standards (Soil Survey Staff 1983); these values have not yet been corrected for the presence of fragments > 2 mm |
| TCEQ | Total carbonate equivalent | $g$ $kg^{-1}$ | |
| TEB  | Total exchangeable bases | $cmol_c$ $kg^{-1}$ | |
| TOTN | Total nitrogen | $g$ $kg^{-1}$ | |


**Temporal Coverage**: NA

**Temporal Resolution**: NA

**Spatial resolution**: 30 arc-second

**Date published**: 2016-05-01

**Data modality**: Soil properties are tabular (provided in Microsoft Access Database file). The soil unit identifier are provided as a GIS file (e.g. .tif).

## Provenance
NOTE: The data is no longer updated or maintained.

**Paper**: [Batjes (2016)](https://doi.org/10.1016/j.geoderma.2016.01.034)

**Report**: [Batjes, 2015](https://library.wur.nl/WebQuery/wurpubs/fulltext/400244)

## License
Data is shared with [CC By 3.0 Attribution license](https://creativecommons.org/licenses/by/3.0/). Redistribution is permitted with appropriate attribution.

## How to cite
Batjes NH 2016. Harmonised soil property values for broad-scale modelling (WISE30sec) with estimates of global soil carbon stocks. Geoderma 2016(269), 61-68 ( http://dx.doi.org/10.1016/j.geoderma.2016.01.034 )

## References
Batjes NH 2015. World soil property estimates for broad-scale modelling (WISE30sec) Report 2015/01, ISRIC — World Soil Information, Wageningen.

Soil Survey Staff 1983. Soil Survey Manual (rev. ed.). United States Agriculture Handbook 18, USDA, Washington.

van Reeuwijk LP 1998. Guidelines for quality management in soil and plant laboratories, FAO, Rome, 143 p.
http://www.fao.org/docrep/W7295E/W7295E00.htm