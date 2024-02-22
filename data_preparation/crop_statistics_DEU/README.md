# German subnational crop yield statistics

## Short description

This dataset includes crop yield and area for Germany from 1979 to 2021. The data are spatially resolved to 397 districts with an average size of 900 km2 and include the crops spring barley, winter barley, grain maize, silage maize, oats, potatoes, winter rape, rye, sugarbeet, triticale and winter wheat. The crop-yield data cover on average about 9.5 million hectares per year and 80% of Germany’s total arable land. The final dataset contains 214,820 yield and area data points. These were obtained by collecting and digitizing crop data from multiple statistical sources and transforming the data to match the district boundaries in 2020.

## Link
https://doi.org/10.1038/s41597-024-02951-8

## Publisher
Open Agrar Repository

## Dataset owner
Thünen-Institut, Institut für Betriebswirtschaft (I'm not sure if this is correct, I have to check.)

## Data card author
Rahel Laudien

## Dataset overview

**Crops**:
- spring barley
- winter barley
- grain maize
- silage maize
- oats
- potatoes
- winter rape
- rye
- sugarbeet
- triticale 
- winter wheat

**Variables**:
name | description | unit
---|---|---
district_no | Official identifier of the district according to the German Federal Statistical Office. The first two digits encode the federal states of Germany, and the remaining three digits encode the district of the federal state  
district | Official district name 
nuts_id | District identifier according to the European NUTS 3 classification scheme
year | Year of harvest
var | Variable name (see below)
measure | Measure of the variable, being either ‘yield’ or ‘area’
value | Value for the respective district, year, variable and measure | Yield values in [t/ha], area values in [ha]
outlier | Flags missing values that were deleted during the outlier-detection procedure (1 if outlier, else 0)

The following variables are includes in the 'var' variable:
variable name | description 
---|---|
ArabLand | Area of arable land
district | Total district area
sb | Spring barley
wb | Winter barley
grain_maize | Grain maize
silage_maize | Silage maize
oats | Oats
potat_tot | Potatoes
wrape | Winter rape
rye | Rye
sugarbeet | Sugarbeet
triticale | Triticale
ww | Winter wheat

**Temporal coverage**: 1979 - 2021

**Temporal resolution**: 397 districts

**Date Published**: 17/11/2023

## Data access API
Not available.

## Provenance 
The dataset available via OpenAgrar is peer reviewed in 2023 and will be maintained.

# License 
CC-BY 4.0 
- The dataset can be used by the general public if the paper and its data are cited (Creative Commons License with attribution; CC-BY 4.0).

## How to cite
Duden, C., Nacke, C. & Offermann, F. Crop yields and area in Germany from 1979 to 2021 at a harmonized district-level. OpenAgrar https://doi.org/10.3220/DATA20231117103252-0 (2023).

## Additional information
*Optional*.

data cleaning 1 and 0 

## References
Duden, C., Nacke, C. & Offermann, F. Crop yields and area in Germany from 1979 to 2021 at a harmonized district-level. OpenAgrar https://doi.org/10.3220/DATA20231117103252-0 (2023).

Duden, C., Nacke, C. & Offermann, F. German yield and area data for 11 crops from 1979 to 2021 at a harmonized spatial resolution of 397 districts. Sci Data 11, 95 (2024). https://doi.org/10.1038/s41597-024-02951-8
