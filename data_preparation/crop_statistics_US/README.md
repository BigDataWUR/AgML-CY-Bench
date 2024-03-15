# Crop statistics of the United States
The National Agricultural Statistics Service (NASS) of the United States Department of Agriculture (USDA) conducts hundreds of surveys every year and prepares reports including those related to crop statistics.

## DATASET LINK
https://quickstats.nass.usda.gov/

## PUBLISHER
The National Agricultural Statistics Service (NASS) of the United States Department of Agriculture (USDA)

## DATASET OWNER
USDA-NASS

## DATA CARD AUTHOR
Pratishtha Poudel, Dilli Paudel

## DATASET OVERVIEW
**Crops**: Grain corn, Winter wheat

More crops can be added in the script if/when needed.

**Variables** [unit]: Acres harvested [acres], Production [bushels], Yield [bushels/acre].

The variables that can be selected and their units vary by crop. The ones above are for "CORN, GRAIN" (Grain Maize). Similarly, for "CORN, GRAIN", statistics are available for IRRIGATED as well as NON-IRRIGATED.

**Temporal coverage**: From: 1944 - To: 2022

**Temporal resolution**: Yearly

**Spatial resolution**: Agricultural District, County, State.

**Date Published**: Regular updates. The last update for CORN was 2023-12-01.

**Data Modality**: Tabular data

## DATA ACCESS API
https://quickstats.nass.usda.gov/api

## PROVENANCE
The yield forecasting program of NASS is described [in this document](https://www.nass.usda.gov/Publications/Methodology_and_Data_Quality/Advanced_Topics/Yield%20Forecasting%20Program%20of%20NASS_2023.pdf). This is a revised version of the [document published in 2012](https://www.nass.usda.gov/Education_and_Outreach/Understanding_Statistics/Yield_Forecasting_Program.pdf).

## LICENSE
Public; https://opendefinition.org/licenses/cc-zero/

## HOW TO CITE
Cite the data source and the R package used to access it. See references.

## ADDITIONAL INFORMATION
NA

## REFERENCES

USDA-NASS, 2023. The Yield Forecasting Program of NASS. Technical Report. United States Department of Agriculture (USDA). https://www.nass.usda.gov/Publications/Methodology_and_Data_Quality/Advanced_Topics/Yield%20Forecasting%20Program%20of%20NASS_2023.pdf, Last accessed: Feb 23, 2024.

Potter NA (2019). “rnassqs: An ‘R' package to access agricultural data via the USDA National Agricultural Statistics Service (USDA-NASS) ’Quick Stats' API.” The Journal of Open Source Software.

Potter N (2022). rnassqs: Access the NASS 'Quick Stats' API. R package version 0.6.1, https://CRAN.R-project.org/package=rnassqs.
