# Data format

## File format
For now, we will assume tabular data and use csv format.

## Yield data
Please prepare or provide the yield data in the following format:

**crop_name**: e.g. winter wheat, grain maize, rice

**country_code**: 3 letter code of the country (check here: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3)

**adm_id**: Identifier for administrative unit. This identifier needs to be unique and could include country code and a way to extract identifier for larger administrative units (e.g. state or province).

**season_name**: local name of the season (if available).

**planting_year**: in YYYY format (if available)

**planting_date**: in YYYY-MM-DD format (if available)

**harvest_year**: in YYYY format

**harvest_date**: in YYYY-MM-DD format (if available)

**yield**: crop yield in mt/ha

**production**: crop production in mt (if available)

**planted_area**: planted area in ha (if available) 

**harvest_area**: harvested area in ha (if available)

**source**: data source with version

### Notes
Please provide the `planting_year`, `planting_date` and `harvest_date` in case this information is provided as part of the yield data set. If this information is not available, we will use a crop calendar to get this information.

This format was proposed by Rahel, Weston, Donghoon and Dilli. Please get in touch with us (laudien@pik-potsdam.de) in case you have questions or other suggestions. 

## Predictor time series
These will be prepared separately for different crops. Appropriate crop masks will be applied during preparation.

**adm_id**: Identifier for administrative unit. This identifier needs to be unique and could include country code and a way to extract identifier for larger administrative units (e.g. state or province).

**date**: (daily for all except for FPAR, which is dekadal, i.e. every ~10 days)

One or more predictor variables

## Static data (e.g. soil)
**adm_id**: Identifier for administrative unit. This identifier needs to be unique and could include country code and a way to extract identifier for larger administrative units (e.g. state or province).

One or more soil properties

## Crop calendar
TBD
