# Adding or onboarding data to AgML Subnational Crop Yield Forecasting
Thank you for your interest in contributing to the AgML Subnational Crop Yield Forecasting benchmark. You could contribute one of the following
1. Crop statistics for a country or region.
2. Alternative or additional predictor data source.
3. Alternative crop mask source.
4. Alternative crop calendar source.

## Contributing crop statistics
1. First check if crop statistics for the country or region of interest is already included.
2. If not, create an issue and follow the [contributing guidelines](../CONTRIBUTING.md). Create a new directory under `data_preparation`.
3. Create a data card (`README.md`) following [the template](DATA-CARD-TEMPLATE.md) in the new directory.
4. As much as possible, look for ways to automatically export and prepare the dataset. This way the steps can be followed by other community members to prepare the data. Use existing APIs if available. If not, document the steps required to export the data. Add a notebook or script to document the steps to download data, to explore the data and to prepare it in [the expected format](DATA-FORMAT.md). Add the notebook or script in the new directory.
5. Define the administrative region identifier as follows: 
* Use existing ids if they are available and they contain 2 letter country codes plus admin level codes (e.g. NUTS_ID in Europe).
* Construct one with 2 letter country code and admin level codes. If more than one admin level is involved, separate the country code and ids for different levels with a hyphen.
6. Follow the [contributing guidelines](../README.md) to get your changes reviewed and merge when approved.

## Contributing predictor data, crop masks and crop calendars
1. First check if the predictor data source is already included.
2. If not, create an issue and follow the [contributing guidelines](../CONTRIBUTING.md). Create a new directory under `data_preparation`.
3. Create a data card (`README.md`) following [the template](DATA-CARD-TEMPLATE.md) in the new directory. Document information about the data source, publisher, provenance information, license, how to cite, etc.
4. Check [data sources comparison document](DATA-SOURCES-SELECTION.md) to see how the new source compares with included data source or other sources considered. Update [data sources comparison document](DATA-SOURCES-SELECTION.md) with pros and cons and justifications for why the new source is preferable.
5. Download the predictor and prepare it for inclusion in the benchmark dataset. An R data preparation script is included in `data_preparation/`. Data preparation will require shapefiles. Make sure the same administrative region identifiers can be constructed for predictor data from shapefiles used. This is crucial for matching predictor data with crop statistics data.
6. For crop masks, data preparation may not be necessary.
7. Send an email to AgML list email (agml@mail.agml.org) to include the new data source in the benchmark. Include "CY-Bench" or "Subnational crop yield forecasting" in the subject.
