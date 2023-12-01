# Global MOD09CMG product
The Global 8-day Climate Modeling Grid (CMG)-scale product provides NDVI and GCVI information at a global scale. This product refers to both MODIS and VIIRS acquisitions.

## DATASET LINK
The dataset can be processed using [the octvi python library](https://pypi.org/project/octvi/).

## PUBLISHER
The publisher of the octvi python library is [NASA Harvest](https://nasaharvest.org/).

## DATASET OVERVIEW
**Coordinate system**: WGS_1984

**Spatial resolution - Pixel Size**: ~5km

**Temporal resolution**: 8 day

**Value range**: 50 to 250 for NDVI. To scale the values into conventional range, apply the formula: (x - 50)/200
The octvi library can also be configured to download GCVI data instead of NDVI

## PROVENANCE
[MODIS and VIIRS NDVI source](https://lpdaac.usgs.gov/products/mod09cmgv061/)