# Global MOD09CMG product

## Short description
The Global 8-day Climate Modeling Grid (CMG)-scale product provides Normalized Difference Vegetation Index (NDVI) and Green Chlorophyll Vegetation Index (GCVI) information at a global scale. This product refers to both MODIS and VIIRS acquisitions.

## Link
https://lpdaac.usgs.gov/products/mod09cmgv061/#tools
https://e4ftl01.cr.usgs.gov/

## Publisher
Land Processes Distributed Active Archive Center (LP DAAC), NASA

## Dataset owner
NASA

## Data card author
Ritvik Sahajpal
Siyabusa Mkuhlani

## Dataset overview
**Coordinate system**: WGS_1984

**Spatial resolution - Pixel Size**: ~5km

**Temporal resolution**: 8 day

**Value range**: 50 to 250 for NDVI. To scale the values into conventional range, apply the formula: (x - 50)/200
The octvi library can also be configured to download GCVI data instead of NDVI

## Provenance
[MODIS and VIIRS NDVI source](https://lpdaac.usgs.gov/products/mod09cmgv061/)
The python package https://pypi.org/project/octvi/ downloads, process and packages vegetation index from various sateliete sensors such as MODIS and VIIRS.

## License
License not shown the terms and conditions, of no restrictions on re-use, sale, or redistribution, denotes, CC BY 4.0 license.

## How to cite
Vermote, E.. MOD09CMG MODIS/Terra Surface Reflectance Daily L3 Global 0.05Deg CMG V006. 2015, distributed by NASA EOSDIS Land Processes Distributed Active Archive Center, https://doi.org/10.5067/MODIS/MOD09CMG.006. Accessed 2024-04-19.

## References
Yan, X., Zang, Z., Li, Z., Luo, N., Zuo, C., Jiang, Y., Li, D., Guo, Y., Zhao, W., Shi, W., and Cribb, M.: A global land aerosol fine-mode fraction dataset (2001–2020) retrieved from MODIS using hybrid physical and deep learning approaches, Earth Syst. Sci. Data, 14, 1193–1213, 2022. https://doi.org/10.5194/essd-14-1193-2022

Zhang X, Jiao Z, Zhao C, Guo J, Zhu Z, Liu Z, Dong Y, Yin S, Zhang H, Cui L, et al. Evaluation of BRDF Information Retrieved from Time-Series Multiangle Data of the Himawari-8 AHI. Remote Sensing. 2022; 14(1):139. https://doi.org/10.3390/rs14010139

Donnelly, A., Yu, R., & Liu, L. (2021). Comparing in situ spring phenology and satellite-derived start of season at rural and urban sites in Ireland. International Journal of Remote Sensing, 42(20), 7821–7841. https://doi.org/10.1080/01431161.2021.1969056
