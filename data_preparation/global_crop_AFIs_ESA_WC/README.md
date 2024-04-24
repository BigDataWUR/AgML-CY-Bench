# Global maize and winter-spring cereals Area Fraction Images

## Short description
JRC compiled maize and winter-spring cereals AFIs fully spatially compatible with the JRC FPAR 500 m from for ESA WorldCereal products.

Maize and winter-spring cereals are identified by masks with 500 m spatial resolution and expressed as Area Fraction Image, AFI, i.e. the percentage of the pixel area occupied by the target crops (either maize or winter-spring cereals), ranging from 0 to 100%.

AFIs are typically used in the process of aggregation (of remote sensing and meteorological data) at the desired administrative level (e.g. GAUL1 or GAUL2 level) where spatial averaging is performed using AFI as weighting factor.

WorldCereal generated global-scale maps for the year 2021. The maize AFI (maize_WC_AFI.tif) was derived by merging the two maize WorldCereal 10 m products related to main and second maize (tc-maize-main, tc-maize-second).

The winter-spring cereals AFI (wintercereals_springcereals_WC_AFI.tif) was derived by merging the winter cereals and spring cereals 10 m products (tc-wintercereals and tc-springcereals, respectively). Note that cereals as defined by WorldCereal include wheat, barley and rye. Information about WorldCereal can be found at https://doi.org/10.5194/essd-15-5491-2023.

The generic cropland mask (asap_mask_crop_v04.tif) is developed by IIASA and JRC as a global temporary cropland map compatible with the JRC FPAR 500 m time series from the hybridization of several high resolution global cropland layers (selecting the best product between ESA WordCereal, GLAD UMD, others). Cropland is identified by masks with 500 m spatial resolution expressed as area fraction image (AFI, i.e. the percentage of the pixel area occupied by temporary crops, ranging from 0 to 100%).

## Link
The AFIs were generated on purpose for AgML.

## Publisher
Food Security Unit of the European Commission’s Joint Research Center (JRC D.5)

## Dataset owner
Food Security Unit of the European Commission’s Joint Research Center (JRC D.5)

## Data card author
Michele Meroni, Francesco Collivignarelli

## Dataset overview

Coordinate system: WGS_1984

Spatial resolution - Pixel Size: 0.004464 deg (about 500 m)

## Provenance
Original base layers come from [ESA WorldCereal](https://esa-worldcereal.org/en). Elaboration to AFI was made by Food Security Unit of the European Commission’s Joint Research Center (JRC D.5).
The processing code to generate the AFIs is fully carried out in Google Earth Engine (GEE) and can be retrieved at the following link: https://code.earthengine.google.com/616f8e3a87e0bf37b9b47e7ba1ca8073. Base WorldCereal products are available at https://zenodo.org/records/7875105 and in GEE catalog.

## License
The layers are shared with Creative Commons Attribution 4.0 License (https://zenodo.org/records/7875105).

## How to cite
Cite the WorldCereal AFIs as follows: elaboration of WorldCereal map layers (Van Tricht et al., 2023).

Cite the generic ASAP crop mask as follows:
Fritz S, Lesiv M, Perez Guzman K, See L, Meroni M, Collivignarelli F, & Rembold F (2024). Development of a new cropland and rangeland Area Fraction Image at 500 m for the ASAP system. European Commission, Joint Research Centre (JRC).

## Reference
Van Tricht, K. and Degerickx, J. and Gilliams, S. and Zanaga, D. and Battude, M. and Grosu, A. and Brombacher, J. and Lesiv, M. and Bayas, J. C. L. and Karanam, S. and Fritz, S. and Becker-Reshef, I. and Franch, B. and Moll\`a-Bononad, B. and Boogaard, H. and Pratihast, A. K. and Koetz, B. and Szantoi, Z. (2023). WorldCereal: a dynamic open-source system for global-scale, seasonal, and reproducible crop and irrigation mapping, Earth System Science Data, 15, 5491--5515, DOI: 10.5194/essd-15-5491-2023.