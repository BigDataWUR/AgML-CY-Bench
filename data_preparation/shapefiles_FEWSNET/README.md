# FEWS NET Crop Statistics Unit Dataset

## Short description
The FEWS NET crop statistics unit dataset contains a unique geocode (FNID) of an administrative unit that is linked to the country's boundary at a given point in time. FEWS NET has tracked changes in the names and geometry of administrative boundaries and created a [database](https://fews.net/data/geographic-boundaries) of historical and current subnational administrative boundaries for a number of countries. The FNID is represented as `adm_id` and is aligned with the `adm_id` of the crop statistics dataset.

## Dataset owner
FEWS NET

## Data card author
Donghoon Lee and Weston Anderson

## Dataset overview
- **Variables**:
adm_id				
  - `adm_id`: Administrative unit (FNID)
  - `admin0`: Country name (admin level 0)
  - `admin1`: Province name (admin level 1)
  - `admin2`: District name (admin level 2); if available
  - `geometry`: Geometries of administrative unit
- **File format**: ESRI Shapefile
- **Geomtry types**: Polygons
- **Coordinate reference system**: EPSG: 4326
- **Date published**: Feb 27, 2024

## License
Data use aligns with [FEWS NET's Data Attribution and Use Policy](https://help.fews.net/fdp/data-and-information-use-and-attribution-policy):
- FEWS NET publishes all data according to permissions provided by the original source institution, and all data sources are fully attributed. There is no attempt to otherwise duplicate the information or systems of other organizations. 
- Data collected directly by FEWS NET are noted as such and are managed according to USAID’s policy on sharing Agency-funded data for public benefit, while ensuring proper protections for privacy and national security ([ADS 579](https://www.usaid.gov/about-us/agency-policy/series-500/579)).

The dataset shared herein adheres to FEWS NET's [terms of use and disclaimer](https://help.fews.net/fdp/data-and-information-use-and-attribution-policy#Dataandinformationuseandattributionpolicy-Termsofuseanddisclaimer) as well as [attribution policy](https://help.fews.net/fdp/data-and-information-use-and-attribution-policy#Dataandinformationuseandattributionpolicy-Attributionpolicy). It is important to note that this dataset represents a version that has undergone additional processing beyond the initial collection by FEWS NET.

## How to cite
A formal citation for FEWS NET data will be provided in due course.