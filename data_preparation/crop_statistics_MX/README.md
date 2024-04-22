# Crop Yield Statistics from Mexican National Institute for Statistics & Geography / Census & Surveys

## Short description

The dataset contains information for the years 2014,2017,2019 and 2022 and is coming from the national agricultural census.

## Link
Multiple links exist since the data is published for each survey / census separately:


**Summary with links to original tables**

| Year  | Scale | Filename | Title | Link | 
|:---------|:----------------|:----------------------------|:------------------------|:------------------------|
| 2022   |  -Estate/-Municipality | ca2022_agr01.xlsx | Número de unidades de producción con agricultura a cielo abierto, superficie cultivada y producción según modalidad hídrica por entidad federativa, municipio y cultivo | https://www.inegi.org.mx/contenidos/programas/ca/2022/tabulados/ca2022_agr01.xlsx |
| 2019    |  -Estate              | ena19_ent_agri02.xlsx| Superficie cultivada y producción de cultivos anuales y perennes según modalidad hídrica por entidad federativa y cultivo seleccionado | https://www.inegi.org.mx/contenidos/programas/ena/2019/tabulados/ena19_ent_agri02.xlsx |
| 2017    |  -Estate              | ena17_ent_agri03.xlsx| Superficie cultivada y producción de cultivos anuales y perennes a cielo abierto según disponibilidad del agua por entidad federativa y cultivo con representatividad en la entidad | https://www.inegi.org.mx/contenidos/programas/ena/2017/tabulados/ena17_ent_agri03.xlsx |
| 2014    |  -Estate              | ena14_agri02.xlsx| Superficie cultivada y producción de cultivos anuales y perennes por entidad federativa y cultivo con representatividad en la muestra| https://www.inegi.org.mx/contenidos/programas/ena/2014/tabulados/ena14_agri02.xlsx |





## Publisher
Instituto Nacional de Estadística y Geografía (INEGI)
Mexican National Institute for Statistics & Geography

## Dataset owner
Instituto Nacional de Estadística y Geografía (INEGI)
Mexican National Institute for Statistics & Geography

## Data card author

Oumnia Ennaji
Inti Ernesto Luna Aviles

## Dataset overview
The dataset has different attributes depending on the year of the census. In general, crop yield statistics are available at state level with the exception of the last census (2022) which in addition has information at the municipality level.

The census provides information for many crop types. However, here we just focus on the maize crop types.

### Census 2022 Specific Information

**Crops**: Maize

**Varieties**: Yellow / White / Forage Corn

**Variables**: 
'State'
'Municipality'
'Crop'
 'Active agricultural production units - Total'
 'Active agricultural production units - With open agriculture'
 'Cultivated area - Sown' [unit]:hectares
 'Cultivated area - Harvested' [unit]:hectares
 'Production ' (yield) [unit]:tonnes
 'Water Modality - Temporary - Production unit',
 'Water Modality - Temporary - Cultivated area - Sown' [unit]:hectares 
 'Water Modality - Temporary - Cultivated area - Harvested' [unit]:hectares
 'Water Modality - Temporary - Production' [unit]:tonnes
 'Water Modality - Irrigation - Production unit' [unit]:tonnes
 'Water Modality - Irrigation - Cultivated area - Sown' [unit]:hectares
 'Water Modality - Irrigation - Cultivated area - Harvested' [unit]:hectares
 'Water Modality - Irrigation - Production' [unit]:tonnes

**Temporal coverage**:

2022-Census:
Data collected from October, 2021 to September, 2022 
Temporal resolution: data for specific census period
Date Published: 2023-11-17

### Survey 2019 Specific Information

**Crops**: Maize

**Varieties** : White / Yellow

**Variables**: 'State','Crop','Total Cultivated area - Sown' [unit]:hectares
'Total Cultivated area - Harvested' [unit]:hectares
'Water Modality - Temporary - Cultivated area - Sown' [unit]:hectares
'Water Modality - Temporary - Cultivated area - Harvested' [unit]:hectares
'Water Modality - Irrigation - Cultivated area - Sown' [unit]:hectares
'Water Modality - Irrigation - Cultivated area - Harvested' [unit]:hectares
'Total production'[unit]:tonnes
'Water Modality - Temporary - Production'[unit]:tonnes
'Water Modality - Irrigation - Production'[unit]:tonnes

**Temporal coverage**:

2019-Census:
Data collected from October, 2018 to September, 2019 
Temporal resolution: data for specific census period
Date Published: 2020

### Survey 2017 Specific Information

**Crops**: Maize

**Varieties**: White / Yellow

**Variables**: 
'State'
'Crop'
'Total Cultivated area - Sown' [unit]:hectares
'Total Cultivated area - Harvested' [unit]:hectares
           	'Total production'[unit]:tonnes
       	'Water Modality - Temporary - Cultivated area - Sown' [unit]:hectares
        	'Water Modality - Temporary - Cultivated area - Harvested' [unit]:hectares
          	'Water Modality - Temporary - Production'[unit]:tonnes
        	'Water Modality - Irrigation - Cultivated area - Sown' [unit]:hectares
          	'Water Modality - Irrigation - Cultivated area - Harvested' [unit]:hectares
          	'Water Modality - Irrigation - Production'[unit]:tonnes

**Temporal coverage**:

2017-Survey:
Data collected from October, 2016 to September, 2017 
Temporal resolution: data for specific census period
Date Published: 8th of January, 2019

### Survey 2014 Specific Information

**Crops**: Maize

**Varieties**: White / Forage

**Variables**:

'State'
'Crop'
'Total Cultivated area - Sown' [unit]:hectares
'Total Cultivated area - Harvested' [unit]:hectares
           	'Total production'[unit]:tonnes

**Temporal coverage**:

2014-Survey:
Data collected from October, 2013 to September, 2014 
Temporal resolution: data for specific census period
Date Published: 2015

## Data Modality:
Original data for all years are in excel files (.xlsx) but processed for cleaning and english translation purposes and available as csv(data)+json(metadata)

## Data access API
Not aware of it

## Provenance 
No specific information available

## License 

According to INEGI “Technical Standard for access and publication of Open Data of Statistical and Geographic Information of National Interest”, published on December 4, 2014 in the Official Gazette of the Federation, whose purpose is to establish the provisions so that the Data Sets within the framework of the Public Statistical and Geographic Information Service, generated and managed by the State Units, are made available as Open Data, with the purpose of facilitating their access, use, consultation, reuse and redistribution for any purpose, all free of charge in accordance with the provisions of the Terms of Free Use published on the INEGI Internet Site (Site), which can be consulted at the link:
 
Terms of use: https://www.inegi.org.mx/inegi/terminos.html



## How to cite
Include the expected citation for the data here. References can be more general (including citations to other data or papers).

Census Project 2022 
Census Data 2022
Survey Data 2019
Survey Data 2017
Survey Data 2014

## Aditional Information

The data set contains information for only 4 years because the national institute (INEGI) have carried out theses surveys and census only for specific years and we selected these ones as they provide more information for our purposes.

We have focused on **maize** as it is the main grain grown in Mexico. In the dataset, there are other crops including wheat but given the size of the data and areas, they were not considered.

## References
(1)	Mexico - Censo Agropecuario 2022, Resultados definitivos. https://www.inegi.org.mx/rnm/index.php/catalog/867/related-materials (accessed 2024-04-10).
(2)	Geografía(INEGI), I. N. de E. y. Censo Agropecuario (CA) 2022. https://www.inegi.org.mx/programas/ca/2022/#tabulados (accessed 2024-04-10).
(3)	Geografía(INEGI), I. N. de E. y. Encuesta Nacional Agropecuaria (ENA) 2019. https://www.inegi.org.mx/programas/ena/2019/#Tabulados (accessed 2024-04-10).
(4)	Encuesta Nacional Agropecuaria 2017. https://www.inegi.org.mx/programas/ena/2017/#tabulados (accessed 2024-04-10).
(5)	Encuesta Nacional Agropecuaria 2014. https://www.inegi.org.mx/programas/ena/2014/#tabulados (accessed 2024-04-10).
