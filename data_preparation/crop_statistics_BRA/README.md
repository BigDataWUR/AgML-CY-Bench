# Brazil subnational crop yield statistics

## Short description
The linked platform allows you to customize and download datasets based on crop, variable, year, spatial aggregation, and region. You can obtain the results in various formats, including CSV, XLSX, TSV, ODS, or HTML, as well as a compressed ZIP file. For larger requests exceeding 200,000 values, you can provide an email address, and once the file is ready, download instructions will be sent to you.

In most cases, the resulting table will have multi-level columns and a header. The structure of the file depends on the information you requested prior to the download. Each file contains a "Notas" sheet with supporting information on the data.

For additional information and to avoid redundancy, I refer to the platform itself and the remarks they provide, e.g. at the end of the page and in the downloaded tables.

## Link
https://sidra.ibge.gov.br/tabela/1612

## Publisher
Brazilian Institute of Geography and Statistics (IBGE)

## Dataset owner
Brazilian Institute of Geography and Statistics (IBGE)

## Data card author
Maximilian Zachow

## Dataset overview
The process of obtaining an exemplary wheat dataset is outlined below. A translation is given in italic. Basic preprocessing of the downloaded xlsx file is given in the notebook `read_and_preprocess_data.ipynb`.

### Selection process via interface on platform 
 **Variável** (*variable*): 
 - Área plantada (Hectares [1988 a 2022]) - *planted area (ha)*
 - Área colhida (Hectares) - *harvested area (ha)* 
 - Quantidade produzida (Toneladas) - *production (t)*
 - Rendimento médio da produção (Quilogramas por Hectare) - *yield (kg/ha)*

**Produto das lavouras temporárias** (*crop selection*): 
 - Trigo (em grão) - *wheat (grains)*

 **Ano** (*year*):
  - 2022, 2021, 2020 

**Unidade Territorial** *(territorial unit)*: 
 - Município - *municipality*

Click download and a dialogue opens. Select formato *(format)* XLSX and check two boxes "Exibier códigos de territórios *(display territory codes)* and Exibir nomes de territórios *(Display territory names)*. Lastly, select "Imediato (até 200.000 valores)" *(immediate download)* and click download again. A new tab opens and after ~7 min the download starts.

## Provenance 

Weekly updates.

## License
Nothing was found on their website regarding data policies. However, IBGE data has been distributed before. See [here](https://figshare.com/articles/dataset/_2008_IBGE_data_for_the_regions_and_states_sampled_in_this_study_/468422?backTo=/collections/The_Genomic_Ancestry_of_Individuals_from_Different_Geographical_Regions_of_Brazil_Is_More_Uniform_Than_Expected/1696325) and [here](https://www.ceicdata.com/en/brazil/sna-2008-gross-domestic-product-per-capita/gross-domestic-product-prices-of-previous-year-ibge)

## Additional information
Not applicable.

### Overview of dataset structure and basic preprocessing

See notebook `read_and_preprocess_data.ipynb`.

## References
IBGE SIDRA. (2022), “Tabela 1612: Área plantada, área colhida, quantidade produzida, rendimento médio e valor da produção das lavouras temporárias”, available at: https://sidra.ibge.gov.br/tabela/1612 (accessed 6 February 2024).
