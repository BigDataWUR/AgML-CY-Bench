"""
@author: Guanyuan Shuai
"""

import geopandas as gpd


EU_COUNTRY_CODE_KEY = "CNTR_CODE"
EU_ADMIN_LEVEL_KEY = "LEVL_CODE"

# Country codes to admin level
# Austria (AT), Belgium (BE), Bulgaria (BG), Czech Republic (CZ), Germany (DE), Denmark (DK), Estonia (EE), Greece (EL),
# Spain (ES), Finland (FI), France (FR), Croatia (HR), Hungary (HU), Ireland (IE), Italy (IT), Lithuania (LT),
# Latvia (LV), The Netherlands (NL), Poland (PL), Portugal (PT), Romania (RO), Sweden (SE), Slovakia (SK)
EU_COUNTRIES = {
    "AT" : 2,
    "BE" : 2,
    "BG" : 2,
    "CZ" : 3,
    "DE" : 3,
    "DK" : 3,
    "EE" : 3,
    "EL" : 3,
    "ES" : 3,
    "FI" : 3,
    "FR" : 3,
    "HR" : 2,
    "HU" : 3,
    "IE" : 2,
    "IT" : 3,
    "LT" : 3,
    "LV" : 3,
    "NL" : 2,
    "PL" : 2,
    "PT" : 2,
    "RO" : 3,
    "SE" : 3,
    "SK" : 3
}

# Angola (AO), Burkina Faso (BF), Ethiopia (ET), Lesotho (LS), Madagascar (MG), Malawi (MW),
# Mozambique (MZ), Niger (NE), Senegal (SN), Chad (TD), South Africa (ZA), Zambia (ZM)
FEWSNET_COUNTRIES = ["AO", "BF", "ET", "LS", "MG", "MW", "MZ", "NE", "SN", "TD", "ZA", "ZM"]
FEWSNET_ADMIN_ID_KEY = "adm_id"



def get_shapes(file_path, country="US", admin_level=None):
    geo_df = gpd.read_file(file_path)
    if (country in EU_COUNTRIES):
        geo_df = geo_df[geo_df[EU_COUNTRY_CODE_KEY] == country]
    elif (country in FEWSNET_COUNTRIES):
        geo_df = geo_df[geo_df[FEWSNET_ADMIN_ID_KEY].str[:2] == country]


    if (admin_level is not None):
        assert (country in EU_COUNTRIES)
        geo_df = geo_df[geo_df[EU_ADMIN_LEVEL_KEY] == admin_level]

    return geo_df

def get_admin_id(df, country):
    # European countries
    if (country in EU_COUNTRIES):
        df["adm_id"] = df["NUTS_ID"]
    elif (country == "US"):
        df["adm_id"] = "US-" + df["STATEFP"] + "-" + df["COUNTYFP"]
    # TODO: add other cases
    elif (country == "CN"):
        df["adm_id"] = df["ADM1_PCODE"].str.replace('CN', 'CN-')
    # MEXICO:adm_id was created for Mexico using the number ID (1-32) for states coming from the original data
    # and prefix "MX" was added so we have "MX01" or "MX32". For more information check Mexico data card.
    elif (country == "MX"):
         df["adm_id"] = df["adm_id"]

    # elif (country == "IN"):
    #     df["adm_id"] = df["adm_id"]  #adm_id already exists
    # elif (country == "ML"):
    #     df["adm_id"] = df["adm_id"] # already exists
    # NOTE for FEWSNET, adm_id already exists
    # elif (country in FEWSNET_COUNTRIES):
        # FEWS NET's "FNID" is converted to "adm_id" during the data preparation
        # df["adm_id"] = df["adm_id"]

    return df