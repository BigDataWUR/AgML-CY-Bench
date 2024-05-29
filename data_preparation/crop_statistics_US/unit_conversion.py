import pandas as pd
import os

data_path = os.path.join("data", "data_US")
yield_csv = os.path.join(data_path, "YIELD_COUNTY_US.csv")

yield_df = pd.read_csv(yield_csv, header=0)
print(yield_df.head())

# unit conversions for maize or corn
# convert corn yield from bushels/acre to t/ha
# See https://www.extension.iastate.edu/agdm/wholefarm/html/c6-80.html
yield_corn = yield_df[yield_df["crop_name"] == "corn_grain "].copy()
print(yield_corn.head())
yield_corn["yield"] = 0.0628 * yield_corn["yield"]
yield_corn["crop_name"] = "Grain maize"

# convert area from acres to hectares
# See https://www.unitconverters.net/area/acres-to-hectare.htm
yield_corn["harvest_area"] = 0.4047 * yield_corn["harvest_area"]

# convert production from bushels to tons
# See https://grains.org/markets-tools-data/tools/converting-grain-units/
yield_corn["production"] = 0.0254 * yield_corn["production"]

# unit conversions for wheat
# convert yields yield from bushels/acre to t/ha
# See https://www.extension.iastate.edu/agdm/wholefarm/html/c6-80.html
yield_wheat = yield_df[yield_df["crop_name"] == "wheat_winter "].copy()
print(yield_wheat.head())
yield_wheat["yield"] = 0.0673 * yield_wheat["yield"]
yield_wheat["crop_name"] = "Winter wheat"

# convert area from acres to hectares
# See https://www.unitconverters.net/area/acres-to-hectare.htm
yield_wheat["harvest_area"] = 0.4047 * yield_wheat["harvest_area"]

# convert production to tons
# See https://grains.org/markets-tools-data/tools/converting-grain-units/
yield_wheat["production"] = 0.0272 * yield_wheat["production"]

yield_df = pd.concat([yield_corn, yield_wheat], axis=0)
print(yield_df.head())
yield_df.to_csv("YIELD_COUNTY_US.csv", index=False)
