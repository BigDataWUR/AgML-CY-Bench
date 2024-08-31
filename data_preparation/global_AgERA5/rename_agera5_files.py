import os
import sys

# NOTE: AgERA5 indicators used in CY-Bench are:
# Maximum_Temperature
# Minimum_Temperature
# Mean_Temperature
# Precipitation_Flux
# Solar_Radiation_Flux
# See the AgERA5_params in download script.

indicator = sys.argv[1]
start_year = int(sys.argv[2])
end_year = int(sys.argv[3])

# rename files in the current directory
for yr in list(range(start_year, end_year + 1)):
    files = [ f for f in os.listdir(".") if f.endswith(".nc") and str(yr) in f]
    for f in files:
        try:
            agera5_date_index = f.index("AgERA5_" + str(yr))
            date_str = f[agera5_date_index + 7:agera5_date_index+15]
            _, extension = os.path.splitext(f)
            new_name = "_".join(["AgERA5", indicator, date_str]) + extension
            os.rename(f, new_name)
        except ValueError:
            continue
