"""
@author: Dilli R. Paudel
"""

import os
from zipfile import ZipFile


def unzipAgERA5andRename(data_dir, country_dir, predictor_source, indicator, year):
    """
    AgERA5 data downloaded by Abdelrahman is stored as .zip files in Google Drive.
    """
    dir_path = os.path.join(data_dir, predictor_source, country_dir, indicator)
    filename = "_".join([country_dir, str(year), indicator]) + ".zip"
    file_path = os.path.join(dir_path, filename)
    files = [ f for f in os.listdir(dir_path) if f.endswith(".nc") and (str(year) in f)]
    # if files do not exist
    if (not files):
        with ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dir_path)

        # rename files
        files = [ f for f in os.listdir(dir_path) if f.endswith(".nc") and (str(year) in f)]
        for f in files:
            old_file = os.path.join(dir_path, f)
            try:
                agera5_date_index = f.index("AgERA5_" + str(year))
                date_str = f[agera5_date_index + 7: agera5_date_index+15]
                _, extension = os.path.splitext(f)
                new_name = "_".join([predictor_source, indicator, date_str]) + extension
                new_file = os.path.join(dir_path,  new_name)
                os.rename(old_file, new_file)

            except ValueError:
                continue

def deleteAgERA5Files(indicator_dir):
    """
    Delete extracted .nc files or converted .tif files.
    NOTE: Must be called after processing corresponding files.
    """
    files = [ f for f in os.listdir(indicator_dir) if f.endswith(".nc")]
    for f in files:
        os.remove(os.path.join(indicator_dir, f))

    # remove tif files
    tiff_dir = os.path.join(indicator_dir, TIFF_DIRECTORY_NAME)
    if (os.path.isdir(tiff_dir)):
        files = [ f for f in os.listdir(tiff_dir) if f.endswith(".tif")]
        for f in files:
            os.remove(os.path.join(tiff_dir, f))

def get_time_series_files(data_path,
                          exclude_filenames=[],
                          exclude_extensions=[],
                          year=2000):
    files = []
    for f in os.listdir(data_path):
        # exclude sub-directories
        if (os.path.isdir(os.path.join(data_path, f))):
            continue

        fname, ext = os.path.splitext(f)
        if ((fname in exclude_filenames) or
            (ext in exclude_extensions)):
            continue

        # we expect the last part of filename to be YYYYMMDD
        date_str = fname[-8:]
        if (int(date_str[:4]) == year):
            files.append(f)

    return files