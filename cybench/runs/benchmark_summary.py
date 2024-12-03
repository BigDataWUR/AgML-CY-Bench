import os

import pandas as pd
import argparse

from cybench.config import (
    DATASETS,
    PATH_DATA_DIR,
    PATH_OUTPUT_DIR,
)

from cybench.datasets.dataset import Dataset


def dataset_summary(
    dataset_name: str = "maize_NL",
    min_year: int = 2003,
    max_year: int = 2024
) -> dict:
    """
    Output a summary of dataset.
    Args:
        dataset_name (str): The name of the dataset to load
        min_year (int): minimum year (soil moisture data starts from 2003)
        max_year (int): maximum year (some regions in AR have data points for 2024)

    Returns:
        pd.DataFrame with a summary for given dataset
    """
    dataset = Dataset.load(dataset_name)
    all_years = sorted(dataset.years)
    data_sizes = []
    year_cols = [str(yr) for yr in range(min_year, max_year + 1)]
    for test_year in range(min_year, max_year + 1):
        train_years = [y for y in all_years if y != test_year]
        test_years = [test_year]
        if (test_year not in all_years):
            data_sizes.append(0)
        else:
            _, test_dataset = dataset.split_on_years((train_years, test_years))
            data_sizes.append(len(test_dataset))

    country_code = dataset_name.split("_")[1]
    ds_summary = { country_code : data_sizes }
    summary_df = pd.DataFrame.from_dict(ds_summary, columns=year_cols, orient="index")
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={"index": "country_code"}, inplace=True)

    return summary_df


def run_benchmark_summary(output_file: str = None):
    if (output_file is None):
        output_file = "dataset_summary.csv"

    summary_df = pd.DataFrame()
    for crop in DATASETS:
        for cn in DATASETS[crop]:
            if os.path.exists(os.path.join(PATH_DATA_DIR, crop, cn)):
                dataset_name = crop + "_" + cn
                ds_df = dataset_summary(dataset_name=dataset_name)
                summary_df = pd.concat([summary_df, ds_df], axis=0)

    summary_df.to_csv(os.path.join(PATH_OUTPUT_DIR, output_file), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_summary.py", description="Output benchmark dataset summary"
    )
    parser.add_argument("-o", "--output_file")
    args = parser.parse_args()
    output_file = args.output_file
    run_benchmark_summary(output_file)
