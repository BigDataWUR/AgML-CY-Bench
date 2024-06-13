import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import pickle

from cybench.config import PATH_RESULTS_DIR, CONFIG_DIR
from cybench.runs.run_benchmark import _compute_evaluation_results

PATH_GRAPHS_DIR = os.path.join(CONFIG_DIR, "output", "graphs")
os.makedirs(PATH_GRAPHS_DIR, exist_ok=True)

# the ones commented out have zero targets
datasets = {
    "maize": [
        "AO",
        "AR",
        "AT",
        "BE",
        "BF",
        "BG",
        "BR",
        "CN",
        "CZ",
        "DE",
        "DK",
        "EE",
        "EL",
        "ES",
        "ET",
        "FI",
        "FR",
        "HR",
        "HU",
        "IE",
        "IN",
        "IT",
        "LS",
        "LT",
        "LV",
        "MG",
        "ML",
        "MW",
        "MX",
        "MZ",
        "NE",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SK",
        "SN",
        "TD",
        "US",
        "ZA",
        "ZM",
    ],
    "wheat": [
        "AR",
        "AT",
        "AU",
        "BE",
        "BG",
        "BR",
        "CN",
        "CZ",
        "DE",
        "DK",
        "EE",
        "EL",
        "ES",
        "FI",
        "FR",
        "HR",
        "HU",
        "IE",
        "IN",
        "IT",
        "LT",
        "LV",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SK",
        "US",
    ],
}


def plot_graph(df, x_col, x_label, metric_label,
               title, file_name, rotation=45):
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=x_col,
        y="value",
        data=df,
        hue=x_col,
        palette="Set2",
        showfliers=False,
        legend=False,
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(metric_label)
    if (rotation is not None):
        plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, file_name))
    plt.close(fig)


def summarize_metrics(
        df: pd.DataFrame,
        metric: str = None,
):

    # Reset index to make it easier to work with
    df_reset = df.reset_index()

    # Filter the data for MAPE and NRMSE
    df_mape = df_reset[df_reset["metric"] == "mape"]
    df_nrmse = df_reset[df_reset["metric"] == "normalized_rmse"]

    sns.set_palette(palette="pastel")

    if metric == "mape_year":
        plot_graph(df_mape, "year", "Year",
                   "Mean Absolute Percentage Error",
                   "MAPE Boxplots by year",
                   "mape_year.jpg", rotation=45)
    elif metric == "nrmse_year":
        plot_graph(df_nrmse, "year", "Year", "Normalized RMSE",
                   "NRMSE Boxplots by year",
                   "nrmse_year.jpg", rotation=45)
    elif metric == "mape_model":
        plot_graph(df_mape, "model", "Model", "Mean Absolute Percentage Error",
                   "MAPE Boxplots by model", "mape_model.jpg")
    elif metric == "nrmse_model":
        plot_graph(df_nrmse, "model", "Model", "Normalized RMSE",
                   "NRMSE Boxplots by model", "nrmse_model.jpg")
    elif metric == "mape_country":
        plot_graph(df_mape, "country", "Country",
                   "Mean Absolute Percentage Error",
                   "MAPE Boxplots by country",
                   "mape_country.jpg", rotation=45)
    elif metric == "nrmse_country":
        plot_graph(df_nrmse, "country", "Country",
                   "Normalized RMSE",
                   "NRMSE Boxplots by country",
                   "nrmse_country.jpg", rotation=45)
    elif metric == "mape":
        plot_graph(df_mape, "metric", "MAPE",
                   "Mean Absolute Percentage Error",
                   "MAPE Boxplots",
                   "mape.jpg")
    elif metric == "nrmse":
        plot_graph(df_nrmse, "metric", "NRMSE",
                   "Normalized RMSE",
                   "NRMSE Boxplots",
                   "nrmse.jpg")
    elif metric == "crop":
        plot_graph(df_mape, "crop", "Crop",
                   "Mean Absolute Percentage Error",
                   "MAPE Boxplots by crop",
                   "mape_crop.jpg")

        plot_graph(df_nrmse, "crop", "Crop",
                   "Normalized RMSE",
                   "NRMSE Boxplots by crop",
                   "nrmse_crop.jpg")
    elif metric == "bars":
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x="country", y="value", data=df_mape, hue="country", palette="Set2")
        plt.title(f"MAPE per Country")
        plt.xlabel("Country")
        plt.ylabel("Mean Average Percentage Error")
        plt.xticks()
        plt.tight_layout()
        plt.savefig(os.path.join(PATH_GRAPHS_DIR, "mape_country_bar.jpg"))

        plt.figure(figsize=(12, 6))
        sns.barplot(
            x="country", y="value", data=df_nrmse, hue="country", palette="Set2"
        )
        plt.title(f"NRMSE per Country")
        plt.xlabel("Country")
        plt.ylabel("Normalized RMSE")
        plt.xticks()
        plt.tight_layout()
        plt.savefig(os.path.join(PATH_GRAPHS_DIR, "nrmse_country_bar.jpg"))
        plt.close(fig)
    elif metric == "box_plots":
        fig = plt.figure(figsize=(14, 10))
        sns.boxplot(
            x="country",
            y="value",
            hue="crop",
            data=df_nrmse,
            dodge=True,
            showfliers=False,
            palette="Set2",
        )
        # plt.title(f'Vertical Stacked Box Plot for NRMSE')
        plt.xlabel("Country")
        plt.ylabel("Normalized RMSE")
        plt.legend(title="Crop", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(PATH_GRAPHS_DIR, "box_country_nrmse.jpg"))

        plt.figure(figsize=(14, 10))
        sns.boxplot(
            x="country",
            y="value",
            hue="crop",
            data=df_mape,
            dodge=True,
            showfliers=False,
            palette="Set2",
        )
        # plt.title(f'Vertical Stacked Box Plot for MAPE')
        plt.xlabel("Country")
        plt.ylabel("Mean Average Precision Error")
        plt.legend(title="Crop", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(PATH_GRAPHS_DIR, "box_country_mape.jpg"))
        plt.close(fig)


def plot_prediction_residuals(data, countries, crop, metric, subplots_per_row=4):
    # Filter data based on selected countries and crop
    data_filtered = data[(data["country"].isin(countries)) & (data["crop"] == crop)]

    # Set up the figure for subplots
    num_countries = len(countries)
    num_rows = (num_countries + subplots_per_row - 1) // subplots_per_row
    fig, axes = plt.subplots(
        num_rows, subplots_per_row, figsize=(21, 4 * num_rows), sharey=True
    )
    axes = axes.flatten()

    if metric == "normalized_rmse":
        a = "Normalized RMSE"
    elif metric == "mape":
        a = "Mean Average Percentage Error"

    # Plot boxplots for each country
    for i, country in enumerate(countries):
        ax = axes[i]
        sns.boxplot(
            x="model",
            y=metric,
            data=data_filtered[data_filtered["country"] == country],
            ax=ax,
            showfliers=False,
            hue="model",
            palette="tab10",
        )
        ax.set_title(f"{country} ({crop})")
        ax.set_xlabel("")
        ax.set_ylabel(a if i % subplots_per_row == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.xaxis.grid(False)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(
        os.path.join(PATH_GRAPHS_DIR, f"box_country_{metric}_{countries}_{crop}.jpg")
    )
    plt.close(fig)


def plot_year_results(data, country, crop, metric):
    fig = plt.figure(figsize=(14, 10))

    data_filtered = data[(data["country"] == country) & (data["crop"] == crop)]
    if metric == "mape":
        data_filtered = data[data["mape"] < 1000]

    sns.boxplot(
        x="year",
        y=metric,
        hue="year",
        data=data_filtered,
        showfliers=False,
        palette="Set2",
    )
    plt.title(f"{country} ({crop})")
    plt.xlabel("Year")
    plt.ylabel(
        "Normalized RMSE"
        if metric == "normalized_rmse"
        else "Mean Average Precision Error"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PATH_GRAPHS_DIR, f"{country}_{metric}_{crop}.jpg")
    )
    plt.close(fig)


if __name__ == "__main__":
    skipped = []
    df_all = pd.DataFrame()
    for crop, cns in datasets.items():
        for cn in cns:
            name_benchmark = crop + "_" + cn
            if not os.path.isdir(os.path.join(PATH_RESULTS_DIR, name_benchmark)):
                continue
            print(name_benchmark)
            df = _compute_evaluation_results(name_benchmark)
            df["country"] = [cn for _ in df.value]
            df["crop"] = [crop for _ in df.value]
            df.reset_index(inplace=True)
            df.set_index(
                ["model", "year", "metric", "country", "crop"], inplace=True
            )
            df_all = pd.concat([df_all, df], axis=0)

    print(df_all)
    # Drop duplicate indexes
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    print(df_all)
    print(skipped)

    df = df_all.reset_index()
    df = df.rename(columns={"model": "model"})
    df_pivot = df.pivot_table(
        index=["model", "year", "country", "crop"], columns="metric", values="value"
    ).reset_index()
    df_pivot["model"].replace(
        {
            "AverageYieldModel": "Naive",
            "LSTM": "LSTM",
            "SklearnRidge": "Ridge",
            "SklearnRF": "RF",
            # "LinearTrend": "Trend",
        },
        inplace=True,
    )
    # df_pivot = df_pivot[df_pivot["model"] != "Trend"]
    # df_pivot['dataset_name'] = df_pivot['country'] + '_' + df_pivot['crop'] + "_" + df_pivot['year'].apply(str)
    # df_pivot = df_pivot.drop(columns=['country', 'crop', "year"])
    # df_pivot = df_pivot[['model', 'dataset_name', 'normalized_rmse', 'mape']]

    print(df_pivot)
    selected_countries = ["BG", "BE", "CN", "DE", "DK"]
    selected_crop = "maize"
    selected_metric = "mape"
    metrics = ["normalized_rmse", "mape"]
    # selected_countriess = [['AT', 'AR', 'BR', 'US', 'NL'], ['AU', 'DE', 'IN', 'FR', 'LT']]
    # selected_crops = ['wheat', 'wheat']
    # for cn, cr in zip(selected_countriess, selected_crops):
    #     for met in metrics:
    #         print(cn)
    #         print(cr)
    #         print(met)
    #         plot_prediction_residuals(df_pivot, cn, cr, metrics,
    #                                   subplots_per_row=len(cn))
    crops = ["maize", "wheat"]
    for metric in metrics:
        for crop in crops:
            for cn in datasets[crop]:
                plot_year_results(df_pivot, cn, crop, metric)

    # plot_prediction_residuals(
    #     df_pivot,
    #     selected_countries,
    #     selected_crop,
    #     selected_metric,
    #     subplots_per_row=len(selected_countries),
    # )

    df_pivot.to_csv(os.path.join(PATH_RESULTS_DIR, "cd_diagram.csv"))

    #
    #
    # graph_types = [
    #     "mape_year",
    #     "nrmse_year",
    #     "mape_model",
    #     "nrmse_model",
    #     "nrmse_country",
    #     "mape_country",
    #     "nrmse",
    #     "mape",
    #     "crop",
    #     "bars",
    #     "box_plots",
    # ]
    # for a in graph_types:
    #     make_graph_mape(df_all, a)

    # with open(os.path.join(PATH_RESULTS_DIR, f'visualize.pkl'), 'wb') as f:
    #     pickle.dump(df_all, f)
