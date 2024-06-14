# @author: Hilmy, Dilli

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import pickle

from cybench.config import (
    PATH_RESULTS_DIR,
    CONFIG_DIR,
    KEY_LOC,
    KEY_YEAR,
)

from cybench.runs.run_benchmark import (
    get_prediction_residuals,
    compute_metrics,
)

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


def plot_bars(df, metric, metric_label, title_label, file_name):
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(
        x="country",
        y=metric,
        data=df,
        hue="model",
        palette="Set2",
        width=0.5,
        native_scale=True,
    )
    plt.title(title_label)
    plt.xlabel("Country")
    plt.ylabel(metric_label)
    plt.xticks()
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, file_name))


def plot_graph(
    df, x_col, hue_col, x_label, metric, metric_label, title, file_name, rotation=45
):
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=x_col,
        y=metric,
        data=df,
        hue=hue_col,
        palette="Set2",
        showfliers=False,
        legend=False,
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(metric_label)
    if rotation is not None:
        plt.xticks(rotation=rotation)
    plt.grid(True)
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, file_name))
    plt.close(fig)


def plot_metrics(
    df: pd.DataFrame,
    metric: str = None,
):
    sns.set_palette(palette="pastel")

    if metric == "mape_country":
        plot_graph(
            df,
            "country",
            "model",
            "Country",
            "mape",
            "Mean Absolute Percentage Error",
            "MAPE Boxplots by country",
            "mape_country.jpg",
            rotation=45,
        )
    elif metric == "nrmse_country":
        plot_graph(
            df,
            "country",
            "model",
            "Country",
            "normalized_rmse",
            "Normalized RMSE",
            "NRMSE Boxplots by country",
            "nrmse_country.jpg",
            rotation=45,
        )
    elif metric == "bars":
        plot_bars(
            df,
            "mape",
            "Mean Absolute Percentage Error",
            "MAPE by country",
            "bars_mape_country.jpg",
        )

        plot_bars(
            df,
            "normalized_rmse",
            "Mean Absolute Percentage Error",
            "MAPE by country",
            "bars_nrmse_country.jpg",
        )


def box_plots_residuals(
    data,
    crop,
    countries,
    residual_cols,
    residual_labels,
    ymin,
    ymax,
    subplots_per_row=4,
):
    # Filter data based on selected countries and crop
    data_filtered = data[(data["country"].isin(countries)) & (data["crop"] == crop)]

    # Set up the figure for subplots
    num_countries = len(countries)
    num_rows = (num_countries + subplots_per_row - 1) // subplots_per_row
    fig, axes = plt.subplots(
        num_rows, subplots_per_row, figsize=(21, 4 * num_rows), sharey="none"
    )
    axes = axes.flatten()
    font = {
        "color": "black",
        "size": 18,
    }

    for i in range(num_countries):
        cn = countries[i]
        data_cn = data_filtered[data_filtered["country"] == cn]
        boxplot_df = data_cn[residual_cols]
        boxplot_df = boxplot_df.rename(columns=residual_labels)
        sel_ax = axes if (num_countries == 1) else axes[i]
        sel_ax.set_title(cn, fontdict=font)
        sel_ax.set_facecolor("w")
        sel_ax.set_ylim([ymin, ymax])
        sel_ax.tick_params(axis="both", which="major", labelsize=28)
        b = sns.boxplot(
            x="variable",
            y="value",
            data=pd.melt(boxplot_df),
            hue="variable",
            ax=sel_ax,
            whis=1.5,
            palette="tab10",
        )
        b.set_xlabel(" ")
        b.set_ylabel("Prediction Residuals", fontdict=font)
        if i > 0:
            b.set(yticklabels=[])
            b.set_ylabel(None)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    countries_str = "_".join(countries)
    plt.savefig(
        os.path.join(PATH_GRAPHS_DIR, f"box_residuals_{crop}_{countries_str}.jpg")
    )
    plt.close(fig)


def box_plots_metric(data, crop, countries, metric, metric_label, subplots_per_row=4):
    # Filter data based on selected countries and crop
    data_filtered = data[(data["country"].isin(countries)) & (data["crop"] == crop)]

    # Set up the figure for subplots
    num_countries = len(countries)
    num_rows = (num_countries + subplots_per_row - 1) // subplots_per_row
    fig, axes = plt.subplots(
        num_rows, subplots_per_row, figsize=(21, 4 * num_rows), sharey=True
    )
    axes = axes.flatten()

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
        ax.set_ylabel(metric_label if i % subplots_per_row == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.margins(x=0.1)
        ax.xaxis.grid(False)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    countries_str = "_".join(countries)
    plt.savefig(
        os.path.join(
            PATH_GRAPHS_DIR, f"box_metrics_{metric}_{crop}_{countries_str}.jpg"
        )
    )
    plt.close(fig)


def plot_year_results(data, crop, country, metric, metric_label):
    data_filtered = data[(data["crop"] == crop) & (data["country"] == country)]
    all_years = sorted(data_filtered[KEY_YEAR].unique())
    num_rows = 2
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 5 * num_rows), sharey=True)
    axes = axes.flatten()

    num_years = len(all_years)
    sel_years = [all_years[: int(num_years / 2)], all_years[int(num_years / 2) :]]
    for i in range(2):
        ax = axes[i]
        sns.barplot(
            x="year",
            y=metric,
            data=data_filtered[data_filtered[KEY_YEAR].isin(sel_years[i])],
            ax=ax,
            hue="model",
            palette="Set2",
            width=0.5,
            native_scale=True,
        )
        ax.set_title(f"{country} ({crop})")
        ax.set_xlabel("Year")
        ax.set_ylabel(metric_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.margins(x=0.1)
        ax.xaxis.grid(False)
        if i != 0:
            ax.get_legend().remove()

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(PATH_GRAPHS_DIR, f"{metric}_{crop}_{country}.jpg"))
    plt.close(fig)


if __name__ == "__main__":
    model_names = {
        "AverageYieldModel": "Naive",
        "LSTM": "LSTM",
        "SklearnRidge": "Ridge",
        "SklearnRF": "RF",
        # "LinearTrend": "Trend",
    }

    df_metrics = pd.DataFrame()
    df_residuals = pd.DataFrame()
    metrics = None
    residual_cols = None
    sel_crop_countries = {}
    for crop, country_codes in datasets.items():
        for cn in country_codes:
            run_name = crop + "_" + cn
            if not os.path.isdir(os.path.join(PATH_RESULTS_DIR, run_name)):
                continue

            if crop in sel_crop_countries:
                sel_crop_countries[crop] = sel_crop_countries[crop] + [cn]
            else:
                sel_crop_countries[crop] = [cn]

            df = compute_metrics(run_name, list(model_names.keys()))
            if metrics is None:
                metrics = list(df.columns)
            df.reset_index(inplace=True)
            df["country"] = cn
            df["crop"] = crop
            # NOTE: Mean Absolute Percentage Error (MAPE) is not converted to percentage.
            # This is because we follow the default from scikit-learn.
            # TODO: Remove this when MAPE is actually a percentage.
            df["mape"] = df["mape"] * 100
            df = df[["crop", "country", KEY_YEAR, "model"] + metrics]
            df_metrics = pd.concat([df_metrics, df], axis=0)

            df_r = get_prediction_residuals(run_name, model_names)
            df_r["country"] = cn
            df_r["crop"] = crop
            df_r.reset_index(inplace=True)
            if residual_cols is None:
                residual_cols = [c for c in df_r.columns if "res" in c]

            df_r = df_r[["crop", "country", KEY_LOC, KEY_YEAR] + residual_cols]
            df_residuals = pd.concat([df_residuals, df_r], axis=0)

    df_metrics["model"].replace(model_names, inplace=True)
    metric_labels = {
        "mape": "Mean Absolute Percentage Error",
        "normalized_rmse": "Normalized RMSE",
    }
    for cr, cns in sel_crop_countries.items():
        for met in metrics:
            print(cr, cns, met)
            box_plots_metric(
                df_metrics, cr, cns, met, metric_labels[met], subplots_per_row=len(cns)
            )

    residual_labels = {}
    model_short_names = list(model_names.values())
    for model_sname in model_short_names:
        residual_labels[model_sname + "_res"] = model_sname

    ymin = None
    ymax = None
    for c in residual_cols:
        if ymin is None:
            ymin = df_residuals[c].min()
        else:
            ymin = min(ymin, df_residuals[c].min())

        if ymax is None:
            ymax = df_residuals[c].max()
        else:
            ymax = max(ymax, df_residuals[c].max())

    for cr, cns in sel_crop_countries.items():
        box_plots_residuals(
            df_residuals,
            cr,
            cns,
            residual_cols,
            residual_labels,
            ymin,
            ymax,
            subplots_per_row=len(cns),
        )

    for met in metrics:
        for cr in sel_crop_countries:
            for cn in sel_crop_countries[cr]:
                plot_year_results(df_metrics, cr, cn, met, metric_labels[met])

    for metric in ["mape_country", "nrmse_country", "bars"]:
        plot_metrics(df_metrics, metric)

    df_metrics.to_csv(os.path.join(PATH_RESULTS_DIR, "cd_diagram.csv"))
