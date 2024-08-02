import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import pymannkendall as trend_mk


def estimate_trend(years, values, target_year):
    assert len(values) >= 5

    trend_x = add_constant(years)
    linear_trend_est = OLS(values, trend_x).fit()

    pred_x = np.array([target_year]).reshape((1, 1))
    pred_x = add_constant(pred_x, has_constant="add")

    return linear_trend_est.predict(pred_x)


def find_optimal_trend_window(years_values, window_years, extend_forward=False):
    min_p = float("inf")
    opt_trend_years = None
    for i in range(5, min(10, len(window_years)) + 1):
        # should the search window be extended forward, i.e. towards later years
        if (extend_forward):
            trend_x = window_years[:i]
        else:
            trend_x = window_years[-i:]

        trend_y = years_values[np.in1d(years_values[:, 0], trend_x)][:, 1]
        # print(sel_yr, trend_x, trend_y)
        result = trend_mk.original_test(trend_y)
        # select based on p-value, lower the better
        if result.h and (result.p < min_p):
            min_p = result.p
            opt_trend_years = trend_x

    return opt_trend_years


def detrend_values(df, value_col="yield"):
    adm_id = df["adm_id"].unique()[0]
    original_values = df[["year", value_col]].values
    trend_values = np.zeros(original_values.shape)
    years = sorted(df["year"].unique())
    for i, sel_yr in enumerate(years):
        lt_sel_yr = [yr for yr in years if yr < sel_yr]
        gt_sel_yr = [yr for yr in years if yr > sel_yr]

        # Case 1: Not enough years to estimate trend
        if (len(lt_sel_yr) < 5) and (len(gt_sel_yr) < 5):
            trend = original_values[:, 1].mean()
        else:
            trend = None
            # Case 2: Estimate trend using years before
            window_years = find_optimal_trend_window(original_values, lt_sel_yr, extend_forward=False)
            if (window_years is not None):
                window_values = original_values[np.in1d(original_values[:, 0], window_years)][:, 1]
                trend = estimate_trend(window_years, window_values, sel_yr)[0]
            else:
                # Case 3: Estimate trend using years after
                window_years = find_optimal_trend_window(original_values, gt_sel_yr, extend_forward=True)
                if (window_years is not None):
                    window_values = original_values[np.in1d(original_values[:, 0], window_years)][:, 1]
                    trend = estimate_trend(window_years, window_values, sel_yr)[0]

            # Case 4: No significant trend exists
            if (trend is None):
                trend = original_values[:, 1].mean()

        # print(adm_id, sel_yr, trend, sel_case)
        trend_values[i, 0] = sel_yr
        trend_values[i, 1] = trend

    trend_cols = ["year", value_col + "_trend"]
    trend_df = pd.DataFrame(trend_values, columns=trend_cols)
    trend_df["adm_id"] = adm_id
    trend_df = trend_df.astype({"year" : int})
    detrended_df = trend_df.merge(df, on=["adm_id", "year"])
    detrended_df[value_col + "_res"] = detrended_df[value_col] - detrended_df[value_col + "_trend"]
    detrended_df = detrended_df[["adm_id", "year", "yield", "yield_trend", "yield_res"]]

    # print(detrended_df.head(20))
    return detrended_df


yield_df = pd.read_csv("C:/Users/paude006/Documents/git-repos/AgML-crop-yield-forecasting/cybench/data/maize/NL/yield_maize_NL.csv")
yield_df = yield_df[yield_df["harvest_year"] >= 2001]
yield_df = yield_df.rename(columns={"harvest_year" : "year"})
yield_df = yield_df[["adm_id", "year", "yield"]]
print(len(yield_df.index))

admin_ids = yield_df["adm_id"].unique()
yield_res_df = pd.DataFrame()
for adm_id in admin_ids:
    yield_sel_df = yield_df[yield_df["adm_id"] == adm_id]
    trend_df = detrend_values(yield_sel_df)
    yield_res_df = pd.concat([yield_res_df, trend_df], axis=0)

print(len(yield_res_df.index))
print(yield_res_df.head(20))
