import pandas as pd

# Read the CSV files
df_nrmse = pd.read_csv("Wheat_NRMSE.csv", index_col=0)
df_mape = pd.read_csv("Wheat_MAPE.csv", index_col=0)

# Replace values larger than 100 with NaN
df_nrmse[df_nrmse > 1000] = float("nan")
df_mape[df_mape > 1000] = float("nan")

# Transpose the dataframes to swap rows and columns
df_nrmse_transposed = df_nrmse.transpose()
df_mape_transposed = df_mape.transpose()

# Find the minimum values for NRMSE and MAPE in each row
min_nrmse_values = df_nrmse_transposed.min(axis=1)
min_mape_values = df_mape_transposed.min(axis=1)

# Mark the minimum NRMSE and MAPE values in each row as bold
for idx, row in df_nrmse_transposed.iterrows():
    min_nrmse_val = min_nrmse_values[idx]
    min_mape_val = min_mape_values[idx]
    df_nrmse_transposed.loc[idx] = [
        f"\\textbf{{{val:.3f}}}" if val == min_nrmse_val else f"{val:.3f}"
        for val in row
    ]
    df_mape_transposed.loc[idx] = [
        f"\\textbf{{{val:.3f}}}" if val == min_mape_val else f"{val:.3f}"
        for val in df_mape_transposed.loc[idx]
    ]

# Combine the NRMSE and MAPE dataframes with a vertical line between them
combined_df = pd.concat(
    [
        df_nrmse_transposed,
        pd.DataFrame(
            {" ": ["\\vline"] * len(df_nrmse_transposed.index)},
            index=df_nrmse_transposed.index,
        ),
        df_mape_transposed,
    ],
    axis=1,
)

# Add column names as the first row of the dataframe
combined_df.columns = pd.MultiIndex.from_tuples(
    [(col, "") for col in combined_df.columns]
)

# Convert dataframe to LaTeX table format with header
latex_table = combined_df.to_latex(escape=False, header=True)

# Write LaTeX code to a file
with open("Maize.tex", "w") as file:
    file.write(latex_table)
