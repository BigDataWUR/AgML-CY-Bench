import pandas as pd

filename = '/path-to-results.csv'
outputfilename = 'output_tables.md'

df_results = pd.read_csv(filename)
df_results[['crop', 'country']] = df_results['crop_cn'].str.split('_', expand=True)
df_results.drop(columns=['crop_cn'], inplace=True)
df_results.set_index(['model', 'crop', 'country'], inplace=True)

crops = df_results.index.get_level_values('crop').unique()
metrics = df_results.columns.unique()

tables = {}
for crop in crops:
    tables[crop] = {}
    crop_df = df_results[df_results.index.get_level_values('crop').isin([crop])]
    crop_df = crop_df.groupby(["model", "crop", "country"]).agg({"normalized_rmse": "mean", "mape": "mean"})
    for metric in metrics:
        tables[crop][metric] = crop_df.reset_index().pivot_table(index=["crop", "country"], columns='model',
                                                                 values=metric)


# Function to format rows with the minimum value in bold
def format_row(row):
    min_value = row.min()
    return ' '.join([f"**{value:.3f}**" if value == min_value else f"{value:.3f}" for value in row])


# Construct the Markdown table
def df_to_markdown(df, formatted_df):
    # Define column headers
    headers = ['crop', 'country'] + df.columns.tolist()

    # Construct table
    table = []
    table.append('| ' + ' | '.join(headers) + ' |')
    table.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

    for idx, formatted_row in formatted_df.items():
        crop, country = idx
        row_values = [crop, country] + formatted_row.split()
        table.append(f'| ' + ' | '.join(row_values) + ' |')

    return '\n'.join(table)


# Open a file to write Markdown content
with open(outputfilename, 'w') as file:

    for crop, error_measures in tables.items():
        for error_measure, values in error_measures.items():
            df = tables[crop][error_measure]
            # Apply the formatting function to each row
            df_formatted = df.apply(format_row, axis=1)
            # Create the Markdown table
            markdown_table = df_to_markdown(df, df_formatted)
            file.write(f"## {crop} {error_measure}\n\n")
            file.write(markdown_table + "\n\n")