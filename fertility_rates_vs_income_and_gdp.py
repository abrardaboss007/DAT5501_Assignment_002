import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Loading csv files in this way ensures portability when moving across different environments
def load_csv_file(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "csv_files", filename)
    df = pd.read_csv(csv_path)
    return df

fertility_rate_df = load_csv_file("fertility_rate.csv")
median_income_df = load_csv_file("daily_median_income.csv")
gdp_df = load_csv_file("gdp.csv")

# Aligns with personal naming convention and allows downstream code to be more readable and consistent and reliable without any long column names
# Functions also facilitate unit testing
def rename_columns(df, rename_dict):
    return df.rename(columns=rename_dict)

def convert_columns_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # ensures that bad data can be handled gracefully by dropping later
    return df

fertility_rate_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "Fertility rate (period), historical": "fertility_rate"
}
fertility_rate_numeric_cols = ["year", "fertility_rate"]

median_income_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "Median (2021 prices)": "daily_median_income"
}
median_income_numeric_cols = ["year", "daily_median_income"]

gdp_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "GDP (constant 2015 US$)": "gdp"
}
gdp_numeric_cols = ["year", "gdp"]

# For fertility_rate_df
fertility_rate_df = rename_columns(fertility_rate_df, fertility_rate_rename_dict)
fertility_rate_df = convert_columns_to_numeric(fertility_rate_df, fertility_rate_numeric_cols)

# For median_income_df
median_income_df= rename_columns(median_income_df, median_income_rename_dict)
median_income_df = convert_columns_to_numeric(median_income_df, median_income_numeric_cols)
median_income_df["annual_median"] = median_income_df["daily_median_income"] * 365.25

# For gdp_df
gdp_df= rename_columns(gdp_df, gdp_rename_dict)
gdp_df = convert_columns_to_numeric(gdp_df, gdp_numeric_cols)


# Left joins ensure only complete records present in all datasets are kept,
# preventing mismatches or missing data from polluting analysis.
fertility_median_merged_df = pd.merge(left = median_income_df, right= fertility_rate_df, 
                                      how="left", on = ["code","country","year"])

fertility_median_gdp_merged_df = pd.merge(left = fertility_median_merged_df, right=gdp_df, 
                                          how="left", on = ["code","country","year"])

# Select years greater than 1950 and drop NA values
fertility_median_gdp_merged_df = fertility_median_gdp_merged_df[fertility_median_gdp_merged_df["year"] >= 1950]
fertility_median_gdp_merged_df = fertility_median_gdp_merged_df.dropna(axis = 0, how="any")

# Group by country to get mean values over years, smoothing year-to-year variability
# and enabling comparison at the country-level rather than individual years.
summary_mean_df = fertility_median_gdp_merged_df.groupby("country").agg({
    "fertility_rate": "mean",
    "annual_median": "mean",
    "gdp":"mean"
}).reset_index()

summary_mean_df = summary_mean_df.rename(columns={"fertility_rate":"mean_fertility_rate",
                                                  "annual_median":"mean_annual_median_income",
                                                  "gdp":"mean_gdp"})
#------------------------------------------------------------------------------------------------------------------
# Plot graph (lines 80-90)
#------------------------------------------------------------------------------------------------------------------
fig1, ax1 = plt.subplots()

x_values = summary_mean_df["mean_annual_median_income"].to_numpy()
y_values = summary_mean_df["mean_fertility_rate"].to_numpy()
bubble_sizes = summary_mean_df["mean_gdp"].to_numpy()

# Manually tuning size_scale to 5e-11 to keep bubble sizes visually meaningful 
# and avoid excessive overlap or minuscule dots, based on data magnitude.
size_scale = 5e-11 
sizes = bubble_sizes * size_scale


# Using raw GDP values for coloring adds an additional dimension of info without needing extra markers,
# while 'viridis' colormap offers perceptually uniform and visually accessible coloring.
scatter = ax1.scatter(
    x_values,y_values, s=sizes, 
    c=bubble_sizes,   cmap='viridis',
    alpha=0.7, edgecolors='w', linewidth=0.5
)

ax1.set_xlabel("Mean Annual Median Income")
ax1.set_ylabel("Mean Fertility Rate")
ax1.set_title("Fertility Rate vs Income sized and\n coloured by GDP")

# Including colorbar helps interpret bubble colors quantitatively,
# making the plot's three-variable representation clearer to viewers.
cbar = fig1.colorbar(scatter, ax=ax1)
cbar.set_label("Mean GDP")

# Saving figures in a controlled output folder keeps project artifacts organised,
# and handling folder creation ensures the save doesn't fail due to missing paths.
def save_figure(fig, fig_name, folder="output_plots", dpi=300):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, folder)
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, fig_name)
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')

save_figure(fig1, "fertility_rates_vs_income_and_gdp_plot")