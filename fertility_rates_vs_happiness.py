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
self_reported_happiness_df = load_csv_file("cantril_ladder_score.csv")
countries_by_continent_df = load_csv_file("countries_by_continent.csv")

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

self_reported_happiness_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "Cantril ladder score": "cantril_score"
}
self_reported_happiness_numeric_cols = ["year", "cantril_score"]

# For fertility_rate_df
fertility_rate_df = rename_columns(fertility_rate_df, fertility_rate_rename_dict)
fertility_rate_df = convert_columns_to_numeric(fertility_rate_df, fertility_rate_numeric_cols)

# For self_reported_happiness_df
self_reported_happiness_df = rename_columns(self_reported_happiness_df, self_reported_happiness_rename_dict)
self_reported_happiness_df = convert_columns_to_numeric(self_reported_happiness_df, self_reported_happiness_numeric_cols)

# Inner joins ensure only complete records present in all datasets are kept,
# preventing mismatches or missing data from polluting analysis.
merged_df = pd.merge(left = self_reported_happiness_df, right = fertility_rate_df, 
                     how = "left", on = ["code", "country", "year"])

merged_df= merged_df.dropna(axis = 0, subset=["code","cantril_score","fertility_rate"])
merged_df = merged_df[merged_df["year"] >= 1950]

# Group by country to get mean values over years, smoothing year-to-year variability
# and enabling comparison at the country-level rather than individual years.
summary_mean_df = merged_df.groupby("country").agg({
    "fertility_rate": "mean",
    "cantril_score": "mean"
}).reset_index()

summary_mean_df = summary_mean_df.rename(columns={"fertility_rate":"mean_fertility_rate",
                                                  "cantril_score":"mean_cantril_ladder_score"})

# Merge continent info to allow continent-level grouping in visualisation
# Dropping rows without continent info to avoid misleading or incomplete color categorisation.
summary_with_continent_df = pd.merge(
    left = summary_mean_df, right = countries_by_continent_df,
    how='left', left_on='country', right_on='Country'
)
summary_with_continent_df = summary_with_continent_df.dropna(subset=["Continent"])

from matplotlib.lines import Line2D

fig1, ax1 = plt.subplots(figsize=(10,6))
x_values = summary_with_continent_df["mean_cantril_ladder_score"].to_numpy()
y_values = summary_with_continent_df["mean_fertility_rate"].to_numpy()

# Use consistent palette and mapping to ensure colors represent continents distinctly
# making interpretation straightforward and consistent across plots.
continents = summary_with_continent_df["Continent"].unique()
palette = sns.color_palette("tab10", len(continents))  
continent_color_dict = dict(zip(continents, palette))
colors = summary_with_continent_df["Continent"].map(continent_color_dict)

# scale for visibility 
sizes = 100

# Scatter plot
scatter = ax1.scatter(
    x_values, y_values,
    s=sizes,
    c=colors,
    alpha=0.7, 
    edgecolors='w', 
    linewidth=0.5
)

ax1.set_xlabel("Mean Cantril Ladder Score")
ax1.set_ylabel("Mean Fertility Rate")
ax1.set_title("Mean Fertility Rate vs. Mean Cantril Ladder Score by Continent")

# Custom legend for continents avoids clutter and ensures clear association of colors.
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=continent,
           markerfacecolor=color, markersize=10)
    for continent, color in continent_color_dict.items()
]

ax1.legend(handles=legend_elements, title="Continent", loc='best')

plt.tight_layout()

# Saving figures in a controlled output folder keeps project artifacts organised,
# and handling folder creation ensures the save doesn't fail due to missing paths.
def save_figure(fig, fig_name, folder="output_plots", dpi=300):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, folder)
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, fig_name)
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')

save_figure(fig1, "fertility_rates_vs_happiness_plot")