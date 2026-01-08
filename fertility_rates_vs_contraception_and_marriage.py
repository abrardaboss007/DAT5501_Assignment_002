import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# Loading csv files in this way ensures portability when moving across different environments
def load_csv_file(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "csv_files", filename)
    df = pd.read_csv(csv_path)
    return df

fertility_rate_df = load_csv_file("fertility_rate.csv")
contraceptive_prevalence_df = load_csv_file("contraceptive_prevalence.csv")
marriage_rate_df = load_csv_file("marriages_per_1000_people.csv")
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

contraceptive_prevalence_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    f"Contraceptive prevalence, any method (% of married women ages 15-49)": "contraceptive_prevalence",
    "World region according to OWID":"world_region"
}
contraceptive_prevalence_numeric_cols = ["year", "contraceptive_prevalence"]

marriage_rate_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "Crude marriage rate": "marriage_rate"
}
marriage_rate_numeric_cols = ["year", "marriage_rate"]

# For fertility_rate_df
fertility_rate_df = rename_columns(fertility_rate_df, fertility_rate_rename_dict)
fertility_rate_df = convert_columns_to_numeric(fertility_rate_df, fertility_rate_numeric_cols)

# For contraceptive_prevalence_df
contraceptive_prevalence_df = rename_columns(contraceptive_prevalence_df, contraceptive_prevalence_rename_dict)
contraceptive_prevalence_df = convert_columns_to_numeric(contraceptive_prevalence_df, contraceptive_prevalence_numeric_cols)

# For marriage_rate_df
marriage_rate_df = rename_columns(marriage_rate_df, marriage_rate_rename_dict)
marriage_rate_df = convert_columns_to_numeric(marriage_rate_df, marriage_rate_numeric_cols)

# Inner joins ensure only complete records present in all datasets are kept,
# preventing mismatches or missing data from polluting analysis.
fertility_contraception_df = pd.merge(
    left = contraceptive_prevalence_df, right= fertility_rate_df,
    how = 'inner', on = ["code","country","year"]
)

fertility_contraception_marriage_df = pd.merge(
    left = fertility_contraception_df, right= marriage_rate_df,
    how = 'inner', on = ["code","country","year"]
)
# Select years greater than 1950 and drop NA values
fertility_contraception_marriage_df = fertility_contraception_marriage_df[fertility_contraception_marriage_df["year"] >= 1950]
fertility_contraception_marriage_df = fertility_contraception_marriage_df.dropna(axis = 0, subset=["contraceptive_prevalence","marriage_rate","fertility_rate"])

# Group by country to get mean values over years, smoothing year-to-year variability
# and enabling comparison at the country-level rather than individual years.
summary_mean_df = fertility_contraception_marriage_df.groupby("country").agg({
    "contraceptive_prevalence": "mean",
    "fertility_rate": "mean",
    "marriage_rate": "mean"
}).reset_index()

# Merge continent info to allow continent-level grouping in visualisation
# Dropping rows without continent info to avoid misleading or incomplete color categorisation.
summary_with_continent_df = pd.merge(
    left = summary_mean_df, right = countries_by_continent_df,
    how='left', left_on='country', right_on='Country'
)

summary_with_continent_df = summary_with_continent_df.dropna(subset=["Continent"])

# Prepare data for plotting
x = summary_with_continent_df["contraceptive_prevalence"].to_numpy()
y = summary_with_continent_df["fertility_rate"].to_numpy()
bubble_sizes = summary_with_continent_df["marriage_rate"].to_numpy()

# Scale bubble sizes for better visibility
sizes = (bubble_sizes ** 2) * 10 

# Use consistent palette and mapping to ensure colors represent continents distinctly
# making interpretation straightforward and consistent across plots.
continents = summary_with_continent_df["Continent"].unique()
palette = sns.color_palette("tab10", n_colors=len(continents))
continent_color_dict = dict(zip(continents, palette))
colors = summary_with_continent_df["Continent"].map(continent_color_dict)

# Plot bubble chart
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    x, y, s=sizes, c=colors,
    alpha=0.7, edgecolors="w", linewidth=0.5
)

ax.set_xlabel("Mean Contraceptive Prevalence (%)")
ax.set_ylabel("Mean Fertility Rate")
ax.set_title("Fertility Rate vs Contraceptive Prevalence \nBubble size = Marriage rate, Colored by Continent")

# Custom legend for continents avoids clutter and ensures clear association of colors.
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=continent,
        markerfacecolor=color,
        markersize=10,
    )
    for continent, color in continent_color_dict.items()
]
ax.legend(handles=legend_elements, title="Continent", loc="best")

plt.tight_layout()

# Saving figures in a controlled output folder keeps project artifacts organised,
# and handling folder creation ensures the save doesn't fail due to missing paths.
def save_figure(fig, fig_name, folder="output_plots", dpi=300):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, folder)
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, fig_name)
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')

save_figure(fig, "fertility_rates_vs_contraception_and_marriage_plot")

