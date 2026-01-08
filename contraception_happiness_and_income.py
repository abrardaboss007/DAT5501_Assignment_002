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

contraceptive_prevalence_df = load_csv_file("contraceptive_prevalence.csv")
self_reported_happiness_df = load_csv_file("cantril_ladder_score.csv")
median_income_df = load_csv_file("daily_median_income.csv")
countries_by_continent_df = load_csv_file("countries_by_continent.csv")

# Aligns with personal naming convention and allows downstream code to be more readable and consistent and reliable without any long column names
# Functions also facilitate unit testing
def rename_columns(df, rename_dict):
    return df.rename(columns=rename_dict)


def convert_columns_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # ensures that bad data can be handled gracefully by dropping later
    return df

contraceptive_prevalence_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    f"Contraceptive prevalence, any method (% of married women ages 15-49)": "contraceptive_prevalence",
    "World region according to OWID":"world_region"
}
contraceptive_prevalence_numeric_cols = ["year", "contraceptive_prevalence"]

self_reported_happiness_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "Cantril ladder score": "cantril_score"
}
self_reported_happiness_numeric_cols = ["year", "cantril_score"]

median_income_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "Median (2021 prices)": "daily_median_income"
}
median_income_numeric_cols = ["year", "daily_median_income"]

# For contraceptive_prevalence_df
contraceptive_prevalence_df = rename_columns(contraceptive_prevalence_df, contraceptive_prevalence_rename_dict)
contraceptive_prevalence_df = convert_columns_to_numeric(contraceptive_prevalence_df, contraceptive_prevalence_numeric_cols)

# For self_report_happiness_df
self_reported_happiness_df = rename_columns(self_reported_happiness_df, self_reported_happiness_rename_dict)
self_reported_happiness_df = convert_columns_to_numeric(self_reported_happiness_df, self_reported_happiness_numeric_cols)

# For median_income_df
median_income_df= rename_columns(median_income_df, median_income_rename_dict)
median_income_df = convert_columns_to_numeric(median_income_df, median_income_numeric_cols)
median_income_df["annual_median"] = median_income_df["daily_median_income"] * 365.25

# Inner joins ensure only complete records present in all datasets are kept,
# preventing mismatches or missing data from polluting analysis.
contraception_happiness_df = pd.merge(
    left = contraceptive_prevalence_df, right= self_reported_happiness_df,
    how = 'inner', on = ["code","country","year"]
)

contraception_happiness_income_df  = pd.merge(
    left = contraception_happiness_df , right= median_income_df,
    how = 'inner', on = ["code","country","year"]
)

contraception_happiness_income_df = contraception_happiness_income_df[contraception_happiness_income_df["year"] >= 1950]
contraception_happiness_income_df = contraception_happiness_income_df.dropna(axis = 0, subset=["contraceptive_prevalence","cantril_score","daily_median_income"])

# Group by country to get mean values over years, smoothing year-to-year variability
# and enabling comparison at the country-level rather than individual years.
summary_mean_df = contraception_happiness_income_df.groupby("country").agg({
    "contraceptive_prevalence": "mean",
    "cantril_score": "mean",
    "annual_median":"mean"
}).reset_index()


# Merge continent info to allow continent-level grouping in visualisation
# Dropping rows without continent info to avoid misleading or incomplete color categorisation.
summary_with_continent_df = pd.merge(
    left = summary_mean_df, right = countries_by_continent_df,
    how='left', left_on='country', right_on='Country'
)
summary_with_continent_df = summary_with_continent_df.dropna(subset=["Continent"])


fig1, ax1 = plt.subplots(figsize=(12,8))

x_values = summary_with_continent_df["annual_median"].to_numpy()
y_values = summary_with_continent_df["cantril_score"].to_numpy()
bubble_sizes = summary_with_continent_df["contraceptive_prevalence"].to_numpy()

# Scale bubble sizes for better visibility
sizes = (bubble_sizes ** 2) * 0.10 

# Use consistent palette and mapping to ensure colors represent continents distinctly
# making interpretation straightforward and consistent across plots.
continents = summary_with_continent_df["Continent"].unique()
palette = sns.color_palette("tab10", len(continents))
continent_color_dict = dict(zip(continents, palette))

colors = summary_with_continent_df["Continent"].map(continent_color_dict)

scatter = ax1.scatter(
    x_values, y_values, s=sizes, c=colors,
    alpha=0.7, edgecolors='w', linewidth=0.5
)

ax1.set_xlabel("Average Annual Median Income")
ax1.set_ylabel("Mean Self-Reported Happiness Score (Cantril Ladder)")
ax1.set_title("Mean Happiness vs Income sized by Contraceptive Prevalence")

# Custom legend for continents avoids clutter and ensures clear association of colors.
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=continent,
           markerfacecolor=color, markersize=10)
    for continent, color in continent_color_dict.items()
]

ax1.legend(handles=legend_elements, title="Continent", loc='best')

plt.tight_layout()
plt.show()