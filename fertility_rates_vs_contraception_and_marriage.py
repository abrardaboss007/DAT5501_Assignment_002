import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

def load_csv_file(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "csv_files", filename)
    df = pd.read_csv(csv_path)
    return df

# Bring in relevant csv files
fertility_rate_df = load_csv_file("fertility_rate.csv")
contraceptive_prevalence_df = load_csv_file("contraceptive_prevalence.csv")
marriage_rate_df = load_csv_file("marriages_per_1000_people.csv")
countries_by_continent_df = load_csv_file("countries_by_continent.csv")

# Rename columns of specified dataframe according to rename_dict
# This will return a df with the renamed columns
def rename_columns(df, rename_dict):
    return df.rename(columns=rename_dict)

# Convert specified columns of a dataframe to numeric data type and return said dataframe
def convert_columns_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
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

# Merge all dataframes together
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

# Aggregate by country mean to reduce data points
summary_mean_df = fertility_contraception_marriage_df.groupby("country").agg({
    "contraceptive_prevalence": "mean",
    "fertility_rate": "mean",
    "marriage_rate": "mean"
}).reset_index()

# Merge continent info to summary dataframe
summary_with_continent_df = pd.merge(
    left = summary_mean_df, right = countries_by_continent_df,
    how='left', left_on='country', right_on='Country'
)

# Drop rows with missing continent info 
summary_with_continent_df = summary_with_continent_df.dropna(subset=["Continent"])

# Prepare data for plotting
x = summary_with_continent_df["contraceptive_prevalence"].to_numpy()
y = summary_with_continent_df["fertility_rate"].to_numpy()
bubble_sizes = summary_with_continent_df["marriage_rate"].to_numpy()

# Scale bubble sizes for better visibility
sizes = (bubble_sizes ** 2) * 10 

# Map continents to colors
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

# Create legend for continents
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

# Show plot in Streamlit
st.pyplot(fig)