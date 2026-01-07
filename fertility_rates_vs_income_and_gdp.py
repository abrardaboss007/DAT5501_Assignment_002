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
median_income_df = load_csv_file("daily_median_income.csv")
gdp_df = load_csv_file("gdp.csv")

#------------------------------------------------------------------------------------------------------------------
# Data cleansing for fertility_rate_df (lines 14-23)
#------------------------------------------------------------------------------------------------------------------
# Rename colums for ease
fertility_rate_df = fertility_rate_df.rename(columns={"Entity": "country", 
                                                      "Code":"code",
                                                      "Year":"year",
                                                      "Fertility rate (period), historical": "fertility_rate"})

# Convert some columns to numeric data type
fertility_rate_df["year"] = pd.to_numeric(fertility_rate_df["year"], errors="coerce")
fertility_rate_df["fertility_rate"] = pd.to_numeric(fertility_rate_df["fertility_rate"], errors="coerce")

#------------------------------------------------------------------------------------------------------------------
# Data cleansing for median_income_df (lines 26-39)
#------------------------------------------------------------------------------------------------------------------
# Rename colums for ease
median_income_df = median_income_df.rename(columns={"Entity":"country",
                                                    "Code":"code",
                                                    "Year":"year",
                                                    "Median (2021 prices)":"daily_median"})

# Convert some columns to numeric data type
median_income_df["year"] = pd.to_numeric(median_income_df["year"], errors="coerce")
median_income_df["daily_median"] = pd.to_numeric(median_income_df["daily_median"], errors="coerce")

median_income_df["annual_median"] = median_income_df["daily_median"] * 365.25
#------------------------------------------------------------------------------------------------------------------
# Data cleansing for gdp_df (lines 40-49)
#------------------------------------------------------------------------------------------------------------------
# Rename colums for ease
gdp_df = gdp_df.rename(columns={"Entity":"country",
                                "Code":"code",
                                "Year":"year",
                                "GDP (constant 2015 US$)":"gdp"})

# Convert some columns to numeric data type
gdp_df["year"] = pd.to_numeric(gdp_df["year"], errors="coerce")
gdp_df["gdp"] = pd.to_numeric(gdp_df["gdp"], errors="coerce")

# Merge all three df's into one df and drop na values
fertility_median_merged_df = pd.merge(left = median_income_df, right= fertility_rate_df, how="left", left_on=["code","country","year"], right_on=["code","country","year"])
fertility_median_gdp_merged_df = pd.merge(left = fertility_median_merged_df, right=gdp_df, how="left", left_on=["code","country","year"], right_on=["code","country","year"])
fertility_median_gdp_merged_df = fertility_median_gdp_merged_df.dropna(axis = 0, how="any")

# Select data from last ten years only
last_ten_years = [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
fertility_median_gdp_merged_df = fertility_median_gdp_merged_df[fertility_median_gdp_merged_df["year"].isin(last_ten_years)]

# Calculate mean fertility rate for each country in the last 10 years
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

# Scale raw GDP for bubble sizes 
size_scale = 5e-11 # The scale I considered to be optimum
sizes = bubble_sizes * size_scale

scatter = ax1.scatter(
    x_values,y_values, s=sizes, 
    c=bubble_sizes,            # color bubbles by GDP (raw values)
    cmap='viridis',            # colormap for colors
    alpha=0.7, edgecolors='w', linewidth=0.5
)

ax1.set_xlabel("Mean Annual Median Income")
ax1.set_ylabel("Mean Fertility Rate")
ax1.set_title("Fertility Rate vs Income sized and coloured by GDP")

# Add colorbar for bubble color
cbar = fig1.colorbar(scatter, ax=ax1)
cbar.set_label("Mean GDP")

st.pyplot(fig1)


