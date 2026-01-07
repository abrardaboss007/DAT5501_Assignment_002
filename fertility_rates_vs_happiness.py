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
fertility_rate_df = load_csv_file("fertility_rates.csv")
self_reported_happiness_df = load_csv_file("cantril_ladder_score.csv")
countries_by_continent_df = load_csv_file("countries_by_continent.csv")
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
# Data cleansing for self_reported_happiness_df (lines 27-35)
#------------------------------------------------------------------------------------------------------------------
# Rename colums for ease
self_reported_happiness_df = self_reported_happiness_df.rename(columns={"Entity": "country",
                                                                        "Code":"code",
                                                                        "Year":"year",                                                                       
                                                                        "Cantril ladder score": "score"})

# Convert some columns to numeric data type
self_reported_happiness_df["year"] = pd.to_numeric(self_reported_happiness_df["year"], errors="coerce")
self_reported_happiness_df["score"] = pd.to_numeric(self_reported_happiness_df["score"], errors="coerce")

#------------------------------------------------------------------------------------------------------------------
# Merge into one df with further data cleansing (lines 40-60)
#------------------------------------------------------------------------------------------------------------------
#Creating one df with relevant information from fertility_rate_df and self_reported_happiness_df
# Applying left join with self_reported_happiness_df on the left due to it having less years with joining on multiple conditions (maybe articulate better)
merged_df = pd.merge(left = self_reported_happiness_df, right = fertility_rate_df, how = "left", left_on = ["code", "country", "year"], right_on=["code", "country", "year"])

# Remove entries with no country code (e.g. entries where country value is 'Africa', 'Asia', 'Upper middle income countries' etc.) as well as countries with no cantril ladder score or fertility rate value for a given year
merged_df= merged_df.dropna(axis = 0, subset=["code","score","fertility_rate"])

# Select data from last ten years only
last_ten_years = [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
merged_df = merged_df[merged_df["year"].isin(last_ten_years)]

# Calculate mean fertility rate for each country in the last 10 years
summary_mean_df = merged_df.groupby("country").agg({
    "fertility_rate": "mean",
    "score": "mean"
}).reset_index()

summary_mean_df = summary_mean_df.rename(columns={"fertility_rate":"mean_fertility_rate",
                                                  "score":"mean_cantril_ladder_score"})

#------------------------------------------------------------------------------------------------------------------
# Plot graph (lines 63-70)
#------------------------------------------------------------------------------------------------------------------
fig1, ax1 = plt.subplots()
x_values = summary_mean_df["mean_cantril_ladder_score"].to_numpy()
y_values = summary_mean_df["mean_fertility_rate"].to_numpy()
ax1.scatter(x_values, y_values)
ax1.set_xlabel("Mean cantril ladder score")
ax1.set_ylabel("Mean fertility rate")
ax1.set_title("Last 10 years")
st.pyplot(fig1)


