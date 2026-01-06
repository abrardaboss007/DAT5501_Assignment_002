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

fertility_rate_df = load_csv_file("fertility_rate.csv")
#------------------------------------------------------------------------------------------------------------------
# Data cleansing for fertility_rate_df (lines 14-23)
#------------------------------------------------------------------------------------------------------------------
# Rename colums for ease
fertility_rate_df = fertility_rate_df.rename(columns={"Entity": "country", 
                                                      "Code":"code",
                                                      "Year":"year",
                                                      "Fertility rate (period), historical": "fertility_rate"})

# Convert some columns to numeric data type and drop NA values
fertility_rate_df["year"] = pd.to_numeric(fertility_rate_df["year"], errors="coerce")
fertility_rate_df["fertility_rate"] = pd.to_numeric(fertility_rate_df["fertility_rate"], errors="coerce")
fertility_rate_df = fertility_rate_df.dropna(axis= 0, how="any")

# Select only fertility rates for the world
world_fertility_df = fertility_rate_df[fertility_rate_df["country"] == "World"]
world_fertility_df = world_fertility_df[world_fertility_df["year"] >= 1950]

train_df = world_fertility_df.iloc[0:-10] # training data
test_df = world_fertility_df.iloc[-10:] # test data

x_values_training = train_df["year"].to_numpy()
y_values_training = train_df["fertility_rate"].to_numpy()


future_years = np.arange(test_df["year"].max() + 1, 2051)
future_predictions = {}

all_predictions = {}

fig1, ax1 = plt.subplots()
ax1.scatter(x_values_training, y_values_training, marker="x", s=7) # Plot scatter points for training data

colours = ["#00AA55", "#1E90FF", "#0000E0", "#E26A6A", "#E000E0", "#939393BF", "#AA8F00", "#FF4500", "#2B9999"]
for degree in range(1, 10):
    # Obtain coefficients for each degree
    coeffs= np.polyfit(x_values_training, y_values_training, deg=degree)
    poly_test = np.poly1d(coeffs)
    poly_future = np.poly1d(coeffs)

    x_test = test_df["year"].to_numpy()
    y_pred = poly_test(x_test) # Line fitting based on x_test values
    all_predictions[degree] = y_pred 

    y_future_pred = poly_future(future_years)
    future_predictions[degree] = y_future_pred

    ax1.plot(x_test, y_pred, label=f"Degree {degree} Prediction", color = colours[degree - 1]) # Plot predicted lines for polynomial degrees 1 to 9
    ax1.plot(future_years, y_future_pred, linestyle="--", label=f"Degree {degree} Forecast to 2050", color = colours[degree -1])

ax1.scatter(test_df["year"], test_df["fertility_rate"], color="red", marker="o", s=10, label="Test Data (Actual)") # Plot scatter points for test data
ax1.set_xlabel("Year")
ax1.set_xlim(1950,2050)
ax1.set_ylim(0,6)
ax1.set_ylabel("Fertility Rate")
ax1.set_title("Fertility rates in USA with Polynomial Forecasts")
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

st.pyplot(fig1)

# ------------------------------------------------------------------------------------------------
# Variables that will help in plotting the graph
chi2_per_dof_vals = []
bic_vals = []
degrees = []

y_true = test_df["fertility_rate"].to_numpy()
n = len(y_true)

for degree, y_pred in all_predictions.items():
    k = degree + 1  # number of parameters
    residuals = y_true - y_pred
    RSS = np.sum(residuals ** 2)
    
    dof = n - k
    chi2_per_dof = RSS / dof if dof > 0 else np.nan
    
    bic = n * np.log(RSS / n) + k * np.log(n)
    
    degrees.append(degree)
    chi2_per_dof_vals.append(chi2_per_dof)
    bic_vals.append(bic)
    

# Plot Chi2 per DoF and BIC by polynomial degree on same graph
fig2, ax2 = plt.subplots()
ax2.plot(degrees, chi2_per_dof_vals, marker='o', label='Chi² per DoF')
ax2.plot(degrees, bic_vals, marker='s', label='BIC')
ax2.set_xlabel('Polynomial Degree')
ax2.set_title('Chi² per DoF and BIC by Polynomial Degree')
ax2.legend()
st.pyplot(fig2)

# Inform user of best model
best_index = np.argmin(bic_vals)
best_degree = degrees[best_index]
st.write(f"Best polynomial degree by BIC: {best_degree}")

# Inform user of coefficient of best model and uncertainties in parameters
mask = (~np.isnan(x_values_training)) & (~np.isnan(y_values_training)) & (~np.isinf(x_values_training)) & (~np.isinf(y_values_training))
x_fit = x_values_training[mask]
y_fit = y_values_training[mask]

coeffs, cov = np.polyfit(x_fit, y_fit, deg=best_degree, cov=True)
st.write(f"Model coefficients for degree {best_degree}:", coeffs)
st.write(f"Covariance matrix:\n", cov)

# Parameter uncertainties are the sqrt of diagonal cov elements
param_uncertainties = np.sqrt(np.diag(cov))
st.write(f"Uncertainties in parameters:", param_uncertainties)