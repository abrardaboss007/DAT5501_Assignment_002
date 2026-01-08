import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Loading csv files in this way ensures portability when moving across different envioronments
def load_csv_file(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "csv_files", filename)
    df = pd.read_csv(csv_path)
    return df

fertility_rate_df = load_csv_file("fertility_rate.csv")

# Aligns with personal naming convention and allows downstream code to be more readable and consistent and reliable without any long column names
# Functions also facilitate unit testing
def rename_columns(df, rename_dict):
    return df.rename(columns=rename_dict)

def convert_columns_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") # ensures that bad data can be handled gracefully by dropping later
    return df

fertility_rate_rename_dict = {
    "Entity": "country", 
    "Code": "code",
    "Year": "year",
    "Fertility rate (period), historical": "fertility_rate"
}
fertility_rate_numeric_cols = ["year", "fertility_rate"]

# For fertility_rate_df
fertility_rate_df = rename_columns(fertility_rate_df, fertility_rate_rename_dict)
fertility_rate_df = convert_columns_to_numeric(fertility_rate_df, fertility_rate_numeric_cols)

fertility_rate_df = fertility_rate_df.dropna(axis= 0, how="any")

# Focus forecast on *global* fertility trends post 1950 to capture modern demographic shifts
world_fertility_df = fertility_rate_df[fertility_rate_df["country"] == "World"]
world_fertility_df = world_fertility_df[world_fertility_df["year"] >= 1950]

train_df = world_fertility_df.iloc[0:-10] # training data
test_df = world_fertility_df.iloc[-10:] # test data

x_values_training = train_df["year"].to_numpy()
y_values_training = train_df["fertility_rate"].to_numpy()

# Predictions extend to 2050 to provide long-term forecasting beyond test set.
future_years = np.arange(test_df["year"].max() + 1, 2051)
future_predictions = {}
all_predictions = {}

fig1, ax1 = plt.subplots()
ax1.scatter(x_values_training, y_values_training, marker="o", s=7) # Plot scatter points for training data

# Colours chosen by myself to keep it distinguishable but subtle 
# Also to ensure the line colour for 2025 - 2050 is the same as 2015-2025 since matplotlib will change them otherwise
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

# Highlight actual test data in red to clearly separate it from predictions.
ax1.scatter(test_df["year"], test_df["fertility_rate"], color="red", marker="o", s=10, label="Test Data (Actual)") # Plot scatter points for test data

ax1.set_xlabel("Year")
ax1.set_xlim(1950,2050)
ax1.set_ylim(0,6)
ax1.set_ylabel("Fertility Rate")
ax1.set_title("World Fertility Rates with Polynomial Forecasts")
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
# ----------------------------------------------------------------------------------
# Evaluate best polynomial degree by Chi² per degree of freedom and BIC criterion,
# providing balance between fit quality and model complexity to avoid overfitting.
# ----------------------------------------------------------------------------------
chi2_per_dof_vals = []
bic_vals = []
degrees = []

y_true = test_df["fertility_rate"].to_numpy()
n = len(y_true)

for degree, y_pred in all_predictions.items():
    k = degree + 1  # Parameters include polynomial coefficients
    residuals = y_true - y_pred
    RSS = np.sum(residuals ** 2)
    
    dof = n - k
    chi2_per_dof = RSS / dof if dof > 0 else np.nan # Guard against division by zero if model is too complex relative to data.
    
    # BIC penalises model complexity stronger than AIC, fitting preference for simplicity.
    bic = n * np.log(RSS / n) + k * np.log(n)
    
    degrees.append(degree)
    chi2_per_dof_vals.append(chi2_per_dof)
    bic_vals.append(bic)

# Plot metrics on the same graph to visualise trade-offs clearly.
fig2, ax2 = plt.subplots(figsize = (12,8))
ax2.plot(degrees, chi2_per_dof_vals, marker='o', label='Chi² per DoF')
ax2.plot(degrees, bic_vals, marker='s', label='BIC')
ax2.set_xlabel('Polynomial Degree')
ax2.set_title('Chi² per DoF and BIC by Polynomial Degree')
ax2.legend()

# Selecting best model by minimal BIC to favor simplest adequate model.
best_index = np.argmin(bic_vals)
best_degree = degrees[best_index]
print(f"Best polynomial degree by BIC: {best_degree}")

# Refitting model on training data, filtering out problematic values in case of any,
# to provide final coefficients and uncertainties for reporting.
mask = (~np.isnan(x_values_training)) & (~np.isnan(y_values_training)) & (~np.isinf(x_values_training)) & (~np.isinf(y_values_training))
x_fit = x_values_training[mask]
y_fit = y_values_training[mask]

coeffs, cov = np.polyfit(x_fit, y_fit, deg=best_degree, cov=True)
print(f"Model coefficients for degree {best_degree}:", coeffs)
print(f"Covariance matrix:\n", cov)

# Extract uncertainties from covariance matrix diagonals to understand parameter confidence,
# important for interpreting how reliable predictions are.
param_uncertainties = np.sqrt(np.diag(cov))
print(f"Uncertainties in parameters:", param_uncertainties)

plt.show()