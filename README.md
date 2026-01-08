# DAT5501_Assignment_002
## Fertility, Contraception, Income & Happiness Data Analysis

This project analyses the relationships between fertility rates, contraceptive prevalence, income, happiness, marriage rates, and GDP across countries using multiple open datasets. It includes advanced forecasting, data merging, and visualisations to uncover demographic and socioeconomic trends globally.

## Project Overview
This repository comprises several Python scripts, each focused on a specific data analysis theme:

contraception_happiness_and_income.py: Explores how contraceptive prevalence relates to happiness and median income.

fertility_rates_forecast.py: Performs polynomial forecasting of global fertility rates up to 2050.

fertility_rates_vs_contraception_and_marriage.py: Analyses fertility rates in the context of contraception usage and marriage rates across countries.

fertility_rate_vs_happiness.py: Studies the association between fertility rates and self-reported happiness scores.

fertility_rate_vs_income_and_gdp.py: Investigates how fertility rates relate to income and GDP levels.

Each script loads relevant CSV files from the csv_files directory using portable path constructions to ensure environment independence. Data cleaning and processing functions (e.g., column renaming and numeric conversions) are consistently applied to guarantee reliable downstream analysis.

## Key Design Principles
Portability: File paths are constructed relative to the script location, making it straightforward to move the project across different file systems or platforms without adjustment.

Code Consistency: Functions with repeated roles across scripts, like rename_columns and convert_columns_to_numeric, follow consistent patterns to ensure predictability and ease of maintenance.

Resilience to Bad Data: Converting columns to numeric uses coercion, gracefully handling invalid or missing entries without breaking the data pipeline.

Data Integrity: Inner joins are used to merge datasets, ensuring only complete records common to all sources are kept, preventing illusory relationships from incomplete data.

Smoothed Aggregations: By grouping data at the country level and averaging over years, the analyses focus on meaningful trends without year-to-year noise.

Clear Visualisation Practices: Bubble chart sizes and colours are carefully scaled and chosen to improve interpretability, with consistent continent colour mappings enhancing comparative insight.

Structured Output: Visualisations are saved automatically into organised output folders, with directory creation handled dynamically to avoid file system errors.