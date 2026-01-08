import os
import unittest
import pandas as pd
import matplotlib.pyplot as plt

import contraception_happiness_and_income
import fertility_rates_forecast
import fertility_rates_vs_contraception_and_marriage
import fertility_rates_vs_happiness
import fertility_rates_vs_income_and_gdp


class TestFertilityRate(unittest.TestCase):
    """
    A unified test case for core utility functions used across all modules.

    Rationale:
    The code base has very similar or identical utility functions across files (load_csv_file,
    rename_columns, convert_columns_to_numeric, save_figure). Hence, we test these once,
    parameterising inputs where applicable to cover typical usage. This approach prevents duplication
    and keeps tests maintainable.
    """
    
    def test_csv_file(self): # Use fertility_rates_forecast here but any file can be used
        # File that is supposed to exist for testing
        self.existing_file = "fertility_rate.csv"

        # A file that is NOT supposed to exist
        self.non_existent_file = "this_file_is_not_real.csv"

        # Test loading a CSV file that exists returns a DataFrame
        existing_df = fertility_rates_forecast.load_csv_file(self.existing_file)
        
        self.assertIsInstance(existing_df, pd.DataFrame)
        self.assertFalse(existing_df.empty, "DataFrame loaded should not be empty")

        with self.assertRaises(FileNotFoundError):
            # Test loading a non-existent file raises FileNotFoundError
            fertility_rates_forecast.load_csv_file(self.non_existent_file)

    def test_save_figure_creates_file(self):  # Use contraception_happiness_and_income here but any file can be used
        # Create a dummy figure
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        
        # Define test file name and folder
        test_folder = "test_output"
        test_filename = "test_plot.png"
        
        # Call save_figure
        contraception_happiness_and_income.save_figure(fig, test_filename, folder=test_folder)
        
        # Build expected file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        expected_path = os.path.join(current_dir, test_folder, test_filename)
        
        # Check if file exists
        self.assertTrue(os.path.exists(expected_path), f"File {expected_path} does not exist")
        
        # Clean up: close figure and remove test file
        plt.close(fig)
        os.remove(expected_path)
        os.rmdir(os.path.join(current_dir, test_folder))  # remove test directory if empty
    
    def test_rename_columns(self): # Use fertility_rates_vs_happiness but any file can be used
        # Create a sample DataFrame with original column names
        data = {
            "Entity": ["CountryA", "CountryB"],
            "Code": ["CTA", "CTB"],
            "Year": [2000, 2001],
            "Fertility rate (period), historical": [2.1, 2.5]
        }
        df = pd.DataFrame(data)

        # Apply renaming
        renamed_df = fertility_rates_vs_happiness.rename_columns(df, fertility_rates_vs_happiness.fertility_rate_rename_dict)

        # Assert columns are renamed properly
        expected_columns = ["country", "code", "year", "fertility_rate"]
        self.assertListEqual(list(renamed_df.columns), expected_columns)

        # Also check data consistency for one row
        self.assertEqual(renamed_df.loc[0, "country"], "CountryA")
        self.assertEqual(renamed_df.loc[1, "fertility_rate"], 2.5)

    def test_convert_columns_to_numeric(self): # Use fertility_rates_vs_income_and_gdp but any file can be used
        # Create DataFrame with columns as strings, including some non-numeric bad data
        data = {
            "year": ["2000", "2001", "not_a_year"],
            "fertility_rate": ["2.1", "bad_data", "3.4"],
            "country": ["CountryA", "CountryB", "CountryC"]
        }
        df = pd.DataFrame(data)

        # Convert columns to numeric
        converted_df = fertility_rates_vs_income_and_gdp.convert_columns_to_numeric(df, fertility_rates_vs_income_and_gdp.fertility_rate_numeric_cols)

        # Check that the columns are floats (numeric dtype)
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_df["year"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_df["fertility_rate"]))

        # Check that bad data is coerced to NaN
        self.assertTrue(pd.isna(converted_df.loc[2, "year"]))  # "not_a_year" -> NaN
        self.assertTrue(pd.isna(converted_df.loc[1, "fertility_rate"]))  # "bad_data" -> NaN

        # Check that good data remains unchanged (but converted to numeric)
        self.assertEqual(converted_df.loc[0, "year"], 2000)
        self.assertEqual(converted_df.loc[2, "fertility_rate"], 3.4)


if __name__ == "__main__":
    unittest.main()


