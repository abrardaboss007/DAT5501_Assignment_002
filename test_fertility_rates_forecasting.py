import unittest
import pandas as pd
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
#project_root = os.path.normpath(os.path.join(current_dir))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import fertility_rates_forecasting as fertility_rates_forecasting


class TestFertilityRateForecasting(unittest.TestCase):
    def test_csv_file(self):
        # File that is supposed to exist for testing
        self.existing_file = "fertility_rate.csv"

        # A file that is NOT supposed to exist
        self.non_existent_file = "this_file_is_not_real.csv"

        # Test loading a CSV file that exists returns a DataFrame
        existing_df = fertility_rates_forecasting.load_csv_file(self.existing_file)
        
        self.assertIsInstance(existing_df, pd.DataFrame)
        self.assertFalse(existing_df.empty, "DataFrame loaded should not be empty")

        with self.assertRaises(FileNotFoundError):
            # Test loading a non-existent file raises FileNotFoundError
            fertility_rates_forecasting.load_csv_file(self.non_existent_file)

if __name__ == "__main__":
    unittest.main()


