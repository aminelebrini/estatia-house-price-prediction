import pandas as pd
from pathlib import Path
from DataCleaner import DataCleaner
class DataExtracter:
    def __init__(self):
        pass
    def get_data(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        data_description_path = BASE_DIR / 'data' / 'bronze' / 'data_description.txt'
        data_house_prices_path = BASE_DIR / 'data' / 'bronze' / 'House_Prices.csv'

        try:
            with open(data_description_path, "r") as file:
                data_description = file.read()
            
            data_house_prices = pd.read_csv(data_house_prices_path)

            # print(data_house_prices)
            
            return data_description, data_house_prices

        except FileNotFoundError:
            print("Error: File not found.")
            return None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None


data_extracter = DataExtracter()
data_extracter.get_data()
