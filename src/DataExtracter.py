import pandas as pd
from pathlib import Path
class DataExtracter:
    def __init__(self):
        pass
    def extract_data(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        # data_description_path = BASE_DIR / 'data' / 'bronze' / 'data_description.txt'
        data_house_prices_path = BASE_DIR / 'data' / 'bronze' / 'House_Prices.csv'

        try:
            # with open(data_description_path, "r") as file:
            #     data_description = file.read()
            
            data_house_prices = pd.read_csv(data_house_prices_path)

            # print(data_house_prices)
            
            return data_house_prices

        except FileNotFoundError as error:
            print(f"Error: File not found: {error.filename}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def data_extracter_file(self, data_frame_path):
        try:
            data_frame = pd.read_csv(data_frame_path)
            return data_frame
        except FileNotFoundError as error:
            print(f"Error: File not found: {error.filename}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
