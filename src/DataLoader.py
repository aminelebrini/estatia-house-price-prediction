from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, ):
        pass

    def load_data_silver(self, df):
        BASE_DIR = Path(__file__).resolve().parent.parent
        try:
            
            silver_data_path = BASE_DIR / 'data' / 'silver' / 'House_Prices_Silver.csv'
            df.to_csv(silver_data_path, index=False)
            print(f"Cleaned data saved to {silver_data_path}")
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")

    def load_data_gold(self, df):
        BASE_DIR = Path(__file__).resolve().parent.parent
        try:
            gold_data_path = BASE_DIR / 'data' / 'gold' / 'House_Prices_Gold.csv'
            df.to_csv(gold_data_path, index=False)
            print(f"Engineered data saved to {gold_data_path}")
        except FileNotFoundError:
            print("Error: File not found.")
        except Exception as e:
            print(f"Error saving engineered data: {e}")