import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.DataCleaner import DataCleaner
from src.DataLoader import DataLoader
from src.DataExtracter import DataExtracter
from notebooks.exploratory_data_analysis import EDA_visualisation

def main():

    data_extracter = DataExtracter()
    data_house_prices = data_extracter.extract_data()

    if data_house_prices is not None:
        data_cleaner = DataCleaner()
        cleaned_data = data_cleaner.clean_data(data_house_prices)
    else:
        print("Error: Failed to extract data.")
        return

    eda = EDA_visualisation(cleaned_data)
    eda.hist_plot()
    eda.scatter_plot()
    eda.box_plot()
    eda.line_plot_year()
    eda.heatmap_plot()


    data_loader = DataLoader()
    data_loader.load_data_silver(cleaned_data)


if __name__ == "__main__":
    main()

