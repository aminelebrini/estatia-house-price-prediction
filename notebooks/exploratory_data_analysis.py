import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class EDA_visualisation:
    def __init__(self, df):
        self.df = df

    def hist_plot(self):
        plt.figure(figsize=(9, 5))
        plt.hist(self.df['SalePrice'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of SalePrice')
        plt.xlabel('SalePrice')
        plt.ylabel('Frequency')
        plt.show()

    def scatter_plot(self):
        plt.figure(figsize=(9, 5))
        plt.scatter(
            self.df['GrLivArea'], self.df['SalePrice'], alpha=0.6, color='orange'
        )
        plt.title('GrLivArea vs SalePrice')
        plt.xlabel('GrLivArea')
        plt.ylabel('SalePrice')
        plt.show()

    def box_plot(self):
        plt.figure(figsize=(9, 5))
        sns.boxplot(
            data=self.df,
            x='OverallQual',
            y='SalePrice',
            hue='OverallQual',
            palette='Blues',
            legend=False,
        )
        plt.title('OverallQual vs SalePrice')
        plt.xlabel('OverallQual')
        plt.ylabel('SalePrice')
        plt.show()

        plt.figure(figsize=(12, 6))
        order = (
            self.df.groupby('Neighborhood')['SalePrice']
            .median()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(
            data=self.df,
            x='Neighborhood',
            y='SalePrice',
            order=order,
            hue='Neighborhood',
            palette='viridis',
            legend=False,
        )
        plt.xticks(rotation=90)
        plt.title('Neighborhood vs SalePrice')
        plt.xlabel('Neighborhood')
        plt.ylabel('SalePrice')
        plt.show()

    def line_plot_year(self):
        plt.figure(figsize=(10, 5))
        df_year = self.df.groupby('YearBuilt')['SalePrice'].median().reset_index()
        plt.plot(
            df_year['YearBuilt'],
            df_year['SalePrice'],
            color='red',
            marker='o',
            markersize=3,
        )
        plt.title('YearBuilt vs Median SalePrice')
        plt.xlabel('YearBuilt')
        plt.ylabel('Median SalePrice')
        plt.show()

    def heatmap_plot(self):
        plt.figure(figsize=(10, 8))
        num_df = self.df.select_dtypes(include=['int64', 'float64'])
        top_corr = (
            num_df.corr()['SalePrice']
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        corr_matrix = num_df[top_corr].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
        plt.title('Top 10 Correlation Matrix')
        plt.show()
