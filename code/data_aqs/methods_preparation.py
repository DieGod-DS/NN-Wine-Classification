import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Methods():
    def __init__(self):
        pass
    
    
    def import_df(self, file_name):
        '''
        esta função serve para importar o dataframe
        '''
        data = pd.read_csv(
            os.getcwd().replace('code', 'data').replace('data_aqs', 'raw').replace('/' or '//', r'\\') 
            + f'\\{file_name}'
            )
        return data
    
    def export_df(self, df, file_name, file_extension = '.csv'):
        df.to_csv(os.getcwd().replace('code', 'data').replace('data_aqs', 'processed')+r'\\'+f'{file_name}+{file_extension}')
        return f'Arquivo salvo!'
    
    
    def create_histogram(self, df,):
        df.hist(figsize=(12,10), bins=30, edgecolor='black')
        plt.subplots_adjust(hspace=0.7, wspace=0.4)
        return plt.show()
    
    
    def create_boxplot(self, df):
        plt.figure(figsize=(21,10))
        plt.boxplot(df, labels=df.columns);
        return plt.show()
    
    
    def create_heatmap(self, df):
        plt.figure(figsize=(12,8))
        sns.heatmap(df.corr(), annot=True)
        return plt.show()
    
    
    def export_df(self, df, file_name, file_extension='.csv'):
        df.to_csv(os.getcwd().replace('code', 'data').replace('data_aqs', 'processed')+r'\\'+f'{file_name}{file_extension}')
        return f'Arquivo salvo!'