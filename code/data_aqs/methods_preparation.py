import os
import warnings
import numpy as np
import pandas as pd
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