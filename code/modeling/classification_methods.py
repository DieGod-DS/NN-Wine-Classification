import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class Classification():
    
    
    def __init__(self):
        pass
    
    
    def import_df(self, file_name):
        data = pd.read_csv(
            os.getcwd().replace('/' or '//', r'\\').replace('code','data').replace('modeling','processed')+f'\\{file_name}'
        )
        return data
    
    
