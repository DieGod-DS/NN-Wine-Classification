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

class Classification_ann():
    
    
    def __init__(self):
        pass
    
    
    def import_df(self, file_name):
        data = pd.read_csv(
            os.getcwd().replace('/' or '//', r'\\').replace('code','data').replace('modeling','processed')+f'\\{file_name}'
        )
        return data
    
    
    def props(self, df, target, y_train, y_val, y_test):
        '''
        esta função retorna as proporções das classes
        '''
        props = pd.DataFrame(df[target].value_counts(normalize=True).values,
                             index = df[target].value_counts(normalize=True).index,
                             columns = ['original'])
        
        props['treino'] = y_train[target].value_counts(normalize=True).values
        props['val'] = y_val[target].value_counts(normalize=True).values
        props['teste'] = y_test[target].value_counts(normalize=True).values
        
        return props
    
    
    def create_model(x, n_neurons:list):
        '''
        esta função cria uma rede neural
        '''
        model = Sequential()
        
        if len(n_neurons) == 1:
            model.add(Dense(n_neurons[0], input_shape=(11,), activation='relu'))
        else:
            for index, n in n_neurons:
                if index == 0:
                    model.add(Dense(n, input_shape=(11,), activation='relu'))
                else:
                    model.add(Dense(n, activation='relu'))
                    
        model.add(Dense(1, activation='sigmoid'))
    
        return model
    
    
    def scal_data(self, x_train, x_val, x_test):
        '''
        esta função normaliza os dados
        '''
        
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(x_train)
        
        x_train_norm = scaler.transform(x_train)
        x_val_norm = scaler.transform(x_val)
        x_test_norm = scaler.transform(x_test)
        
        return x_train_norm, x_val_norm, x_test_norm
