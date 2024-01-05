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


    def plot_history(self, history, n_epochs):
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,2,1)
        plt.plot(range(1, n_epochs+1), history.history['recall'],label='Treinamento')
        plt.plot(range(1, n_epochs+1), history.history['val_loss'], label='Validação')
        plt.legend(loc='best')
        plt.xlabel('N_epochs')
        plt.ylabel('Recall')
        plt.title('Curva de aprendizado - Recall')
        
        ax = fig.add_subplot(1,2,2)
        plt.plot(range(1, n_epochs+1), history.history['loss'], label='Treinamento')
        plt.plot(range(1, n_epochs+1), history.history['val_loss'], label='Validação')
        plt.legend(loc='best')
        plt.xlabel('N_epochs')
        plt.ylabel('Função de Perda')
        plt.title('Curva de aprendizado - Função de Perda')
        
        return plt.show()
    
    
    def analyze_results(x_test_norm, list_models, list_model_names, THRESHOLD=0.6):
        for i in range(len(list_models)):

            # Faz a previsão na base de teste
            yhat_test = list_models[i].predict(x_test_norm)

            # Transforma em classes
            yhat_test_class = [1 if x >= THRESHOLD else 0 for x in yhat_test]

        # Imprime resultados
    
        print(f'DESEMPENHO DE CLASSIFICAÇÃO - MODELO {i+1} - {list_model_names[i]} - Prob. min = {THRESHOLD}')
        print(classification_report(y_test, yhat_test_class))
   
        print('*' * 50)
        print('\n')