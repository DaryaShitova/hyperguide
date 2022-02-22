import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def prepare_dataset(dataset_name='iris', test_size=0.25):
    if dataset_name=='iris':
        data_df = pd.read_csv('data/iris.csv')
        data_df.dropna(0, inplace=True)
        encoder_species = LabelEncoder()
        X = data_df.iloc[:,:-1].values
        y = np.ravel(encoder_species.fit_transform(data_df['species']))

    elif dataset_name=='penguins':
        data_df = pd.read_csv('data/penguins.csv')
        data_df.dropna(0, inplace=True)
        encoder_island = LabelEncoder()
        encoder_sex = LabelEncoder()
        encoder_species = LabelEncoder()
        data_df['island'] = np.ravel(encoder_island.fit_transform(data_df['island']))
        data_df['sex'] = np.ravel(encoder_sex.fit_transform(data_df['sex']))
        X = data_df.iloc[:,1:].values
        y = np.ravel(encoder_species.fit_transform(data_df['species']))

    else:
        print('Please prepare dataset yourself.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test