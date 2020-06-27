import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
data = pd.read_csv('data/train.csv')

temp = data[['Embarked', 'Survived', 'Ticket']]

data.groupby('Embarked')['Survived'].agg(['count', 'mean'])

def cabin(cabin):
    if pd.isnull(cabin):
        return 9
    if 'A' in cabin:
        return 1
    if 'B' in cabin:
        return 2
    if 'C' in cabin:
        return 3
    if 'D' in cabin:
        return 4
    if 'E' in cabin:
        return 5
    if 'F' in cabin:
        return 6
    if 'G' in cabin:
        return 7
    return 8
    
    
data['CC'] = data['Cabin'].apply(cabin)

data.groupby('CC')['Survived'].mean()


data['FamSize'] = data['SibSp'] + data.Parch
data.groupby('FamSize')['Survived'].mean()
