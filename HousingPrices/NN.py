import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
import tensorflow as tf
from tensorflow import keras

train_data = pd.read_csv('data/train.csv')
train_data = train_data[train_data['SalePrice'] < 500000]
train_data = train_data[train_data['LotArea'] < 100000]
train_data[['LotFrontage']] = train_data['LotFrontage'].fillna(0)
train_data = train_data[train_data['LotFrontage'] < 200]
train_data = train_data[train_data['OpenPorchSF'] < 350]
train_data = train_data[train_data['TotalBsmtSF'] < 3000]

test_data = pd.read_csv('data/test.csv')

data = train_data.append(test_data)

data[['BsmtCond']] = data['BsmtCond'].fillna('O')
data[['BsmtExposure']] = data['BsmtExposure'].fillna('O')
data[['BsmtFinType1']] = data['BsmtFinType1'].fillna('O')
data[['BsmtFinType2']] = data['BsmtFinType2'].fillna('O')
data[['BsmtQual']] = data['BsmtQual'].fillna('Fa')
data[['Electrical']] = data['Electrical'].fillna(data['Electrical'].mode().values[0])
data[['Exterior1st']] = data['Exterior1st'].fillna(data['Exterior1st'].mode().values[0])
data[['Exterior2nd']] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode().values[0])
data[['FireplaceQu']] = data['FireplaceQu'].fillna('O')
data[['Functional']] = data['Functional'].fillna(data['Functional'].mode().values[0])
data[['GarageCond']] = data['GarageCond'].fillna('Fa')
data[['GarageFinish']] = data['GarageFinish'].fillna('Fa')
data[['GarageQual']] = data['GarageQual'].fillna('Fa')
data[['GarageType']] = data['GarageType'].fillna('O')
data[['KitchenQual']] = data['KitchenQual'].fillna(data['KitchenQual'].mode().values[0])
data[['MasVnrType']] = data['MasVnrType'].fillna('O')
data[['MSZoning']] = data['MSZoning'].fillna(data['MSZoning'].mode().values[0])
data[['PavedDrive']] = data['PavedDrive'].replace('P', 'N')
data[['SaleType']] = data['SaleType'].fillna(data['SaleType'].mode().values[0])
data[['BsmtFullBath']] = data['BsmtFullBath'].fillna(data['BsmtFullBath'].mode().values[0])
data[['GarageCars']] = data['GarageCars'].fillna(data['GarageCars'].mode().values[0])

data[['TotalFlrSF']]= data[['1stFlrSF']] + data[['1stFlrSF']]
data[['GarageArea']] = data[['GarageArea']].fillna(0)
data[['TotalBsmtSF']] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean())
data['HasBasement'] = data['TotalBsmtSF'] > 0
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(1900)
data['GarageYrBltCat'] = ((data['GarageYrBlt'] - 1900)/10).astype('int')
data['YearBuiltCat'] = ((data['YearBuilt'] - 1870)/10).astype('int')
data[['LotFrontage']] = data['LotFrontage'].fillna(0)
data[['MasVnrArea']] = data['MasVnrArea'].fillna(0)
data[['LotFrontage']] = data['LotFrontage'].fillna(0)
 
cat_columns = [
    'BldgType',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'BsmtQual',
    'CentralAir',
    'Condition1',
    'Electrical',
    'ExterCond',
    'ExterQual',
    'Exterior1st', 
    'Exterior2nd',
    'FireplaceQu',
    'Foundation', 
    'Functional',
    'GarageCond',
    'GarageFinish',
    'GarageQual',
    'GarageType',
    'HeatingQC',
    'HouseStyle',
    'KitchenQual',
    'LandContour',
    'LotConfig',
    'LotShape',
    'MSZoning', 
    'MasVnrType',
    'Neighborhood',
    'PavedDrive',
    'RoofStyle',
    'SaleCondition',
    'SaleType',
    
    'BedroomAbvGr',
    'BsmtFullBath',
    'Fireplaces',
    'FullBath',
    'GarageCars',
    'HalfBath',
    'KitchenAbvGr',
    'MSSubClass',
    'OverallCond',
    'OverallQual',
    'TotRmsAbvGrd',
    
    'HasBasement',
    'GarageYrBltCat',
    'YearBuiltCat'
    ]

num_columns = [
    'TotalBsmtSF',
    'TotalFlrSF',
    'GarageArea',
    'GrLivArea',
    'LotArea',
    'LotFrontage',
    # 'MasVnrArea',
    'OpenPorchSF',
    'ScreenPorch',
    'WoodDeckSF'
    
    ]

transformers = []
for column in cat_columns:
    transformers.append((column, OneHotEncoder(drop = 'first'), [column]))
    
for column in num_columns:
    transformers.append((column, PowerTransformer(), [column]))
    
columns = cat_columns + num_columns

transformer = ColumnTransformer(transformers, remainder = 'passthrough')

X = data[columns]
features = transformer.fit_transform(X)

train_features = features.toarray()[:len(train_data)]
train_labels = np.log(data[:len(train_data)].SalePrice)

tf.keras.backend.set_floatx('float64')
model = tf.keras.Sequential([
    keras.layers.Dense(128, activation = 'relu', input_shape = (293,)),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(1)
    ])


print(model.summary())

model.compile(optimizer = keras.optimizers.Adam(), loss = 'mean_squared_error', metrics = ['mean_squared_error'])

model.fit(train_features, train_labels, epochs = 80, validation_split = .2)

test_features = features.toarray()[-len(test_data):]
predictions = np.exp(model.predict(test_features)).ravel()

output = pd.DataFrame({'Id': test_data.iloc[:,0], 'SalePrice': predictions})
output.to_csv('output.csv', index = False)

