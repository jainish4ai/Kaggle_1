import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
data = train_data.append(test_data)

train_data['SalePrice'] = np.log(train_data.SalePrice)
train_data.SalePrice.hist()
sns.boxplot(train_data.SalePrice)
cat_columns = train_data.select_dtypes(exclude = 'number').columns.sort_values()
cat_columns

num_columns = train_data.select_dtypes('number').columns.sort_values()
data[num_columns].describe()

col_name = 'TotalBsmtSF'

# train_data[col_name]=train_data[col_name].fillna('O')

train_data[col_name].isna().sum()
(train_data[col_name] == 0).isna().sum()
train_data[col_name].describe()
train_data[col_name].value_counts(dropna = False)
train_data[col_name].sort_values(ascending=False)[0:10]


sns.boxplot(x=train_data[col_name], y=train_data.SalePrice)
train_data[col_name].hist()
sns.scatterplot(x=train_data[col_name], y=train_data.SalePrice) 


data[col_name].value_counts(dropna = False)

# Index(['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
#        'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2',
#        'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',
#        'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond',
#        'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC',
#        'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
#        'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood',
#        'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition',
#        'SaleType', 'Street', 'Utilities'],

# Index(['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1',
#        'BsmtFinSF2', 'BsmtUnfSF',
#        'GarageArea', 'GarageCars',
#        'GarageYrBlt', 'GrLivArea', 'Id', 'LotArea',
#        'LotFrontage', 'MasVnrArea',
#        , 'OpenPorchSF', ,
#        'SalePrice', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF',
#        'YearBuilt', 'YearRemodAdd', 'YrSold'],
      
data[col_name].mode()

totalFloorSF = train_data['BsmtFinSF1'] + train_data['BsmtFinSF2']
totalFloorSF.value_counts(dropna=False)
sns.scatterplot(x=totalFloorSF, y=train_data.SalePrice)
totalFloorSF.hist()

train_data = pd.read_csv('data/train.csv')
train_data = train_data[train_data['TotalBsmtSF'] < 3000]
sns.scatterplot(x=train_data[col_name], y=train_data.SalePrice) 
train_data.groupby('TotalBsmtSF')['SalePrice'].mean().plot(kind = 'bar')
