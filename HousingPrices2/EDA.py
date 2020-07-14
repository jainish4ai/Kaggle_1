import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
data = train_data.append(test_data)

data[['BsmtFullBath']] = data[['BsmtFullBath']].replace(2,1)
data[['BsmtFullBath']] = data[['BsmtFullBath']].replace(3,1)

cat_columns = train_data.select_dtypes(include = 'object')
temp = train_data.corr()['SalePrice']
train_data['SalePrice'] = np.log(train_data.SalePrice)
train_data.SalePrice.hist()
sns.boxplot(train_data.SalePrice)
cat_columns = train_data.select_dtypes(exclude = 'number').columns.sort_values()
cat_columns

num_columns = train_data.select_dtypes('number').columns.sort_values()
data[num_columns].describe()

col_name = 'SaleType'

# train_data[col_name]=train_data[col_name].fillna('O')

train_data[col_name].isna().sum()
(data[col_name] == 0).isna().sum()
data[col_name].describe()
data[col_name].value_counts(dropna = False)
train_data[col_name].sort_values(ascending=False)[0:10]


sns.boxplot(x=train_data[col_name], y=train_data.SalePrice)
train_data[col_name].hist()
sns.scatterplot(x=train_data[col_name], y=train_data.SalePrice) 


data[col_name].value_counts(dropna = False)

# KitchenAbvGr    -0.135907
# EnclosedPorch   -0.128578
# MSSubClass      -0.084284
# OverallCond     -0.077856
# YrSold          -0.028923
# LowQualFinSF    -0.025606
# Id              -0.021917
# MiscVal         -0.021190
# BsmtHalfBath    -0.016844
# BsmtFinSF2      -0.011378
# 3SsnPorch        0.044584
# MoSold           0.046432
# PoolArea         0.092404
# ScreenPorch      0.111447
# BedroomAbvGr     0.168213
# BsmtUnfSF        0.214479
# BsmtFullBath     0.227122
# LotArea          0.263843
# HalfBath         0.284108
# OpenPorchSF      0.315856
# 2ndFlrSF         0.319334
# WoodDeckSF       0.324413
# LotFrontage      0.351799
# BsmtFinSF1       0.386420
# Fireplaces       0.466929
# MasVnrArea       0.477493
# GarageYrBlt      0.486362
# YearRemodAdd     0.507101
# YearBuilt        0.522897
# TotRmsAbvGrd     0.533723
# FullBath         0.560664
# 1stFlrSF         0.605852
# TotalBsmtSF      0.613581
# GarageArea       0.623431
# GarageCars       0.640409
# GrLivArea        0.708624
# OverallQual      0.790982
# SalePrice        1.000000

# 'Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
#        'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2',
#        'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',
#        'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond',
#        'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC',
#        'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
#        'LotShape', 'MSZoning', 'MasVnrType', 'MiscFeature', 'Neighborhood',
#        'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition',
#        'SaleType', 'Street', 'Utilities'
       
data[col_name].mode()

totalFloorSF = train_data['BsmtFinSF1'] + train_data['BsmtFinSF2']
totalFloorSF.value_counts(dropna=False)
sns.scatterplot(x=totalFloorSF, y=train_data.SalePrice)
totalFloorSF.hist()

train_data = pd.read_csv('data/train.csv')
train_data = train_data[train_data['TotalBsmtSF'] < 3000]
sns.scatterplot(x=train_data[col_name], y=train_data.SalePrice) 
train_data.groupby('TotalBsmtSF')['SalePrice'].mean().plot(kind = 'bar')
