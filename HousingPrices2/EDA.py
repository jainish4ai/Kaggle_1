import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

pd.set_option("display.precision", 4)
train_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')
data = train_data.append(test_data)

# data[['GarageCars']] = data['GarageCars'].fillna(data['GarageCars'].mode().values[0])
data[['TotalBsmtSF']] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].median())
data[['GarageArea']] = data['GarageArea'].fillna(data['GarageArea'].median())
data[['MasVnrArea']] = data['MasVnrArea'].fillna(0)
data[['MasVnrAreaMask']] = (data[['MasVnrArea']] != 0).astype('int')
data[['BsmtFinSF1']] = data['BsmtFinSF1'].fillna(0)
data[['LotFrontage']] = data['LotFrontage'].fillna(0)
data[['BsmtFullBath']] = data[['BsmtFullBath']].replace(2,1)
data[['BsmtFullBath']] = data[['BsmtFullBath']].replace(3,1)
data[['BsmtFullBath']] = data[['BsmtFullBath']].fillna(0)
data[['BsmtUnfSF']] = data[['BsmtUnfSF']].fillna(0)
data[['KitchenAbvGr']] = data[['KitchenAbvGr']].replace(0,1)
data[['KitchenAbvGr']] = data[['KitchenAbvGr']].replace(3,2)
data[['BsmtCond']] = data['BsmtCond'].fillna(data['BsmtCond'].mode().values[0])
data[['BsmtExposure']] = data['BsmtExposure'].fillna(data['BsmtExposure'].mode().values[0])
data[['BsmtFinType1']] = data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode().values[0])
data[['BsmtQual']] = data['BsmtQual'].fillna(data['BsmtQual'].mode().values[0])
data[['Electrical']] = data['Electrical'].fillna(data['Electrical'].mode().values[0])
data[['Exterior1st']] = data['Exterior1st'].fillna(data['Exterior1st'].mode().values[0])
data[['Exterior2nd']] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode().values[0])
data[['GarageFinish']] = data['GarageFinish'].fillna(data['GarageFinish'].mode().values[0])
data[['GarageType']] = data['GarageType'].fillna(data['GarageType'].mode().values[0])
data[['KitchenQual']] = data['KitchenQual'].fillna(data['KitchenQual'].mode().values[0])
data[['MSZoning']] = data['MSZoning'].fillna(data['MSZoning'].mode().values[0])
data[['MasVnrType']] = data['MasVnrType'].fillna(data['MasVnrType'].mode().values[0])
data[['SaleType']] = data['SaleType'].fillna(data['SaleType'].mode().values[0])
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['TotalPorchSF'] = data['ScreenPorch'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['WoodDeckSF']
data['TotalBsmtFinSF'] = data['BsmtFinSF1'] + data['BsmtFinSF2'] + data['BsmtUnfSF']
data['YrSold'] = data['YrSold'] - 2000


# Compute the correlation matrix
corr = train_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,
            square=True, linewidths=.5)

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

col_name = 'BsmtFinSF2'
data['TotalB'] = data['HalfBath'] + data['FullBath']
train_data['TotalB'] = train_data['HalfBath'] + train_data['FullBath']
# train_data[col_name]=train_data[col_name].fillna('O')

data[col_name].isna().sum()
(data[col_name] == 0).isna().sum()
data[col_name].describe()

data[col_name].value_counts(dropna = False)
train_data[col_name].sort_values(ascending=True)[0:10]


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
