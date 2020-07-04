import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate

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
# data['GarageYrBltCat'] = ((data['GarageYrBlt'] - 1900)/15).astype('int')
# data['YearBuiltCat'] = ((data['YearBuilt'] - 1870)/15).astype('int')
data[['LotFrontage']] = data['LotFrontage'].fillna(0)
data[['MasVnrArea']] = data['MasVnrArea'].fillna(0)
data[['LotFrontage']] = data['LotFrontage'].fillna(0)
data['TotalBsmtSF_cat'] = pd.qcut(data['TotalBsmtSF'], 10, labels = False, duplicates='drop')
data['TotalFlrSF_cat'] = pd.qcut(data['TotalFlrSF'], 10, labels = False, duplicates='drop')
data['GarageArea_cat'] = pd.qcut(data['GarageArea'], 10, labels = False, duplicates='drop')
data['GrLivArea_cat'] = pd.qcut(data['GrLivArea'], 10, labels = False, duplicates='drop')
data['LotArea_cat'] = pd.qcut(data['LotArea'], 10, labels = False, duplicates='drop')
data['LotFrontage_cat'] = pd.qcut(data['LotFrontage'], 10, labels = False, duplicates='drop')
data['OpenPorchSF_cat'] = pd.qcut(data['OpenPorchSF'], 10, labels = False, duplicates='drop')
data['ScreenPorch_cat'] = pd.qcut(data['ScreenPorch'], 15, labels = False, duplicates='drop')
data['WoodDeckSF_cat'] = pd.qcut(data['WoodDeckSF'], 10, labels = False, duplicates='drop')



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
    # 'GarageYrBltCat',
    # 'YearBuiltCat'
    
    'TotalBsmtSF_cat',
    'TotalFlrSF_cat',
    'GarageArea_cat',
    'GrLivArea_cat',
    'LotArea_cat',
    'LotFrontage_cat',
    'OpenPorchSF_cat',
    'ScreenPorch_cat',
    'WoodDeckSF_cat'
    ]


transformers = []
for column in cat_columns:
    transformers.append((column, OneHotEncoder(drop = 'first'), [column]))
    
columns = cat_columns

transformer = ColumnTransformer(transformers, remainder = 'passthrough')

X = data[columns]
features = transformer.fit_transform(X)

train_features = features.toarray()[:len(train_data)]
train_labels = np.log(data[:len(train_data)].SalePrice)


models = [
    RandomForestRegressor(),
    ExtraTreesRegressor(),
    # AdaBoostRegressor(),
    XGBRegressor()
    ]

results = pd.DataFrame(columns=['Model', 'Fit Time', 'Train MSE', 'Train R2', 'Val MSE', 'Val R2'])
for model in models:
    cv_results = cross_validate(model, train_features, train_labels, cv = 5, 
                                scoring = ['neg_mean_squared_error', 'r2'],
                                n_jobs= -1, return_train_score = True)
    
    results = results.append(pd.Series({'Model': type(model).__name__,
                                        'Fit Time': cv_results['fit_time'].mean(),
                                        'Train MSE': np.sqrt(np.abs(cv_results['train_neg_mean_squared_error'].mean())),
                                        'Train R2': cv_results['train_r2'].mean(),
                                        'Val MSE': np.sqrt(np.abs(cv_results['test_neg_mean_squared_error'].mean())),
                                        'Val R2': cv_results['test_r2'].mean()}),
                             ignore_index = True)

print(results)

best_model = models[results['Val MSE'].argmin()]

print ('Best Model selected:', type(best_model).__name__)
best_model.fit(train_features, train_labels)
predictions = best_model.predict(train_features)

test_features = features.toarray()[-len(test_data):]
predictions = np.exp(best_model.predict(test_features))

output = pd.DataFrame({'Id': test_data.iloc[:,0], 'SalePrice': predictions})
output.to_csv('output.csv', index = False)

