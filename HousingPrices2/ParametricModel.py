import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate

pd.set_option("display.precision", 4)
train_data = pd.read_csv('data/train.csv')
train_data = train_data[train_data.SalePrice < 500000]
train_data = train_data[train_data['TotalBsmtSF'] < 3000]
train_data = train_data[train_data['MasVnrArea'] < 1200]
train_data = train_data[train_data['LotFrontage'] < 200]
train_data = train_data[train_data['OpenPorchSF'] < 400]
train_data = train_data[train_data['LotArea'] < 100000]

test_data = pd.read_csv('data/test.csv')
data = train_data.append(test_data)

data[['GarageCars']] = data['GarageCars'].fillna(data['GarageCars'].mode().values[0])
data[['TotalBsmtSF']] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean())
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

cat_columns = ['OverallQual', 'GarageCars', 'FullBath', 'TotRmsAbvGrd','Fireplaces',
               'HalfBath', 'BsmtFullBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'CentralAir', 'Condition1',
               'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'Foundation', 
               'GarageFinish', 'GarageType','HeatingQC','HouseStyle', 'KitchenQual', 
               'LandContour', 'LotShape', 'LotConfig', 'MSZoning','MasVnrType','Neighborhood',
               'PavedDrive', 'RoofStyle', 'SaleCondition', 'SaleType']

num_columns = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 
               'MasVnrArea', 'BsmtFinSF1', 'LotFrontage', 'OpenPorchSF','WoodDeckSF',
               'LotArea', 'BsmtUnfSF']

transformers = []
for column in cat_columns:
    transformers.append((column, OneHotEncoder(drop = 'first'), [column]))
    
for column in num_columns:
    transformers.append((column, PowerTransformer(), [column]))
    
column_transformer = ColumnTransformer(transformers, remainder = 'passthrough')

X = data[cat_columns + num_columns]
features = column_transformer.fit_transform(X)

train_features = features[:len(train_data)]
train_labels = np.log(data[:len(train_data)].SalePrice)

models = [
    LinearRegression(),
    Ridge(),
    Lasso(alpha = 0),
    SVR(),
    KNeighborsRegressor()
    ]

estimators = []
for model in models:
    estimators.append((type(model).__name__, model))


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

test_features = features[-len(test_data):]
predictions = np.exp(best_model.predict(test_features))

output = pd.DataFrame({'Id': test_data.iloc[:,0], 'SalePrice': predictions})
output.to_csv('output.csv', index = False)