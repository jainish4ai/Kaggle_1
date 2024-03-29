import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel, chi2, f_regression, SelectPercentile, mutual_info_classif, RFE

pd.set_option("display.precision", 4)
train_data = pd.read_csv('data/train.csv')
train_data = train_data[train_data['LotFrontage'] < 200]
train_data = train_data[train_data['OpenPorchSF'] < 400]
train_data = train_data[train_data['LotArea'] < 100000]
# train_data = train_data[train_data['WoodDeckSF'] < 610]
# train_data = train_data[train_data['TotalBsmtSF'] < 3100]


test_data = pd.read_csv('data/test.csv')
data = train_data.append(test_data)

# data[['GarageCars']] = data['GarageCars'].fillna(data['GarageCars'].mode().values[0])
data[['TotalBsmtSF']] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].median())
data[['GarageArea']] = data['GarageArea'].fillna(data['GarageArea'].median())
data[['MasVnrArea']] = data['MasVnrArea'].fillna(0)
data[['MasVnrAreaMask']] = (data[['MasVnrArea']] != 0).astype('int')
data[['BsmtFinSF1']] = data['BsmtFinSF1'].fillna(0)
data[['BsmtFinSF2']] = data['BsmtFinSF2'].fillna(0)
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
data['YearBuilt'] = ((data['YearBuilt'] - 1870)/20)

cat_columns = ['OverallQual','Fireplaces','HalfBath', 'FullBath',
                'BsmtFullBath', 'BedroomAbvGr', 'KitchenAbvGr',
               'BldgType', 'BsmtCond', 'BsmtExposure', 'CentralAir', 'Condition1',
               'ExterCond', 'ExterQual', 'Exterior1st', 'Foundation', 
               'HeatingQC','HouseStyle', 'KitchenQual', 'BsmtFinType1',
               'LandContour', 'LotShape', 'LotConfig', 'MSZoning','MasVnrType','Neighborhood',
               'PavedDrive', 'SaleCondition', 'MSSubClass', 'OverallCond']

num_columns = ['GrLivArea', 'TotalBsmtSF', 'YearRemodAdd', 'GarageArea','YearBuilt',
               'MasVnrArea','TotalPorchSF','WoodDeckSF', 'BsmtFinSF1','TotalBsmtFinSF',
               'LotArea', 'TotalSF']

data.isna().sum()
transformers = []
for column in cat_columns:
    transformers.append((column, OneHotEncoder(drop = 'first'), [column]))
    
for column in num_columns:
    transformers.append((column, PowerTransformer(), [column]))
    
column_transformer = ColumnTransformer(transformers, remainder = 'passthrough')

X = data[cat_columns + num_columns]
features = column_transformer.fit_transform(X)

train_features = features[:len(train_data)].toarray()
train_labels = np.log(data[:len(train_data)].SalePrice)

# model_selector = SelectPercentile(f_regression, 100)
# model_selector = SelectFromModel(Lasso(alpha = 0.000302, selection = 'random'), 200)
model_selector = RFE(Ridge(alpha=  3.473684210526316, solver = 'sparse_cg'), 150)
train_features = model_selector.fit_transform(train_features, train_labels)
models = [
    # LinearRegression(),
    # Ridge(alpha=  3.473684210526316, solver = 'sparse_cg'),
    # Lasso(alpha = 0.00021836734693877552, selection = 'random', max_iter=1e5),
    # SVR(C = 0.06736842105263158, kernel = 'linear'),
    # KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    xgb.XGBRegressor(),
    lgb.LGBMRegressor()
    # KNeighborsRegressor()
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
np.isnan(features[-len(test_data):].toarray()).sum(axis = 1)
test_features = model_selector.transform(features[-len(test_data):])
# test_features = features[-len(test_data):]
predictions = np.exp(best_model.predict(test_features))

output = pd.DataFrame({'Id': test_data.iloc[:,0], 'SalePrice': predictions})
output.to_csv('output.csv', index = False)