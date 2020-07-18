import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
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


# params_svr = {
#     'kernel': ['poly', 'rbf', 'sigmoid', 'linear'],
#     'C': np.linspace(.05,.08, 20),
#     # 'coef0' : np.linspace(.0000, .0001, 5)
#     }

# model_selector = GridSearchCV(SVR(), params_svr, cv = 5, n_jobs = -1, verbose = True, refit=True)

# params_ridge = {
#     'alpha': np.linspace(3,4, 20),
#     'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
#     }
# model_selector = GridSearchCV(Ridge(), params_ridge, cv = 5, n_jobs = -1, verbose = True, refit=True)

# params_lasso = {
#     'alpha': np.linspace(.0001,.0003, 50),
#     'selection':['cyclic', 'random']
# }
# model_selector = GridSearchCV(Lasso(), params_lasso, cv = 5, n_jobs = -1, verbose = True, refit=True)

params_kernel_ridge = {
    'alpha': np.linspace(.001,.01, 10),
    'gamma': np.linspace(.001,.01, 10),
    'coef0': np.linspace(.001,.01, 10)
}
model_selector = GridSearchCV(KernelRidge(), params_kernel_ridge, cv = 5, n_jobs = -1, verbose = True, refit=True)

result = model_selector.fit(train_features, train_labels)

best_model = model_selector.best_estimator_
print(model_selector.best_params_)
print(model_selector.best_score_)

print ('Best Model selected:', type(best_model).__name__)
best_model.fit(train_features, train_labels)
predictions = best_model.predict(train_features)

test_features = features[-len(test_data):]
predictions = np.exp(best_model.predict(test_features))

output = pd.DataFrame({'Id': test_data.iloc[:,0], 'SalePrice': predictions})
output.to_csv('output.csv', index = False)