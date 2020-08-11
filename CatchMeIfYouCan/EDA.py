import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('data/train_sessions.csv')

time_cols = ['time'+str(i) for i in range(1,11)]
for i in range(1,11):
    time_col = 'time'+str(i)
    train[time_col] = pd.to_datetime(train[time_col])
    train['hr'+str(i)] = train[time_col].dt.hour
Alice = train[train.target == 1]
Others = train[train.target == 0]

plt.hist(Alice[['hr'+str(i) for i in range(1,11)]].values.ravel())
np.unique(Alice[['hr'+str(i) for i in range(1,11)]].values)

site_cols = ['site'+str(i) for i in range(1,11)]
Alice[site_cols].isna().sum(axis = 0)/len(Alice)
Others[site_cols].isna().sum(axis = 0)/len(Others)
sites = pd.Series()

groups = []
for i in range(1,11):
    group = train.groupby('site' + str(i)).target.mean()
    sites = sites.append(pd.Series(group.index))
    groups.append(group)

sites = np.sort(sites.unique())

transformed_df = pd.DataFrame(index = sites)

for i, group in enumerate(groups):
    transformed_df[str(i)] = group
    
sites_data = transformed_df.mean(axis = 1)

Alice_sites = sites_data[sites_data == 1]
Other_sites = sites_data[sites_data == 0]
# for i in range(1, 11):
#     print('Processing Col:' + str(i))
#     train['site' + str(i)] = train['site' + str(i)].replace(Alice_sites, 50000)
    
    
    
# for i in range(1, 11):
#     print('Processing Col:' + str(i))
#     train['site' + str(i)] = train['site' + str(i)].replace(Other_sites, 100000)
    
# train.to_csv('transformed_data.csv', index=False)

# test = pd.read_csv('data/test_sessions.csv')
# for i in range(1, 11):
#     print('Processing Col:' + str(i))
#     test['site' + str(i)] = test['site' + str(i)].replace(Alice_sites, 50000)
    
# for i in range(1, 11):
#     print('Processing Col:' + str(i))
#     test['site' + str(i)] = test['site' + str(i)].replace(Other_sites, 100000)
    
# test.to_csv('test_transformed_data.csv', index=False)

# import pickle

# with open('data/site_dic.pkl', 'rb') as f:
#     data = pickle.load(f)
    
# sites_df = pd.DataFrame([data.keys(), data.values()]).transpose()
# sites_df.columns = ['site', 'index']
# sites_df = sites_df.sort_values('index').set_index('index')
# sites_df.to_csv('site_data.csv', index=False)

# train = pd.read_csv('data/train_sessions.csv')


#DATA PREPARATION
# for i in range(1,11):
#     col_name = 'time' + str(i)
#     train[col_name] = pd.to_datetime(train[col_name])
#     train['date' + str(i)] = pd.to_datetime(train[col_name].dt.date)
#     train[col_name] = train[col_name].dt.time
#     train['site' + str(i)] = train['site' + str(i)].fillna(0)
#     train['site' + str(i)] = train['site' + str(i)].astype('int64')

# col_names = ['session_id']
# for i in range(1,11):
#     col_names.extend(['site' + str(i), 'date' + str(i), 'time' + str(i)])

# col_names.append('target')
# train = train[col_names]

# train.to_csv('train_date_split.csv', index = False)

# test = pd.read_csv('data/test_sessions.csv')

# for i in range(1,11):
#     col_name = 'time' + str(i)
#     test[col_name] = pd.to_datetime(test[col_name])
#     test['date' + str(i)] = pd.to_datetime(test[col_name].dt.date)
#     test[col_name] = test[col_name].dt.time
#     test['site' + str(i)] = test['site' + str(i)].fillna(0)
#     test['site' + str(i)] = test['site' + str(i)].astype('int64')

# col_names = ['session_id']
# for i in range(1,11):
#     col_names.extend(['site' + str(i), 'date' + str(i), 'time' + str(i)])

# test = test[col_names]

# test.to_csv('test_date_split.csv', index = False)

