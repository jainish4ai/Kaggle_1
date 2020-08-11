import pandas as pd
import pickle
import numpy as np

train = pd.read_csv('data/train_sessions.csv')

temp = train.head()
sites = pd.Series()
temp.info()

groups = []
for i in range(1,11):
    group = train.groupby('site' + str(i)).target.mean()
    sites = sites.append(pd.Series(group.index.astype('int64')))
    groups.append(group)

sites = np.sort(sites.unique())

transformed_df = pd.DataFrame(index = sites.astype('int64'))

for i, group in enumerate(groups):
    transformed_df[str(i)] = group
    
sites_data = transformed_df.sum(axis = 1)

Alice_sites = sites_data[sites_data == 1].index
Other_sites = sites_data[sites_data == 0].index
Alice_site = sites_data.iloc[Alice_sites]
for i in range(1, 11):
    print('Processing Col:' + str(i))
    train['site' + str(i)] = train['site' + str(i)].replace(Alice_sites, 50000)
    
    
    
for i in range(1, 11):
    print('Processing Col:' + str(i))
    train['site' + str(i)] = train['site' + str(i)].replace(Other_sites, 100000)
    
train.to_csv('transformed_data.csv', index=False)

test = pd.read_csv('data/test_sessions.csv')
for i in range(1, 11):
    print('Processing Col:' + str(i))
    test['site' + str(i)] = test['site' + str(i)].replace(Alice_sites, 50000)
    
for i in range(1, 11):
    print('Processing Col:' + str(i))
    test['site' + str(i)] = test['site' + str(i)].replace(Other_sites, 100000)
    
test.to_csv('test_transformed_data.csv', index=False)

import pickle

with open('data/site_dic.pkl', 'rb') as f:
    data = pickle.load(f)
    
sites_df = pd.DataFrame([data.keys(), data.values()]).transpose()
sites_df.columns = ['site', 'index']
sites_df = sites_df.sort_values('index').set_index('index')
sites_df.to_csv('site_data.csv', index=False)

train = pd.read_csv('data/train_sessions.csv')


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

