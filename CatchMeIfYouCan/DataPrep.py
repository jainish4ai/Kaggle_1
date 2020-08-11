import pandas as pd
import pickle

train_df = pd.read_csv('data/train_sessions.csv')


with open('data/site_dic.pkl', 'rb') as f:
    sites = pickle.load(f)
    
sites=dict(zip(sites.values(), sites.keys()))
sites[0] = 'None'
sites_cols = ['site' + str(i) for i in range(1, 11)]

def transform_df(df):
    df[sites_cols] = df[sites_cols].fillna(0).astype('int64')
    df[sites_cols] = df[sites_cols].applymap(lambda x: sites[x])
    
    def in_office_hours(x):
        if pd.isnull(x):
            return -1
        elif x.hour >= 9 and x.hour <=18:
            return 1
        else:
            return 0
    
    for i in range(1, 11):
        time_col = 'time'+str(i)
        df[time_col] = pd.to_datetime(df[time_col])
        df['Office_Hours' + str(i)] = df['time' + str(i)].transform(in_office_hours)
    
    df['TotalSites'] = (df[sites_cols] != 'None').sum(axis = 1)
    # for i in range(1,10):
    #     df['diff' + str(i)] = (df['time'+str(i+1)] - df['time'+str(i)]).dt.total_seconds()
    return df

X_train = transform_df(train_df)

test = pd.read_csv('data/test_sessions.csv')
test_df = transform_df(test)

X_train.to_csv('Processed_train.csv', index = False)
test_df.to_csv('Processed_test.csv', index = False)
