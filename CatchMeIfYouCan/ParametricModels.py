import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv('Processed_train.csv')
test_df = pd.read_csv('Processed_test.csv')

with open('data/site_dic.pkl', 'rb') as f:
    sites = pickle.load(f)
    
sites['None'] = 0
sites_col = ['site'+str(i) for i in range(1, 11)]

train_df[sites_col] = train_df[sites_col].applymap(lambda x: str(sites[x]))

train_df['sites'] = train_df[sites_col].apply(lambda x: x.str.cat(sep = ' '), axis = 1)

cv = CountVectorizer(max_features=40000)
X_train = cv.fit_transform(train_df['sites']).todense()

lr = LogisticRegression(n_jobs=-1)
lr.fit(X_train, train_df.target)
predictions = lr.predict_proba(X_train)

print('AUC score:', roc_auc_score(train_df.target, predictions))