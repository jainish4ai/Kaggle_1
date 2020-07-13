import pandas as pd
import string
from nltk.corpus import stopwords
import re
import numpy as np
train = pd.read_csv('data/train.csv')
train = train.drop_duplicates(['text'])

stop_words = stopwords.words('english')
puncs_to_remove = string.punctuation.replace('@', '')
table = str.maketrans('', '', puncs_to_remove)

def process_text(text):
    words = text.split()
    words = [re.sub(r'^http?:\/\/.*[\r\n]*', '', word) for word in words]
    words = [word if word == word.upper() else word.lower() for word in words]
    words = [word for word in words if word not in stop_words]
    words = [word.translate(table) for word in words]
    return set(words)

def clean_data(dataset):
    dataset['keyword'] = dataset.keyword.fillna('None')
    dataset['words'] = dataset.text.apply(process_text)
    return dataset

train=clean_data(train)

word_counter = {}
def counter(row):
    for word in row['words']:
        if word in word_counter:
            word_counter[word][row.target] =  word_counter[word][row.target] + 1
        else:
            word_counter[word] = [0,0]
            word_counter[word][row.target] = 1
    return row

train.apply(counter, axis = 1)
target_0 = []
target_1 = []
for val in word_counter.values():
    target_0.append(val[0])
    target_1.append(val[1])
data = pd.DataFrame([word_counter.keys(), target_0, target_1]).T
data.columns = ['Word', 'Target_0', 'Target_1']
data['Total'] = data['Target_0'] + data['Target_1']

data['Prop'] = data['Target_0']/data['Total']
temp = data[(data.Prop > .47) & (data.Prop < .53) ]
temp.sort_values('Total', ascending = False)[:300].Word.values
