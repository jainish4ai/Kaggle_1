import tensorflow as tf
from tensorflow import keras
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re, string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import symspellpy

train = pd.read_csv('data/train.csv')
train = train.drop_duplicates(['text'])
test = pd.read_csv('data/test.csv')

stop_words = stopwords.words('english')
stop_words.append(' ')
lemmatizer = WordNetLemmatizer()
puncs_to_remove = string.punctuation
table = str.maketrans('', '', puncs_to_remove)
domain_stopwords =['', 'A', 'emergency', '4', '1', '2015',
       'another', '\x89Û' , 'live', 'said', 'set', 'house', 'IN',
        'call', 'im','rt','lt',
        'must', 'use', 'out', 'days', 'nearby', 'half', '20',
        'second', 'place', '12', 'order', 'plans',
       'daily', 'crazy', 'worst', 'happened', 'united', '13', 'green',
       'thousands', 'america', 'ready', 'drive', 'true', 'ppl', 'taking',
       'USA', 'told', 'moment', 'parents', 'small', 'information', 'lies',
       'act', 'fully', 'phoenix', '24', 'wall', 'company', 'account',
       'industry', 'mountain', 'five', 'sky', 'bit', 'board', 'late',
       'release', 'drunk', 'keeps', 'seems', 'states', 'picture',
       'pathogens', 'system', 'points', '26', 'ft', 'stupid', 'cancer',
       'began', 'heres', 'streets', 'WAR', 'sadly', 'held',
       'queen', 'machine', 'fox', 'lion', 'created', 'solar', 'nature',
       'economic', 'progress', 'iphone', 'down', 'bells', 'ultimate',
       'church', 'remembering', 'price', 'certain', 'larger', 'salem',
       'waste', 'hunters', 'blamed', 'everyday', 'concerned', 'festival',
       'looked', 'loss', 'tech', 'heads', 'roads', 'jam',
       'especially', 'sparked', 'mentions', 'MS', 'greatest', 'largest',
       'aussie', 'wealth', 'roll', 'have', 'predict', 'dinner', 'fired',
       'reward', 'base', 'exist', 'extra', 'sexual', '5000', '45',
       'decisions', 'daughters', 'just', 'values', 'minds', 'bath',
       'wire', 'bell', 'recover', 'recently', 'network', 'innocent',
       'estate', 'jordan', 'vets', '3rd', 'concern',
       'fund', 'b4', 'factory', 'apocalyptic', 'blame', 'tons', 'clouds',
       'challenge', 'how', 'transit', 'psychiatric', 'humidity', 'DOWN',
       'am', 'materials', 'van', 'starter', 'mess', 'utter', 'fly',
       'childhood', 'wicked', 'prompts', '5pm', 'WERE', 'upper', 'leg',
       'attacking', 'mike', 'TIME', 'FIRST', 'passed', 'groups', 'recall',
       'HOW', 'outdoor', '48', 'birds', 'AREA', 'GPS', 'biological',
       'boot', 'pipe', 'revealed', 'correction', 'crowd', 'ep', 'houston',
       'posted', 'goal', 'dress', 'sources', 'concerns', 'finds', 'mo',
       'targeting', 'charts', 'hosting', 'gift', 'angeles', 'los',
       'fergusons', 'ian', 'richard', 'orange', 'ANOTHER', 'click',
       'crater', 'prince', 'abuse', 'leaves', 'speaking', '3G',
       'produced', 'realise', '360wisenews', 'thru', 'you\x89ûªve',
       'accused', 'sanctions', 'willing', 'previous', 'private',
       'reveals', 'apple', 'ability', '125', 'springs', 'kendall',
       'yellow', 'vietnamese', '@originalfunko', 'mouth', 'wud', 'chick',
       'cain', 'electricity', 'cafe', 'jenner', 'surfers', 'philadelphia',
       'incase', 're', 'hysteria', 'shanghai', 'performing', 'freeway',
       'tidal', 'henry', 'shutdown', 'emerges', 'summit', 'personnel',
       'mohammed', 'mentioned', 'census', 'acquire', 'tanzania',
       'nowhere', 'CANNOT', 'FTE', '@spencers', 'BUDDYS', 'glenn', 'DONE',
       'sunnis', 'wi', 'matches', '64', 'vital', 'federal', 'bn', '1979',
       'JOKE', 'testing', 'language', 'journeys']

# spellcheck = symspellpy.SymSpell()
def process_text(text):
    words = text.split()
    words = [re.sub(r'^http?:\/\/.*[\r\n]*', '', word) for word in words]
    words = [re.sub(r'^https?:\/\/.*[\r\n]*', '', word) for word in words]
    words = nltk.word_tokenize(str.join(' ', words))
    words = [word.lower() for word in words]
    words = [word for word in words if word not in stop_words]
    words = [word.translate(table) for word in words]
    words = [word for word in words if word != '']
    words = [word for word in words if word.strip() not in domain_stopwords]
    words = [lemmatizer.lemmatize(word) for word in words]
    # words = [spellcheck(word) for word in words]
    return str.join(' ',words).strip()

def clean_data(dataset):
    dataset['keyword'] = dataset.keyword.fillna('None')
    dataset['text1'] = dataset.text.apply(process_text)
    return dataset

train=clean_data(train)
train = train.drop_duplicates(subset = ['text1'])
test = clean_data(test)

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(train.text1)

X_train = tokenizer.texts_to_sequences(train.text1)
X_test = tokenizer.texts_to_sequences(test.text1)

vocab_size = len(tokenizer.word_index) + 1

maxlen = max(train.text1.apply(lambda x: len(x.split())))
minlen = min(train.text1.apply(lambda x: len(x.split())))

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

model = keras.Sequential([
        keras.layers.Embedding(vocab_size, 40, input_length=maxlen),
        keras.layers.Dropout(.4),
        keras.layers.Bidirectional(keras.layers.LSTM(32, activation = 'relu')),
        keras.layers.Dropout(.4),
        keras.layers.Dense(1, activation = 'sigmoid')
    ]
    )

print(model.summary())
model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

model.fit(X_train, train.target, epochs =20, validation_split=.2, batch_size=256)

predictions = np.round(model.predict(X_train)).astype('int').ravel()
incorrect_predictions = train[predictions != train.target]
plt.show()
print (classification_report(train.target, predictions))

predictions = np.round(model.predict(X_test)).astype('int64').ravel()

output = pd.DataFrame({'id': test.iloc[:,0], 'target': predictions})
output.to_csv('output.csv', index = False)