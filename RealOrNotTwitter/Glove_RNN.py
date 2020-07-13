import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import string
import nltk
import ekphrasis
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB, CategoricalNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.initializers import Constant
import re
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from spellchecker import SpellChecker
from tqdm import tqdm

train = pd.read_csv('data/train.csv')
train = train.drop_duplicates(['text'])
test = pd.read_csv('data/test.csv')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()
domain_stop_words = ['us', 'place', 'officers', 'people', 'school', 'streets',
       'emergency', 'second', 'live', 'set', 'plus', 'sky', '\x89',
       'cruz', 'outside', 'taking', 'chicago', 'thousands', 'leave',
       'first', 'year', 'û', 'students', '2', 'blocked', 'among', 'use',
       'happen', 'sunday', 'another', 'fort', 'property', '3', 'plans',
       'happened', 'drive', 'b', 'daily', 'could', 'moment', 'study',
       'remembering', 'cannot', 'family', 'gov', 'via', 'began', 'death',
       'bit', 'trauma', 'omg', 'house', 'cause', 'call', 'hospital',
       'ems', 'target', 'crazy', 'ny', '÷', 'said', 'world', 'drunk',
       'hit', 'must', 'since', 'river', 'national', 'false', 'less',
       'order', 'mo', 'x', 'queen', 'e', 'points', 'late', 'ûªs',
       'toddler', 'library', 'sadly', 'tracks', 'lion', 'women', 'truth',
       'burn', 'trial', 'hunt', 'system', 'burning', 'blamed', 'green',
       'story', 'civil', 'texas', 'worry', 'shift', 'machine', 'act',
       'female', 'abuse', 'worst', 'program', 'flames', 'fully', 'hits',
       'taken', 'dc', 'australia', 'longer', 'usatoday', 'pathogens',
       'certain', 'problem', 'health', 'lies', 'ft', 'drill', 'concerned',
       'americans', 'weapons', 'respect', 'ways', 'fine', 'solar', 'told',
       'mentions', 'stories', 'leaving', 'wall', 'morning', 'associated',
       'smh', 'looked', 'strike', 'half', 'k', 'kept', 'small', 'calls',
       'pics', 'picture', 'board', 'pray', 'bombed', 'elephant', 'ground',
       'mission', 'bells', 'plan', 'ashes', 'five', 'nearby', 'upset',
       'babies', 'nation', 'co', 'camp', 'progress', 'pro', 'sparked',
       'keeps', 'parents', 'accidentally', 'ppl', 'stupid', 'seems',
       'casualty', 'arsenal', 'industry', 'ultimate', 'company', 'felt',
       'chemical', 'seattle', 'pipe', 'landslide', 'tent', 'force',
       'heads', 'rescued', 'loss', 'mr', 'phoenix', 'release', 'blast',
       'especially', 'plague', 'workers', 'evening', 'enugu', 'articles',
       'derail', 'recovery', 'updates', 'chief', 'message', 'jam',
       'information', 'seismic', 'jst', 'senator', 'engulfed', 'hijacker',
       'hijacking', 'held', 'radiation', 'reactor', 'larger', 'snowstorm']

puncs_to_remove = str.maketrans('', '', string.punctuation)

text_processor = TextPreProcessor(
    # terms that will be normalized
    # normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
    #     'time', 'url', 'date', 'number'],
    # normalize = ['number', 'date', 'time'],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
        # 'emphasis', 'censored'},
    # annotate = {'hashtag', 'allcaps'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def process_text(text):
    url1 = re.compile(r'https?://\S+|www\.\S+')
    url2 = re.compile(r'https?://\S+|www\.\S+')
    text = url1.sub(r'',text)
    text = url2.sub(r'',text)
    text = " ".join(text_processor.pre_process_doc(text))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [word.translate(puncs_to_remove) for word in words]
    words = [word for word in words if word != '']
    words = [word for word in words if word not in domain_stop_words]
    # words = [spell.correction(word) for word in words]
    return str.join(' ',words).strip()

def clean_data(dataset):
    dataset['keyword'] = dataset.keyword.fillna('None')
    dataset['text1'] = dataset.text.apply(process_text)
    return dataset

train=clean_data(train)
train = train.drop_duplicates(subset = ['text1'])
test = clean_data(test)

tokenizer = Tokenizer(filters = '', lower = False)
tokenizer.fit_on_texts(train.text1)

X_train = tokenizer.texts_to_sequences(train.text1)
X_test = tokenizer.texts_to_sequences(test.text1)

vocab_size = len(tokenizer.word_index) + 1

maxlen = max(train.text1.apply(lambda x: len(x.split())))

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
word_index=tokenizer.word_index



embedding_dict={}
with open('/Users/jainish/Embeddings/glove.6B/glove.6B.200d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

vocab_size=len(word_index)+1
embedding_matrix=np.zeros((vocab_size,200))

for word,i in tqdm(word_index.items()):
    if i > vocab_size:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
        
model = keras.Sequential([
        keras.layers.Embedding(vocab_size,200,embeddings_initializer=Constant(embedding_matrix),
                   input_length=maxlen,trainable=False),
        # keras.layers.Dropout(0.3),
        keras.layers.Bidirectional(keras.layers.LSTM(128)),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(64),
        keras.layers.Dense(1, activation = 'sigmoid')
    ]
    )


print(model.summary())
model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=0.00005), metrics = ['accuracy'])

model.fit(X_train, train.target, epochs =15, validation_split=.2, batch_size=128)

predictions = np.round(model.predict(X_train)).astype('int').ravel()
incorrect_predictions = train[predictions != train.target]
plt.show()
print (classification_report(train.target, predictions))

predictions = np.round(model.predict(X_test)).astype('int64').ravel()

output = pd.DataFrame({'id': test.iloc[:,0], 'target': predictions})
output.to_csv('output.csv', index = False)