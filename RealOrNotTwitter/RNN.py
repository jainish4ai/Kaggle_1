import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import string
import re
import datetime
from nltk.corpus import stopwords
from tqdm import tqdm
import os
from tensorboard.plugins import projector

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_data = train_data.drop_duplicates(['text']).reset_index()

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
         'time', 'date', 'number'],

    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},

    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

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

class TwitterTextConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.puncs_table = str.maketrans('', '', string.punctuation)
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        transformed_text = X.apply(lambda x: re.sub(r'(.)\1+', r'\1\1', x))
        transformed_text = transformed_text.apply(lambda text: " ".join(text_processor.pre_process_doc(text)))
        transformed_text = transformed_text.apply(lambda text: " ".join(text.translate(self.puncs_table).strip().split()))
        transformed_text = transformed_text.apply(lambda text: " ".join([word for word in text.split() if word not in (self.stop_words)]))
        transformed_text = transformed_text.apply(lambda text: " ".join([word for word in text.split() if word not in (domain_stop_words)]))
        return transformed_text
    
train_data['transformed_text'] = TwitterTextConverter().fit_transform(train_data['text'])
test_data['transformed_text'] = TwitterTextConverter().fit_transform(test_data['text'])

ohe = OneHotEncoder(drop = 'first', sparse=False)
train_data[['keyword']] = train_data[['keyword']].fillna('None')
test_data[['keyword']] = test_data[['keyword']].fillna('None')
train_keyword = ohe.fit_transform(train_data[['keyword']])
test_keyword = ohe.transform(test_data[['keyword']])

tokenizer = Tokenizer(filters = '', lower = False)
tokenizer.fit_on_texts(train_data.transformed_text)

X_train = tokenizer.texts_to_sequences(train_data.transformed_text)
X_test=tokenizer.texts_to_sequences(test_data.transformed_text)
vocab_size = len(tokenizer.word_index) + 1

maxlen = max(train_data.transformed_text.apply(lambda x: len(x.split())))

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
word_index=tokenizer.word_index

# Set up a logs directory, so Tensorboard knows where to look for files
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


      
embedding_dict={}
with open('/Users/jainish/dev/Embeddings/glove.twitter.27B/glove.twitter.27B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

vocab_size=len(word_index)+1
embedding_matrix=np.zeros((vocab_size,100))
for word,i in tqdm(word_index.items()):
    if i > vocab_size:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec

# model = keras.Sequential([
#         keras.layers.Embedding(vocab_size,embedding_matrix.shape[1],
#                                embeddings_initializer=Constant(embedding_matrix),
#                                input_length=maxlen, trainable=False, mask_zero=True),
#         keras.layers.SpatialDropout1D(.2),
#         keras.layers.LSTM(100, return_sequences=True, recurrent_dropout=.2),
#         keras.layers.LSTM(50, recurrent_dropout=.2, return_sequences = False),
#         keras.layers.Dense(20, activation = 'eelu'),
#         keras.layers.Dense(1, activation = 'sigmoid')
#     ]
#     )

def dump_embeddings(layer, subwords, vocab_size):
    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
      for subword in subwords:
        f.write("{}\n".format(subword))
      # Fill in the rest of the labels with "unknown"
      for unknown in range(1, vocab_size - len(subwords)):
        f.write("unknown #{}\n".format(unknown))
    
    
    # Save the weights we want to analyse as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, so
    # we will remove that value.
    weights = tf.Variable(layer.get_weights()[0][1:])
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    
input_ = keras.layers.Input(shape=X_train.shape[1:])
embedding = keras.layers.Embedding(vocab_size,embedding_matrix.shape[1],
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=maxlen, trainable=False, mask_zero=True)(input_)
model = keras.layers.SpatialDropout1D(.1)(embedding)
model = keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True, recurrent_dropout=.1))(model)
model = keras.layers.LSTM(50, recurrent_dropout=.1, return_sequences = False)(model)

input_2 = keras.layers.Input(shape=train_keyword.shape[1:])

model = keras.layers.concatenate([input_2, model])
model = keras.layers.Dense(64, activation = 'elu')(model)
model = keras.layers.Dense(32, activation = 'elu')(model)
model = keras.layers.Dense(1, activation = 'sigmoid')(model)
model = keras.Model(inputs=[input_, input_2], outputs=[model])

print(model.summary())
model.layers[0].trainable = True
model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.SGD(lr = .08), metrics = ['accuracy'])


checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')
model.fit((X_train, train_keyword), train_data.target, epochs=40, callbacks=[tensorboard_callback, checkpoint, early_stopping], 
          shuffle=True, validation_split=.2, batch_size=64)

model.load_weights('best_model.h5')

dump_embeddings(model.layers[1], word_index.keys(), vocab_size)
# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

predictions = np.round(model.predict(X_train)).astype('int').ravel()
incorrect_predictions = train_data[predictions != train_data.target]
print (classification_report(train_data.target, predictions))


predictions = np.round(model.predict((X_test, test_keyword))).astype('int64').ravel()

output = pd.DataFrame({'id': test_data.iloc[:,0], 'target': predictions})
output.to_csv('output.csv', index = False)