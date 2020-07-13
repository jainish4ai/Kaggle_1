import pandas as pd
import string
import nltk
import ekphrasis
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB, CategoricalNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import re
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from spellchecker import SpellChecker

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
    normalize = ['number', 'date', 'time'],
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
test=clean_data(test)
transformer = ColumnTransformer(
    [
      ('tf_idf', TfidfVectorizer(lowercase=False, stop_words=None, ngram_range=(1,1)), 'text1')
    ])

train_features = transformer.fit_transform(train[['text1']]).toarray()
train_labels = train['target'].values.tolist()
models = [
            MultinomialNB(),
            # BernoulliNB(),
         ]

results = pd.DataFrame(columns=['Model', 'Train Score', 'Mean Val Score'])
for model in models:
    cv_results = cross_val_score(model, train_features, train_labels, cv = 8, scoring = 'accuracy', n_jobs= -1)
    results = results.append(pd.Series({'Model': type(model).__name__,
                                        'Train Score': model.fit(train_features, train_labels)
                                       .score(train_features, train_labels),
                                        'Mean Val Score': cv_results.mean()}),
                             ignore_index = True)

print(results)

best_model = models[results['Mean Val Score'].argmax()]
print ('Best Model selected:', type(best_model).__name__)
best_model.fit(train_features, train_labels)
predictions = best_model.predict(train_features)
plot_confusion_matrix(best_model, train_features, train_labels)
plt.show()
print (classification_report(train_labels, predictions))

incorrect_predictions = train[predictions != train_labels]

test_features = transformer.transform(test[['text1']]).toarray()
predictions = best_model.predict(test_features).astype('int64')

output = pd.DataFrame({'id': test.iloc[:,0], 'target': predictions})
output.to_csv('output.csv', index = False)