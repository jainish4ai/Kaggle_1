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
# from spellchecker import SpellChecker

train = pd.read_csv('data/train.csv')
train = train.drop_duplicates(['text'])
test = pd.read_csv('data/test.csv')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

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
    # words = [spell.correction(word) for word in words]
    return words

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
temp = data[(data.Prop > .45) & (data.Prop < .55) ]
temp = temp[temp.Total > 5]
temp.Word.values
