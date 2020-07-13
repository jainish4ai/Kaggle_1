import pandas as pd
import string
import nltk
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

train = pd.read_csv('data/train.csv')
train = train.drop_duplicates(['text'])
test = pd.read_csv('data/test.csv')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

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

puncs_to_remove = string.punctuation
table = str.maketrans('', '', puncs_to_remove)

def process_text(text):
    words = text.split()
    words = [re.sub(r'^http?:\/\/.*[\r\n]*', '', word) for word in words]
    words = [re.sub(r'^https?:\/\/.*[\r\n]*', '', word) for word in words]
    words = [word.lower() for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if word not in stop_words]
    words = [word.translate(table) for word in words]
    words = [word for word in words if word.strip() not in domain_stopwords]
    return str.join(' ',words)

def clean_data(dataset):
    dataset['keyword'] = dataset.keyword.fillna('None')
    dataset['text1'] = dataset.text.apply(process_text)
    return dataset

train=clean_data(train)
test=clean_data(test)
transformer = ColumnTransformer(
    [
       # ('keyword', OneHotEncoder(drop = 'first')),
      # ('text', CountVectorizer(), 'text1'),
      ('tf_idf', TfidfVectorizer(lowercase=False, stop_words=None, ngram_range=(1,1)), 'text1')
     ])

train_features = transformer.fit_transform(train[['text1']]).toarray()
train_labels = train['target'].values.tolist()
models = [
            MultinomialNB(),
            BernoulliNB(),
          # CategoricalNB()
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
# result = best_model.predict_proba(train_features)
# result = result[:,0 ] - result[:,1]
# predictions = (result < -.1).astype('int')
predictions = best_model.predict(train_features)
plot_confusion_matrix(best_model, train_features, train_labels)
plt.show()
print (classification_report(train_labels, predictions))

incorrect_predictions = train[predictions != train_labels]

test_features = transformer.transform(test[['text1']]).toarray()
predictions = best_model.predict(test_features).astype('int64')

output = pd.DataFrame({'id': test.iloc[:,0], 'target': predictions})
output.to_csv('output.csv', index = False)