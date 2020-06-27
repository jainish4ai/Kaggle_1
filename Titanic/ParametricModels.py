import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.svm import SVC
import pickle

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

data = train_data.append(test_data)

def break_fare(group):
    return group.mean()

data['Fare'] = data.groupby('Ticket')['Fare'].transform(lambda x: x.mean()/len(x))
data['Fare'] = data.groupby('Pclass')['Fare'].transform(lambda x: x.replace(0, x.sum()/len(x!=0)))
data['Fare'] = data.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))

def transform_name(name):
    titles = {'Mr.':1, 'Mrs.':2, 'Mlle.':2, 'Miss.':3, 'Mme.':3, 'Master.':4,
              'Rev.':5, 'Dr.':6, 'Col.':7, 'Major.':7}
    
    for title in titles.keys():
        if title in name:
            return titles[title]

    return 8 

def transform_familysize(familysize):
    sizes = {0: 'XS', 2: 'S', 3:'M', 6: 'L'}
    for size in sizes.keys():
        if familysize <= size:
            return sizes[size]
        
    return 'XL'

def transform_cabin(cabin):
    if pd.isnull(cabin):
        return 'None'
    
    cabins_cat = {'Safe': ['B', 'D', 'E'],
              'Med': ['C', 'F'],
              'Ok': ['A', 'G']}
    
    for _, (cat, cabins) in enumerate(cabins_cat.items()):
        for cc in cabins:
            if cc in cabin:
                return cat
    
    return 'None'
        

data['LS'] = data['Name'].apply(transform_name)
data['FamSize'] = data['Parch'] + data['SibSp']
data['FamSize'] = data['FamSize'].apply(transform_familysize)
data.loc[[61,829]]['Embarked'] = 'C'
data['Embarked'] = data['Embarked'].apply(lambda x: 'C' if x == 'C' else 'O')
data['Cabin_Grade'] = data['Cabin'].apply(transform_cabin)

columns = [
    'Pclass', 
    'Sex',
    'LS',
    # 'Fare',
    'FamSize',
    'Embarked',
    'Cabin_Grade'
    ]

X = data[columns]
y = data[['Survived']]

transformer = ColumnTransformer(
    [
         ('Pclass', OneHotEncoder(drop = 'first'), ['Pclass']),
         ('Sex', OneHotEncoder(drop = 'first'), ['Sex']),
         ('LS', OneHotEncoder(drop = 'first'), ['LS']),
         ('FamSize', OneHotEncoder(drop = 'first'), ['FamSize']),
         ('Embarked', OneHotEncoder(drop = 'first'), ['Embarked']),
         ('Cabin_Grade', OneHotEncoder(drop = 'first'), ['Cabin_Grade']),
         # ('Fare', PowerTransformer(), ['Fare'])
    ],
    remainder = 'passthrough'
)

features = transformer.fit_transform(X)

train_features = features[train_data.index]
train_labels = y.iloc[train_data.index,:].values.ravel()
test_features = features[-len(test_data.index):].toarray()

models = [LogisticRegression(random_state=0), SVC(random_state=0), 
          KNeighborsClassifier(), RidgeClassifier()]

results = pd.DataFrame(columns=['Model', 'Mean Train Score', 'Mean Val Score'])
for model in models:
    cv_results = cross_val_score(model, train_features, train_labels, cv = 10, scoring = 'accuracy', n_jobs= -1)
    results = results.append(pd.Series({'Model': type(model).__name__,
                                        'Mean Train Score': model.fit(train_features, train_labels)
                                       .score(train_features, train_labels),
                                        'Mean Val Score': cv_results.mean()}),
                             ignore_index = True)

print(results)

modified_test_data = pd.DataFrame({'PassengerId': test_data['PassengerId']})
modified_test_data = pd.concat([modified_test_data, pd.DataFrame(test_features)], ignore_index=True, axis=1)
modified_test_data.to_csv('test_features.csv', index = False)


filename = 'best_model.sav'
models[3].coef_
best_model = models[results['Mean Val Score'].argmax()]
best_model.fit(train_features, train_labels)
print ('Best Model selected:', type(best_model).__name__)
pickle.dump(best_model, open(filename, 'wb'))