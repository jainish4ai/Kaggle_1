import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.metrics import plot_confusion_matrix, classification_report
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
              'Rev.':5, 'Dr.':6, 'Col.':6, 'Major.':6}
    
    for title in titles.keys():
        if title in name:
            return titles[title]

    return 6

def transform_familysize(familysize):
    sizes = {0: 0, 2: 1, 3:2, 6: 3}
    for size in sizes.keys():
        if familysize <= size:
            return sizes[size]
        
    return 4

def transform_cabin(cabin):
    if pd.isnull(cabin):
        return 0
    
    cabins_cat = {1: ['B', 'D', 'E'],
              2: ['C', 'F'],
              3: ['A', 'G']}
    
    for _, (cat, cabins) in enumerate(cabins_cat.items()):
        for cc in cabins:
            if cc in cabin:
                return cat
    
    return 0


data['CategoricalFare'] = pd.qcut(data['Fare'],4, labels = False)
data['LS'] = data['Name'].apply(transform_name)
data['FamSize'] = data['Parch'] + data['SibSp']
data['FamSize'] = data['FamSize'].apply(transform_familysize)
data['Embarked'] = data.Embarked.fillna('C')
data['Embarked'] = data['Embarked'].map({'C':1, 'S': 2, 'Q': 3})
data['Cabin_Grade'] = data['Cabin'].apply(transform_cabin)
data['Age'] = data.groupby('LS')['Age'].transform(lambda x: x.fillna(x.mean()))
data['CategoricalAge'] = pd.qcut(data['Age'], 4, labels=False)
data['HasCabin'] = (~data['Cabin'].isna()).astype(int)
data['IsAlone'] = (data.FamSize == 0).astype(int)


columns = [
    'Pclass', 
    # 'Sex',
    'LS',
    'CategoricalFare',
    'CategoricalAge',
    'FamSize',
    'Embarked',
    'Cabin_Grade',
    'HasCabin',
    'IsAlone',
    ]

X = data[columns]
y = data[['Survived']]

transformer = ColumnTransformer(
    [
           ('Pclass', OneHotEncoder(drop = 'first'), ['Pclass']),
            # ('Sex', OneHotEncoder(drop = 'first'), ['Sex']),
         ('LS', OneHotEncoder(drop = 'first'), ['LS']),
         ('FamSize', OneHotEncoder(drop = 'first'), ['FamSize']),
         ('Embarked', OneHotEncoder(drop = 'first'), ['Embarked']),
          ('Cabin_Grade', OneHotEncoder(drop = 'first'), ['Cabin_Grade']),
          ('CategoricalFare', OneHotEncoder(drop = 'first'), ['CategoricalFare']),
         ('CategoricalAge', OneHotEncoder(drop = 'first'), ['CategoricalAge'])
    ],
    remainder = 'passthrough'
)

features = transformer.fit_transform(X)

train_features = features[train_data.index].toarray()
train_labels = y.iloc[train_data.index,:].values.ravel()
test_features = features[-len(test_data.index):].toarray()

estimators = [
        ('XGB',XGBClassifier(gamma = 0.0001, learning_rate = 0.01, max_depth = 11, n_estimators = 525, reg_alpha = 0, reg_lambda =  2.45)), 
        ('RF', RandomForestClassifier(max_depth = 7, max_features = 0.8111111111111111, n_estimators= 225)), 
        ('ABC',AdaBoostClassifier()),
        ('ET', ExtraTreesClassifier(max_depth = 7, n_estimators = 15, max_features=.9166666, min_impurity_decrease=.001)),
        ('GBM', GradientBoostingClassifier())
    ]
models = [
    XGBClassifier(gamma = 0.0001, learning_rate = 0.01, max_depth = 11, n_estimators = 525, reg_alpha = 0, reg_lambda =  2.45), 
    AdaBoostClassifier(), 
    GradientBoostingClassifier(),
    RandomForestClassifier(max_depth = 7, max_features = 0.8111111111111111, n_estimators= 225),
    ExtraTreesClassifier(max_depth = 7, n_estimators = 15, max_features=.9166666, min_impurity_decrease=.001),
    VotingClassifier(estimators),
    BaggingClassifier(XGBClassifier())
    ]


results = pd.DataFrame(columns=['Model', 'Mean Train Score', 'Mean Val Score'])
for model in models:
    cv_results = cross_val_score(model, train_features, train_labels, cv = 5, scoring = 'accuracy', n_jobs= -1)
    results = results.append(pd.Series({'Model': type(model).__name__,
                                        'Mean Train Score': model.fit(train_features, train_labels)
                                        .score(train_features, train_labels),
                                        'Mean Val Score': cv_results.mean()}),
                              ignore_index = True)

print(results)
best_model = models[results['Mean Val Score'].argmax()]
best_model = models[4]
print ('Best Model selected:', type(best_model).__name__)
best_model.fit(train_features, train_labels)
predictions = best_model.predict(train_features)
plot_confusion_matrix(best_model, train_features, train_labels)
plt.show()
print (classification_report(train_labels, predictions))

incorrect_predictions = data.iloc[0:891,:][predictions != train_labels]

predictions = best_model.predict(test_features).astype('int64')

output = pd.DataFrame({'PassengerId': test_data.iloc[:,0], 'Survived': predictions})
output.to_csv('output.csv', index = False)