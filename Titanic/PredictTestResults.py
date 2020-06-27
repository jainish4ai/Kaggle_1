import pickle
import pandas as pd

test_features = pd.read_csv('test_features.csv')

# load the model from disk
filename = 'best_model.sav'
model = pickle.load(open(filename, 'rb'))
predictions = model.predict(test_features.iloc[:,1:]).astype('int64')

output = pd.DataFrame({'PassengerId': test_features.iloc[:,0], 'Survived': predictions})
output.to_csv('output.csv', index = False)
