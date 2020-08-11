import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
import datetime

train = pd.read_csv('data/train_sessions.csv')

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train.target),
                                                 train.target)

X = train[['site' + str(i) for i in range(1,11)]]
X = X.fillna(0).astype('int64').values
sites_count = len(np.unique(X))

model = keras.Sequential([
    keras.layers.Input(shape=X.shape[1:]),
    keras.layers.Embedding(input_dim=sites_count, output_dim=10, 
                           mask_zero=True, trainable = True, 
                           embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=1)),
    keras.layers.Flatten(),
    keras.layers.Dense(5, activation = 'elu'),
    keras.layers.Dense(1, activation = 'sigmoid')
    ])

print(model.summary())

model.compile(optimizer = keras.optimizers.SGD(), loss = keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.AUC(name = 'auc')])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=10)
es = keras.callbacks.EarlyStopping(patience=10)
cp = keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
model.fit(X, train.target, epochs = 100, batch_size = 256, validation_split=.2, 
          class_weight=dict(enumerate(class_weights)), callbacks = [tb, es, cp])

model.load_weights('best_model.h5')
#######################################################################################

test = pd.read_csv('data/test_sessions.csv')
X_test = test[['site' + str(i) for i in range(1,11)]]
X_test = X_test.where(X_test < sites_count, 0)
X_test = X_test.fillna(0).astype('int64').values
np.unique(X_test)
predictions= model.predict(X_test).ravel()

output = pd.DataFrame({'session_id': test.iloc[:,0], 'target': predictions})
output.to_csv('output.csv', index = False)