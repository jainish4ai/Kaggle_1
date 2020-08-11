import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
from tensorflow import keras
from sklearn.utils import class_weight
import datetime

train_df = pd.read_csv('Processed_train.csv')

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_df.target),
                                                 train_df.target)

with open('data/site_dic.pkl', 'rb') as f:
    sites = pickle.load(f)
    
sites['None'] = 0
sites_col = ['site'+str(i) for i in range(1, 11)]

train_df[sites_col] = train_df[sites_col].applymap(lambda x: str(sites[x]))
X_sites = train_df[sites_col].astype('int64')

for site_col in site_cols:
    print('Processing COl:', site_col)
    X_sites[site_col] = X_sites[site_col].replace(Alice_sites.index.astype('int64').values, 48372)
    X_sites[site_col] = X_sites[site_col].replace(Other_sites.index.astype('int64').values, 48373)
    
X_hrs = train_df[['Office_Hours'+str(i) for i in range(1,11)]]
X_NonSites = pd.concat([X_hrs, train_df['TotalSites']], axis = 1).astype('int64')

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    def _log_gradients(self, epoch):
        writer = self._writers['train']

        with writer.as_default(), tf.GradientTape() as g:
            # here we use test data to calculate the gradients
            features1 = tf.convert_to_tensor(X_sites)
            features2 = tf.convert_to_tensor(X_NonSites)
            y_true =tf.convert_to_tensor(train_df.target)

            y_pred = self.model([features1, features2])  # forward-propagation
            loss = self.model.compiled_loss(y_true=y_true, y_pred=y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function.
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)
            
input_1 = keras.layers.Input(shape=(10,))
embeddings = keras.layers.Embedding(input_dim=len(sites)+2, output_dim=20,
                           trainable = True, name='Embedding',
                           embeddings_initializer=tf.keras.initializers.random_uniform())(input_1)
lstm = keras.layers.Flatten()(embeddings)
input_2 = keras.layers.Input(shape=(11,))
concat = keras.layers.Concatenate()([lstm, input_2])
dense_1 = keras.layers.Dense(64, activation = 'elu', name = 'Dense1')(concat)
dense_2 = keras.layers.Dense(32, activation = 'elu', name = 'Dense2')(dense_1)
output = keras.layers.Dense(1, activation = 'sigmoid', name = 'Dense3')(dense_2)

model = keras.Model([input_1, input_2], output)

print(model.summary())

model.compile(optimizer = keras.optimizers.SGD(lr = 0.006), loss = keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.AUC(name = 'auc')])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = ExtendedTensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1)
es = keras.callbacks.EarlyStopping(patience=20)
cp = keras.callbacks.ModelCheckpoint('best_model1.h5', save_best_only=True)
model.fit([X_sites, X_NonSites], train_df.target, epochs = 100, batch_size = 128, validation_split=.2, 
          class_weight=dict(enumerate(class_weights)), callbacks = [tb, es, cp])

model.load_weights('best_model1.h5')
#######################################################################################


test_df = pd.read_csv('Processed_test.csv')
test_df[sites_col] = test_df[sites_col].applymap(lambda x: str(sites[x]))
test_sites = test_df[sites_col].astype('int64')

for site_col in site_cols:
    print('Processing COl:', site_col)
    test_sites[site_col] = test_sites[site_col].replace(Alice_sites.index.astype('int64').values, 48372)
    test_sites[site_col] = test_sites[site_col].replace(Other_sites.index.astype('int64').values, 48373)
    
test_hrs = test_df[['Office_Hours'+str(i) for i in range(1,11)]]
test_NonSites = pd.concat([test_hrs, test_df['TotalSites']], axis = 1).astype('int64')

predictions= model.predict([test_sites, test_NonSites]).ravel()

output = pd.DataFrame({'session_id': test_df.iloc[:,0], 'target': predictions})
output.to_csv('output.csv', index = False)