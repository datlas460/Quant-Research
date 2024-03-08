import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from pylab import plt, mpl
#
# hours = np.array([0.5, 0.75, 1., 1.25, 1.5, 1.75, 1.75, 2.,
#                   2.25, 2.5, 2.75, 3., 3.25, 3.5, 4., 4.25,
#                   4.5, 4.75, 5., 5.5])
#
# success = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
#                     0, 1, 1, 1, 1, 1, 1])
#
# data = pd.DataFrame({'hours': hours, 'success': success})
#
# model = MLPClassifier(hidden_layer_sizes=[32], max_iter=1000, random_state=100)
#
# model.fit(data['hours'].values.reshape(-1, 1), data['success'])
#
# data['prediction'] = model.predict(data['hours'].values.reshape(-1, 1))
#
# data.plot(x='hours', y=['success', 'prediction'], style=['ro', 'b-'], ylim=[-.1, 1.1], figsize=(10,6))
# plt.show()

raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                  index_col=0,
                  parse_dates=True).dropna()

symbol = 'EUR='

data = pd.DataFrame(raw[symbol])
data.rename(columns={symbol: 'price'}, inplace= True)

data['return'] = np.log(data['price']/data['price'].shift(1))
data['direction'] = np.where(data['return']>0, 1, 0)

lags = 5

cols = []
for lag in range(1, lags+1):
    col = f'lag_{lag}'
    data[col] = data['return'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

print(data.round(4).tail())

optimizer = Adam(learning_rate=.0001)

import random
def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(100)

set_seeds()
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

cutoff = '2017-12-31'

training_data = data[data.index < cutoff].copy()
mu, std = training_data.mean(), training_data.std()
training_data_ = (training_data-mu)/std
test_data = data[data.index>cutoff].copy()

test_data_ =(test_data - mu)/std

model.fit(training_data[cols],
          training_data['direction'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False)

print(model)

res = pd.DataFrame(model.history.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')
plt.show()

model.evaluate(training_data_[cols], training_data['direction'])

pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)

pred[:30].flatten()

training_data['prediction'] = np.where(pred > 0, 1, -1)
training_data['strategy'] = (training_data['prediction'] *
                             training_data['return'])

training_data[['return', 'strategy']].sum().apply(np.exp)
training_data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))

model.evaluate(test_data_[cols], test_data['direction'])






