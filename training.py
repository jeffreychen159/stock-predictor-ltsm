import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential  #removed python from each layer
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import datetime

import matplotlib.pyplot as plt

import format

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ReduceLROnPlateau

from copy import deepcopy

print(format.dates_test)

model = Sequential([
    layers.Input(shape=(1, 1)),
    layers.LSTM(64, return_sequences=True), 
    layers.LSTM(128, return_sequences=True),  
    layers.LSTM(256), 
    layers.Dense(256, activation='relu'), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(32, activation='relu'), 
    layers.Dense(1)
])


model.summary()

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.0001),
              metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)

# model.fit(format.X_train, format.y_train, validation_data=(format.X_val, format.y_val), epochs=10, batch_size=16, callbacks=[early_stopping, lr_scheduler])


recursive_predictions = []
recursive_dates = np.concatenate([format.dates_val, format.dates_test])
last_window = deepcopy(format.X_train[-1])

for target_date in recursive_dates:
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    last_window = np.concatenate([last_window[1:], [next_prediction]])

# Clearing old plots
plt.cla() 

# Plotting for training done
# train_predictions = model.predict(format.X_train).flatten()
# val_predictions = model.predict(format.X_val).flatten()
# test_predictions = model.predict(format.X_test).flatten()

# plt.plot(format.dates_train, train_predictions, color='blue')
# plt.plot(format.dates_train, format.y_train, color='slateblue')
# plt.plot(format.dates_val, val_predictions, color='green')
# plt.plot(format.dates_val, format.y_val, color='lime')
# plt.plot(format.dates_test, test_predictions, color='orange')
# plt.plot(format.dates_test, format.y_test, color='wheat')

# plt.plot(recursive_dates, recursive_predictions, color='black')

# plt.xlim(xmin=datetime.datetime(2022, 1, 1), xmax=datetime.datetime(2024, 1, 1))

  
# plt.show()

plt.cla() 

recursive_predictions = []
recursive_dates = np.concatenate([format.date_list])
last_window = deepcopy(format.X_train[-1])

for target_date in recursive_dates:
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    last_window = np.concatenate([last_window[1:], [next_prediction]])