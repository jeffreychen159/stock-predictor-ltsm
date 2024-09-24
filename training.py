import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential  #removed python from each layer
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import datetime

import matplotlib.pyplot as plt

import format

import numpy as np

model = Sequential([layers.Input((12, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.0001),
              metrics=['mean_absolute_error'])

model.fit(format.X_train, format.y_train, validation_data=(format.X_val, format.y_val), epochs=300)

train_predictions = model.predict(format.X_train).flatten()

plt.plot(format.dates_train, train_predictions)
plt.plot(format.dates_train, format.y_train)
plt.legend(['Training Predictions', 'Training Observations'])

val_predictions = model.predict(format.X_val).flatten() 

plt.plot(format.dates_val, val_predictions)
plt.plot(format.dates_val, format.y_val)
plt.legend(['Validation Predictions', 'Validation Observations'])

test_predictions = model.predict(format.X_test).flatten()

plt.plot(format.dates_test, test_predictions)
plt.plot(format.dates_test, format.y_test)
plt.legend(['Testing Predictions', 'Testing Observations'])

plt.plot(format.dates_train, train_predictions)
plt.plot(format.dates_train, format.y_train)
plt.plot(format.dates_val, val_predictions)
plt.plot(format.dates_val, format.y_val)
plt.plot(format.dates_test, test_predictions)
plt.plot(format.dates_test, format.y_test)
plt.legend(['Training Predictions', 
            'Training Observations',
            'Validation Predictions', 
            'Validation Observations',
            'Testing Predictions', 
            'Testing Observations'])


plt.xlim(xmin=datetime.datetime(2020, 1, 1), xmax=datetime.datetime(2025, 1, 1))

# # Predict future
# from copy import deepcopy
# recursive_predictions = []
# recursive_dates = np.concatenate([format.dates_val, format.dates_test])
# last_window_new =  deepcopy(format.X_train[-1])

# for target_date in recursive_dates:

#   next_prediction = model.predict(np.array([last_window_new])).flatten()

#   recursive_predictions.append(next_prediction)
#   last_window_new[0] = last_window_new[1]
#   last_window_new[1] = last_window_new[2]
#   last_window_new[-1] = next_prediction

#   print(format.X_train[-2:])  # Print the last 2 elements of X_train
#   print(np.array([last_window_new]))  # Print the value of np.array([last_window])
  
  
plt.show()