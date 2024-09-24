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

# 1. Stack more LSTM layers to capture hierarchical time dependencies.
# 2. Increase sequence length to capture longer-term dependencies.
# 3. Use Bidirectional LSTMs to capture both forward and backward temporal dependencies.
# 4. Add Dropout layers for regularization to avoid overfitting.
# 5. Ensure your data is normalized or standardized.
# 6. Try using GRUs for faster training and fewer parameters.
# 7. Add L2 regularization to prevent overfitting in Dense layers.
# 8. Experiment with LeakyReLU or other activations.
# 9. Use a learning rate scheduler to reduce the learning rate during training.
# 10. Tune hyperparameters like batch size, optimizer, and the number of LSTM units.

model = Sequential([
    layers.Input(shape=(10, 1)),
    # layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    # layers.Bidirectional(layers.LSTM(64)),
    # layers.Bidirectional(layers.LSTM(32)),
    # layers.GRU(64),  # Replacing LSTM with GRU
    # layers.GRU(32),
    # layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # punishes higher weights
    # layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.LSTM(16, return_sequences=True), 
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.LSTM(32, return_sequences=True), # LTSM(32) and LTSM(64) seems to run well
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    layers.LSTM(64), 
    # layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    # layers.LSTM(128),
    # layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

# This seemed to run well
    # layers.Input(shape=(12, 1)),
    # layers.LSTM(32, return_sequences=True), 
    # layers.Dense(32, activation='relu'), 
    # layers.LSTM(64), 
    # layers.Dense(64, activation='relu'), 
    # layers.Dense(32, activation='relu'),
    # layers.Dense(16, activation='relu'),
    # layers.Dense(1)

model.summary()

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.0001),
              metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=400, restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=1e-6)

model.fit(format.X_train, format.y_train, validation_data=(format.X_val, format.y_val), epochs=5000, batch_size=16, callbacks=[early_stopping, lr_scheduler])

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