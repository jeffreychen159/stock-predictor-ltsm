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

print(format.dates_test)

model = Sequential([
    layers.Input(shape=(3701 // 2, 2)),
    layers.LSTM(32, return_sequences=True),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.LSTM(64, return_sequences=True),  
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.LSTM(128, return_sequences=True), 
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.LSTM(256, return_sequences=True),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.LSTM(512), 
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

# Model for train = [:q_90], val = [q_90:], testing [q_95:]
# model = Sequential([
#     layers.Input(shape=(10, 1)),
#     layers.LSTM(16, return_sequences=True), 
#     layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#     layers.LSTM(32, return_sequences=True),
#     layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
#     layers.LSTM(64), 
#     layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)), 
#     layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#     layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#     layers.Dense(1)
# ])


model.summary()

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.0001),
              metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
# callbacks=[early_stopping, lr_scheduler]

model.fit(format.X_train, format.y_train, validation_data=(format.X_val, format.y_val), epochs=500, batch_size=16, callbacks=[early_stopping, lr_scheduler])

model.save('train.h5')

# Clearing old plots
plt.cla() 

# # Plotting for training done
# train_predictions = model.predict(format.X_train).flatten()

# plt.plot(format.dates_train, train_predictions)
# plt.plot(format.dates_train, format.y_train)
# plt.legend(['Training Predictions', 'Training Observations'])

# val_predictions = model.predict(format.X_val).flatten() 

# plt.plot(format.dates_val, val_predictions)
# plt.plot(format.dates_val, format.y_val)
# plt.legend(['Validation Predictions', 'Validation Observations'])

# test_predictions = model.predict(format.X_test).flatten()

# plt.plot(format.dates_test, test_predictions)
# plt.plot(format.dates_test, format.y_test)
# plt.legend(['Testing Predictions', 'Testing Observations'])

# plt.plot(format.dates_train, train_predictions, color='blue')
# plt.plot(format.dates_train, format.y_train, color='slateblue')
# plt.plot(format.dates_val, val_predictions, color='green')
# plt.plot(format.dates_val, format.y_val, color='lime')
# plt.plot(format.dates_test, test_predictions, color='orange')
# plt.plot(format.dates_test, format.y_test, color='wheat')

  
# plt.show()