import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json

import time
start_time = time.time()

train = pd.read_csv('data/processed/train.csv', header=0, index_col=0).values.astype('float32')
valid = pd.read_csv('data/processed/valid.csv', header=0, index_col=0).values.astype('float32')
test = pd.read_csv('data/processed/test.csv', header=0, index_col=0).values.astype('float32')


def plot_loss(history, title):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(title)
    plt.xlabel('Nb Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    val_loss = history.history['val_loss']
    min_idx = np.argmin(val_loss)
    min_val_loss = val_loss[min_idx]
    print('Minimum validation loss of {} reached at epoch {}'.format(min_val_loss, min_idx))

n_lag = 1

train_data_gen = TimeseriesGenerator(train, train, length=n_lag, sampling_rate=1, stride=1, batch_size = 5)
valid_data_gen = TimeseriesGenerator(train, train, length=n_lag, sampling_rate=1, stride=1, batch_size = 1)
test_data_gen = TimeseriesGenerator(test, test, length=n_lag, sampling_rate=1, stride=1, batch_size = 1)

stacked_lstm = Sequential()
stacked_lstm.add(LSTM(1, input_shape=(n_lag, 1), return_sequences=True))
stacked_lstm.add(LSTM(24, return_sequences=True))
stacked_lstm.add(LSTM(24, return_sequences=True))
stacked_lstm.add(LSTM(24, return_sequences=True))
stacked_lstm.add(LSTM(24, return_sequences=True))
stacked_lstm.add(LSTM(24, return_sequences=True))
stacked_lstm.add(LSTM(24, return_sequences=True))
stacked_lstm.add(LSTM(24, return_sequences=True))
stacked_lstm.add(LSTM(24))
stacked_lstm.add(Dense(1))
stacked_lstm.compile(loss='mae', optimizer=RMSprop())

earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

stacked_lstm_history = stacked_lstm.fit(train_data_gen
                                                  , epochs=100
                                                  , validation_data=valid_data_gen
                                                  , verbose=1
                                                  , callbacks=[earlystopper])


end_time = time.time()
total_time = end_time - start_time

print(f"Time running: {total_time} s")

plot_loss(stacked_lstm_history, 'Stacked LSTM - Train & Validation Loss')
