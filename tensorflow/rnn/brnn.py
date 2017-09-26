import numpy as np
from random import random
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# generate random training examples
def get_sequence(n_timesteps):
    limit = n_timesteps / 4.0
    X = np.random.uniform(low=0.0, high=1.0, size=n_timesteps)
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
    X = X.reshape(1, n_timesteps, 1) 
    y = y.reshape(1, n_timesteps, 1)
    return (X, y)

def get_lstm_model(n_timesteps, backwards):
	model = Sequential()
	model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def get_bi_lstm_model(n_timesteps, mode):
	model = Sequential()
	model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def train_model(model, n_timesteps):
	loss = list()
	for _ in range(250):
		X, y = get_sequence(n_timesteps)
		# fit model for one epoch on this sequence
		hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
		loss.append(hist.history['loss'][0])
	return loss

X, y = get_sequence(10)

n_timesteps = 10
results = DataFrame()

# # lstm forwards
# model = get_lstm_model(n_timesteps, False)
# results['lstm_forw'] = train_model(model, n_timesteps)

# # lstm backwards
# model = get_lstm_model(n_timesteps, True)
# results['lstm_back'] = train_model(model, n_timesteps)

# # bidirectional concat
# model = get_bi_lstm_model(n_timesteps, 'concat')
# results['bilstm_con'] = train_model(model, n_timesteps)

# sum merge
model = get_bi_lstm_model(n_timesteps, 'sum')
results['bilstm_sum'] = train_model(model, n_timesteps)

# mul merge
model = get_bi_lstm_model(n_timesteps, 'mul')
results['bilstm_mul'] = train_model(model, n_timesteps)

# avg merge
model = get_bi_lstm_model(n_timesteps, 'ave')
results['bilstm_ave'] = train_model(model, n_timesteps)

# concat merge
model = get_bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)

# line plot of results
results.plot()
pyplot.show()