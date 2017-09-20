import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

# generate random training examples
def get_sequence(n_timesteps):
    limit = n_timesteps / 4.0
    X = np.random.uniform(low=0.0, high=1.0, size=n_timesteps)
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
    X = X.reshape(1, n_timesteps, 1) 
    y = y.reshape(1, n_timesteps, 1)
    return (X, y)

X, y = get_sequence(10)

n_timesteps = 10
n_hidden_units = 20
model = Sequential()
model.add(LSTM(n_hidden_units, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

for epoch in range(1000):
	X, y = get_sequence(n_timesteps)
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)

X, y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
