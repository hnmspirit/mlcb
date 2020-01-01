import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
np.random.seed(21)

X = np.random.rand(100, 2)
y = 2*X[:,0] + 3*X[:,1] + 4 + .2*np.random.randn(100)

model = Sequential([Dense(1, input_shape=(2,), activation='linear')])

sgd = optimizers.SGD(lr=.1)
model.compile(loss='mse', optimizer=sgd)

model.fit(X, y, epochs=100, batch_size=2)
print('\n+ weight:',model.get_weights())