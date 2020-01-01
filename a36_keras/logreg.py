import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
np.random.seed(21)

means = [[2,3], [4,1]]
cov = [[.3,0], [0,.3]]
N = 50
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.vstack((X0, X1))
y = np.array([0]*N + [1]*N)

model = Sequential([Dense(1, input_shape=(2,), activation='sigmoid')])

sgd = optimizers.SGD(lr=.2)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X, y, epochs=3, batch_size=4)
print()

w, b = model.get_weights()
print(model.get_weights())


xv = np.array([0,5])
yv = -(b[0] + w[0,0]*xv)/w[1,0]

plt.plot(xv, yv, 'k')
plt.plot(X0[:,0], X0[:,1], 'b.')
plt.plot(X1[:,0], X1[:,1], 'r.')
plt.grid(alpha=0.5)
plt.show()