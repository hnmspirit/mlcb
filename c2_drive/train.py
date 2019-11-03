import os
from os.path import isdir, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from utils import INPUT_SHAPE, batch_generator

data_dir = 'sdc'
data_names = ['c','l','r', 'st', 'th', 're', 'sp']
data_df = pd.read_csv(join(data_dir, 'driving_log.csv'), names=data_names)

X = data_df[['c', 'l', 'r']].values
y = data_df['st'].values
plt.hist(y)

id_zero = np.where(y==0)[0]
id_nonz = np.where(y!=0)[0]
id_zero = np.random.permutation(id_zero)[:600]
ids = np.concatenate((id_zero, id_nonz))

X = X[ids]
y = y[ids]

# remove dir in fname
X = np.array([['IMG/{}'.format(p.split('\\')[-1]) for p in x] for x in X])

# split train val
X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2, random_state=21)

model = Sequential([
    Lambda(lambda x: x/127.5 - 1, input_shape=INPUT_SHAPE),
    Conv2D(24, 5, 2, activation='elu'),
    Conv2D(36, 5, 2, activation='elu'),
    Conv2D(48, 5, 2, activation='elu'),
    Conv2D(64, 3, 1, activation='elu'),
    Conv2D(64, 3, 1, activation='elu'),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='elu'),
    Dropout(0.5),
    Dense(50, activation='elu'),
    Dense(10, activation='elu'),
    Dense(1),
])
print(model.summary())

epochs = 10
samples_epoch = 1000
batch_size = 32
lr = 1e-4

if not isdir('models'):
    os.makedirs('models')
callback = ModelCheckpoint('models/model-{epoch:04d}.h5',
                             monitor='val_loss',verbose=0,
                             save_best_only=True, mode='auto')

model.compile(loss='mse',
              optimizer=Adam(lr))

H = model.fit_generator(batch_generator(data_dir, X_trn, y_trn, batch_size, True),
                        steps_per_epoch=samples_epoch, epochs=epochs, verbose=1,
                        validation_data=batch_generator(data_dir, X_val, y_val, batch_size, False),
                        validation_steps=len(X_val),
                        max_queue_size=1, callbacks=[callback])