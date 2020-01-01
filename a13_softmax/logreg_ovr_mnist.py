import sys
sys.path.append('..')

import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score as acc
from utils.mnist_loader import load_mnist
np.random.seed(21)

## DATA
X_path = 'mnist/t10k-images-idx3-ubyte'
y_path = 'mnist/t10k-labels-idx1-ubyte'
X, y = load_mnist(X_path, y_path)

## FEATURE
id_shuffle = np.random.permutation(len(y))
id_trn = id_shuffle[:6000]
id_tst = id_shuffle[-4000:]

X = X.reshape(X.shape[0],-1)/127.5 - 1.
X_trn, y_trn = X[id_trn], y[id_trn]
X_tst, y_tst = X[id_tst], y[id_tst]

print('train: ', X_trn.shape, y_trn.shape)
print('test: ', X_tst.shape, y_tst.shape)

## MODEL
model = linear_model.LogisticRegression(C=1e5,
        solver='liblinear', multi_class='ovr')
model.fit(X_trn, y_trn)

y_prd = model.predict(X_tst)
accuracy = acc(y_tst, y_prd)
print('accuracy: ', accuracy)