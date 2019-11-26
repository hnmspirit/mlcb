import os
from os.path import join
import numpy as np
from scipy.sparse import coo_matrix as cmat
from sklearn.naive_bayes import MultinomialNB as MNB, BernoulliNB as BNB
from sklearn.metrics import accuracy_score

path = 'ex6_prep/'
X_trn_fn = join(path, 'train-features.txt')
y_trn_fn = join(path, 'train-labels.txt')
X_tst_fn = join(path, 'test-features.txt')
y_tst_fn = join(path, 'test-labels.txt')

nwords = 2500

def read_data(X_fn, y_fn):
    X = np.loadtxt(X_fn)
    y = np.loadtxt(y_fn)
    row = X[:,0] - 1
    col = X[:,1] - 1
    dat = X[:,2]

    X = cmat((dat, (row, col)), shape=(len(y), nwords))
    return X, y

X_trn, y_trn = read_data(X_trn_fn, y_trn_fn)
X_tst, y_tst = read_data(X_tst_fn, y_tst_fn)
print('train size: ', y_trn.shape)

model = MNB()
model.fit(X_trn, y_trn)

y_prd = model.predict(X_tst)
score = accuracy_score(y_tst, y_prd)
print('score: ', score)
