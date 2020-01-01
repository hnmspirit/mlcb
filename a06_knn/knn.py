import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=40)

print('train: ', X_trn.shape, y_trn.shape)
print('test: ', X_tst.shape, y_tst.shape)

# model
dist = cdist(X_tst, X_trn, metric='cosine')
ids = dist.argmin(axis=1)
y_prd = y_trn[ids]

# eval
acc = accuracy_score(y_tst, y_prd)
print('Accuracy of 1NN: {:.3f} %'.format(acc))