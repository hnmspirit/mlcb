import numpy as np
import scipy.io as sio
from sklearn import linear_model
from sklearn.metrics import accuracy_score as acc
np.random.seed(0)

## data
root = 'randomfaces4ar/randomfaces4ar.mat'
data = sio.loadmat(root)

X = data['featureMat'].T
y = data['labelMat'].T
print('origin: ', X.shape, y.shape)

y1 = np.where(y==1)[1]
y1 = (y1 > 50)*1

id_trn = list(range(0    , 25*26)) + list(range(50*26, 75*26))
id_tst = list(range(25*26, 50*26)) + list(range(75*26, 100*26))

X_trn = X[id_trn]
y_trn = y1[id_trn]
X_tst = X[id_tst]
y_tst = y1[id_tst]

print('train: ', X_trn.shape, y_trn.shape)
print('test: ', X_tst.shape, y_tst.shape)


## model
model = linear_model.LogisticRegression(C=10, solver='liblinear')
model.fit(X_trn, y_trn)


## eval
y_prd = model.predict(X_tst)
accuracy = acc(y_tst, y_prd)
print('accuracy: ', accuracy)