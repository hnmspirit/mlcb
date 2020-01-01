import os, glob
from os.path import join
import numpy as np
from sklearn import linear_model, tree, neighbors
from sklearn.metrics import accuracy_score as acc
import cv2
np.random.seed(21)

# DATA
root1 = 'caltech101/sunflower'
root2 = 'caltech101/lotus'

ps1 = sorted(glob.glob(join(root1, '*jpg')))
ps2 = sorted(glob.glob(join(root2, '*jpg')))

# FEATURE
def vectorize_img(p):
    gray = cv2.imread(p,0)
    h, w = gray.shape
    top, bot, lef, rig = 0, 0, 0, 0
    if h > w:
        lef = (h - w)//2
        rig = h - w - lef
    if h < w:
        top = (w - h)//2
        bot = w - h - top
    if h != w:
        gray = cv2.copyMakeBorder(gray, top, bot, lef, rig, cv2.BORDER_CONSTANT, value=255)

    gray = cv2.resize(gray, (32,32))
    vec = gray.reshape(-1)
    return vec

def feature_img(ps):
    vecs = np.array([vectorize_img(p) for p in ps])
    vecs = (vecs - vecs.mean(0)) / vecs.std(0)
    print('feature shape: ', vecs.shape)
    return vecs

X1 = feature_img(ps1)
X2 = feature_img(ps2)

ids1 = np.random.permutation(len(X1))
ids2 = np.random.permutation(len(X2))
lim1 = int(len(ids1)*0.7)
lim2 = int(len(ids2)*0.7)

X_trn = np.vstack((X1[ids1[:lim1]], X2[ids2[:lim2]]))
X_tst = np.vstack((X1[ids1[lim1:]], X2[ids2[lim2:]]))

y_trn = np.array([0]*lim1 + [1]*lim2)
y_tst = np.array([0]*(len(ids1)-lim1) + [1]*(len(ids2)-lim2))

print('train: ', X_trn.shape, y_trn.shape)
print('test: ', X_tst.shape, y_tst.shape)

# MODEL
model = tree.DecisionTreeClassifier()
model.fit(X_trn, y_trn)

y_prd = model.predict(X_tst)
accuracy = acc(y_tst, y_prd)
print('accuracy: ', accuracy)