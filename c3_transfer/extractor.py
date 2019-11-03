import os, glob
from os.path import basename, dirname, join
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import VGG16

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

model = VGG16(weights='imagenet',
              input_shape=(224,224,3),
              pooling='avg',
              include_top=False)

ps = sorted(glob.glob(join('dataset', '*/*')))
labels = [basename(dirname(p)) for p in ps]

batch_size = 64
X = []
for i in tqdm(range(0, len(ps), batch_size)):
    pi = ps[i: i+batch_size]
    imgs = np.array([np.asarray(Image.open(p).convert('RGB').resize((224,224))) for p in pi])
    features = model.predict(imgs)
    X.extend(features)


LE = LabelEncoder()
y = LE.fit_transform(labels)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=21)

params = {'C': [0.1, 1.0, 10.0, 100.0]}
model = GridSearchCV(LogReg(), params)
model.fit(X_trn, y_trn)
print('\n*** Best parameter for the model ***\n', model.best_params_)

y_prd = model.predict(X_tst)
print('\n*** Report ***\n', classification_report(y_tst, y_prd))
