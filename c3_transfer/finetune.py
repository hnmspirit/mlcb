import os, glob
from os.path import basename, dirname, join
import numpy as np
from PIL import Image
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout

ps = sorted(glob.glob(join('dataset', '*/*')))
X = np.array([np.asarray(Image.open(p).convert('RGB').resize((224,224))) for p in ps])
Y = [basename(dirname(p)) for p in ps]

LE = LabelEncoder()
Y = LE.fit_transform(Y)
LB = LabelBinarizer()
Y = LB.fit_transform(Y)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, Y, test_size=0.2, random_state=21, shuffle=True)



baseModel = VGG16(weights='imagenet',
                  input_shape=(224, 224, 3),
                  pooling='avg',
                  include_top=False)

outs = baseModel.output
outs = Dense(256, activation='relu')(outs)
outs = Dropout(0.5)(outs)
outs = Dense(17, activation='softmax')(outs)
model = Model(inputs=baseModel.input, outputs=outs)


aug_trn = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             fill_mode='nearest', horizontal_flip=True)
aug_tst = ImageDataGenerator(rescale=1./255)
batch_size = 64



# freeze VGG model
for layer in baseModel.layers:
    layer.trainable = False

opt = Adam(0.002)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
epochs = 25
H = model.fit_generator(aug_trn.flow(X_trn, y_trn, batch_size=batch_size),
                        steps_per_epoch=len(X_trn)//batch_size, epochs=epochs,
                        validation_data=(aug_tst.flow(X_tst, y_tst, batch_size=batch_size)),
                        validation_steps=len(X_tst)//batch_size)


# unfreeze some last CNN layer:
for layer in baseModel.layers[15:]:
    layer.trainable = True

epochs = 30
opt = SGD(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
H = model.fit_generator(aug_trn.flow(X_trn, y_trn, batch_size=batch_size),
                        steps_per_epoch=len(X_trn)//batch_size, epochs=epochs,
                        validation_data=(aug_tst.flow(X_tst, y_tst, batch_size=batch_size)),
                        validation_steps=len(X_tst)//batch_size)

