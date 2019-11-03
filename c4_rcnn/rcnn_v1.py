import os, glob
from os.path import basename, join
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical


def _annotate(img, annots):
    im1 = img.copy()
    for x1,y1,x2,y2 in annots:
        cv2.rectangle(im1, (x1,y1), (x2,y2), (0,255,0), 1, cv2.LINE_AA)
    plt.figure(figsize=(5,5))
    plt.imshow(im1)


def get_iou(bbA, bbB):
    x1A, y1A, x2A, y2A = bbA
    x1B, y1B, x2B, y2B = bbB

    x1I = max(x1A, x1B)
    y1I = max(y1A, y1B)
    x2I = min(x2A, x2B)
    y2I = min(y2A, y2B)

    if (x2I < x1I) or (y2I < y1I):
        return 0.

    areaI = (x2I - x1I) * (y2I - y1I)
    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)
    iou = areaI / float(areaA + areaB - areaI)
    return iou

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

dir1 = 'Airplanes_Annotations'
dir2 = 'Images'

ps1 = sorted(glob.glob(join(dir1, '*csv')))
ps2 = sorted(glob.glob(join(dir2, '*jpg')))


images = []
labels = []
n_pos = 0

for j, (p1, p2) in enumerate(zip(ps1, ps2)):
    if n_pos > 2000:
        break
    print('+ {}: {} {}'.format(j, basename(p1), basename(p2)))
    annots = np.loadtxt(p1, dtype='int', delimiter=' ', skiprows=1, ndmin=2)

    if annots.shape[0] == 0:
        print('\n\t ***! annotations empty !\n')
        continue

    img = cv2.imread(p2)

    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    preds = ss.process()
    preds = np.array([(x,y,x+w,y+h) for (x,y,w,h) in preds])

    list_pos = []
    list_neg = []

    for bb1 in annots:
        ious = np.array([get_iou(bb1, bb2) for bb2 in preds])
        pos = np.argsort(ious)[-5:]
        pos = [i for i in pos if ious[i] > 0.7]
        neg = np.where(ious < 0.3)[0][:len(pos)]
        list_pos.extend(pos)
        list_neg.extend(neg)
    print('\t\t pos: {}, neg: {} '.format(len(list_pos), len(list_neg)))
    n_pos += len(list_pos)

    img = [cv2.resize(img[y1:y2, x1:x2], (224,224)) for (x1,y1,x2,y2) in preds[list_pos+list_neg]]
    lbl = [1]*len(list_pos) + [0]*len(list_neg)

    images.extend(img)
    labels.extend(lbl)

images = np.array(images)
labels = np.array(labels)


X_trn, X_val, y_trn, y_val = train_test_split(images, labels, test_size=0.1, random_state=21, shuffle=True)
Y_trn = to_categorical(y_trn)
Y_val = to_categorical(y_val)

# train
baseModel = VGG16(weights='imagenet',
                  input_shape=(224, 224, 3),
                  pooling='avg',
                  include_top=False)

outs = baseModel.output
outs = Dense(2, activation='softmax')(outs)
model = Model(inputs=baseModel.input, outputs=outs)

for layer in baseModel.layers[:15]:
    layer.trainable = False


aug_trn = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                             rescale=1./255, rotation_range=90)
aug_val = ImageDataGenerator(rescale=1./255)

batch_size = 128
data_trn = aug_trn.flow(X_trn, Y_trn, batch_size=batch_size)
data_val = aug_val.flow(X_val, Y_val, batch_size=batch_size)


checkpoint = ModelCheckpoint('models/model-{epoch:04d}.h5',
                             monitor='val_loss',verbose=1,
                             save_best_only=True, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100,
                      verbose=1, mode='auto')

model.compile(Adam(0.0001), 'categorical_crossentropy', ['accuracy'])
H = model.fit_generator(data_trn,
                        steps_per_epoch=len(X_trn)//batch_size,
                        epochs=10,
                        validation_data=data_val,
                        validation_steps=len(X_val)//batch_size,
                        callbacks=[checkpoint, early])


plt.figure(figsize=(5,5))
plt.plot(H.history['loss'], label='loss_trn')
plt.plot(H.history['val_loss'], label='loss_val')
plt.legend()
plt.title('loss')


# test
def suppress_box(bbs):
    list_rm = []
    for i in range(len(bbs)-1):
        if i in list_rm:
            continue
        cmps = np.array([j for j in range(i+1, len(bbs)) if j not in list_rm])
        if len(cmps) == 0:
            continue
        ious = np.array([get_iou(bbs[i], bbs[j]) for j in cmps])
        rm = cmps[ious > 0.5]
        if len(rm) > 0:
            list_rm.extend(rm)
    list_keep = [i for i in range(len(bbs)) if i not in list_rm]
    print('Total remove: {} boxs.'.format(len(list_rm)))
    return list_keep


def annot(i):
    annots = np.loadtxt(ps1[i], dtype='int', delimiter=' ', skiprows=1, ndmin=2)
    if annots.shape[0] == 0:
        print('\n***************\n\t!anno empty!!!')
        return
    img = cv2.imread(ps2[i])
    _annotate(img, annots)


def boxify(i, batch_size=64):
    img = cv2.imread(ps2[i])
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    bboxs = ss.process()
    bboxs = np.array([(x,y,x+w,y+h) for (x,y,w,h) in bboxs])

    list_pos = []
    list_prob = []
    for j in range(0, len(bboxs), batch_size):
        ids = np.arange(j, min(j+batch_size, len(bboxs)))
        print('j = {}, ids = {}:{}'.format(j, ids[0], ids[-1]))
        rois = np.array([cv2.resize(img[y1:y2, x1:x2], (224,224)) for (x1,y1,x2,y2) in bboxs[ids]])
        rets = model.predict(rois)[:,1]

        pos = ids[rets > 0.99]
        pos_prob = rets[rets > 0.99]
        list_pos.extend(pos)
        list_prob.extend(pos_prob)

    list_pos = np.array(list_pos)
    list_pos = list_pos[np.argsort(list_prob)]

    bboxs = bboxs[list_pos]
    list_pos = suppress_box(bboxs)
    _annotate(img, bboxs[list_pos])

