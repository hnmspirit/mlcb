import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 1, 2])
classes = [0, 1, 2]
cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='CM', cmap='Blues'):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)
    plt.xlabel('label pred')
    plt.ylabel('label true')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha='center', va='center',
                     color="w" if cm[i, j] > thresh else "k")

    plt.tight_layout()


plot_confusion_matrix(cm, classes, title='confusion matrix')
plt.show()