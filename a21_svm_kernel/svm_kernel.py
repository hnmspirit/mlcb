import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

X = np.array([(0, 0),(1, 1),(1, 0),(0, 1)])
Y = [0,0,1,1]

x_min, x_max = -2, 3
y_min, y_max = -2, 3

xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
xy = np.c_[xx.ravel(), yy.ravel()]

# fit the model
for kernel in ('sigmoid', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=4, coef0 = 0)
    clf.fit(X, Y)

    zz = clf.decision_function(xy)
    zz = zz.reshape(xx.shape)

    fig = plt.figure(figsize=(4, 4))
    plt.scatter(X[:2, 0], X[:2, 1], s=60, c='b', edgecolor='k')
    plt.scatter(X[2:, 0], X[2:, 1], s=60, c='r', edgecolor='k')

    plt.contourf(xx, yy, np.sign(zz), 100, cmap='jet', alpha=.2)
    plt.contour(xx, yy, zz, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.title(kernel, fontsize = 15)
    plt.axis('tight')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
plt.show()