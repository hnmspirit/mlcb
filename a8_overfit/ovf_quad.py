import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
np.random.seed(21)

N = 30
N2 = 20
X = np.random.rand(N, 1) * 5
y = 3 * (X - 2) * (X - 3) * (X - 4) + 10 * np.random.randn(N, 1)

X2 = (np.random.rand(N2, 1) - 1 / 8) * 10
y2 = 3 * (X2 - 2) * (X2 - 3) * (X2 - 4) + 10 * np.random.randn(N2, 1)


def oneX(X, d):
    bar = np.ones((X.shape[0], 1))
    for i in range(1, d + 1):
        bar = np.hstack((bar, X**i))
    return bar


xx = np.linspace(-2, 7, 200, endpoint=True)
yy = 5 * (xx - 2) * (xx - 3) * (xx - 4)


def fit(X, y, d):
    Xbar = oneX(X, d)
    print('Xbar: {}'.format(Xbar.shape))
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(Xbar, y)

    # visualize
    w = regr.coef_[0]
    ypred = np.zeros_like(xx)
    for i in range(d + 1):
        ypred += w[i] * xx**i

    plt.scatter(X, y, c='r', s=10, label='trn pt')
    plt.scatter(X2, y2, c='y', s=5, label='tst pt')

    plt.plot(xx, ypred, 'b', linewidth=2, label='pred ln')
    plt.plot(xx, yy, 'g', linewidth=2, label='true ln')

    plt.xticks([]), plt.yticks([])
    plt.axis([-4, 10, np.amin(yy) - 50, np.amax(yy) + 50])
    plt.title('d={}'.format(d))


plt.figure(figsize=(10, 5))
for j, d in enumerate([1, 2, 3, 4, 8, 16]):
    plt.subplot(2, 3, j + 1)
    fit(X, y, d)
plt.legend(loc='best')
plt.show()
