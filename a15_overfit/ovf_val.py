import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
np.random.seed(5)

N1 = 50
N2 = 10
X1 = np.random.rand(N1, 1) * 5
y1 = 3 * (X1 - 2) * (X1 - 3) * (X1 - 4) + 10 * np.random.randn(N1, 1)
X2 = np.random.rand(N2, 1) * 5
y2 = 3 * (X2 - 2) * (X2 - 3) * (X2 - 4) + 10 * np.random.randn(N2, 1)

X_trn, X_val, y_trn, y_val = train_test_split(
    X1, y1, test_size=0.33, random_state=0)
X_tst, y_tst = X2, y2


def buildX(X, d=2):
    res = np.ones((X.shape[0], 1))
    for i in range(1, d + 1):
        res = np.hstack((res, X**i))
    return res


def poly(a, x):
    # return a[0] + a[1].x + a[2].x^2 + ...
    res = np.zeros_like(x)
    for i in range(len(a) - 1, -1, -1):
        # print('res*x + a[i] = {}*{} + {} '.format(res, x, a[i]))
        res = res * x + a[i]
    return res


def mse(x, y, w):
    y_prd = poly(w, x)
    return np.mean(np.abs(y - y_prd))


def myfit(d):
    Xbar = buildX(X_trn, d)
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(Xbar, y_trn)

    w = regr.coef_[0]
    trn_err = mse(X_trn, y_trn, w)
    val_err = mse(X_val, y_val, w)
    tst_err = mse(X_tst, y_tst, w)

    return trn_err, val_err, tst_err


trn_errs = []
val_errs = []
tst_errs = []

degree = 9
for d in range(1, degree):
    trn_err, val_err, tst_err = myfit(d)
    trn_errs.append(trn_err)
    val_errs.append(val_err)
    tst_errs.append(tst_err)

degree = range(1, degree)

fig = plt.figure(figsize=(7, 5))
plt.plot(degree, trn_errs, 'c-', linewidth=2, label='train error')
plt.plot(degree, val_errs, 'm-', linewidth=2, label='valid error')
plt.plot(degree, tst_errs, 'g-', linewidth=2, label='test error')
plt.legend(loc='best')
plt.xlabel('degree')
plt.ylabel('error')

plt.text(1.5, 14, 'underfit', fontsize=20)
plt.text(4.5, 14, 'overfit', fontsize=20)
plt.plot([3.5, 3.5], [6, 14], 'k-', linewidth=2)
plt.show()
