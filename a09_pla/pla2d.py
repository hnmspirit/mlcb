import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(21)

## data
means = [[2,2], [4,2]]
cov = [[.3,.2], [.2,.3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.vstack((X0, X1))
y = np.array([1.]*N + [-1.]*N)

one = np.ones((2*N,1))
X1 = np.hstack((one, X))
print('X1 shape: {}, y shape: {} '.format(X1.shape, y.shape))

## perceptron
def h(x, w):
    return np.sign(x.dot(w))

def converged(X, y, w):
    return np.array_equal(h(X,w), y)

def perceptron(X, y, w_init):
    w = w_init
    ws = [w]
    N, d = X.shape
    for i in range(100):
        shuffle_id = np.random.permutation(N)
        for j in shuffle_id:
            xi = X[j]
            yi = y[j]
            if h(xi, w) != yi:
                w = w + yi*xi
                ws.append(w)
        if converged(X, y, w):
            break
    print('update {} times after {} epoch.'.format(len(ws),i))
    return w, ws

w0 = np.random.randn(3)
w, ws = perceptron(X1, y, w0)

## plot
xv = np.array([0, 6])
plt.scatter(X[:,0], X[:,1], c=y, cmap='Accent')
line, = plt.plot([], [], 'r')

plt.grid(alpha=0.5)
plt.axis([-1,7, -2,7])
txt = plt.title('')

def update(i):
    a,b,c = ws[i]
    yv = -b/c*xv - a/c
    line.set_data(xv, yv)
    txt.set_text('time {:02d}/{:02d}'.format(i+1, len(ws)))

for i in range(len(ws)):
    update(i)
    plt.pause(0.1)
plt.show()
