import numpy as np
import matplotlib.pyplot as plt
np.random.seed(21)

def sigmoid(z): return 1 / (1 + np.exp(-z))

means = [[2,3], [4,1]]
cov = [[.3,0], [0,.3]]
N = 50
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.vstack((X0, X1))
y = np.array([0]*N + [1]*N)

Xbar = np.hstack((np.ones((2*N,1)), X))
w_init = np.random.randn(3)


def gd(w_init, lr, epochs, eps=1e-4):
    w = w_init.copy()
    for epoch in range(epochs):
        w_check = w.copy()
        z = sigmoid(Xbar.dot(w))
        w -= lr * Xbar.T.dot(z - y)
        if np.linalg.norm(w - w_check) / w.size < eps:
            break
    return w, epoch


w, epoch = gd(w_init, lr=.1, epochs=1000)
print('gd run {} epoch.'.format(epoch))

xv = np.array([0,5])
yv = -(w[0] + w[1]*xv)/w[2]

plt.plot(xv, yv, 'k')
plt.plot(X0[:,0], X0[:,1], 'b.')
plt.plot(X1[:,0], X1[:,1], 'r.')
plt.grid(alpha=0.5)
plt.show()