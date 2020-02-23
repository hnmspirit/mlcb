import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LinReg


X = np.array([147, 150, 153, 158, 160, 163, 168, 170,
              173, 175, 178, 180, 183]).reshape(-1, 1)
y = np.array([49, 50, 51, 54, 56, 58, 60, 62,
              63, 64, 66, 67, 68]).reshape(-1, 1)
X1 = np.hstack([np.ones((X.shape[0], 1)), X])


A = X1.T.dot(X1)
b = X1.T.dot(y)
w_fml = np.linalg.pinv(A).dot(b)
print(w_fml[:,0].tolist())


model = LinReg()
model.fit(X, y)
w_lib = [model.intercept_[0], model.coef_[0,0]]
print(w_lib)


xv = np.array([145, 185])
yv = w_lib[0] + w_lib[1]*xv
plt.plot(X[:, 0], y[:,0], 'co')
plt.plot(xv, yv, 'g:')
plt.grid(alpha=0.5)
plt.show()
