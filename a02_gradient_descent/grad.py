import numpy as np
import matplotlib.pyplot as plt
np.random.seed(21)


def f(x): return x**2 + 5*np.sin(x)
def g(x): return 2*x + 5*np.cos(x)


def gradient_descent(x=None, lr=0.1, eps=1e-4):
    if not x:
        x = np.random.randn()
    xs = [x]
    for it in range(100):
        gx = g(x)
        if np.abs(gx) < 1e-4:
            break
        x -= lr * gx
        xs.append(x)
    print('gd stopped after %d iter' % (it+1))
    return np.array(xs), it


x1, _ = gradient_descent(-5, 0.1)
y1 = f(x1)

x2, _ = gradient_descent(5, 0.1)
y2 = f(x2)

xv = np.linspace(-6, 6, 90)
yv = f(xv)

plt.plot(xv, yv, 'c-', alpha=0.5)
plt.plot(x1, y1, 'r.')
plt.plot(x2, y2, 'b.')
plt.grid(alpha=0.5)
plt.show()