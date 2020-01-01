import numpy as np

def vanilla(x_init, grad, lr=0.1, epochs=100, eps=1e-3):
    x = x_init.copy()
    xs = [x]
    for it in range(epochs):
        x_check = x
        gx = grad(x)
        x = x - lr * gx
        if np.linalg.norm(x - x_check) / x.size < eps:
            break
        xs.append(x)
    return np.array(xs), it


def momentum(x_init, grad, lr=0.1, beta1=0.9, epochs=100, eps=1e-3):
    x = x_init.copy()
    m = np.zeros_like(x)
    xs = [x]
    for it in range(epochs):
        x_check = x
        gx = grad(x)
        m = beta1*m - lr*gx
        x = x + m
        if np.linalg.norm(x - x_check) / x.size < eps:
            break
        xs.append(x)
    return np.array(xs), it


def nesterov(x_init, grad, lr=0.1, beta1=0.9, epochs=100, eps=1e-3):
    x = x_init.copy()
    m = np.zeros_like(x)
    xs = [x]
    for it in range(epochs):
        x_check = x
        gx = grad(x + beta1*m)
        m = beta1*m - lr*gx
        x = x + m
        if np.linalg.norm(x - x_check) / x.size < eps:
            break
        xs.append(x)
    return np.array(xs), it


def adagrad(x_init, grad, lr=0.1, eps2=1e-8, epochs=100, eps=1e-3):
    x = x_init.copy()
    v = np.zeros_like(x)
    xs = [x]
    for it in range(epochs):
        x_check = x.copy()
        gx = grad(x)
        v += gx**2
        x = x - lr * gx / (np.sqrt(v) + eps2)
        if np.linalg.norm(x - x_check) / x.size < eps:
            break
        xs.append(x)
    return np.array(xs), it


def rmsprop(x_init, grad, lr=0.1, beta2=0.999, eps2=1e-8, epochs=100, eps=1e-3):
    x = x_init.copy()
    v = np.zeros_like(x)
    xs = [x]
    for it in range(epochs):
        x_check = x.copy()
        gx = grad(x)
        v = beta2*v + (1-beta2)*gx**2
        x = x - lr * gx / (np.sqrt(v) + eps2)
        if np.linalg.norm(x - x_check) / x.size < eps:
            break
        xs.append(x)
    return np.array(xs), it


def adam(x_init, grad, lr=0.1, beta1=0.9, beta2=0.999, eps2=1e-8, epochs=100, eps=1e-3):
    x = x_init.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    xs = [x]
    for it in range(epochs):
        x_check = x.copy()
        gx = grad(x)
        m = beta1*m + (1-beta1)*gx
        v = beta2*v + (1-beta2)*gx**2
        x = x - lr * m / (np.sqrt(v) + eps2)
        if np.linalg.norm(x - x_check) / x.size < eps:
            break
        xs.append(x)
    return np.array(xs), it