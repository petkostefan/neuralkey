import numpy as np


'''
Pravila učenja mašine za uparivanje stabala
'''

def theta(t1, t2):
    return int(t1 == t2)

def hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] += X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j] , -l, l)

def anti_hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] -= X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j] , -l, l)

def random_walk(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] -= X[i, j] * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j] , -l, l)