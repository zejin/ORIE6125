#!/usr/local/bin/python2.7

import numpy as np

# Estimation of Quadratic Functionals

# function

def BS_quad(X):  
    p, n = X.shape

    X_bar = np.mean(X, axis=1)
    sigma_hat = np.dot((X.T - X_bar).T, X.T - X_bar) / float(n)
    tr_sigma_hat = sum(np.diag(sigma_hat))
  
    return float(n) ** 2 / (n + 2) / (n - 1) * (np.linalg.norm(sigma_hat, 'fro') ** 2 \
    - 1. / n * (tr_sigma_hat ** 2))

def CQ_quad(X):
    p, n = X.shape
  
    X_bar = np.mean(X, axis=1)
  
    s = 0
    for j in range(n):
        for k in range(n):
            if j != k:
                X_bar_jk = (X_bar * n - X[:, j] - X[:, k]) / float(n - 2)
                s += np.dot(X[:, j].T, X[:, k] - X_bar_jk) \
                * np.dot(X[:, k].T, X[:, j] - X_bar_jk)  
                          
    return float(s) / n / (n - 1)

def threshold_quad(X, C=1.5):
    p, n = X.shape
  
    tau = C * np.sqrt(np.log(p) / float(n))
    sigma_hat_tau = np.dot(X, X.T) / float(n)
    np.fill_diagonal(sigma_hat_tau, 0)

    sigma_ij_hat_tau = sigma_hat_tau[abs(sigma_hat_tau) > tau]
    Q_hat = sum(sigma_ij_hat_tau ** 2)
    
    D_matrix = np.dot((X ** 2).T, X ** 2)  
    np.fill_diagonal(D_matrix, 0)
    D_hat = np.sum(D_matrix) / float(n) / (n - 1)

    return [Q_hat, D_hat, Q_hat + D_hat]

