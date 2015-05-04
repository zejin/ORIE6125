import numpy as np
from numpy import linalg as la
from est import *

def BS_testing(X1,X2,C=1.5):
    p1,n1 = X1.shape
    p2,n2 = X2.shape
    n = n1+n2
    if p1 !=p2 or n1!=n2:
        raise Exception("X1 and X2 do not have the same dimension")
        return 0
    X1_bar = np.mean(X1, axis=1) # O(np)
    X2_bar = np.mean(X2,axis=1)
    sigma_hat = (np.dot((X1.T - X1_bar).T, X1.T - X1_bar) +  np.dot((X2.T - X2_bar).T, X2.T - X2_bar))/ float(n) # O(np^2)
    tr_sigma_hat = sum(np.diag(sigma_hat)) # O(p)

    M_stat = np.dot((X1_bar-X2_bar).T, X1_bar-X2_bar) - n*tr_sigma_hat/n1/n2

    B_square = float(n)**2/(n+2)/(n-1) * ((la.norm(sigma_hat, 'fro')**2-1)/n * (tr_sigma_hat)**2)

    var_M_Original = 2*n*(n+1)/(n1*n2)**2 * B_square

    Z_original = float(M_stat/np.sqrt(var_M_Original))


    tau = C*np.sqrt(np.log(p1)/n)

    sigma_hat_tau = (np.dot(X1,X1.T) + np.dot(X2,X2.T))/n

    sigma_ij_hat_tau = sigma_hat_tau[abs(sigma_hat_tau) > tau]
    Q_hat = sum(sigma_ij_hat_tau * sigma_ij_hat_tau)

    var_M_Modified = 2*n*(n+1)/(n1*n2)**2 * Q_hat

    Z_modified = float(M_stat/np.sqrt(var_M_Modified))
  
    return (2*(1-np.random.normal(abs(Z_original)))), 2*(1-np.random.normal(abs(Z_modified)))

def CQ_testing(X1,X2,C=1.5):
    p1,n1 = X1.shape
    p2,n2 = X2.shape
    n = n1+n2
    if p1 !=p2 or n1!=n2:
        raise Exception("X1 and X2 do not have the same dimension")
        return 0
    X1_bar = np.mean(X1, axis=1) # O(np)
    X2_bar = np.mean(X2,axis=1)
    sigma_hat = (np.dot((X1.T - X1_bar).T, X1.T - X1_bar) +  np.dot((X2.T - X2_bar).T, X2.T - X2_bar))/ float(n) # O(np^2)
    tr_sigma_hat = sum(np.diag(sigma_hat)) # O(p)

    X1X1 = np.dot(X1.T,X1)
    X2X2 = np.dot(X2.T,X2)
    X1X2 = np.dot(X1.T, X2)

    T_stat = (sum(np.sum(X1X1,axis=1))-sum(np.diag(X1X1)))/n1/(n1-1)+(sum(np.sum(X2X2,axis=1))-sum(np.diag(X2X2)))/n2/(n2-1) - sum(np.sum(X1X2,axis=1))*2/n1/n2

    s1 = 0
    for j in range(n1):
        for k in range(n1):
            if j!=k:
                X1_bar_jk = np.mean(X1[:,j:k],axis=1)
                s1 = s1+ np.dot(np.dot(X1[:,j].T,(X1[:,k]-X1_bar_jk)),np.dot(X1[:,k].T,(X1[:,j]-X1_bar_jk)))
    sigma1_F_norm_square = s1/n1/(n1-1)

    s2 = 0
    for j in range(n2):
        for k in range(n2):
            if j!=k:
                X2_bar_jk = np.mean(X2[:,j:k],axis=1)
                s2 = s2+ np.dot(np.dot(X2[:,j].T,(X2[:,k]-X2_bar_jk)),np.dot(X2[:,k].T,(X2[:,j]-X2_bar_jk)))
    sigma2_F_norm_square = s2/n2/(n2-1)

    sigma12_innerprod = np.sum(np.diag(np.dot(np.dot((X1-X1_bar),X1.T),np.dot((X2-X2_bar),X2.T))))/(n1-1)/(n2-1)

    var_T_original = sigma1_F_norm_square*2/n1/(n1-1)+sigma2_F_norm_square*2/n2/(n2-1)+sigma12_innerprod*4/n1/n2

    Z_original = float(T_stat/np.sqrt(var_T_original))

    tau = C*np.sqrt(np.log(p1)/n)
    sigma1_hat_tau = np.dot(X1,X1.T)/n1

    sigma1_ij_hat_tau = sigma1_hat_tau[abs(sigma1_hat_tau) > tau]
    Q1_hat = np.sum(sigma1_ij_hat_tau * sigma1_ij_hat_tau)

    sigma2_hat_tau = np.dot(X2,X2.T)/n2

    sigma2_ij_hat_tau = sigma2_hat_tau[abs(sigma2_hat_tau) > tau]
    Q2_hat = np.sum(sigma2_ij_hat_tau * sigma2_ij_hat_tau)

    sigma12_hat_tau = np.dot(X1,X2.T)/n1

    sigma12_ij_hat_tau = sigma12_hat_tau[abs(sigma12_hat_tau) > tau]
    Q12_hat = np.sum(sigma12_ij_hat_tau * sigma12_ij_hat_tau)

    var_T_modified = Q1_hat * 2 / n1 / (n1 - 1) + Q2_hat * 2 / n2 / (n2 - 1) + Q12_hat * 4 / n1 / n2
    Z_modified = float(T_stat/np.sqrt(var_T_modified))

    return 2*(1-np.random.normal(abs(Z_original))), 2*(1-np.random.normal(abs(Z_modified)))


    