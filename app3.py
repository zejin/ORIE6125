#!/usr/local/bin/python2.7

from scipy.stats import norm
from sklearn import linear_model
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/huweici/Documents/orie6125')
from est import *

# High Dimensional Multifactor Pricing Model Test

# application

date = []
factor = []
FFF = []
with open('/Users/huweici/Documents/orie6125/F-F_Research_Data_Factors.txt', 'r') as f:
    for i, line in enumerate(f):
        if i == 0:
            factor += line.split()[1:]
        else:
            seq = line.split()
            date += [seq[0]]
            FFF.append([float(elem) for elem in seq[1:]])

FFF = np.array(FFF)

time_start = date.index('199001')
time_end = date.index('201312')

FFF = FFF[time_start:(time_end + 1), :]
T_num = FFF.shape[0]

ticker = []
with open('//Users/huweici/Documents/orie6125/stock_return.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):    
        if i == 0:
            missing = np.array([True] * (len(row) - 1))
            ticker += row[1:]
        elif 1 <= i <= T_num:
            missing[np.where(np.array(row[1:]) == 'NA')] = False

N_num = sum(missing == True)
stock_return = []
with open('//Users/huweici/Documents/orie6125/stock_return.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):    
        if 1 <= i <= T_num:
            stock_return.append([float(elem) for elem in np.array(row[1:])[missing]]) 
                                           
stock_return = np.array(stock_return).reshape((T_num, N_num)) * 100
stock_excessive_return = (stock_return.T - FFF[:, 3]).T

period = 60
window_num = T_num - period + 1

period_train = int(round(period / np.log(period)))
period_test = period - period_train

cv_num = 20
C_length = 201
C_list = np.linspace(0, 20, C_length)

def PMT(K_num):
    np.random.seed(12345)
    
    PM_p_est = np.zeros(window_num) 
    
    for t in range(window_num):
        stock_alpha = np.zeros(N_num)   
        stock_residual = np.zeros((period, N_num))
      
        for i in range(N_num):
            clf = linear_model.LinearRegression()
            clf.fit(FFF[t:(t + period), 0:K_num], stock_excessive_return[t:(t + period), i])
            stock_alpha[i] = clf.intercept_
            stock_residual[:, i] = stock_excessive_return[t:(t + period), i] - \
            clf.predict(FFF[t:(t + period), 0:K_num])
    
        v = period - K_num - 1
        sigma_tilt = np.dot(stock_residual.T, stock_residual) / float(v)
      
        D_hat = np.diag(sigma_tilt)
        F_space = FFF[t:(t + period), 0:K_num]
      
        W_d = (period - np.sum(np.dot(np.dot(F_space, np.linalg.inv(np.dot(F_space.T, F_space))), 
                                      F_space.T))) * sum(1. / D_hat * stock_alpha * stock_alpha)
        
        E_W_d = float(v) * N_num / (v - 2)
      
        CQ_threshold_diff_est = np.zeros((cv_num, C_length))
      
        max_sigma_ii = np.sqrt(np.max(sigma_tilt))

        for k in range(cv_num):
            index_total = np.arange(period)
            np.random.shuffle(index_total)
            index_train = index_total[0:period_train]      
            stock_residual_train = stock_residual.T[:, index_train]
            
            index_test = index_total[period_train:] 
            stock_residual_test = stock_residual.T[:, index_test]
        
            CQ_temp = CQ_quad(stock_residual_test)
        
            for j in range(C_length):
                threshold_temp = threshold_quad(stock_residual_train, 
                                                C=C_list[j] * max_sigma_ii)
                CQ_threshold_diff_est[k, j] = threshold_temp[2] - CQ_temp
        
        CQ_threshold_diff_mean_abs_error = np.mean(abs(CQ_threshold_diff_est), axis=0)
    
        index = np.argmin(CQ_threshold_diff_mean_abs_error)  
          
        tau = C_list[index] * max_sigma_ii * np.sqrt(np.log(N_num) / float(period))      
          
        pho_hat_tau = (sigma_tilt.T * (np.diag(sigma_tilt) ** (-1. / 2))).T \
        * (np.diag(sigma_tilt) ** (-1. / 2))
    
        np.fill_diagonal(pho_hat_tau, 0)
          
        pho_ij_hat_tau = pho_hat_tau[abs(sigma_tilt) > tau]
    
        pho_square = sum(pho_ij_hat_tau ** 2)
          
        var_W_d = 2 * float(N_num) * (v - 1) / (v - 4) * (v ** 2) / ((v - 2) ** 2) \
        * (1 + pho_square / float(N_num))
          
        J_alpha = (W_d - E_W_d) / np.sqrt(var_W_d)
          
        PM_p_est[t] = 2 * (1 - norm.cdf(abs(J_alpha)))

    return PM_p_est

# CAPM: K = 1 

CAPM_p_est = PMT(1)

# Fama-French Model: K = 3

FFM_p_est = PMT(3)

plt.plot(range(window_num), CAPM_p_est, linestyle='-', color='r')
plt.plot(range(window_num), FFM_p_est, linestyle='-', color='b')
plt.axhline(0.05, linestyle='--', color='k')  
plt.xlim((0, window_num))
plt.xticks(range(0, window_num, 24), range(1995, 2015, 2))
plt.ylim((0, 1))
plt.xlabel('time')
plt.ylabel('p-value')
plt.title('p-values of testing multifactor pricing models')
plt.legend(['CAPM', 'Fama-French', 'p = 0.05'], 1, fontsize=10)

plt.savefig('/Users/huweici/Documents/orie6125/app-3-' + str(1) + '.png')
plt.close("all")






















