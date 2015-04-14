#!/usr/local/bin/python2.7

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/fs01/zj58/orie6125')
from fun1 import *

# Estimation of Quadratic Functionals

# application

n = 100 
p = 500 
num_rep = 500 

C_length = 41
C_list = np.linspace(0.8, 1.6, C_length)

def QFE(M, M_quad, M_offdiag_quad, order):
    np.random.seed(12345)
    
    BS_est = np.zeros(num_rep)
    CQ_est = np.zeros(num_rep)

    threshold_est = np.zeros((num_rep, C_length))
    threshold_offdiag_est = np.zeros((num_rep, C_length))

    for i in range(num_rep):
      X = np.random.multivariate_normal(mean=np.zeros(p), cov=M, size=n).T
                                                  
      BS_est[i] = BS_quad(X) 
      CQ_est[i] = CQ_quad(X)
      
      for j in range(C_length):
        threshold_temp = threshold_quad(X, C = C_list[j])
        threshold_est[i, j] = threshold_temp[2]
        threshold_offdiag_est[i, j] = threshold_temp[0]
      
    threshold_error = M_quad - threshold_est
    threshold_mean_abs_error = np.mean(abs(threshold_error), axis=0)
    
    index = np.argmin(threshold_mean_abs_error)
    
    plt.plot(range(num_rep), threshold_error, 'k-')
    plt.plot(range(num_rep), threshold_error[:, index], 'r-')       
    plt.xlabel('sample')
    plt.ylabel('error')
    plt.title('error of squared F-norm')
    fake1 = plt.Line2D([0, 0], [0, 1], color='k', linestyle='-')
    fake2 = plt.Line2D([0, 0], [0, 1], color='r', linestyle='-')
    plt.legend([fake1, fake2], ['threshold', 'optimal threshold'], 4, fontsize=10)
    
    plt.savefig('/home/fs01/zj58/orie6125/app-1-' + str(order) + '-' + str(1) + '.png')
    plt.close("all")
    
    BS_mean_abs_error = np.mean(abs(M_quad - BS_est))
    CQ_mean_abs_error = np.mean(abs(M_quad - CQ_est))
    
    y_max = np.log(max(threshold_mean_abs_error)) + 0.1
    y_min = np.log(min(threshold_mean_abs_error)) - 0.1
    
    plt.plot(C_list, np.log(threshold_mean_abs_error), linestyle='-', color='k', marker='o')
    plt.axhline(np.log(BS_mean_abs_error), linestyle='--', color='b')
    plt.axhline(np.log(CQ_mean_abs_error), linestyle='--', color='r')
    plt.axhline(np.log(threshold_mean_abs_error[index]), linestyle='--', color='g')       
    plt.xlabel(r'$\tau$')
    plt.ylabel('log mean absolute error')
    plt.title('log mean absolute error of squared F-norm')
    plt.legend(['threshold', 'BS', 'CQ', 'optimal threshold'], 1, fontsize=10)
    
    plt.savefig('/home/fs01/zj58/orie6125/app-1-' + str(order) + '-' + str(2) + '.png')
    plt.close("all")
    
    threshold_offdiag_error = M_offdiag_quad - threshold_offdiag_est
    threshold_mean_abs_offdiag_error = np.mean(abs(threshold_offdiag_error), axis=0)
    
    offdiag_index = np.argmin(threshold_mean_abs_offdiag_error)
    
    plt.plot(range(num_rep), threshold_offdiag_error, 'k-')
    plt.plot(range(num_rep), threshold_offdiag_error[:, offdiag_index], 'r-')       
    plt.xlabel('sample')
    plt.ylabel('error')
    plt.title('error of off-diagonal squared F-norm')
    fake1 = plt.Line2D([0, 0], [0, 1], color='k', linestyle='-')
    fake2 = plt.Line2D([0, 0], [0, 1], color='r', linestyle='-')
    plt.legend([fake1, fake2], ['threshold', 'optimal threshold'], 4, fontsize=10)
    
    plt.savefig('/home/fs01/zj58/orie6125/app-1-' + str(order) + '-' + str(3) + '.png')
    plt.close("all")
    
    threshold_optimal_diag_est = threshold_est[:, offdiag_index] - threshold_offdiag_est[:, offdiag_index]
    
    BS_offdiag_est = BS_est - threshold_optimal_diag_est
    CQ_offdiag_est = CQ_est - threshold_optimal_diag_est
    
    BS_mean_abs_offdiag_error = np.mean(abs(M_offdiag_quad - BS_offdiag_est))
    CQ_mean_abs_offdiag_error = np.mean(abs(M_offdiag_quad - CQ_offdiag_est))
    
    y_max = np.log(max(threshold_mean_abs_offdiag_error)) + 0.1
    y_min = np.log(min(threshold_mean_abs_offdiag_error)) - 0.1
    
    plt.plot(C_list, np.log(threshold_mean_abs_offdiag_error), linestyle='-', color='k', marker='o')
    plt.axhline(np.log(BS_mean_abs_offdiag_error), linestyle='--', color='b')
    plt.axhline(np.log(CQ_mean_abs_offdiag_error), linestyle='--', color='r')
    plt.axhline(np.log(threshold_mean_abs_offdiag_error[offdiag_index]), linestyle='--', color='g')       
    plt.xlabel(r'$\tau$')
    plt.ylabel('log mean absolute error')
    plt.title('log mean absolute error of off-diagonal squared F-norm')
    plt.legend(['threshold', 'BS', 'CQ', 'optimal threshold'], 1, fontsize=10)
    
    plt.savefig('/home/fs01/zj58/orie6125/app-1-' + str(order) + '-' + str(4) + '.png')
    plt.close("all")

# auto-correlation AR(1) covariance matrix

M1 = np.array([1. / 4 ** (abs(i - j)) for i in range(p) for j in range(p)]).reshape((p, p))
M1_quad = np.linalg.norm(M1, 'fro') ** 2
M1_offdiag_quad = M1_quad - p

QFE(M1, M1_quad, M1_offdiag_quad, 1)

# banded correlation matrix

M2 = np.zeros((p, p))

for i in range(p):
  if i >= 1:
    M2[i, i - 1] = 0.3
  if i <= p - 2:
    M2[i, i + 1] = 0.3
  M2[i, i] = 1

M2_quad = 0.3 ** 2 * (p - 1) * 2 + p
M2_offdiag_quad = M2_quad - p

QFE(M2, M2_quad, M2_offdiag_quad, 2)

# sparse block matrix 

num_block = 20
size_block = p / num_block

M3_small = np.ones((size_block, size_block)) * 0.3
np.fill_diagonal(M3_small, 1)

M3 = np.zeros((p, p)) 
for i in range(num_block):
  M3[(i * size_block):((i + 1) * size_block), 
     (i * size_block):((i + 1) * size_block)] = M3_small

M3_quad = np.linalg.norm(M3_small, 'fro') ** 2 * num_block
M3_offdiag_quad = M3_quad - p

QFE(M3, M3_quad, M3_offdiag_quad, 3)

# identity matrix

M4 = np.eye(p)
M4_quad = p
M4_offdiag_quad = 0

QFE(M4, M4_quad, M4_offdiag_quad, 4)




