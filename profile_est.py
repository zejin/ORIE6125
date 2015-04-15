#!/usr/local/bin/python2.7

import sys
sys.path.append('/home/fs01/zj58/orie6125')
from est import *

# example

np.random.seed(12345)

X = np.random.normal(size=(500, 100))

BS_quad(X)
CQ_quad(X)
threshold_quad(X, C=1.5)

