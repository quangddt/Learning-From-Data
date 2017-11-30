#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:24:15 2017

@author: quang
"""

from tools import generate_points
from tools import visualize_points
from tools import cal_error
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import numpy as np
from math import pi
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression

Nruns = 1000
Ein_neq_0 = 0

for i in range(Nruns):
    X = generate_points()
    
    #Evaluate outputs for samples in X, f(x) = sign(x2 - x1 + 0.25 * sin(pi * x1))
    
    y = np.sign(X[:, 1] - X[:, 0] + 0.25 * np.sin(pi * X[:, 0]))
    
    #Visualize dataset
    #fig, ax = plt.subplots()
    #visualize_points(X, y, ax=ax)
    #plt.grid('off')
    
    clf_rbf_svm = SVC(C=10**5, kernel='rbf', gamma=1.5)
    
    clf_rbf_svm.fit(X, y)
    
    if (1 - clf_rbf_svm.score(X, y)) != 0:
        Ein_neq_0 += 1
        
print('How often a data set is not separable by RBF kernel'
      ' using hard-margin SVM: {}'.format(Ein_neq_0 / Nruns))