#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:40:09 2017

@author: quang
"""

from tools import generate_points
from tools import RBF_normal
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import numpy as np
from math import pi
from sklearn.svm import SVC

#target function
def f(X):
    return np.sign(X[:, 1] - X[:, 0] + 0.25 * np.sin(pi * X[:, 0]))

Nruns = 10000
Ein_eq_0 = 0

for i in range(Nruns):
    X = generate_points()
    
    #Evaluate outputs for samples in X, f(x) = sign(x2 - x1 + 0.25 * sin(pi * x1))
    
    y = f(X)
    
    #RBF-normal (clustering -> gaussian RBF -> linear regression)
    n_clusters = 9
        
    normal = RBF_normal(K=n_clusters)
           
    normal.fit(X, y)

    #Evaluate E_in
    E_in = 1 - normal.score(X, y)   
    
    if E_in < 10**-3:
        Ein_eq_0 += 1
        
print('Percentage of time that regular RBF achivies Ein = 0: {}'
      .format(Ein_eq_0 / Nruns))