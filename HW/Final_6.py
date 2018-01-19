#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:47:49 2017

@author: quang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:24:15 2017

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
Ein_neq_0 = 0

svm_beat_normal = 0

for i in range(Nruns):
    X = generate_points()
    
    #Evaluate outputs for samples in X, f(x) = sign(x2 - x1 + 0.25 * sin(pi * x1))
    
    y = f(X)
    
    #SVM-kernal
    svm = SVC(C=10**5, kernel='rbf', gamma=1.5)
    
    svm.fit(X, y)
    
    #RBF-normal (clustering -> gaussian RBF -> linear regression)
    n_clusters = 12
        
    normal = RBF_normal(K=n_clusters)
           
    normal.fit(X, y)

    #Evaluate E_out
    N_out = 10000
    X_out = generate_points(N_out)
    y_out = f(X_out)
    
    E_out_svm = (1 - svm.score(X_out, y_out))
    E_out_rbf = (1 - normal.score(X_out, y_out))
    
    if E_out_svm < E_out_rbf:
        svm_beat_normal += 1        
    
print('How often does the kernel form beats the normal form: {}'
      .format(svm_beat_normal / Nruns))
    