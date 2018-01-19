#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:31:34 2017

@author: quang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:05:03 2017

@author: quang
"""

from tools import generate_points
from tools import RBF_normal
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import numpy as np
from math import pi
import pandas as pd

#target function
def f(X):
    return np.sign(X[:, 1] - X[:, 0] + 0.25 * np.sin(pi * X[:, 0]))

Nruns = 10000

Ein_trend = []
Eout_trend = []

for i in range(Nruns):
    X = generate_points()
    
    #Evaluate outputs for samples in X, f(x) = sign(x2 - x1 + 0.25 * sin(pi * x1))    
    y = f(X)
    
    #RBF-normal (clustering -> gaussian RBF -> linear regression)
    gamma_1 = 1.5
    gamma_2 = 2
        
    normal_1 = RBF_normal(gamma=gamma_1)
    normal_2 = RBF_normal(gamma=gamma_2)
           
    normal_1.fit(X, y)
    normal_2.fit(X, y)

    #Evaluate E_in
    E_in_1 = 1 - normal_1.score(X, y)  
    E_in_2 = 1 - normal_2.score(X, y)
    Ein_trend.append(np.sign(E_in_2 - E_in_1))
    
    #Evaluate E_out
    N_out = 10000
    X_out = generate_points(N_out)
    y_out = f(X_out)
    
    E_out_1 = (1 - normal_1.score(X_out, y_out))
    E_out_2 = (1 - normal_2.score(X_out, y_out))
    Eout_trend.append(np.sign(E_out_2 - E_out_1))
    
print(pd.Series(list(zip(Ein_trend, Eout_trend))).value_counts())