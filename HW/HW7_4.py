#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:44:03 2017

@author: quang
"""

import numpy as np
import matplotlib.pyplot as plt
import tools
from sklearn.svm import SVC

def compare_pla_svc(N, N_runs=1000):
    g_svm_isbetter = np.zeros(N_runs)
    n_support = np.zeros([N_runs, 2])
    for i in range(N_runs):
        #Generate N points in the range [-1, 1] x [-1, 1]  
        while True:
            X = tools.generate_points(N)
            
            #Choose a random line in 2D as a target function
            f = tools.choose_boundary()
            
            #Assign label to X
            y = tools.evaluate_output(X, f)
            
            if abs(sum(y)) != y.shape[0]:
                break
        
        #Visualize data and target function
        #fig, ax = plt.subplots()
        #tools.visualize_points(X, y, ax)
        #tools.visualize_line(f, 'k', ax)
        
        #Fit Perceptron Learning Algorithm
        g_pla = tools.perceptron_learning(X, y)
        
        #Visualize the learned PLA function
        #tools.visualize_line(g_pla, '--k', ax)
        
        #Generate out-of-sample data
        N_out = 10000
        X_out = tools.generate_points(N_out)
        y_out = tools.evaluate_output(X_out, f)
        
        #Evaluate E_out of PLA
        y_pla_pred = tools.evaluate_output(X_out, g_pla)
        E_out_pla = tools.cal_error(y_out, y_pla_pred)
        
        #Visualize out-of-sample data
        #tools.visualize_points(X_out, y_out, ax, 's')
        #tools.visualize_points(X_out, y_out_pred, ax, 'x')
        
        #Create Linear Support Vector Classification
        svc = SVC(kernel='linear', C=1000)
        svc.fit(X, y)
        g_svc = np.concatenate([svc.intercept_, svc.coef_[0]])
        n_support[i,:] = svc.n_support_
        #Visualize the learned SVC function
        #tools.visualize_line(g_svc, ':k', ax)
        #Evaluate E_out of PLA
        y_svc_pred = tools.evaluate_output(X_out, g_svc)
        E_out_svc = tools.cal_error(y_out, y_svc_pred)
        
        g_svm_isbetter[i] = (E_out_svc < E_out_pla)
    
    return g_svm_isbetter.mean(), n_support.sum(axis=1).mean()
        
print('For N = 10, g_svm is better than g_pla {:.2%}'.format(compare_pla_svc(N=10)[0]))
print('For N = 100, g_svm is better than g_pla {:.2%}'.format(compare_pla_svc(N=100)[0]))
print('For N = 100, the average number of support vector of g_svm is {}'.format(compare_pla_svc(N=100)[1]))

