#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:17:12 2017

@author: quang
"""

import tools
import pandas as pd
import numpy as np

#Import train and test set data in numpy array
df_train = pd.read_table('in.dta', sep="\s+", header=None).values
df_test = pd.read_table('out.dta', sep="\s+", header=None).values

#Sepatate features and labels
X = df_train[:,:2]
y = df_train[:,2]

X_test = df_test[:,:2]
y_test = df_test[:,2]

#Transform to non-linear space
X = tools.nonlinear_transform(X)
X_test = tools.nonlinear_transform(X_test)

#Split 25 first examples into training and 10 last samples into validation, 
X_train = X[:25,:]
X_val = X[-10:,:]

y_train = y[:25]
y_val = y[-10:]

def evaluate_error(X_train, y_train, X_val, y_val, X_test, y_test):
    E_val = []
    E_test = []
    for k in [3, 4, 5, 6, 7]:
        #Fit transformed train data using linear regression without regularization
        #, using only k features of X_train
        w_lin = tools.linear_regression(X_train[:,:k+1], y_train)
        
        #Predict class of validation set
        y_val_pred = tools.predict(X_val[:,:k+1], w_lin)
        
        #Calculate classification error on validation set
        E_val.append(tools.cal_error(y_val, y_val_pred))
        
        #Predict class of test set
        y_test_pred = tools.predict(X_test[:,:k+1], w_lin)
        
        #Calculate classification error on test set
        E_test.append(tools.cal_error(y_test, y_test_pred))
    
    print(E_val)
    print(E_test)
    print('Smallest error on validation set is achieved when k = {}'.format(3 + np.argmin(E_val)))
    print('Smallest error on test set is {}, achieved when k = {}'.format(E_test[np.argmin(E_test)], 3 + np.argmin(E_test)))

print("Training with 25 first samples and validating with 10 last samples")
evaluate_error(X_train, y_train, X_val, y_val, X_test, y_test)

#Reverse the role of training and validation set
X_train, X_val = X_val, X_train
y_train, y_val = y_val, y_train

print("\nTraining with 10 last samples and validation with 25 first samples")
evaluate_error(X_train, y_train, X_val, y_val, X_test, y_test)