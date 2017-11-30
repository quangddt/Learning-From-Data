#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 08:00:30 2017

@author: quang
"""
import tools
import pandas as pd
import numpy as np

#Import train and test set data in numpy array
df_train = pd.read_table('in.dta', sep="\s+", header=None).values
df_test = pd.read_table('out.dta', sep="\s+", header=None).values

#Sepatate features and labels
X_train = df_train[:,:2]
y_train = df_train[:,2]

X_test = df_test[:,:2]
y_test = df_test[:,2]

#Transform to non-linear space
X_train = tools.nonlinear_transform(X_train)
X_test = tools.nonlinear_transform(X_test)

#Fit transformed train data using linear regression without regularization
w_lin = tools.linear_regression(X_train, y_train)

#Classifies train, and test data 
y_in = tools.predict(X_train, w_lin)
y_out = tools.predict(X_test, w_lin)

#Calculate in-sample and out-of-sample classification errors
E_in = tools.cal_error(y_train, y_in)
E_out = tools.cal_error(y_test, y_out)

print("In-sample error without regularization: ", E_in)
print("Out-of-sample error without regularization: ", E_out)

ks = [-3, -2, -1, 0, 1, 2, 3]

for k in ks:
    #Fit transformed train data using linear regression with regularization
    w_reg = tools.linear_regression(X_train, y_train, alpha=10**k)
    
    #Classifies train, and test data 
    y_in = np.sign(np.dot(X_train, w_reg))
    y_out = np.sign(np.dot(X_test, w_reg))

    #Calculate in-sample and out-of-sample classification errors
    E_in = (y_in != y_train).sum() / y_train.size
    E_out = (y_out != y_test).sum() / y_test.size

    print("\nIn-sample error with regularization, k = {}: {}".format(k, E_in))
    print("Out-of-sample error with regularization, k = {}: {}".format(k, E_out))
    