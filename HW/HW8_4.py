#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:34:56 2017

@author: quang
"""

import pandas as pd
from sklearn.svm import SVC
from tools import subset_data

#Load training and testing data
X_train = pd.read_csv('features.train', sep="\s+", usecols=[1, 2], 
                      header=None)
y_train = pd.read_csv('features.train', sep="\s+", usecols=[0], 
                      header=None).iloc[:, 0]

X_test = pd.read_csv('features.test', sep="\s+", usecols=[1, 2], 
                      header=None)
y_test = pd.read_csv('features.test', sep="\s+", usecols=[0], 
                      header=None).iloc[:, 0]    

#Consider the 1 vs 5 classsifier
train_model = [1, 5]
X_train_, y_train_ = subset_data(X_train, y_train, train_model)
X_test_, y_test_ = subset_data(X_test, y_test, train_model)

C_list = [0.01, 1, 100, 10**4, 10**6]

E_in = []
E_out = []

for C in C_list:
    clf = SVC(C=C, kernel='rbf', gamma=1.0)
    
    clf.fit(X_train_, y_train_)
    
    #Calculate E_in
    E_in.append(1 - clf.score(X_train_, y_train_))
    
    #Calculate E_out
    E_out.append(1 - clf.score(X_test_, y_test_))
    
print('C = {}'.format(C_list))
print('E_in = {}'.format(E_in))
print('E_out = {}'.format(E_out))