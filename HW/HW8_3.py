#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:30:25 2017

@author: quang
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
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

#Model selection for SVM with 10-fold cross validation
tuned_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1], 'kernel':['poly'], 
                'degree':[2], 'coef0':[1.0], 'gamma':[1.0]}

N_runs = 100
idx_choice = np.zeros(N_runs)
E_cv = np.zeros(N_runs)

for i in range(N_runs):
    kf = KFold(n_splits=10, shuffle=True)
    clf = GridSearchCV(SVC(decision_function_shape='ovo'), tuned_params, cv=kf)

    #Fit data
    clf.fit(X_train_, y_train_)

    idx_choice[i] = clf.best_index_
    
    E_cv[i] = 1 - clf.best_score_
    
print(pd.Series(idx_choice).value_counts())
print(np.mean(E_cv))

