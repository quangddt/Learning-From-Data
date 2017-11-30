#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:07:19 2017

@author: quang
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import zero_one_loss, make_scorer
from sklearn.model_selection import cross_val_score
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

X_train_ = X_train_.values
y_train_ = y_train_.values

C_list = [0.0001, 0.001, 0.01, 0.1, 1]

choice_idx = []

for i in range(100):
    kf = KFold(n_splits=10, shuffle=True)
    
    E_cv_avg = []
    
    for C in C_list:
        E_cv = []
        
        clf = SVC(C=C, kernel='poly', degree=2, coef0=1.0, gamma=1.0)
        
        for train_idx, val_idx in kf.split(X_train_):
            clf.fit(X_train_[train_idx], y_train_[train_idx])    
            score = clf.score(X_train_[val_idx], y_train_[val_idx])
            E_cv.append(1 - score)
            
        E_cv_avg.append(np.mean(E_cv))
    
    choice_idx.append(np.argmin(E_cv_avg))

print(pd.Series(choice_idx).value_counts())    
#print(np.mean(E_cv))
#print(1 - cross_val_score(clf, X_train_, y_train_, cv=kf).mean())