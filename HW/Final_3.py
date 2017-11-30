#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:23:55 2017

@author: quang
"""

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from tools import subset_data

# Load training and testing data
X_train = pd.read_csv('features.train', sep="\s+", usecols=[1, 2],
                      header=None)
y_train = pd.read_csv('features.train', sep="\s+", usecols=[0],
                      header=None).iloc[:, 0]

X_test = pd.read_csv('features.test', sep="\s+", usecols=[1, 2],
                     header=None)
y_test = pd.read_csv('features.test', sep="\s+", usecols=[0],
                     header=None).iloc[:, 0]

X_train_, y_train_ = subset_data(X_train, y_train, [1, 5])
X_test_, y_test_ = subset_data(X_test, y_test, [1, 5])

alpha_list = [0.01, 1]

E_in, E_out = [], []

estimators = [('poly_transform', PolynomialFeatures()), 
              ('clf', RidgeClassifier())]

pipe = Pipeline(estimators)

for alpha in alpha_list:
    pipe.set_params(clf__alpha=alpha)
    pipe.fit(X_train_, y_train_)
    
    E_in.append(1 - pipe.score(X_train_, y_train_))
    
    E_out.append(1 - pipe.score(X_test_, y_test_))
    
print('E_in: {}\nE_out: {}'.format(E_in, E_out))