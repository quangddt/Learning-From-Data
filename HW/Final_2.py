#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:37:31 2017

@author: quang
"""

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from tools import transform_y

# Load training and testing data
X_train = pd.read_csv('features.train', sep="\s+", usecols=[1, 2],
                      header=None)
y_train = pd.read_csv('features.train', sep="\s+", usecols=[0],
                      header=None).iloc[:, 0]

X_test = pd.read_csv('features.test', sep="\s+", usecols=[1, 2],
                     header=None)
y_test = pd.read_csv('features.test', sep="\s+", usecols=[0],
                     header=None).iloc[:, 0]

def clf_model(with_transform=False):
    if with_transform == True:
        estimators = [('poly_transform', PolynomialFeatures()), ('clf', RidgeClassifier(alpha=1.0))]
        return Pipeline(estimators)
    else:
        return RidgeClassifier(alpha=1.0)

E_in, E_out, E_in_tf, E_out_tf = [], [], [], []

for i in range(10):
    y_train_ = y_train.apply(transform_y, value=i)
    y_test_ = y_test.apply(transform_y, value=i) 
    
    clf = clf_model()
    clf_tf = clf_model(with_transform=True)
    
    clf.fit(X_train, y_train_)
    clf_tf.fit(X_train, y_train_)
    
    E_in.append(1 - clf.score(X_train, y_train_)) 
    E_out.append(1 - clf.score(X_test, y_test_))
    
    E_in_tf.append(1 - clf_tf.score(X_train, y_train_))
    E_out_tf.append(1 - clf_tf.score(X_test, y_test_))

print('E_in: {}\nE_in_tf: {}\nE_out: {}\nE_out_tf: {}'.format(E_in, E_in_tf, E_out, E_out_tf))