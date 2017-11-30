#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:40:47 2017

@author: quang
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from tools import transform_y
from tools import insert_column

# Load training and testing data
X_train = pd.read_csv('features.train', sep="\s+", usecols=[1, 2],
                      header=None)
y_train = pd.read_csv('features.train', sep="\s+", usecols=[0],
                      header=None).iloc[:, 0]

X_test = pd.read_csv('features.test', sep="\s+", usecols=[1, 2],
                     header=None)
y_test = pd.read_csv('features.test', sep="\s+", usecols=[0],
                     header=None).iloc[:, 0]

clf_list_1 = [5, 6, 7, 8, 9]
E_in = []

clf = RidgeClassifier(alpha=1.0)

for i in clf_list_1:
    y_train_ = y_train.apply(transform_y, value=i)
    
    clf.fit(X_train, y_train_)

    E_in.append(1 - clf.score(X_train, y_train_))

X_train_padded = insert_column(X_train.values, col=np.ones([X_train.shape[0], 1]))

clf_list_2 = [0, 1, 2, 3, 4]
E_out = []

estimators = [('poly_transform', PolynomialFeatures()), ('clf', RidgeClassifier(alpha=1.0))]
pipe = Pipeline(estimators)
    
for i in clf_list_2:
    y_train_ = y_train.apply(transform_y, value=i)
    y_test_ = y_test.apply(transform_y, value=i)
    
    pipe.fit(X_train, y_train_)
    
    E_out.append(1 - pipe.score(X_test, y_test_))
    
print('E_in: {}\nE_out: {}'.format(E_in, E_out))