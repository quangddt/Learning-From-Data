#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:02:38 2017

@author: quang
"""

import pandas as pd
from sklearn.svm import SVC
from tools import transform_y

# Load training and testing data
X_train = pd.read_csv('features.train', sep="\s+", usecols=[1, 2],
                      header=None)
y_train = pd.read_csv('features.train', sep="\s+", usecols=[0],
                      header=None).iloc[:, 0]

X_test = pd.read_csv('features.test', sep="\s+", usecols=[1, 2],
                     header=None).values
y_test = pd.read_csv('features.test', sep="\s+", usecols=[0],
                     header=None).iloc[:, 0]


def compare_svm_E_in_ovr(X_train, y_train, train_model, C=0.01):
    E_in = []
    n_supports = []

    # Inititate an SVC with polinomial kernel, C=0.01, and Q=2
    clf = SVC(C=C, kernel='poly', degree=2, coef0=1.0, gamma=1.0)

    for i in train_model:
        X_train_ = X_train
        y_train_ = y_train.apply(transform_y, value=i)

        # Fit training data
        clf.fit(X_train_, y_train_)

        # Calculate E_in which is 1 - mean accuracy 
        E_in.append(1 - clf.score(X_train, y_train_))

        # Get the total number of suport vectors
        n_supports.append(clf.n_support_.sum())

    return E_in, n_supports


train_model = [0, 2, 4, 6, 8]
E_in, n_support_even = compare_svm_E_in_ovr(X_train, y_train, train_model)
for i, v in enumerate(E_in):
    print('E_in of {} versus all is: {}'.format(i * 2, v))

train_model = [1, 3, 5, 7, 9]
E_in, n_support_odd = compare_svm_E_in_ovr(X_train, y_train, train_model)
for i, v in enumerate(E_in):
    print('E_in of {} versus all is: {}'.format(i * 2 + 1, v))

print("The difference between the number of support vectors of 0 vs all"
      "and 1 vs all is: {}".format(abs(n_support_even[0] - n_support_odd[0])))
