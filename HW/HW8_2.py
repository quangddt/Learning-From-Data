#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:31:00 2017

@author: quang
"""

import pandas as pd
from sklearn.svm import SVC
from tools import subset_data

# Load training and testing data
X_train = pd.read_csv('features.train', sep="\s+", usecols=[1, 2],
                      header=None)
y_train = pd.read_csv('features.train', sep="\s+", usecols=[0],
                      header=None).iloc[:, 0]

X_test = pd.read_csv('features.test', sep="\s+", usecols=[1, 2],
                     header=None).values
y_test = pd.read_csv('features.test', sep="\s+", usecols=[0],
                     header=None).iloc[:, 0]


def compare_svm_E_in_ovo(X_train, y_train, Q, train_model, C_list):
    E_in = []
    E_out = []
    n_supports = []

    # Subsetting the data such that only data of 2 class label,
    # specified in train_model, are remained
    X_train_, y_train_ = subset_data(X_train, y_train, train_model)
    X_test_, y_test_ = subset_data(X_test, y_test, train_model)

    for C in C_list:
        # Inititate an SVC with polinomial kernel, C=C, and Q=2
        clf = SVC(C=C, kernel='poly', degree=Q, coef0=1.0, gamma=1.0)

        # Fit training data
        clf.fit(X_train_, y_train_)

        # Calculate E_in which is 1 - mean accuracy
        E_in.append(1 - clf.score(X_train_, y_train_))

        # Calculate E_out
        E_out.append(1 - clf.score(X_test_, y_test_))

        # Get the total number of suport vectors
        n_supports.append(clf.n_support_.sum())

    return E_in, E_out, n_supports


C_list = [0.0001, 0.001, 0.01, 0.1, 1]
E_in, E_out, n_supports = compare_svm_E_in_ovo(X_train, y_train,
                                               Q=2, train_model=[1, 5], C_list=C_list)
print('Q = 2\nC = {}\nE_in: {}\nE_out: {}\nNumber of support vectors: {}'
      .format(C_list, E_in, E_out, n_supports))

E_in, E_out, n_supports = compare_svm_E_in_ovo(X_train, y_train,
                                               Q=5, train_model=[1, 5], C_list=C_list)

print('\nQ = 5\nC = {}\nE_in: {}\nE_out: {}\nNumber of support vectors: {}'
      .format(C_list, E_in, E_out, n_supports))
