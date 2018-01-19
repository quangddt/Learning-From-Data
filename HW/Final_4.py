#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:09:00 2017

@author: quang
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from tools import visualize_points
from tools import visualize_line
from sklearn.svm import SVC
from cvxopt import matrix
from cvxopt import solvers

X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
y = np.array([-1, -1, -1, 1, 1, 1, 1])

Z = np.array([[x2**2 - 2 * x1 - 1, x1**2 -2 * x2 + 1] for [x1, x2] in X])

fig, ax = plt.subplots()
visualize_points(Z, y, ax=ax)
visualize_line([-0.5, 1, 0], style='-k', ax=ax, range_=[-4, 4])

clf = SVC(C=10**5, kernel='linear')
clf.fit(Z[:6], y[:6])

w = clf.coef_[0].tolist()
w.insert(0, clf.intercept_[0])
print(w)
visualize_line(w, style='--r', ax=ax, range_=[-4, 4])

clf2 = SVC(C=10**5, kernel='poly', degree=2, coef0=1.0, gamma=1.0)
clf2.fit(X, y)

print(clf2.n_support_)

P_np = np.dot(y[:,np.newaxis], y[np.newaxis,:]) * (1 + np.dot(X, X.transpose()))**2
q_np = -np.ones(X.shape[0])
G_np = -np.eye(X.shape[0])
h_np = np.zeros(X.shape[0])
A_np = y[np.newaxis, :]
b_np = 0

P = matrix(P_np, tc='d')
q = matrix(q_np, tc='d')
G = matrix(G_np, tc='d')
h = matrix(h_np, tc='d')
A = matrix(A_np, tc='d')
b = matrix(b_np, tc='d')

sol = solvers.qp(P,q,G,h,A,b)
print(sol['x'])