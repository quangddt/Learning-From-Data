#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:43:11 2017

@author: quang
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(np.pi * x)

def learn_a(x1, x2):
    a = (x1 * f(x1) + x2 * f(x2)) / (x1 ** 2 + x2 ** 2)
    return a

def learn_b(x1, x2):
    return (f(x1) + f(x2)) /2

def find_g(Nrun=10000, learn=learn_a):
    g = []
    for i in range(Nrun):
        x1 = np.random.uniform(-1.0, 1.0)
        x2 = np.random.uniform(-1.0, 1.0)
        g.append(learn(x1, x2))
    return np.array(g)

a_hat = np.mean(find_g(learn = learn_a))
b_hat = np.mean(find_g(learn = learn_b))

def h1(x):
    return a_hat * x

def h2(x):
    return b_hat

def compute_bias(Nrun=10000, g=h1):
    bias = []
    for i in range(Nrun):
        x = np.random.uniform(-1.0, 1.0)
        bias.append((g(x) - f(x)) ** 2)
    return np.mean(bias)

def compute_variance(Nrun=10000, learn=learn_a):
    variance = []
    g_set = find_g(learn = learn)
    for i in range(Nrun):
        x = np.random.uniform(-1.0, 1.0)
        variance.append(np.mean((g_set * x - np.mean(g_set) * x) ** 2))
    return np.mean(variance)

print('bias =', compute_bias())
print('variance =', compute_variance())

print('\nbias =', compute_bias(g=h2))
#plt.plot(x, f(x), 'b')
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.grid()
#plt.xlim([-1, 1])
#plt.ylim([-1, 1])
#
#x1 = 0.1
#x2 = -0.1
#a_hat = learn_a(x1, x2)

#plt.scatter(x1, f(x1), color='r')
#plt.scatter(x2, f(x2), color='r')
#plt.plot(x, a_hat * x, 'r')