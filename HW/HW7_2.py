#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:27:42 2017

@author: quang
"""
import numpy as np
def find_E_of_min(n_runs=1000):
    E = 0
    for i in range(n_runs):
        e1 = np.random.uniform()
        e2 = np.random.uniform()
        E += min(e1, e2)
    return E / n_runs

print('\nExpected value of min(e1, e2) is {}'.format(find_E_of_min()))