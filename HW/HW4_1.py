#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:33:42 2017

@author: quang
"""
import numpy as np

eps = 0.05
delta = 0.05
dvc = 10

N = 1000

for i in range(100):
    N = 8 / (eps ** 2) * np.log(4 * ((2 * N) ** dvc + 1) / delta)

print(N)