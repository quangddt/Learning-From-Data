# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 01:12:05 2016

@author: quang
"""
import numpy as np
import scipy as sp
import math

M = 100
e = 0.05
N = - math.log(0.03/(2*M)) / (2*e**2)
print (N)