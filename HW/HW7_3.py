#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:28:03 2017

@author: quang
"""
import numpy as np
import tools

rho_s = [np.sqrt(np.sqrt(3) + 4), np.sqrt(np.sqrt(3) - 1), np.sqrt(9 + 4*np.sqrt(6)), np.sqrt(9 - np.sqrt(6))]

def cross_validation(rho=0, model='constant'):
    x1 = np.array([-1, 0])
    x2 = np.array([1, 0])
    x3 = np.array([rho, 1])
    
    if model == 'constant':
        e1 = 1
        e2 = 0.5**2
        e3 = 0.5**2
        
    if model == 'linear':
        e1 = 1
        
        slope, interp = tools.choose_boundary(np.stack([x1,x3]))
        e2 = abs(slope*x2[0] - x2[1] + interp)**2 / (slope**2 + 1)
        
        slope, interp = tools.choose_boundary(np.stack([x2,x3]))
        e3 = abs(slope*x1[0] - x1[1] + interp)**2 / (slope**2 + 1)
    return np.mean([e1, e2, e3])

for rho in rho_s:
    print('\nrho = {}'.format(rho))
    print('E_val of constant model is {}'.format(cross_validation(rho)))
    print('E_val of linear model is {}'.format(cross_validation(rho, model='linear')))
