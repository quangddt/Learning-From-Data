#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:41:09 2017

@author: quang
"""

d = 8
delta = 0.1
E_in = 0.008

N = (d + 1) / (1 - E_in / delta**2)
print(N)