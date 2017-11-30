#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:28:20 2017

@author: quang
"""
import numpy as np
import matplotlib.pyplot as plt
import math

d_vc = 50
delta = 0.05
N = np.array(range(5,10010, 5), dtype="float64")

def m_H(N):
    return N**d_vc

original_bound = np.sqrt(8 / N * np.log(4 * m_H(2 * N) / delta))
rade_bound = np.sqrt(2 * np.log(2 * N * m_H(N)) / N) + np.sqrt(2 / N  * \
                    np.log(1 / delta)) + 1 / N
paron_bound = (1 / N + np.sqrt(1 / N ** 2 + 4 / N * np.log(6 * m_H(2 * N) \
                                                           / delta))) / 2
                                                           
N_new = np.array(range(5,10010, 5))                                                           
devroye_bound = np.zeros(len(N), dtype="float64")
for i in range(len(N)):
    devroye_bound[i] = (4 / (2 * int(N_new[i])) + np.sqrt(16 / (4 * int(N_new[i])**2) + 4 / (2 * int(N_new[i])) * (1 - \
                      4 / (2 * int(N_new[i]))) * math.log(4 * m_H(int(N_new[i]) ** 2) * 100 // 5))) / (2 * \
                                                           (1 - 4 / (2 * int(N_new[i]))))

print('For large N: N = 10000')
print('Original VC bound: %s' %original_bound[-1])
print('Rademacher Penalty Bound: %s' %rade_bound[-1])
print('Parrondo and Van den Broek: %s' %paron_bound[-1])
print('Devroye: %s' %devroye_bound[-1])

print('\n For small N: N = 5')
print('Original VC bound: %s' %original_bound[0])
print('Rademacher Penalty Bound: %s' %rade_bound[0])
print('Parrondo and Van den Broek: %s' %paron_bound[0])
print('Devroye: %s' %devroye_bound[0])

plt.figure()
plt.semilogy(N, original_bound, '-b', label='Original')
plt.semilogy(N, rade_bound, '-r', label='Rademacher')
plt.semilogy(N, paron_bound, '-g', label='Parrondo')
plt.semilogy(N, devroye_bound, '-k', label='Devroye')
plt.legend()
plt.grid(which='major', axis='x')
plt.grid(which='minor', axis='y')
plt.xlabel('N')
plt.title('For larger N')

plt.figure()
plt.semilogy(N[:10], original_bound[:10], '-b', label='Original')
plt.semilogy(N[:10], rade_bound[:10], '-r', label='Rademacher')
plt.semilogy(N[:10], paron_bound[:10], '-g', label='Parrondo')
plt.semilogy(N[:10], devroye_bound[:10], '-k', label='Devroye')
plt.legend()
plt.grid(which='major', axis='x')
plt.grid(which='minor', axis='y')
plt.xlabel('N')
plt.title('For small N')