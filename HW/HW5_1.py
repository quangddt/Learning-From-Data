#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:52:49 2017

@author: quang
"""
import numpy as np
import math
import tools
import matplotlib.pyplot as plt
import seaborn as sns

def E_in(u, v):
    return (u * math.exp(v) - 2 * v * math.exp(-u))**2

def gradientEin(u, v):
    gradient_u = 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (math.exp(v) + 2 * v * math.exp(-u))
    gradient_v = 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * math.exp(-u))
    return gradient_u, gradient_v

def update_uv(u, v, eta):
    gradient_u, gradient_v = gradientEin(u, v)
    u = u - eta * gradient_u
    v = v - eta * gradient_v
    return u, v

def main():
    u, v = (1.0, 1.0)
    eta = 0.1
    iteration = 0
    
    while E_in(u, v) >= 10**(-14):
        u, v = update_uv(u, v, eta)
        iteration += 1
        
    print(iteration)
    print(u, v)
    print(E_in(u, v))
    
def coordinate_decent():
    u, v = (1.0, 1.0)
    eta = 0.1
    iteration = 15
    
    for i in range(iteration):
        gradient_u = gradientEin(u, v)[0]
        u = u - eta * gradient_u
        gradient_v = gradientEin(u, v)[1]
        v = v - eta * gradient_v
        
    print(E_in(u, v))

def logistic_regression_one_run(eta=0.01, visualize=False):
    X = tools.generate_points(100)
    f = tools.choose_boundary()
    y = tools.evaluate_output(X, f)
    
    #Fit logistic Regression using stochastic gradient descent
    w, num_iter = tools.logistic_regression(X, y, eta)
    
    #Estimate E_out by generating separate set of points to evaluate error
    X_out = tools.generate_points(1000)
    y_out = tools.evaluate_output(X_out, f)
    E_out = tools.error_measure_log(X_out, y_out, w)
    
    if visualize == True: 
        fig, ax = plt.subplots()
        tools.visualize_points(X, y, ax)
        tools.visualize_line(f, '-k', ax)
        tools.visualize_line(w, '--k', ax)
        plt.grid("off")
        sns.despine(bottom=True, left=True)
        plt.show()
    return E_out, num_iter

def experiment(n_runs=100):
    E_total = 0
    iter_total =0
    for i in range(n_runs):
        E_out, num_iter = logistic_regression_one_run()
        E_total += E_out
        iter_total += num_iter
    return E_total / n_runs, iter_total / n_runs

#print(experiment())
logistic_regression_one_run(visualize=True)
