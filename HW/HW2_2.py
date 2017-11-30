# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:31:37 2016

@author: quang
"""

import HW1 
import numpy as np

def generate_random_x(N=100, u_range=[-1, 1], v_range=[-1, 1]):
    """
    Generate a set of N random points in the specified ranges
    
    Inputs
    ------
    - N: number of points
    - u_range: horizontal range
    - v_range: vertival range
    
    Outputs
    ------
    - x: 2D array where the rows has length 3 which represents the locations of 
        a data point with the first one is set to 1, and the columns is the 
        number of data points considered
    """
    x = HW1.generate_random_points(N, u_range, v_range)
    x_vec = np.vstack((np.ones(N), x))
    return x_vec
    
    
def generate_x_f_y(N=100, u_range=[-1, 1], v_range=[-1, 1]):
    """
    Generate a set of N points in the specified ranges, generate a random 
    function f, and evaluate f on N points
    
    Inputs
    ------
    - N: number of points
    - u_range: horizontal range
    - v_range: vertival range
    
    Outputs
    ------
    - x: 2D array where the rows has length 3 which represents the locations of 
        a data point with the first one is set to 1, and the columns is the 
        number of data points considered
    - y: classification value of x
    - f: target function
    """
    x = generate_random_x(N)
    f = HW1.generate_random_line(u_range, v_range)
    y = np.array([1 if item > 0 else -1 for item in np.dot(f, x)])
    return x, y, f
    
def linear_regression_algorithm(x, y):
    """
    Perform linear regression 
    
    Inputs
    ------
    - x: 2D array where the rows has length 3 which represents the locations of 
        a data point with the first one is set to 1, and the columns is the 
        number of data points considered
    - y: classification value of x
    
    Outputs
    ------
    - w: 1D array of length 3
    """
    X_matrix = np.transpose(x)
    w = np.dot(np.linalg.pinv(X_matrix), y)
    return w

def evaluate_E(x, y, g):
    """
    Evaluate the fraction of sample points which got classified incorrectly,
        which means sign(g^T*x_n) != y_n
    
    Inputs
    ------
    - x: 2D array where the rows has length 3 which represents the locations of 
        a data point with the first one is set to 1, and the columns is the 
        number of data points considered
    - y: 1D array of length equal to number of columns of x; 
        classification value of x
    - g: 1D array of length 3
    
    Outputs:
    ------
    E: fraction of sample error
    """
    y_estimate = np.array([1 if item > 0 else -1 for item in np.dot(g, x)])
    E = (y_estimate != y).sum()/len(y_estimate)
    return E

def experiment_E_in(N=100, u_range=[-1, 1] , v_range=[-1, 1], N_experiment=1000):
    E_in_sum = 0
    for i in range(N_experiment):
        x, y, f = generate_x_f_y(N, u_range, v_range)
        g = linear_regression_algorithm(x, y)
        E_in = evaluate_E(x, y, g)
        E_in_sum += E_in
    return E_in_sum / N_experiment
        
def experiment_E_out(N_fresh_points=1000, u_range=[-1, 1], v_range=[-1, 1], N_experiment = 1000):
    E_out_sum = 0
    for i in range(N_experiment):
        x, y, f = generate_x_f_y()
        g = linear_regression_algorithm(x, y)
        x_fresh = generate_random_x(N_fresh_points)
        y_fresh = np.array([1 if item > 0 else -1 for item in np.dot(f, x_fresh)])
        E_out = evaluate_E(x_fresh, y_fresh, g)
        E_out_sum += E_out
    return E_out_sum / N_experiment
        
def pla_combine_linear_regression(x, y):
    w_initial = linear_regression_algorithm(x, y)
    g, numberOfIteration = HW1.pla_algorithm(x, y, w=w_initial)
    return g, numberOfIteration
        
def experiment_pla_combine_linear_regression(N=10, u_range=[-1, 1], v_range=[-1, 1], N_experiment = 1000):
    numberOfIterationSum = 0    
    for i in range(N_experiment):
        x, y, f = generate_x_f_y(N, u_range, v_range)
        g, numberOfIteration = pla_combine_linear_regression(x, y)
        numberOfIterationSum += numberOfIteration
    return numberOfIterationSum / N_experiment
        
    
def main():          
    u_range = [-1, 1]      
    v_range = [-1, 1]
    N = 100
    x, y, f = generate_x_f_y(N, u_range, v_range)
    HW1.plot_a_line_2D(f, u_range, v_range)
    HW1.scatterplot(x, y)
    g = linear_regression_algorithm(x, y)
    HW1.plot_a_line_2D(g, u_range, v_range)
    E_in = evaluate_E(x, y, g)
    print(E_in)
    g_pla, numberOfIteration = pla_combine_linear_regression(x, y)
    print(numberOfIteration)