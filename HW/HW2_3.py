# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:23:48 2016

@author: quang
"""

import HW2_linear_regression
import numpy as np

def apply_function_f(x):
    """
    Generate the output of data points x based on function f = sign(x1^2+x2^2-0.6)
    
    Inputs
    ------
    - x: 2D array; rows are location of a points; columns are distinc points
    
    Outputs
    ------
    - y: 1D array; output of applying f on x
    """
    y = np.array([1 if item > 0 else -1 for item in ((x[1:,:])**2).sum(axis = 0) - 0.6])
    return y
    
def apply_random_noise(y, percent = 0.1):
    """
    Apply random noise to the output of the generated training set
    
    Inputs:
    - y: outcome of generated training set
    - percent: percent of outcome will be randomly flipped 
    
    Outputs:
    - y_noised: noise affected outcome

    """
    indexBeingFlipped = np.random.choice(len(y), size = int(percent * len(y)))
    y_noised = y.copy()
    y_noised[indexBeingFlipped]=-y_noised[indexBeingFlipped]
    return y_noised
    
def experiment_E_in(N=1000, N_experiment=1000):
    E_in_sum = 0
    for i in range(N_experiment):
        x = HW2_linear_regression.generate_random_x(N)
        y = apply_function_f(x)
        y_noised = apply_random_noise(y)
        g = HW2_linear_regression.linear_regression_algorithm(x, y_noised)
        E_in = HW2_linear_regression.evaluate_E(x, y, g)
        E_in_sum += E_in
    return E_in_sum / N_experiment
    
def transform_into_nonlinear(x):
    """
    Transform data x into the non linear vector (1,x1,x2,x1x2,x1^2,x2^2)
    
    Inputs:
    - x: 2D array; rows are location of a points; columns are distinc points
    
    Outputs:
    - x_transformed: 2D array; 6 rows
    """
    x_transformed = np.vstack((x, x[1,:]*x[2,:], (x[1,:])**2, (x[2,:])**2))
    return x_transformed
    
def find_g(N_runs = 10):
    g_sum = np.zeros(6)
    for i in range(N_runs):
        x = HW2_linear_regression.generate_random_x(N=1000)
        y = apply_function_f(x)
        y_noised = apply_random_noise(y)
        x_transformed = transform_into_nonlinear(x)
        g_sum += HW2_linear_regression.linear_regression_algorithm(x_transformed, y_noised)
    g_average = g_sum / N_runs
    return g_average
        
def experiment_E_out(N=1000, N_experiment=1000):
    E_out_sum = 0
    for i in range(N_experiment):
        x = HW2_linear_regression.generate_random_x(N)
        y = apply_function_f(x)
        y_noised = apply_random_noise(y)
        x_transformed = transform_into_nonlinear(x)
        g = HW2_linear_regression.linear_regression_algorithm(x_transformed, y_noised)
        x_new = HW2_linear_regression.generate_random_x(N)
        y_new = apply_function_f(x_new)
        y_noised_new = apply_random_noise(y_new)
        x_new_transformed = transform_into_nonlinear(x_new)
        E_out = HW2_linear_regression.evaluate_E(x_new_transformed, y_noised_new, g)
        E_out_sum += E_out
    return E_out_sum / N_experiment        
    
def main():
    x = HW2_linear_regression.generate_random_x(N=1000)
    y = apply_function_f(x)
    y_noised = apply_random_noise(y)
    x_transformed = transform_into_nonlinear(x)
    g = HW2_linear_regression.linear_regression_algorithm(x_transformed, y_noised)
    return g