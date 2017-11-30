# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:09:57 2016

@author: quang
"""
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib as mlb

def generate_random_points(N, u_range=[-1, 1], v_range=[-1,1]):
    # generate N points uniformly distributed in the range u_range x v_range
    # input: 
    #       N: number of points
    #       u_range: range of horizontal axis
    #       v_range: range of vertical axis
    # output: 
    #       array of position
    import numpy as np
    u = np.random.uniform(u_range[0], u_range[1], N)
    v = np.random.uniform(v_range[0], v_range[1], N)
    return np.array([u,v])

def extractpoints(x_vec, y, sign):
        #Return horizontal and vertival location of points
        #depends on the sign       
        if sign == "positive":
            Index = np.array([True if item > 0 else False for item in y])
        elif sign == "negative":
            Index = np.array([True if item < 0 else False for item in y])    
        pointsLocation = x_vec[1:,Index]
        horizontalLocation = pointsLocation[0,:]
        verticalLocation = pointsLocation[1,:]
        return horizontalLocation, verticalLocation
        
def scatterplot(x_vec, y):
    pointPositive = extractpoints(x_vec, y, "positive")
    pointNegative = extractpoints(x_vec, y, "negative")
    plt.scatter(pointPositive[0], pointPositive[1], c='r', marker='x')
    plt.scatter(pointNegative[0], pointNegative[1], c='g', marker='o')
    
    
def generate_random_line(u_range=[-1,1], v_range=[-1,1]):
    # generate a random in the range u_range x v_range
    # output: 
    #       a target function f, f = [f0, f1, -1]
    x = generate_random_points(2, u_range, v_range)
    u = x[0,:]
    v = x[1,:]
    f1 = (v[1] - v[0]) / (u[1] - u[0])
    f0 = v[0] - f1*u[0]
    return np.array([f0, f1, -1])

def plot_a_line_2D(f, u_range=[-1,1], v_range=[-1,1]):
    u = np.array(u_range)
    v = (-f[1]*u - f[0]) / f[2]
    plt.plot(u, v)
    plt.xlim([-1.2,1.2])
    plt.xlim([-1.2,1.2])
    
def pla_algorithm(x, y, w = np.array([0.0, 0.0, 0.0])):
    num_iter = 0
    while True:
        y_of_w = np.array([1 if item > 0 else -1 for item in np.dot(w, x)])
        missclassified_bool = y_of_w != y
        if missclassified_bool.any() == False:
            break
        missclassified_set = np.vstack((x[:, missclassified_bool], y[missclassified_bool]))
        choose_index = np.random.choice(len(missclassified_set[0,:]))
        point_choose = missclassified_set[:-1, choose_index]
        y_point_choose = missclassified_set[-1, choose_index]
        w += y_point_choose*point_choose
        num_iter += 1
    return w, num_iter        
    
def experiment_pla_num_iter(N, N_run, u_range, v_range):
    total_iter = 0
    for i in range(N_run):
        x = generate_random_points(N, u_range, v_range)
        f = generate_random_line(u_range, v_range)
        x_vec = np.vstack((np.ones(N), x))
        y = np.array([1 if item > 0 else -1 for item in np.dot(f, x_vec)])
        g, num_iter = pla_algorithm(x_vec, y)
        total_iter += num_iter
    return total_iter/N_run
    
def experiment_pla_f_diff_g(N, N_run, N_random_points, u_range, v_range):         
    total_prob_error = 0
    for i in range(N_run):
        x = generate_random_points(N, u_range, v_range)
        f = generate_random_line(u_range, v_range)
        x_vec = np.vstack((np.ones(N), x))
        y = np.array([1 if item > 0 else -1 for item in np.dot(f, x_vec)])
        g, num_iter = pla_algorithm(x_vec, y)
        x_random = generate_random_points(N_random_points, u_range, v_range)
        x_random_vec = np.vstack((np.ones(N_random_points), x_random))
        y_of_f = np.array([1 if item > 0 else -1 for item in np.dot(f, x_random_vec)])
        y_of_g = np.array([1 if item > 0 else -1 for item in np.dot(g, x_random_vec)])
        prob_error = sum(y_of_f != y_of_g) / N_random_points
        total_prob_error += prob_error
    return total_prob_error/N_run
 
#u_range = [-1, 1]      
#v_range = [-1, 1]
#N = 100
#x = generate_random_points(N, u_range, v_range)
#f = generate_random_line(u_range, v_range)
#plot_a_line_2D(f, u_range, v_range)
#
#
#x_vec = np.vstack((np.ones(N), x))
#y = np.array([1 if item > 0 else -1 for item in np.dot(f, x_vec)])
#scatterplot(x_vec, y)
#
#g, num_iter = pla_algorithm(x_vec, y)
#
#plot_a_line_2D(g, u_range, v_range)
##
#N_run = 1000
#average_iter = experiment_pla_num_iter(N, N_run, u_range, v_range)
#print(average_iter)
#
#N_random_points = N
#average_prob_error = experiment_pla_f_diff_g(N, N_run, N_random_points, u_range, v_range)
#print(average_prob_error)




