#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:16:33 2017

@author: quang
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression

def generate_points(n_points=100, low=-1.0, high=1.0):
    """
    Generate points in 2D with uniform distribution
        
    Parameters:
        n_points: number of points
        
        low: lower boundary of first dimension
            
        high: upper boundary of second dimension
        
    Returns:
        X: array of shape (n_points, 2)
    """
    return np.random.uniform(low, high, (n_points, 2))

def insert_column(X, col, pos=0):
    """
    Insert a column to X
    
    Parameters:
        X: array-like, shape (n_samples, n_features)
        
        col: array-like, shape (n_samples, 1)
        
        pos: interger, index of inserted column
        
    Returns:
        +X: array-like, shape (n_samples, n_features + 1)
    """
    return np.concatenate([X[:,:pos], col, X[:,pos:]], axis=1)

def choose_boundary(X=None, low=-1.0, high=1.0):
    """
    Choose a line in 2D as a boundary or fit a line to 2 points in 2D
    
    Parameters:
        X: array-like, shape (n_samples, n_features)

        low: lower boundary of first dimension
            
        high: upper boundary of second dimension
        
    Returns:
        w: array-like, [intercept, slope, -1]

    """
    
    #Generate 2 random points in 2D in case of choosing a line
    if X is None:
        X = np.random.uniform(low, high, (2, 2))
    
    slope = (X[0, 1] - X[1, 1]) / (X[0, 0] - X[1, 0])
    interception = X[0, 1] - slope * X[0, 0]
    
    return np.array([interception, slope, -1])

def evaluate_output(X, w):
    """
    Evaluate outputs for each of points in X
    
    Parameters:
        X: Array-like, if X has shape (n_samples, n_features) then insert a 
        column of 1 to X to deal with intercept w0
        
        w: Array-like, shape (n_features + 1,)
        
    Returns:
        y: Array-like, shape (n_points, )
           
    """
    #Insert a column of 1 to X to deal with intercept w0
    if X.shape[1] < w.shape[0]:
        X = insert_column(X, np.ones([X.shape[0], 1]))
    
    return np.sign(np.dot(X, w))

def visualize_points(X, y, ax, m='o'):
    for i in range(X.shape[0]):
        if y[i] == 1:
            ax.scatter(X[i, 0], X[i, 1], c='r', marker=m)
        else:
            ax.scatter(X[i, 0], X[i, 1], c='b', marker=m)
            
def visualize_line(w, style, ax, range_=[-1,1]): 
    """
    Parameters:
        w: array-like, [w0, w1, w2]
    """ 
    if abs(w[2]) < 10**-4:
        x2 = np.linspace(-4, 6, 10)
        x1 = (-w[0] / w[1]) * np.ones(x2.shape[0])
        ax.plot(x1, x2, style)
        ax.set_xlim(range_[0],range_[1])
    else:
        slope = -w[1] / w[2]
        intercept = -w[0] / w[2]
        
        x1 = np.linspace(range_[0], range_[1], 10)
        x2 = slope * x1 + intercept
        ax.plot(x1, x2, style)
        ax.set_xlim(range_[0],range_[1])
    

def linear_regression(X, y, alpha=0.0):
    """ 
    Linear regression using one-step learning
    
    Parameters:
        X: Array of shape (n_samples, n_features), already padded with 1.
        
        y: Array of shape (n_samples,).
        
        alpha: regularization parameter
        
    Returns:
        w_lin: Array of shape (n_features,), linear regeression coefficients.
    """
    n_features = X.shape[1]
    w_lin = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X) + 
                                alpha * np.eye(n_features)), X.transpose()), y)
    
    return w_lin

def compute_stochastic_gradient(x_n, y_n, w):
    """
    Compute the stochastic gradient of a single data point 
    
    Parameters:
        x_n: Array-like,  shape (n_features,)
        
        y_n: +1 or -1
        
        w: Array-like, shape (n_features + 1,), learning coefficients
        
    Returns:
        e_n: the gradient of the point
    """
    x = np.concatenate((np.array([1]), x_n))
    e_n = -y_n * x / (1 + np.exp(y_n * np.dot(w, x)))
    
    return e_n

def logistic_regression(X, y, eta=0.01):
    """
    Perform logistic regression using stochastic gradient descent
    
    Parameters:
        X: Array-like, shape (n_samples, n_features)
        
        y: Array-like, shape (n_samples,).
        
        eta: float, learning rate
        
    Returns:
        w: Array of shape (n_features,), logistic regeression coefficients.
    """
    #Initialize the weight vector to all zeros
    w = np.zeros(3)
    w_previous = w
    permutation = np.random.permutation(X.shape[0])
    num_iter = 0
    
    while True:
        for i in range(X.shape[0]):
            #index of each point
            point_idx = permutation[i]
            #compute the gradient of the point
            e_n = compute_stochastic_gradient(X[point_idx, :], y[point_idx], w)
            #update w
            w = w - eta * e_n
        #Stopping condition
        if np.linalg.norm(w-w_previous) >= 0.01:
            w_previous = w
            num_iter += 1
        else:
            break
    return w, num_iter

def nonlinear_transform(X):
    """
    Apply non-linear transformation to X
    
    Parameters:
        X: Array of shape (n_samples, 2). Two features of each sample 
           (x1, x2).
        
    Returns:
        X_trans: Array of shape (n_samples, 8). Eight features of each 
                 samples are (1, x1, x2, x1^2, x2^2, x1x2, |x1-x2|, |x1+x2|).  
    """
    n_samples = X.shape[0]
    X_trans = np.stack([np.ones(n_samples), X[:, 0], X[:, 1], X[:, 0]**2, 
            X[:, 1]**2, X[:, 0] * X[:, 1], np.abs(X[:, 0] - X[:, 1]), 
            np.abs(X[:, 0] + X[:, 1])], axis=1)
    
    return X_trans
        
def predict(X, w):
    """
    Predict output of samples, rows of X
    
    Parameters:
        X: array-like, shape (n_samples, n_features), already padded with 1.
        
        w: array-like, shape (n_features, ), model coefficients
        
    Returns:
        y: array-lie, shape (n_samples,). Predicted outputs.
    """
    return np.sign(np.dot(X, w))

def cal_error(y_true, y_pred):
    """
    Calculation the classification error
    
    Parameters:
        y_true: array-like, shape (n_samples, ), true values
        
        y_pred: array-like, shape (n_samples, ), predicted values
        
    Returns:
        error: float, classification error in range (0, 1)
    """
    return float((y_pred != y_true).sum() / y_true.size)

def error_measure_log(X, y, w):
    """
    Measure error of logistic regression fit
    
    Parameters:
        X: Array of shape (n_samples, n_features), already padded with 1.
        
        y: Array of shape (n_samples,).
        
        w: Array of shape (n_features,), logistic regeression coefficients.
        
    Returns:
        error: float
    """
    X_0 = np.ones([X.shape[0],1])
    X = np.concatenate((X_0, X), axis=1)
    return np.sum(np.log(np.exp(np.dot(X, w) * -y) + 1)) / X.shape[0]

def perceptron_learning(X, y):
    """
    Perceptron Learning Algorithm
    
    Parameters:
        X: Array-like, shape (n_samples, n_features)
        
        y: Array-like, shape (n_samples,).
        
    Returns:
        w: Array of shape (n_features + 1,), logistic regeression coefficients.
    
    """
    #Insert 1 to deal with intercept
    X = insert_column(X, np.ones([X.shape[0], 1]))
    
    #Initialize w
    w = np.random.randn(X.shape[1])
    
    #Evaluate output y_pred based on w
    y_pred = evaluate_output(X, w)
    
    #Pick a misclassified point and update the weight vector
    while (y_pred != y).sum() != 0:
        #Index of a misclassified point
        i = (y_pred != y).argmax()
        
        #Update the weight vector
        w = w + y[i] * X[i, :]
        
        #Update y_pred
        y_pred = evaluate_output(X, w)
        
    return w


def transform_y(label, value):

    if label == value:
        return 1
    else:
        return -1


def subset_data(X, y, model):

    return X[y.isin(model)], y[y.isin(model)]

class RBF_normal():
    
    def __init__(self, K=9, gamma=1.5):
        self.K = K
        self.gamma = gamma
        self.centroids = None
        self.regr = None
        
    def fit(self, X, y):
        init_centroids = generate_points(self.K)
        clustering = KMeans(n_clusters=self.K, init=init_centroids, n_init=1)
        clustering.fit(X)
        self.centroids = clustering.cluster_centers_
        gaussian_kernel = rbf_kernel(X, self.centroids, gamma=self.gamma)
        regr = LinearRegression()
        self.regr = regr.fit(gaussian_kernel, y)
        
        return self
    
    def predict(self, X, y):
        gaussian_kernel = rbf_kernel(X, self.centroids, gamma=self.gamma)
        
        return np.sign(self.regr.predict(gaussian_kernel))
        
    def score(self, X, y):
        y_pred = self.predict(X, y)
        
        return (y_pred == y).mean()