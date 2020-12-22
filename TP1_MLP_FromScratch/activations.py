# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:49:43 2019

@author: moi
"""

import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_prime(x):
    #sigm = 1. / (1. + np.exp(-x))
    #return sigm * (1. - sigm)
    return (np.exp(-x) / (1+np.exp(-x))**2)
