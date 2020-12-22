# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:04:04 2019

@author: moi
"""

import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
#from activations import tanh, tanh_prime
from losses import mse, mse_prime
from activations import sigmoid, sigmoid_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(mse, mse_prime)
cost_, myerr= net.fit(x_train, y_train, epochs=10000, learning_rate=0.2)

# test
out = net.predict(x_train)
print(out)

import matplotlib.pyplot as plt
plt.plot(cost_)
