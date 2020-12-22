# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 22:52:57 2019

@author: moi
"""

import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils




# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train[2])
from matplotlib import pyplot
for i in range(9):	
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    
    
# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

print(y_train[2])
y_train = np_utils.to_categorical(y_train)

print(y_train[2])

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)







# Network
net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
cost_, myerr=net.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.1)
cost2_, myerr2=net.fit(x_train[0:1000], y_train[0:1000], epochs=50, learning_rate=0.1)
#cost3_, myerr3=net.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=2)

# Network2
# =============================================================================
net2 = Network()
net2.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net2.add(ActivationLayer(tanh, tanh_prime))
net2.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net2.add(ActivationLayer(tanh, tanh_prime))
net2.add(FCLayer(50, 25))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net2.add(ActivationLayer(tanh, tanh_prime))
net2.add(FCLayer(25, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net2.add(ActivationLayer(tanh, tanh_prime))
# =============================================================================

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
# =============================================================================
net2.use(mse, mse_prime)
cost22_,err22=net2.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.1)
# =============================================================================

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])

# =============================================================================
import matplotlib.pyplot as plt
plt.plot(cost_,'b')
plt.plot(cost22_,'r')
plt.plot(cost2_,'g')
# =============================================================================
