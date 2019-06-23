#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.14.2019

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

mcycle = np.genfromtxt('./data/mcycle.csv', delimiter=',', skip_header = 1)

x = mcycle[:,0]
y = mcycle[:,1]
x /= max(x)
y = (y-min(y)) / (max(y) - min(y))

plt.scatter(x, y)

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu, input_shape = (1,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse')

model.fit(x, y, epochs = 2000)

fig = plt.figure()
plt.scatter(x, model.predict(x))
plt.scatter(x, y)
plt.savefig('temp.pdf')
