#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/anomaly_detect.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.22.2019

## Try out a new neural net based anomaly detection technique
import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
exec(open("./python/lib.py").read())

np.random.seed(123)
import tensorflow as tf
from scipy.optimize import line_search
tf.enable_eager_execution()
tf.set_random_seed(123)

# N gives the total number of simulation runs (including replicates).
# Q Gives the number of time periods with observations
# P Gives the number of sim params (5)
N = 1000#Assumed odd in this file
Q = 1
P = 0

## GAN Params
H = 20# Number of hidden units
epochs = 10000

## Generate data from two gaussian clusters.
y = np.vstack([np.random.normal(size=[int(N/2),Q]) - 10, np.random.normal(size=[int(N/2),Q]) + 10])
x = None

# Define a discriminator
disc = tf.keras.Sequential()
disc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))

disc.summary()
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')

# Training
for epoch in tqdm(range(epochs)):
    random_bois = np.random.uniform(low = -20, high = 20, size=[N,Q])

    with tf.GradientTape() as t:
        preds_true = disc(tf.cast(y, tf.float32))
        preds_false = disc(tf.cast(random_bois, tf.float32))
        cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape((N,1)), tf.cast(preds_true, tf.float64)))
        cost_false = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape((N,1)), tf.cast(preds_false, tf.float64)))
        cost = 0.5 * (cost_true + cost_false)
    grads = t.gradient(cost, disc.trainable_variables)

    grads_n_vars = [(grads[i], disc.trainable_variables[i]) for i in range(len(grads))]
    disc.optimizer.apply_gradients(grads_n_vars)

# Evaluate result
plotx = np.linspace(-20, 20, num =  1000)
ploty = disc(tf.cast(plotx.reshape((N,1)), tf.float32))

# Plot the data
fig = plt.figure()
plt.hist(y, density = True)
plt.plot(plotx, ploty)
plt.savefig("temp.pdf")
plt.close(fig)

