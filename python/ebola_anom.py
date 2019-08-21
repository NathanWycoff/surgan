#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ebola_anom.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.23.2019

## Fit an anomaly detection algo to the ebola data.
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

# Read in ebola data
ebola_sim = np.load("./data/ebola_sim.npy")
sim_design = np.genfromtxt("./data/sim_design.txt", delimiter = " ")

# The following comment is taken from code written by Arindam Fadikar:
## sima is an nrep x m x nweek dimensional array consisting of 
## all simulations. The simulation corresponding to
## i-th row in d (i.e. d[i,]) can be obtained by 
## calling sima[,i,].

# N gives the total number of simulation runs (including replicates).
# Q Gives the number of time periods with observations
# P Gives the number of sim params (5)
N = 100 * 100
Q = 56
P = 5

# Y - an N by Q array giving all observed runs
# X - An N by P array giving the settings at all observed runs (since there is much replication, many consecutive rows will be identical).
Y = np.moveaxis(ebola_sim,0,1).reshape([N, Q])
X = sim_design[np.repeat(range(sim_design.shape[0]), ebola_sim.shape[0])]

# Standardize Y data
Y = (Y - np.mean(Y)) / np.std(Y)

y = Y
x = X

## GAN Params
#H = 1000 # Number of hidden units
R = Q # Dim of latent error variable
H = 100# Number of hidden units
epochs = 200000
save_every = 500

disc = tf.keras.Sequential()
disc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))

disc.summary()
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')

## Execute training.
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size

    #random_bois = np.random.uniform(low = 2*np.min(y), high = 2*np.max(y), size=[N,R])

    # Get gradient
    with tf.GradientTape(persistent = True) as g:
        prob_true = disc(tf.cast(tf.concat([x, y], 1), tf.float32))
        prob_fake = disc(tf.cast(tf.concat([x, random_bois], 1), tf.float32))
        cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(prob_true, tf.float64)))
        cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
        cost_total = 0.5 * (cost_true + cost_fake)
    graddisc = g.gradient(cost_total, disc.trainable_variables)
    del g

    # Update gen/disc
    dirsnvars_disc = [(graddisc[i], disc.trainable_variables[i]) for i in range(len(graddisc))]
    disc.optimizer.apply_gradients(dirsnvars_disc)

## See if it makes a good anomaly detector.
fig = plt.figure()

#########
# Plot the results
otp1 = 1# Which observation to plot?
minr1 = otp1*100
maxr1 = (otp1+1)*100

for i in range(minr1, maxr1):
    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
plt.savefig("temp.pdf")

disc(np.concatenate([x[100,:], y[100,:]]).astype(np.float32).reshape([1,61]))

fake_y = np.repeat(-np.min(y), y.shape[1])
disc(np.concatenate([x[100,:], fake_y]).astype(np.float32).reshape([1,61]))
