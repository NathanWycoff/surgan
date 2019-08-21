#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ebola_constr.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.27.2019

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

# Sqrt Transform
Y = np.sqrt(Y)

# Standardize Y data
Y = (Y - np.mean(Y)) / np.std(Y)

y = Y
x = X

## GAN Params
#H = 1000 # Number of hidden units
R = Q # Dim of latent error variable
H = 100# Number of hidden units
epochs = 10000
save_every = 500
dp_scale = 0.1

## Create our adversaries.
gen = tf.keras.Sequential()
gen.add(tf.keras.layers.Dense(H, input_dim = P + R, activation = tf.keras.activations.elu))
gen.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
gen.add(tf.keras.layers.Dense(Q))

pdisc = tf.keras.Sequential()
pdisc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.elu))
pdisc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))

xin = tf.keras.layers.Input(shape=(P,))
yin = tf.keras.layers.Input(shape=(Q,))

#genout = gen(tf.cast(tf.concat([xin, yvar], 1), tf.float32))
yindiff = yin[:,1:]-yin[:,:-1]
yindiff.set_shape(tf.TensorShape([None, yin.shape[1]-1]))
yin_smooth = tf.reduce_sum(tf.square(yindiff), axis = 1, keepdims = True)# Measure of smoothness
yin_mon = tf.math.maximum(-tf.reduce_min(yindiff, axis = 1, keepdims = True), 0)# Measure of nonmonotonicity (0 means perfectly monotone).

xycat = tf.keras.layers.Concatenate(1)([xin, yin])
pdiscout = pdisc(xycat)

concated = tf.keras.layers.Concatenate(1)([pdiscout, discin_smooth, discin_mon])
discout = tf.keras.layers.Dense(1, input_dim = H + Q,  activation = tf.keras.activations.sigmoid)(concated)

disc = tf.keras.models.Model(inputs = [xvar, discin], outputs = discout)

gen.summary()
disc.summary()
gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')

## Execute training.
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size
    noise = np.random.normal(size=[N,R])

    # Get gradient g and Hg
    with tf.GradientTape(persistent = True) as tt:
        with tf.GradientTape(persistent = True) as t:
            gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
            prob_true = disc(tf.cast(tf.concat([x, y], 1), tf.float32))
            prob_fake = disc(tf.cast(tf.concat([x, gendat], 1), tf.float32))
            cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(prob_true, tf.float64)))
            cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
            cost_total = 0.5 * (cost_true + cost_fake)
        graddisc = t.gradient(cost_total, disc.trainable_variables)
        gradgen = t.gradient(cost_total, gen.trainable_variables)
        gradsvec = tf.concat([tf.reshape(vec, [np.prod(vec.shape)]) for vec in graddisc + gradgen],0)
        ip = tf.tensordot(gradsvec, gradsvec,1)
    Hgdisc = tt.gradient(ip, disc.trainable_variables)
    Hggen = tt.gradient(ip, gen.trainable_variables)
    del t
    del tt

    # Update gen/disc
    dirsnvars_disc = [(graddisc[i] + dp_scale * Hgdisc[i], disc.trainable_variables[i]) for i in range(len(graddisc))]
    dirsnvars_gen = [(-gradgen[i] + dp_scale * Hggen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
    disc.optimizer.apply_gradients(dirsnvars_disc)
    gen.optimizer.apply_gradients(dirsnvars_gen)

    if epoch % save_every == 0:
        #########
        # Plot the results
        otp1 = 1# Which observation to plot?
        minr1 = otp1*100
        maxr1 = (otp1+1)*100
        otp2 = 2# Which observation to plot?
        minr2 = otp2*100
        maxr2 = (otp2+1)*100

        # Plot the data
        # Just the first setting for now
        fig = plt.figure()
        plt.subplot(1,2,1)
        for i in range(minr1, maxr1):
            plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
        # Plot the predictions
        # Just the first setting for now
        some_noise = np.random.normal(size=[100,R])
        preds = gen.predict(np.hstack([x[minr1:maxr1,:], some_noise]))
        for i in range(100):
            plt.plot(preds[i,:], color = 'orange', alpha = 0.4)

        plt.subplot(1,2,2)
        for i in range(minr2, maxr2):
            plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
        # Plot the predictions
        # Just the first setting for now
        some_noise = np.random.normal(size=[100,R])
        preds = gen.predict(np.hstack([x[minr2:maxr2,:], some_noise]))
        for i in range(100):
            plt.plot(preds[i,:], color = 'orange', alpha = 0.4)

        #plt.savefig("./images/prog/epoch_%d.pdf"%epoch)
        plt.savefig("temp.pdf")
        plt.close(fig)
