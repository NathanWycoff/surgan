#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/motorcycle.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.23.2019

# Run a CGAN on the motorcycle data.
import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(123)
import tensorflow as tf
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
epochs = 10000
save_every = 500
doubleback_const = 1

# Build the generator, accepts X and Z as inputs
gen = tf.keras.Sequential()
gen.add(tf.keras.layers.Dense(H, input_dim = P + R, activation = tf.keras.activations.elu))
#gen.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
gen.add(tf.keras.layers.Dense(Q))

# Build the discriminator, accepts an X and a Y as inputs.
disc = tf.keras.Sequential()
disc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.elu))
#disc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))

gen.summary()
disc.summary()

# NOTE: Compilation of discriminator needs to occur BEFORE we set its weights untrainable below, as these changes will not be reflected until disc is compiled again. So also be wary of compiling disc later, as its weights may not change.
#TODO: the above is a mess, find a better way.
#disc.compile(tf.keras.optimizers.Adam(), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(1.0), 'binary_crossentropy')

noise = tf.keras.layers.Input(shape = (R,))
xdat = tf.keras.layers.Input(shape = (P,))

genin = tf.keras.layers.concatenate([xdat, noise])
genout = gen(genin)

discin = tf.keras.layers.concatenate([xdat, genout])
validity = disc(discin)

#NOTE: Next lin possible issue in ordering of inputs?
both_mod = tf.keras.models.Model([xdat, noise], validity)
both_mod.layers[5].trainable = False

#both_mod.compile(tf.keras.optimizers.Adam(), 'binary_crossentropy')
both_mod.compile(tf.train.GradientDescentOptimizer(1.0), 'binary_crossentropy')

## Custom training with double backprop
#genloss = lambda: both_mod.output
#genopt = tf.keras.optimizers.Adam(genloss, both_mod.trainable_variables)

# Do the training!
for epoch in tqdm(range(epochs)):
    # Sample some noise
    #TODO: Batch size
    some_noise = np.random.normal(size=[N,R])

    gen_dat = gen.predict(np.hstack([x, some_noise]))

    # Train discriminator
    #NOTE: Minor discrepency in losses from the manual loop below and from keras's built in: follow up if there appears to be bugs.
    #disc_rl = disc.train_on_batch(np.hstack([x, y]), np.ones(N))
    #disc_fl = disc.train_on_batch(np.hstack([x, gen_dat]), np.zeros(N))
    #disc_loss = 0.5 * np.add(disc_rl, disc_fl)

    disc.trainable = True
    with tf.GradientTape() as td:
        with tf.GradientTape() as t:
            #preds_real = disc(tf.cast(np.concatenate([x, y]).reshape([N,P+Q]), tf.float32))
            #preds_fake = disc(tf.cast(np.concatenate([x, gen_dat]).reshape([N,P+Q]), tf.float32))
            preds_real = disc(tf.cast(np.hstack([x, y.reshape([N,Q])]), tf.float32))
            preds_fake = disc(tf.cast(np.hstack([x, gen_dat]), tf.float32))
            dl_real = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(preds_real, tf.float64)))
            dl_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(preds_fake, tf.float64)))
            dl = 0.5*tf.add(dl_real, dl_fake)

        grads = t.gradient(dl, disc.trainable_variables)
        grads_norm = 0
        for i in range(len(grads)):
            #grads_norm += tf.reduce_sum(tf.square(grads[i]))
            grads_norm += tf.reduce_mean(tf.square(grads[i]))
        grads_norm /= float(len(grads))

    double_grads = td.gradient(grads_norm, disc.trainable_variables)

    grads_n_vars = [(grads[i] + doubleback_const * double_grads[i], disc.trainable_variables[i]) for i in range(len(grads))]
    disc.optimizer.apply_gradients(grads_n_vars)
    disc.trainable = False

    # Train generator
    #both_mod.train_on_batch([x, some_noise], np.ones(N))
    # Manually compute and apply gradient
    with tf.GradientTape() as td:
        with tf.GradientTape() as t:
            preds = both_mod([tf.cast(x, tf.float32), tf.cast(some_noise, tf.float32)])
            bl = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(preds, tf.float64)))
            #bl = tf.losses.sigmoid_cross_entropy(preds, np.ones(N).reshape([N,1]))

        grads = t.gradient(bl, both_mod.trainable_variables)
        grads_norm = 0
        for i in range(len(grads)):
            #grads_norm += tf.reduce_sum(tf.square(grads[i]))
            grads_norm += tf.reduce_mean(tf.square(grads[i]))
        grads_norm /= float(len(grads))

    double_grads = td.gradient(grads_norm, both_mod.trainable_variables)

    grads_n_vars = [(grads[i] + doubleback_const*double_grads[i], both_mod.trainable_variables[i]) for i in range(len(grads))]
    both_mod.optimizer.apply_gradients(grads_n_vars)

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

        plt.savefig("./images/prog/epoch_%d.pdf"%epoch)
        #plt.savefig("temp.pdf")
        plt.close(fig)
