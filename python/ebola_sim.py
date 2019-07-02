#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ebola_sim.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.26.2019

## Try to fit a CGAN on the ebola simulator.
import keras
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
#tf.enable_eager_execution()

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
R = Q # Gives the dimension of the latent space for the GAN.
#H = 1000 # Number of hidden units
epochs = 8000
batch_size = 500
save_every = 100
droprate = 0.5
#optmzr = keras.optimizers.Adam()
optmzr = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Build the generator, accepts X and Z as inputs
Hx = 100
genx = keras.Sequential()
genx.add(keras.layers.Dense(Hx, input_dim = P, activation = keras.activations.relu))
genx.add(keras.layers.Dropout(rate = droprate))
#genx.add(keras.layers.Dense(Hx, activation = keras.activations.relu))
#genx.add(keras.layers.Dropout(rate = droprate))

Hz = 100
genz = keras.Sequential()
genz.add(keras.layers.Dense(Hz, input_dim = R, activation = keras.activations.relu))
genz.add(keras.layers.Dropout(rate = droprate))
#genz.add(keras.layers.Dense(Hz, activation = keras.activations.relu))
#genz.add(keras.layers.Dropout(rate = droprate))

xdat = keras.layers.Input(shape = (P,))
noise = keras.layers.Input(shape = (R,))

proc_xdat = genx(xdat)
proc_noise = genz(noise)

Hxz = 256
genxz = keras.Sequential()
genxz.add(keras.layers.Dense(Hxz, input_dim = Hx + Hz))
genxz.add(keras.layers.Dropout(rate = droprate))
#genxz.add(keras.layers.Dense(Hxz))
#genxz.add(keras.layers.Dropout(rate = droprate))
genxz.add(keras.layers.Dense(Q, activation = keras.activations.linear))

xzin = keras.layers.concatenate([proc_xdat, proc_noise])
genimg = genxz(xzin)

gen = keras.models.Model([xdat, noise], genimg)

# Build the discriminator, accepts an X and a Y as inputs.
#disc = keras.Sequential()
#disc.add(keras.layers.Dense(H, input_dim = P + Q, activation = keras.activations.elu))
#disc.add(keras.layers.Dense(H, activation = keras.activations.elu))
#disc.add(keras.layers.Dense(H, activation = keras.activations.elu))
#disc.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))

#TODO: Seperate this Hx from the other maybe?
discx = keras.Sequential()
discx.add(keras.layers.MaxoutDense(Hx, input_dim = P, nb_feature = 5))
discx.add(keras.layers.Dropout(droprate))

Hy = 100
discy = keras.Sequential()
discy.add(keras.layers.MaxoutDense(Hy, input_dim = Q, nb_feature = 5))
discy.add(keras.layers.Dropout(droprate))

Hxy = 200
discxy = keras.Sequential()
discxy.add(keras.layers.MaxoutDense(Hxy, input_dim = Hx + Hy, nb_feature = 4))
discxy.add(keras.layers.Dropout(droprate))
discxy.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))

xdat_disc = keras.layers.Input(shape = (P,))
ydat = keras.layers.Input(shape = (Q,))

proc_xdat_disc = discx(xdat_disc)
proc_ydat = discy(ydat)

xyin = keras.layers.concatenate([proc_xdat_disc, proc_ydat])
discprob = discxy(xyin)

disc = keras.models.Model([xdat_disc, ydat], discprob)

# NOTE: Compilation of discriminator needs to occur BEFORE we set its weights untrainable below, as these changes will not be reflected until disc is compiled again. So also be wary of compiling disc later, as its weights may not change.
#TODO: the above is a mess, find a better way.
#disc.compile(keras.optimizers.Adam(), 'binary_crossentropy')
disc.compile(optmzr, 'binary_crossentropy')

xdat_both = keras.layers.Input(shape = (P,))
noise_both = keras.layers.Input(shape = (R,))

genout = gen([xdat_both, noise_both])

validity = disc([xdat_both, genout])

both_mod = keras.models.Model([xdat_both, noise_both], validity)
#both_mod.layers[3].trainable = False

disc.trainable = False
both_mod.compile(optmzr, 'binary_crossentropy')
disc.trainable = True

gen.summary()
disc.summary()
both_mod.summary()

# Can we do double backprop?


# Do the training!
dloss = np.empty(epochs)
for epoch in tqdm(range(epochs)):
    rand_ord = np.random.choice(N,size=N, replace = False)

    n_batches = int(np.ceil(N/batch_size))
    for batch in range(n_batches):
        batch_idx = rand_ord[(batch_size*batch):(batch_size*(batch+1))]

        # If batch_size does not divide N evenly, last batch will have fewer elements.
        actual_bs = len(batch_idx)

        # Sample some noise, and a batch
        some_noise = np.random.normal(size=[actual_bs,R])

        gen_dat = gen.predict([x[batch_idx,:], some_noise])

        # Train discriminator
        disc_rl = disc.train_on_batch([x[batch_idx,:], y[batch_idx,:]], np.ones(actual_bs))
        disc_fl = disc.train_on_batch([x[batch_idx,:], gen_dat], np.zeros(actual_bs))
        disc_loss = 0.5 * np.add(disc_rl, disc_fl)

        # Store disc loss to see if we're oscillating.
        dloss[epoch] = disc_loss

        # Train generator
        both_mod.train_on_batch([x[batch_idx,:], some_noise], np.ones(actual_bs))

    if epoch % save_every == 0:

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
        preds = gen.predict([x[minr1:maxr1,:], some_noise])
        for i in range(100):
            plt.plot(preds[i,:], color = 'orange', alpha = 0.4)

        plt.subplot(1,2,2)
        for i in range(minr2, maxr2):
            plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
        # Plot the predictions
        # Just the first setting for now
        some_noise = np.random.normal(size=[100,R])
        preds = gen.predict([x[minr2:maxr2,:], some_noise])
        for i in range(100):
            plt.plot(preds[i,:], color = 'orange', alpha = 0.4)

        plt.savefig("./images/prog/epoch_%d.pdf"%epoch)
        plt.close(fig)
