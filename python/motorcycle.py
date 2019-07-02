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
tf.set_random_seed(123)

N = 100
P = 1 # Dim of X data (to be conditioned on)
R = 1 # Dim of latent error variable
Q = 1 # Dim of y data (to be generated)
H = 40# Number of hidden units
epochs = 50000

# Load and pre-process data
mcycle = np.genfromtxt('./data/mcycle.csv', delimiter=',', skip_header = 1)
N = mcycle.shape[0]
x = mcycle[:,0].reshape([N,P])
y = mcycle[:,1].reshape([N,Q])
#x /= max(x)
#y = (y-min(y)) / (max(y) - min(y))
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

# Build the generator, accepts X and Z as inputs
gen = keras.Sequential()
gen.add(keras.layers.Dense(H, input_dim = P + R, activation = keras.activations.elu))
gen.add(keras.layers.Dense(H, activation = keras.activations.elu))
gen.add(keras.layers.Dense(Q))

# Build the discriminator, accepts an X and a Y as inputs.
disc = keras.Sequential()
disc.add(keras.layers.Dense(H, input_dim = P + Q, activation = keras.activations.elu))
disc.add(keras.layers.Dense(H, activation = keras.activations.elu))
disc.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))

gen.summary()
disc.summary()

# NOTE: Compilation of discriminator needs to occur BEFORE we set its weights untrainable below, as these changes will not be reflected until disc is compiled again. So also be wary of compiling disc later, as its weights may not change.
#TODO: the above is a mess, find a better way.
disc.compile(keras.optimizers.Adam(), 'binary_crossentropy')

noise = keras.layers.Input(shape = (R,))
xdat = keras.layers.Input(shape = (P,))

genin = keras.layers.concatenate([xdat, noise])
genout = gen(genin)

discin = keras.layers.concatenate([xdat, genout])
validity = disc(discin)

#NOTE: Next lin possible issue in ordering of inputs?
both_mod = keras.models.Model([xdat, noise], validity)
both_mod.layers[5].trainable = False

both_mod.compile(keras.optimizers.Adam(), 'binary_crossentropy')

# Do the training!
for epoch in tqdm(range(epochs)):
    # Sample some noise
    #TODO: Batch size
    some_noise = np.random.normal(size=[N,P])

    gen_dat = gen.predict(np.hstack([x, some_noise]))

    # Train discriminator
    disc_rl = disc.train_on_batch(np.hstack([x, y]), np.ones(N))
    disc_fl = disc.train_on_batch(np.hstack([x, gen_dat]), np.zeros(N))
    disc_loss = 0.5 * np.add(disc_rl, disc_fl)

    # Train generator
    both_mod.train_on_batch([x, some_noise], np.ones(N))

# Plot the results
fig = plt.figure()
plt.scatter(x, y)
some_noise = np.random.normal(size=[N,P])
preds = gen.predict(np.hstack([x, some_noise]))
plt.scatter(x, preds)
plt.savefig("images/motor_scatter.pdf")
