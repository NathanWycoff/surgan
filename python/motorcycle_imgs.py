#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/motorcycle_imgs.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.23.2019

# Convert the motorcycle data to 1D "image" data by showing a normal pdf centered at whatever datapoint.

# Run a CGAN on the motorcycle data.
import keras
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

P = 1 # Dim of X data (to be conditioned on)
R = 1 # Dim of latent error variable
Q = 10 # Dim of y data (to be generated)
H = 40# Number of hidden units
epochs = 20000
bandwidth_coef = 1.0/60.0# Bandwidth for image creation.

# Load and pre-process data
mcycle = np.genfromtxt('./data/mcycle.csv', delimiter=',', skip_header = 1)
N = mcycle.shape[0]
x = mcycle[:,0].reshape([N,P])
y_dat = mcycle[:,1].reshape([N,1])
#x /= max(x)
#y_dat = (y_dat-min(y_dat)) / (max(y_dat) - min(y_dat))
x = (x - np.mean(x)) / np.std(x)
y_dat = (y_dat - np.mean(y_dat)) / np.std(y_dat)

# Turn each datum into an image
y_seq = np.linspace(np.min(y_dat), np.max(y_dat), num = Q)
normdens = lambda d, l: np.exp(-np.square(d) / (2*np.square(l)))
y = np.empty([N,Q])
for n in range(N):
    y[n] = normdens(y_seq - y_dat[n], bandwidth_coef * n)

fig = plt.figure()
plt.imshow(y.T, origin = 'lower')
plt.savefig("temp.pdf")

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

both_mod = keras.models.Model([xdat, noise], validity)
both_mod.layers[5].trainable = False

both_mod.compile(keras.optimizers.Adam(), 'binary_crossentropy')

# Do the training!
dloss = np.empty(epochs)
for epoch in tqdm(range(epochs)):
    # Sample some noise
    #TODO: Batch size
    some_noise = np.random.normal(size=[N,R])

    gen_dat = gen.predict(np.hstack([x, some_noise]))

    # Train discriminator
    disc_rl = disc.train_on_batch(np.hstack([x, y]), np.ones(N))
    disc_fl = disc.train_on_batch(np.hstack([x, gen_dat]), np.zeros(N))
    disc_loss = 0.5 * np.add(disc_rl, disc_fl)

    # Store disc loss to see if we're oscillating.
    dloss[epoch] = disc_loss

    # Train generator
    both_mod.train_on_batch([x, some_noise], np.ones(N))

# Plot the results

fig = plt.figure()
plt.subplot(2,1,1)
plt.imshow(y.T, origin = 'lower')
some_noise = np.random.normal(size=[N,R])
preds = gen.predict(np.hstack([x, some_noise]))
plt.subplot(2,1,2)
plt.imshow(preds.T, origin = 'lower')
plt.savefig("images/motor_img.pdf")
