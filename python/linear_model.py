#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/nobs.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.23.2019

import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

N = 100
P = 1 # Dim of X data (to be conditioned on)
R = 1 # Dim of latent error variable
Q = 1 # Dim of y data (to be generated)
H = 40# Number of hidden units
epochs = 2000

# A simple linear model
sigma = 0.1
x = np.random.normal(size=[N,P])
z = np.random.normal(0, sigma, size=[N,P])
y = x + z

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

both_mod = keras.models.Model([noise, xdat], validity)
both_mod.layers[5].trainable = False

both_mod.compile(keras.optimizers.Adam(), 'binary_crossentropy')

# Do the training!
#TODO: The discriminator's weights seem to be fixed.
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
plt.savefig("temp.pdf")
