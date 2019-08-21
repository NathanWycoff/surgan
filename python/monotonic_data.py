#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/monotonic_data.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.18.2019

# Fit a GAN to some synthetic logistic-ish functions.
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

P = 1 # Dim of X data (to be conditioned on)
Q = 56 # Dim of y data (to be generated)
R = Q # Dim of latent error variable
H = 100# Number of hidden units
epochs = 10000
save_every = 500

## Load and pre-process data
monfunc = np.genfromtxt('./data/synthetic_mon_func_Y.csv', delimiter=',', skip_header = 1)
N = monfunc.shape[0]
x = np.array([1.0 for _ in range(N)]).reshape([N,P])
y = monfunc
#x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

## Create our adversaries.
gen = tf.keras.Sequential()
gen.add(tf.keras.layers.Dense(H, input_dim = P + R, activation = tf.keras.activations.elu))
gen.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
gen.add(tf.keras.layers.Dense(Q))

disc = tf.keras.Sequential()
disc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))

gen.summary()
disc.summary()
gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')

## Pretraining the generator via l1 loss with no onise
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size
    noise = np.zeros(shape=[N,R])

    # Get gradient
    with tf.GradientTape(persistent = True) as g:
        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
        l1_loss = tf.reduce_mean(tf.abs(gendat - y))
    gradgen = g.gradient(l1_loss, gen.trainable_variables)
    del g

    # Update gen/disc
    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
    gen.optimizer.apply_gradients(dirsnvars_gen)

# Visualize pre-trained solution
Ns = 100# Number of samples to plot
# Plot the results
fig = plt.figure()
for i in range(Ns):
    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
# Plot the predictions
some_noise = np.zeros(shape=[N,R])
preds = gen.predict(np.hstack([x[:Ns,:], some_noise[:Ns,:]]))
plt.plot(preds[i,:], color = 'orange')
plt.savefig("pretraining.pdf")
plt.close(fig)

## Execute training.
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size
    noise = np.random.normal(size=[N,R])

    # Get gradient
    with tf.GradientTape(persistent = True) as g:
        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
        prob_true = disc(tf.cast(tf.concat([x, y], 1), tf.float32))
        prob_fake = disc(tf.cast(tf.concat([x, gendat], 1), tf.float32))
        cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(prob_true, tf.float64)))
        cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
        cost_total = 0.5 * (cost_true + cost_fake)
        gen_cost = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
    graddisc = g.gradient(cost_total, disc.trainable_variables)
    gradgen = g.gradient(gen_cost, gen.trainable_variables)
    del g

    # Update gen/disc
    dirsnvars_disc = [(graddisc[i], disc.trainable_variables[i]) for i in range(len(graddisc))]
    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
    disc.optimizer.apply_gradients(dirsnvars_disc)
    gen.optimizer.apply_gradients(dirsnvars_gen)

    #########
    if epoch % save_every == 0:
        # Number of samples to plot
        Ns = 100
        # Plot the results
        fig = plt.figure()
        # Plot the predictions
        some_noise = np.random.normal(size=[Ns,R])
        preds = gen.predict(np.hstack([x[:Ns,:], some_noise]))
        for i in range(Ns):
            plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
        # Plot the data
        for i in range(Ns):
            plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
        plt.savefig("./images/prog/epoch_%d.pdf"%epoch)
        #plt.savefig("temp.pdf")
        plt.close(fig)
