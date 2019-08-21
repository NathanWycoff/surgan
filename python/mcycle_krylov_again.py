#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/mcycle_krylov_again.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.15.2019

## Do the Krylov thing again, but this time we take both gradients at the same time.
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
R = 1 # Dim of latent error variable
Q = 1 # Dim of y data (to be generated)
H = 40# Number of hidden units
epochs = 10000

## Load and pre-process data
mcycle = np.genfromtxt('./data/mcycle.csv', delimiter=',', skip_header = 1)
N = mcycle.shape[0]
x = mcycle[:,0].reshape([N,P])
y = mcycle[:,1].reshape([N,Q])
x = (x - np.mean(x)) / np.std(x)
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

## Execute training.
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size
    noise = np.random.normal(size=[N,R])

    # Get the gradient (g), the action of the Hessian on the gradient (Hg), and H2g
    with tf.GradientTape() as ggg:
        with tf.GradientTape() as gg:
            with tf.GradientTape() as g:
                gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
                prob_true = disc(tf.cast(tf.concat([x, y], 1), tf.float32))
                prob_fake = disc(tf.cast(tf.concat([x, gendat], 1), tf.float32))
                cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(prob_true, tf.float64)))
                cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
                cost_total = 0.5 * (cost_true + cost_fake)
            grads = g.gradient(cost_total, disc.trainable_variables + gen.trainable_variables)
            gradsvec = tf.concat([tf.reshape(grad, [np.prod(grad.shape)]) for grad in grads],0)
            ip = tf.tensordot(gradsvec, gradsvec, 1)
        Hg = gg.gradient(ip, disc.trainable_variables + gen.trainable_variables)
        Hgvec = tf.concat([tf.reshape(vec, [np.prod(vec.shape)]) for vec in Hg],0)
        ip2 = tf.tensordot(gradsvec, Hgvec, 1)
    H2g = ggg.gradient(ip2, disc.trainable_variables + gen.trainable_variables)
    H2gvec = tf.concat([tf.reshape(vec, [np.prod(vec.shape)]) for vec in H2g],0)

    # Do 1 step of MINRES to get an optimization direction
    krylov_coefs = tf.linalg.lstsq(tf.stack([Hgvec, H2gvec], 1), tf.reshape(gradsvec, [gradsvec.shape[0], 1]))
    d = krylov_coefs[0] * gradsvec + krylov_coefs[1] * Hgvec

    d_discvec = d[:disc.count_params()]
    d_disc = par2mat(d_discvec.numpy(), disc)
    d_genvec = d[disc.count_params():]
    d_gen = par2mat(d_genvec.numpy(), gen)

    # Update gen/disc
    dirsnvars_disc = [(d_disc[i], disc.trainable_variables[i]) for i in range(len(d_disc))]
    dirsnvars_gen = [(-d_gen[i], gen.trainable_variables[i]) for i in range(len(d_gen))]
    disc.optimizer.apply_gradients(dirsnvars_disc)
    gen.optimizer.apply_gradients(dirsnvars_gen)

# Plot the results
fig = plt.figure()
plt.scatter(x, y)
some_noise = np.random.normal(size=[N,P])
preds = gen.predict(np.hstack([x, some_noise]))
plt.scatter(x, preds)
#plt.savefig("images/motor_scatter.pdf")
plt.savefig("temp.pdf")
