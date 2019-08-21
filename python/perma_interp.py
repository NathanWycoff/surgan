#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/mcycle_krylov_again.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.15.2019

## Can we get the footballs?
import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(123)
import tensorflow as tf
from scipy.optimize import line_search
tf.enable_eager_execution()
tf.set_random_seed(123)
exec(open("./python/lib.py").read())
### Generate Data
## A simple linear model between 0 and 1 but we never observe data between 0.4 and 0.6
N = 50#Should be even
x = np.vstack([np.random.uniform(0.0,0.3, size = [int(N/2),1]),np.random.uniform(0.7,1.0, size = [int(N/2),1])])
y = x*(x<0.5) + x*(x>=0.5)

## GAN Params
P = 1 # Dim of X data (to be conditioned on)
R = 1 # Dim of latent error variable
Q = 1 # Dim of y data (to be generated)
C = N#Number of constraints
H = 40# Number of hidden units
epochs = 10000
droprate = 0.1

## Create our adversaries.
gen = tf.keras.Sequential()
gen.add(tf.keras.layers.Dense(H, input_dim = P + R, activation = tf.keras.activations.relu))
gen.add(PermanentDropout(droprate))
gen.add(tf.keras.layers.Dense(Q))

pdisc = tf.keras.Sequential()
pdisc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.relu))

xin = tf.keras.layers.Input(shape=(P,))
yin = tf.keras.layers.Input(shape=(Q,))
other_inputs = tf.keras.layers.Input(shape=(C,))

xycat = tf.keras.layers.Concatenate(1)([xin, yin])
pdisco = pdisc(xycat)

concated = tf.keras.layers.Concatenate(1)([pdisco, other_inputs])
discout = tf.keras.layers.Dense(1, input_dim = H + C, \
        activation = tf.keras.activations.sigmoid)(concated)

disc = tf.keras.models.Model([xin, yin, other_inputs], discout)

gen.summary()
disc.summary()
gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')

## Execute Adversarial training.
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size
    noise = np.random.normal(size=[N,R])

    # Get gradient
    with tf.GradientTape(persistent = True) as g:
        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
        prob_true = disc([tf.cast(x, tf.float32), tf.cast(y, tf.float32), tf.cast(interp_constr(x, y, y), tf.float32)])
        prob_fake = disc([tf.cast(x, tf.float32), tf.cast(gendat, tf.float32), tf.cast(interp_constr(x, y, gendat), tf.float32)])
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

## Plot the results
#fig = plt.figure()
#xpred = np.linspace(0,1).reshape([50,1])
#plt.scatter(x, y)
#some_noise = np.random.normal(size=[50,P])
#preds = gen.predict(np.hstack([xpred, some_noise]))
#plt.scatter(xpred, preds)
##plt.savefig("images/motor_scatter.pdf")
#plt.savefig("temp.pdf")

# Get confidence intervals
fig = plt.figure()
xpred = np.linspace(0,1).reshape([50,1])
plt.scatter(x, y)
npred = 100
preds = np.empty([npred,50,P])
for i in range(npred):
    some_noise = np.random.normal(size=[50,P])
    preds[i,:,:] = gen.predict(np.hstack([xpred, some_noise]))

# 75, 50 and 25 quantiles
percs = np.percentile(preds, q = [25., 50., 75.], axis = 0)

plt.plot(xpred, percs[0,:,:], c = 'orange')
plt.plot(xpred, percs[2,:,:], c = 'orange')
plt.plot(xpred, percs[1,:,:], c = 'red')

#plt.savefig("images/motor_scatter.pdf")
plt.savefig("temp.pdf")
plt.close()
