#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/motorcycle_linesearch.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.12.2019
#TODO: Does consensus optimization require us to consider the effect of nets on one another? Here, I am doing the gradient norms separately. Is that a problem?

# Run a CGAN on the motorcycle data.
import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(123)
import tensorflow as tf
from scipy.optimize import line_search
tf.enable_eager_execution()
tf.set_random_seed(123)

P = 1 # Dim of X data (to be conditioned on)
R = 1 # Dim of latent error variable
Q = 1 # Dim of y data (to be generated)
H = 40# Number of hidden units
epochs = 1000
doubleback_const = 0.1

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
gen = tf.keras.Sequential()
gen.add(tf.keras.layers.Dense(H, input_dim = P + R, activation = tf.keras.activations.elu))
gen.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
gen.add(tf.keras.layers.Dense(Q))

# Build the discriminator, accepts an X and a Y as inputs.
disc = tf.keras.Sequential()
disc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.elu))
disc.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))

gen.summary()
disc.summary()

# NOTE: Compilation of discriminator needs to occur BEFORE we set its weights untrainable below, as these changes will not be reflected until disc is compiled again. So also be wary of compiling disc later, as its weights may not change.
#TODO: the above is a mess, find a better way.
#disc.compile(tf.keras.optimizers.Adam(), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 1.0), 'binary_crossentropy')

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
#both_mod.compile(tf.train.AdamOptimizer(), 'binary_crossentropy')
both_mod.compile(tf.train.GradientDescentOptimizer(learning_rate = 1.0), 'binary_crossentropy')

## Custom training with double backprop
#genloss = lambda: both_mod.output
#genopt = tf.keras.optimizers.Adam(genloss, both_mod.trainable_variables)

def par2mat(weights, nnet):
    """
    Turn a weight vector into a list of weights suitable for assignment to a neural net. First arg is the vec, second the net.
    """
    weights_list = []
    used_weights = 0
    for l in nnet.layers:
        nin = l.input_shape[1]
        nout = l.output_shape[1]
        weights_list.append(weights[(used_weights):(used_weights+nin*nout)].reshape((nin,nout)))
        used_weights += nin*nout
        weights_list.append(weights[(used_weights):(used_weights+nout)].flatten())
        used_weights += nout

    if used_weights < len(weights):
        print("Warning: weight input is longer than required for this neural network.")

    return(weights_list)

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

    #TODO: mat2par
    def disc_obj(weights, grad = False):

        weights_list = par2mat(weights, disc)
        disc.set_weights(weights_list)

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
            total_cost = tf.cast(dl, tf.float32) + doubleback_const * grads_norm

        double_grads = td.gradient(grads_norm, disc.trainable_variables)

        both_grads = [grads[i] + doubleback_const*double_grads[i] for i in range(len(grads))]

        if grad:
            return(np.concatenate([x.numpy().flatten() for x in both_grads]))
        else:
            return(total_cost.numpy())

    # Do SGD with line search
    weight_vec = np.concatenate([x.flatten() for x in disc.get_weights()])
    dgrad = disc_obj(weight_vec, grad = True)
    ls = line_search(disc_obj, lambda x: disc_obj(x, grad = True), weight_vec, -dgrad)
    if ls[0] is not None:
        step = min(ls[0], 1.0)
    else:
        step = 1.0
    new_weight_vec = weight_vec - step * dgrad
    disc.set_weights(par2mat(new_weight_vec, disc))
    #grads_n_vars = [(double_grads[i], disc.trainable_variables[i]) for i in range(len(grads))]
    #disc.optimizer.apply_gradients(grads_n_vars)
    #disc.trainable = False

    print("Disc step: %f"%step)

    # Train generator
    #both_mod.train_on_batch([x, some_noise], np.ones(N))
    # Manually compute and apply gradient

    def gen_obj(weights, grad = False):

        weights_list = par2mat(weights, gen)
        gen.set_weights(weights_list)

        with tf.GradientTape() as td:
            with tf.GradientTape() as t:
                #preds = both_mod([tf.cast(x, tf.float32), tf.cast(some_noise, tf.float32)])
                #gen_dat = gen(tf.cast(np.hstack([x, some_noise]), tf.float32))
                #preds = disc(tf.cast(np.hstack([x, gen_dat]), tf.float32))
                gen_dat = gen(tf.cast(tf.concat([x, some_noise], 1), tf.float32))
                preds = disc(tf.cast(tf.concat([x, gen_dat], 1), tf.float32))
                bl = tf.reduce_mean(keras.losses.binary_crossentropy(np.ones(N).reshape([N,1]), tf.cast(preds, tf.float64)))
                #bl = tf.losses.sigmoid_cross_entropy(preds, np.ones(N).reshape([N,1]))

            grads = t.gradient(bl, gen.trainable_variables)
            grads_norm = 0
            for i in range(len(grads)):
                #grads_norm += tf.reduce_sum(tf.square(grads[i]))
                grads_norm += tf.reduce_mean(tf.square(grads[i]))
            grads_norm /= float(len(grads))
            total_cost = tf.cast(bl, tf.float32) + doubleback_const * grads_norm

        double_grads = td.gradient(grads_norm, gen.trainable_variables)

        both_grads = [grads[i] + doubleback_const*double_grads[i] for i in range(len(grads))]

        if grad:
            return(np.concatenate([x.numpy().flatten() for x in both_grads]))
        else:
            return(total_cost.numpy())

    # Do SGD with line search
    weight_vec = np.concatenate([x.flatten() for x in gen.get_weights()])
    dgrad = gen_obj(weight_vec, grad = True)
    ls = line_search(gen_obj, lambda x: gen_obj(x, grad = True), weight_vec, -dgrad)
    if ls[0] is not None:
        step = min(ls[0], 1.0)
    else:
        step = 1.0
    new_weight_vec = weight_vec - step * dgrad
    gen.set_weights(par2mat(new_weight_vec, gen))
    #grads_n_vars = [(grads[i] + doubleback_const*double_grads[i], both_mod.trainable_variables[i]) for i in range(len(grads))]
    #both_mod.optimizer.apply_gradients(grads_n_vars)

    print("Gen step: %f"%step)

# Plot the results
fig = plt.figure()
plt.scatter(x, y)
some_noise = np.random.normal(size=[N,P])
preds = gen.predict(np.hstack([x, some_noise]))
plt.scatter(x, preds)
#plt.savefig("images/motor_scatter.pdf")
plt.savefig("temp.pdf")
