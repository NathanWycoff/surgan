#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ebola_constr_again.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.28.2019

# Try to impose smoothness and monotonicity

import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(123)
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import line_search
tf.enable_eager_execution()
tf.set_random_seed(123)
exec(open("./python/lib.py").read())

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
Y = np.sqrt(np.moveaxis(ebola_sim,0,1).reshape([N, Q]))
X = sim_design[np.repeat(range(sim_design.shape[0]), ebola_sim.shape[0])]

# Standardize Y data
Y = (Y - np.mean(Y)) / np.std(Y)

y = Y
x = X

# Out of sample stuff
otp1 = 1# Which observation to plot?
minr1 = otp1*100
maxr1 = (otp1+1)*100
yall = y
xall = x
y = np.vstack([y[:minr1,:],y[maxr1:,:]])
x = np.vstack([x[:minr1,:],x[maxr1:,:]])
N -= 100

## GAN Params
#H = 1000 # Number of hidden units
R = Q # Dim of latent error variable
C = 2#Number of constraints
#C = 0#Number of constraints
H = 100# Number of hidden units
epochs = 20000
save_every = 500
label_smooth = 0.9

## Create our adversaries.
gen = tf.keras.Sequential()
gen.add(tf.keras.layers.Dense(H, input_dim = P + R, activation = tf.keras.activations.relu))
gen.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.relu))
gen.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.relu))
gen.add(tf.keras.layers.Dense(Q))

pdisc = tf.keras.Sequential()
pdisc.add(tf.keras.layers.Dense(H, input_dim = P + Q, activation = tf.keras.activations.relu))
pdisc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.relu))
pdisc.add(tf.keras.layers.Dense(H, activation = tf.keras.activations.relu))

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
gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')

# Pretraining the generator via l2 loss with no onise
l2_losses = np.empty([epochs])
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size
    noise = np.zeros(shape=[N,R])

    # Get gradient
    with tf.GradientTape(persistent = True) as g:
        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(gendat - y)))
    gradgen = g.gradient(l2_loss, gen.trainable_variables)
    del g

    l2_losses[epoch] = l2_loss

    # Update gen/disc
    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
    gen.optimizer.apply_gradients(dirsnvars_gen)


fig = plt.figure()
plt.plot(np.log(l2_losses))
plt.savefig("l2_loss.pdf")

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
# Plot the predictions
# Just the first setting for now
some_noise = np.random.normal(size=[100,R])
preds = gen.predict(np.hstack([x[minr1:maxr1,:], some_noise]))
for i in range(100):
    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
for i in range(minr1, maxr1):
    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)

plt.subplot(1,2,2)
some_noise = np.random.normal(size=[100,R])
preds = gen.predict(np.hstack([x[minr2:maxr2,:], some_noise]))
for i in range(100):
    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
for i in range(minr2, maxr2):
    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
# Plot the predictions
# Just the first setting for now


plt.savefig("pretraining.pdf")
plt.close(fig)


## Pretraining via Quantile Alignment
epochs = 5000

#gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
#disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
gen.compile(tf.train.AdamOptimizer(), 'binary_crossentropy')
disc.compile(tf.train.AdamOptimizer(), 'binary_crossentropy')

#def quantile_loss(truth, guess, quant):
#    """
#    This function should be given a block of 100 truths and guesses (with the same x).
#
#    It looks at the squared sums of differences between quantiles at each of 56 points.
#
#    It will not work on more general problems: we should do something cute involving tilted losses.
#    """
#    true_quants = tf.map_fn(lambda x: tfp.stats.percentile(x, quant), tf.transpose(truth))
#    guess_quants = tf.map_fn(lambda x: tfp.stats.percentile(x, quant), tf.transpose(guess))
#    return(tf.reduce_mean(tf.square(tf.cast(true_quants, tf.float32) - guess_quants)))

def quantile_loss(truth, draws, quant):
    """
    Truth is a matrix.
    draws is an array, the first index gives draws for the same datapoint, the two others give the data dimensions.
    quant is the vector of quantiles
    """
    U = len(quant)
    guess_quant = tfp.stats.percentile(draws, q = 100.0*quant, axis = 0)
    resids = truth - guess_quant
    tilt_loss = 0
    for u in range(U):
        tilt_loss += (quant[u] * tf.cast(resids[u,:,:] > 0, tf.float32) * resids[u,:,:] + \
                (quant[u]-1) * tf.cast(resids[u,:,:] < 0, tf.float32) * resids[u,:,:]) / float(U)
    tot_loss = tf.reduce_mean(tilt_loss)
    return(tot_loss)

K = 20#Number of draws for each point
U = 10#Number of percentiles to sample
qls = np.empty([epochs])
## Also tilted quantile loss
for epoch in tqdm(range(epochs)):

    #TODO: Batch Size
    noise = tf.cast(np.random.normal(size=[K,N,R]), tf.float32)
    #quant = np.random.uniform(size=U)
    quant = np.linspace(0.1, 0.9, num = U)

    # Get gradient
    with tf.GradientTape(persistent = True) as g:
        draws = tf.map_fn(lambda r: gen(tf.cast(tf.concat([x, r], 1), tf.float32)), noise)
        tilt_loss = quantile_loss(y, draws, quant)
    gradgen = g.gradient(tilt_loss, gen.trainable_variables)
    del g

    qls[epoch] = tilt_loss

    # Update gen/disc
    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
    gen.optimizer.apply_gradients(dirsnvars_gen)

fig = plt.figure()
plt.plot(np.log10(qls[:epoch]))
plt.savefig("quantile_loss.pdf")

### Pretraining the generator via elementwise quantile alignment
#for epoch in tqdm(range(epochs)):
#    #TODO: Batch Size
#
#    # Get gradient
#    with tf.GradientTape(persistent = True) as g:
#        #NOTE: 100's are specific to our dataset 
#        noise = np.random.normal(size=[N,R])
#        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
#        p10_loss = 0
#        p90_loss = 0
#
#        #for m in np.random.choice(100,2):
#        for m in range(100):
#            p10_loss += quantile_loss(y[(m*100):((m+1)*100)], gendat[(m*100):((m+1)*100),:], 0.1) / 100.0
#            p90_loss += quantile_loss(y[(m*100):((m+1)*100)], gendat[(m*100):((m+1)*100),:], 0.9) / 100.0
#
#        quant_loss = p10_loss + p90_loss
#    gradgen = g.gradient(quant_loss, gen.trainable_variables)
#    del g
#
#    # Update gen/disc
#    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
#    gen.optimizer.apply_gradients(dirsnvars_gen)

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
# Plot the predictions
# Just the first setting for now
#some_noise = np.zeros(shape=[100,R])
some_noise = np.random.normal(size=[100,R])
preds = gen.predict(np.hstack([x[minr1:maxr1,:], some_noise]))
for i in range(100):
    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
for i in range(minr1, maxr1):
    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)

plt.subplot(1,2,2)
some_noise = np.random.normal(size=[100,R])
preds = gen.predict(np.hstack([x[minr2:maxr2,:], some_noise]))
for i in range(100):
    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
for i in range(minr2, maxr2):
    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
# Plot the predictions
# Just the first setting for now

plt.savefig("pretraining2.pdf")
plt.close(fig)

gen.save_weights('loo_pretrained_gen.hdf5')
gen.load_weights('loo_pretrained_gen.hdf5')

def ebola_constr(y):
    return(tf.zeros([N,C]))

gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')

## Pretrain the discriminator a little bit as well on the trained generator.
epochs = 100000
for epoch in tqdm(range(epochs)):
    #TODO: Batch Size

    noise = np.random.normal(size=[N,R])

    # Get gradient
    with tf.GradientTape(persistent = True) as g:
        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
        #prob_true = disc(tf.cast(tf.concat([x, y], 1), tf.float32))
        #prob_fake = disc(tf.cast(tf.concat([x, gendat], 1), tf.float32))
        prob_true = disc([tf.cast(x, tf.float32), tf.cast(y, tf.float32), tf.cast(ebola_constr(y), tf.float32)])
        prob_fake = disc([tf.cast(x, tf.float32), tf.cast(gendat, tf.float32), tf.cast(ebola_constr(gendat), tf.float32)])
        cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(label_smooth*np.ones(N).reshape([N,1]), tf.cast(prob_true, tf.float64)))
        cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
        cost_total = 0.5 * (cost_true + cost_fake)
    graddisc = g.gradient(cost_total, disc.trainable_variables)
    del g

    # Update gen/disc
    dirsnvars_disc = [(graddisc[i], disc.trainable_variables[i]) for i in range(len(graddisc))]
    disc.optimizer.apply_gradients(dirsnvars_disc)

pretrain_disc_loss = cost_total
disc_pre_fake = prob_fake.numpy()
disc.save_weights('loo_pretrained_disc.hdf5')
disc.load_weights('loo_pretrained_disc.hdf5')

# Get the ebola constraints back in
exec(open("./python/lib.py").read())

## Execute Adversarial training.
gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')
disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')
meanprob_true = []
meanprob_fake = []
epochs = 200000
#for epoch in tqdm(range(epochs)):
with tqdm(range(epochs)) as t:
    for epoch in t:
        #TODO: Batch Size

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
                plt.plot(yall[i,:], color = 'skyblue', alpha = 0.4)
            # Plot the predictions
            # Just the first setting for now
            some_noise = np.random.normal(size=[100,R])
            preds = gen.predict(np.hstack([xall[minr1:maxr1,:], some_noise]))
            for i in range(100):
                plt.plot(preds[i,:], color = 'orange', alpha = 0.4)

            plt.subplot(1,2,2)
            for i in range(minr2, maxr2):
                plt.plot(yall[i,:], color = 'skyblue', alpha = 0.4)
            # Plot the predictions
            # Just the first setting for now
            some_noise = np.random.normal(size=[100,R])
            preds = gen.predict(np.hstack([xall[minr2:maxr2,:], some_noise]))
            for i in range(100):
                plt.plot(preds[i,:], color = 'orange', alpha = 0.4)

            plt.savefig("./images/prog/epoch_%d.pdf"%epoch)
            #plt.savefig("temp.pdf")
            plt.close(fig)

        noise = np.random.normal(size=[N,R])

        # Do sequential updates
        with tf.GradientTape(persistent = True) as g:
            gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
            #prob_true = disc(tf.cast(tf.concat([x, y], 1), tf.float32))
            #prob_fake = disc(tf.cast(tf.concat([x, gendat], 1), tf.float32))
            prob_true = disc([tf.cast(x, tf.float32), tf.cast(y, tf.float32), tf.cast(ebola_constr(y), tf.float32)])
            prob_fake = disc([tf.cast(x, tf.float32), tf.cast(gendat, tf.float32), tf.cast(ebola_constr(gendat), tf.float32)])
            cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(label_smooth*np.ones(N).reshape([N,1]), tf.cast(prob_true, tf.float64)))
            cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
            cost_total = 0.5 * (cost_true + cost_fake)
            gen_cost = tf.reduce_mean(keras.losses.binary_crossentropy(label_smooth*np.ones(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
        graddisc = g.gradient(cost_total, disc.trainable_variables)
        gradgen = g.gradient(gen_cost, gen.trainable_variables)
        del g

        #Track some statistics
        meanprob_true.append(np.mean(prob_true))
        meanprob_fake.append(np.mean(prob_fake))

        # Update gen/disc
        dirsnvars_disc = [(graddisc[i], disc.trainable_variables[i]) for i in range(len(graddisc))]
        dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
        # Alternative updates
#        if epoch % 2 == 0:
        disc.optimizer.apply_gradients(dirsnvars_disc)
#        else:
        gen.optimizer.apply_gradients(dirsnvars_gen)

        t.set_postfix({"True Prob" : np.mean(prob_true), "Fake Prob" : np.mean(prob_fake)})

fig = plt.figure()
plt.plot(meanprob_true)
plt.plot(meanprob_fake)
plt.savefig('adversarial_advantage.pdf')
plt.close()

# Plot all of them
