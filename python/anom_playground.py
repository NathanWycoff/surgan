#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/ebola_constr_again.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.28.2019

# Try to impose smoothness and monotonicity

import keras
import numpy as np
from keras.datasets import mnist
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(123)
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import line_search
tf.enable_eager_execution()
tf.set_random_seed(123)
exec(open("./python/lib.py").read())

# Unsupervised for now
train, test = mnist.load_data()
y = train[0].reshape([60000,28*28])
x = np.zeros([60000, 0])
labels = train[1]
#test_x = test[0].reshape([10000,28*28])

N = y.shape[0]
Q = y.shape[1]
P = 0

# Y - an N by Q array giving all observed runs
# Standardize Y data
shift = np.mean(y)
scale = np.std(y)
y = (y - shift) / scale

## GAN Params
#H = 1000 # Number of hidden units
R = Q # Dim of latent error variable
C = 0#Number of constraints
H = 100# Number of hidden units
epochs = 20000
save_every = 500
label_smooth = 0.9
mb_size = N
pass_length = int(np.ceil(N / mb_size))
constr_func = empty_constr

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

#gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
#disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
#
## Pretraining the generator via l2 loss with no onise
#l2_losses = np.empty([epochs])
#for epoch in tqdm(range(epochs)):
#    #TODO: Batch Size
#    noise = np.zeros(shape=[N,R])
#
#    # Get gradient
#    with tf.GradientTape(persistent = True) as g:
#        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
#        l2_loss = tf.sqrt(tf.reduce_mean(tf.square(gendat - y)))
#    gradgen = g.gradient(l2_loss, gen.trainable_variables)
#    del g
#
#    l2_losses[epoch] = l2_loss
#
#    # Update gen/disc
#    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
#    gen.optimizer.apply_gradients(dirsnvars_gen)
#
#
#fig = plt.figure()
#plt.plot(np.log(l2_losses))
#plt.savefig("l2_loss.pdf")
#
##########
## Plot the results
#otp1 = 1# Which observation to plot?
#minr1 = otp1*100
#maxr1 = (otp1+1)*100
#otp2 = 2# Which observation to plot?
#minr2 = otp2*100
#maxr2 = (otp2+1)*100
#
## Plot the data
## Just the first setting for now
#fig = plt.figure()
#plt.subplot(1,2,1)
## Plot the predictions
## Just the first setting for now
#some_noise = np.random.normal(size=[100,R])
#preds = gen.predict(np.hstack([x[minr1:maxr1,:], some_noise]))
#for i in range(100):
#    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
#for i in range(minr1, maxr1):
#    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
#
#plt.subplot(1,2,2)
#some_noise = np.random.normal(size=[100,R])
#preds = gen.predict(np.hstack([x[minr2:maxr2,:], some_noise]))
#for i in range(100):
#    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
#for i in range(minr2, maxr2):
#    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
## Plot the predictions
## Just the first setting for now
#
#
#plt.savefig("pretraining.pdf")
#plt.close(fig)
#
#
### Pretraining via Quantile Alignment
#epochs = 5000
#
##gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
##disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.1), 'binary_crossentropy')
#gen.compile(tf.train.AdamOptimizer(), 'binary_crossentropy')
#disc.compile(tf.train.AdamOptimizer(), 'binary_crossentropy')
#
##def quantile_loss(truth, guess, quant):
##    """
##    This function should be given a block of 100 truths and guesses (with the same x).
##
##    It looks at the squared sums of differences between quantiles at each of 56 points.
##
##    It will not work on more general problems: we should do something cute involving tilted losses.
##    """
##    true_quants = tf.map_fn(lambda x: tfp.stats.percentile(x, quant), tf.transpose(truth))
##    guess_quants = tf.map_fn(lambda x: tfp.stats.percentile(x, quant), tf.transpose(guess))
##    return(tf.reduce_mean(tf.square(tf.cast(true_quants, tf.float32) - guess_quants)))
#
#def quantile_loss(truth, draws, quant):
#    """
#    Truth is a matrix.
#    draws is an array, the first index gives draws for the same datapoint, the two others give the data dimensions.
#    quant is the vector of quantiles
#    """
#    U = len(quant)
#    guess_quant = tfp.stats.percentile(draws, q = 100.0*quant, axis = 0)
#    resids = truth - guess_quant
#    tilt_loss = 0
#    for u in range(U):
#        tilt_loss += (quant[u] * tf.cast(resids[u,:,:] > 0, tf.float32) * resids[u,:,:] + \
#                (quant[u]-1) * tf.cast(resids[u,:,:] < 0, tf.float32) * resids[u,:,:]) / float(U)
#    tot_loss = tf.reduce_mean(tilt_loss)
#    return(tot_loss)
#
#K = 20#Number of draws for each point
#U = 10#Number of percentiles to sample
#qls = np.empty([epochs])
### Also tilted quantile loss
#for epoch in tqdm(range(epochs)):
#
#    #TODO: Batch Size
#    noise = tf.cast(np.random.normal(size=[K,N,R]), tf.float32)
#    #quant = np.random.uniform(size=U)
#    quant = np.linspace(0.1, 0.9, num = U)
#
#    # Get gradient
#    with tf.GradientTape(persistent = True) as g:
#        draws = tf.map_fn(lambda r: gen(tf.cast(tf.concat([x, r], 1), tf.float32)), noise)
#        tilt_loss = quantile_loss(y, draws, quant)
#    gradgen = g.gradient(tilt_loss, gen.trainable_variables)
#    del g
#
#    qls[epoch] = tilt_loss
#
#    # Update gen/disc
#    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
#    gen.optimizer.apply_gradients(dirsnvars_gen)
#
#fig = plt.figure()
#plt.plot(np.log10(qls[:epoch]))
#plt.savefig("quantile_loss.pdf")

#### Pretraining the generator via elementwise quantile alignment
##for epoch in tqdm(range(epochs)):
##    #TODO: Batch Size
##
##    # Get gradient
##    with tf.GradientTape(persistent = True) as g:
##        #NOTE: 100's are specific to our dataset 
##        noise = np.random.normal(size=[N,R])
##        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
##        p10_loss = 0
##        p90_loss = 0
##
##        #for m in np.random.choice(100,2):
##        for m in range(100):
##            p10_loss += quantile_loss(y[(m*100):((m+1)*100)], gendat[(m*100):((m+1)*100),:], 0.1) / 100.0
##            p90_loss += quantile_loss(y[(m*100):((m+1)*100)], gendat[(m*100):((m+1)*100),:], 0.9) / 100.0
##
##        quant_loss = p10_loss + p90_loss
##    gradgen = g.gradient(quant_loss, gen.trainable_variables)
##    del g
##
##    # Update gen/disc
##    dirsnvars_gen = [(gradgen[i], gen.trainable_variables[i]) for i in range(len(gradgen))]
##    gen.optimizer.apply_gradients(dirsnvars_gen)
#
##########
## Plot the results
#otp1 = 1# Which observation to plot?
#minr1 = otp1*100
#maxr1 = (otp1+1)*100
#otp2 = 2# Which observation to plot?
#minr2 = otp2*100
#maxr2 = (otp2+1)*100
#
## Plot the data
## Just the first setting for now
#fig = plt.figure()
#plt.subplot(1,2,1)
## Plot the predictions
## Just the first setting for now
##some_noise = np.zeros(shape=[100,R])
#some_noise = np.random.normal(size=[100,R])
#preds = gen.predict(np.hstack([x[minr1:maxr1,:], some_noise]))
#for i in range(100):
#    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
#for i in range(minr1, maxr1):
#    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
#
#plt.subplot(1,2,2)
#some_noise = np.random.normal(size=[100,R])
#preds = gen.predict(np.hstack([x[minr2:maxr2,:], some_noise]))
#for i in range(100):
#    plt.plot(preds[i,:], color = 'orange', alpha = 0.4)
#for i in range(minr2, maxr2):
#    plt.plot(y[i,:], color = 'skyblue', alpha = 0.4)
## Plot the predictions
## Just the first setting for now
#
#plt.savefig("pretraining2.pdf")
#plt.close(fig)
#
#gen.save_weights('pretrained_gen.hdf5')
#gen.load_weights('pretrained_gen.hdf5')

#def constr_func(y):
#    return(tf.zeros([N,C]))

#gen.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')
#disc.compile(tf.train.GradientDescentOptimizer(learning_rate = 0.01), 'binary_crossentropy')
#
### Pretrain the discriminator a little bit as well on the trained generator.
#epochs = 100000
#for epoch in tqdm(range(epochs)):
#    #TODO: Batch Size
#
#    noise = np.random.normal(size=[N,R])
#
#    # Get gradient
#    with tf.GradientTape(persistent = True) as g:
#        gendat = gen(tf.cast(tf.concat([x, noise], 1), tf.float32))
#        #prob_true = disc(tf.cast(tf.concat([x, y], 1), tf.float32))
#        #prob_fake = disc(tf.cast(tf.concat([x, gendat], 1), tf.float32))
#        prob_true = disc([tf.cast(x, tf.float32), tf.cast(y, tf.float32), tf.cast(constr_func(y), tf.float32)])
#        prob_fake = disc([tf.cast(x, tf.float32), tf.cast(gendat, tf.float32), tf.cast(constr_func(gendat), tf.float32)])
#        cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(label_smooth*np.ones(N).reshape([N,1]), tf.cast(prob_true, tf.float64)))
#        cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(N).reshape([N,1]), tf.cast(prob_fake, tf.float64)))
#        cost_total = 0.5 * (cost_true + cost_fake)
#    graddisc = g.gradient(cost_total, disc.trainable_variables)
#    del g
#
#    # Update gen/disc
#    dirsnvars_disc = [(graddisc[i], disc.trainable_variables[i]) for i in range(len(graddisc))]
#    disc.optimizer.apply_gradients(dirsnvars_disc)
#
#pretrain_disc_loss = cost_total
#disc_pre_fake = prob_fake.numpy()
#disc.save_weights('pretrained_disc.hdf5')
#disc.load_weights('pretrained_disc.hdf5')

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

        for ei in range(pass_length):

            # Get a draw from the current boi
            if epoch % save_every == 0:
                n = 1
                noise = np.random.normal(size=[n,R])
                gendat = gen(tf.cast(tf.concat([x[:n,:], noise], 1), tf.float32))
                # Put back into pixel space
                gendat = gendat * scale + shift
                # Get rid of anything outside of [0,255]
                gendat = np.clip(gendat, 0, 255)
                fig = plt.figure()
                plt.imshow(gendat.reshape([28,28]), interpolation = 'nearest')
                plt.savefig("./images/anom/epoch_%d.pdf"%epoch)
                #plt.savefig("temp.pdf")
                plt.close(fig)

            # Subset Data 
            xi = x[(ei*mb_size):((ei+1)*mb_size)]
            yi = y[(ei*mb_size):((ei+1)*mb_size)]
            Ni = yi.shape[0]
            noise = np.random.normal(size=[Ni,R])

            # Do sequential updates
            with tf.GradientTape(persistent = True) as g:
                gendat = gen(tf.cast(tf.concat([xi, noise], 1), tf.float32))
                #prob_true = disc(tf.cast(tf.concat([xi, yi], 1), tf.float32))
                #prob_fake = disc(tf.cast(tf.concat([xi, gendat], 1), tf.float32))
                prob_true = disc([tf.cast(xi, tf.float32), tf.cast(yi, tf.float32), tf.cast(constr_func(xi, yi, yi), tf.float32)])
                prob_fake = disc([tf.cast(xi, tf.float32), tf.cast(gendat, tf.float32), tf.cast(constr_func(xi, yi, gendat), tf.float32)])
                cost_true = tf.reduce_mean(keras.losses.binary_crossentropy(label_smooth*np.ones(Ni).reshape([Ni,1]), tf.cast(prob_true, tf.float64)))
                cost_fake = tf.reduce_mean(keras.losses.binary_crossentropy(np.zeros(Ni).reshape([Ni,1]), tf.cast(prob_fake, tf.float64)))
                cost_total = 0.5 * (cost_true + cost_fake)
                gen_cost = tf.reduce_mean(keras.losses.binary_crossentropy(label_smooth*np.ones(Ni).reshape([Ni,1]), tf.cast(prob_fake, tf.float64)))
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
