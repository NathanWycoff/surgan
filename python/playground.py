#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.14.2019

# Some of this code was either adapted or directly copied from 
# https://raw.githubusercontent.com/eriklindernoren/Keras-GAN/

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

tf.set_random_seed(123)

# Params
noise_dim = 1
y_dim = 1
x_dim = 1
input_dist = lambda bs: np.random.normal(size=bs)
drop_prob = 0.5
H = 20

# Load and pre-process data
mcycle = np.genfromtxt('./data/mcycle.csv', delimiter=',', skip_header = 1)
N = mcycle.shape[0]
x = mcycle[:,0].reshape([N,x_dim])
y = mcycle[:,1].reshape([N,y_dim])
#x /= max(x)
#y = (y-min(y)) / (max(y) - min(y))
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

batch_size = N
epochs = 50000

# Set up neural nets
generator = keras.Sequential([
    keras.layers.Dense(H, activation=tf.nn.relu, input_shape = (x_dim+noise_dim,)),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(y_dim)
])

generator.compile(optimizer='adam',
              loss='mse')

discriminator = keras.Sequential([
    keras.layers.Dense(H, activation=tf.nn.relu, input_shape = (x_dim+y_dim,)),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu, input_shape = (x_dim+y_dim,)),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu, input_shape = (x_dim+y_dim,)),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(H, activation=tf.nn.relu, input_shape = (x_dim+y_dim,)),
    keras.layers.Dropout(drop_prob),
    keras.layers.Dense(1, activation = tf.nn.sigmoid)
])

discriminator.compile(optimizer='adam',
              loss='binary_crossentropy')

# See if the discriminator is sufficiently unsure about our training data
#discriminator.predict(np.hstack([y, x]))

# Link them
noise = keras.layers.Input(shape=(noise_dim,))
x_var = keras.layers.Input(shape=(x_dim,))

genin = keras.layers.concatenate([x_var, noise])
gendat = generator(genin)
discriminator.trainable = False
denin = keras.layers.concatenate([x_var, gendat])
valdty = discriminator([denin])

both = keras.models.Model([x_var, noise], valdty)
both.compile(loss=['binary_crossentropy'],
    optimizer='adam')

fig = plt.figure()
z = np.random.normal(size=[N,1])
gin = np.hstack([x, z])
plt.scatter(x, y)
plt.scatter(x, generator.predict(gin))
plt.savefig('before.pdf')

disc_ability = np.empty([epochs])
for epoch in tqdm(range(epochs), unit = ' Epochs'):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half batch of images
    idx = np.random.randint(0, N, batch_size)
    x_batch , y_batch= x[idx,:], y[idx,:]

    # Sample noise as generator input
    noise = np.random.normal(0, 1, (batch_size, noise_dim))

    # Generate a half batch of new images
    gendat = generator.predict(np.hstack([x_batch, noise]))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(np.hstack([x_batch, y_batch]), np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(np.hstack([x_batch, gendat]), np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    disc_ability[epoch] = d_loss

    # ---------------------
    #  Train Generator
    # ---------------------

    # Generate random inputs with respect to which to generate data
    #sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
    rand_xs = input_dist(batch_size)

    # Train the generator
    g_loss = both.train_on_batch([rand_xs, noise], np.ones((batch_size, 1)))

    # Print the progress
    #print ("%d [D loss: %.4f] [G loss: %.4f]" % (epoch, d_loss, g_loss))

    # If at save interval => save generated image samples
    #if epoch % sample_interval == 0:
    #    self.sample_images(epoch)

fig = plt.figure()
z = np.random.normal(size=[N,1])
gin = np.hstack([x, z])
plt.scatter(x, y)
plt.scatter(x, generator.predict(gin))
plt.savefig('after.pdf')

fig = plt.figure()
plt.plot(disc_ability)
plt.savefig('disc_ability.pdf')
