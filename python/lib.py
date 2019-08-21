#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 07.15.2019


def par2mat(weights, nnet):
    """
    Turn a weight vector into a list of weights suitable for assignment to a neural net. First arg is the vec, second the net.
    """
    weights_list = []
    used_weights = 0
    for l in disc.layers:
        nin = l.input_shape[1]
        nout = l.output_shape[1]
        weights_list.append(weights[(used_weights):(used_weights+nin*nout)].reshape((nin,nout)))
        used_weights += nin*nout
        weights_list.append(weights[(used_weights):(used_weights+nout)].flatten())
        used_weights += nout

    if used_weights < len(weights):
        print("Warning: weight input is longer than required for this neural network.")

    return(weights_list)


def rand_mon(lims, N):
    """
    Generate random monotonic functions
    """
    pass

def ebola_constr(traj):
    trajdiff = traj[:,1:]-traj[:,:-1]

#    trajdiff.set_shape(tf.TensorShape([None, traj.shape[1]-1]))
    traj_smooth = tf.sqrt(tf.reduce_mean(tf.square(trajdiff), axis = 1, keepdims = True))# Measure of smoothness
    traj_mon = tf.math.maximum(-tf.reduce_min(trajdiff, axis = 1, keepdims = True), 0)# Measure of nonmonotonicity (0 means perfectly monotone).
    return(tf.concat([traj_smooth, traj_mon], 1))

def interp_constr(x, y, gendat):
    #TODO: This assume that the x coords line up well.
    return(tf.zeros([N,N]))
    #return(tf.linalg.tensor_diag(tf.reshape(tf.square(y-gendat), [y.shape[0]])))


#TODO: --WARNING -- This function seems not to work if applied to several different neural networks.
# Seems to break when I apply it to a discriminator too.
class PermanentDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = tf.keras.backend.dropout(x, self.rate, noise_shape)
        return x
