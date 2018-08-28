#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""DAFx2018 Audio Style Transfer with Rhythmic Constraints
Modules by Shaun Barry available at https://github.com/anonymousiclr2018/Style-Transfer-for-Musical-Audio
"""
import tensorflow as tf
import numpy as np
import scipy
from tensorflow.python.client import device_lib

def get_logmagnitude_STFT(x_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop):
    """ Projects input audio onto STFT bases. Returns STFT phase, mag and logmag
    Module by Shaun Barry available at 
        https://github.com/anonymousiclr2018/Style-Transfer-for-Musical-Audio
    """
    STFT_real = tf.nn.conv2d(x_,
                            dft_real_kernels_tf,
                            strides=[1, n_hop, 1, 1],
                            padding="SAME",
                            name="conv_dft_real")
    
    STFT_imag = tf.nn.conv2d(x_,
                        dft_imag_kernels_tf,
                        strides=[1, n_hop, 1, 1],
                        padding="SAME",
                        name="conv_dft_imag")
    
    STFT_phase = atan2(STFT_imag, STFT_real)
    STFT_magnitude = tf.sqrt(tf.square(STFT_imag)+tf.square(STFT_real))
    STFT_magnitude = tf.transpose(STFT_magnitude, (0,2,1,3))
    
    STFT_logmagnitude = tf.log1p(STFT_magnitude)
    return STFT_phase, STFT_magnitude, STFT_logmagnitude

def atan2(y, x, epsilon=1.0e-12):
    # Add a small number to all zeros, to avoid division by zero:
    x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y+epsilon, y)

    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
    return angle

def get_stft_kernels(n_dft):
    """ This is the tensorflow version of a function created by
    Keunwoo Choi shown here: https://github.com/keunwoochoi/kapre/blob/master/kapre/stft.py 
    
    Return dft kernels for real/imagnary parts assuming
        the input signal is real.
    An asymmetric hann window is used (scipy.signal.hann).
    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        Number of dft components.
    keras_ver : string, 'new' or 'old'
        It determines the reshaping strategy.
    Returns
    -------
    dft_real_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    dft_imag_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    * nb_filter = n_dft/2 + 1
    * n_win = n_dft
    """
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)

    nb_filter = n_dft / 2 + 1

    # prepare DFT filters
    timesteps = np.arange(n_dft)
    w_ks = (2*np.pi/float(n_dft)) * np.arange(n_dft)
    
    grid = np.dot(w_ks.reshape(n_dft, 1), timesteps.reshape(1, n_dft))
    dft_real_kernels = np.cos(grid)
    dft_imag_kernels = np.sin(grid)
    
    # windowing DFT filters
    dft_window = scipy.signal.hann(n_dft, sym=False)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    dft_real_kernels = dft_real_kernels[:nb_filter].transpose()
    dft_imag_kernels = dft_imag_kernels[:nb_filter].transpose()
    dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    dft_real_kernels = dft_real_kernels.astype(np.float32)
    dft_imag_kernels = dft_imag_kernels.astype(np.float32)
    
    return dft_real_kernels, dft_imag_kernels


def truncated_normal(shape, mean=0.0, stddev=1.0):
    x = np.random.normal(0.0, stddev, shape)
    inds = np.where(np.logical_or(x > 2.0*stddev, x < -2.0*stddev))
    for i in range(10):
        x[inds] = np.random.normal(0.0, stddev, x[inds].shape)
        val = x[inds].shape[0]/float(np.product(shape)) 
        inds = np.where(np.logical_or(x > 2.0*stddev, x < -2.0*stddev))
        if np.all(x < 2.0*stddev) and np.all(x > -2.0*stddev):
            return x + mean

def weight_fn(shape, mode="fan_in", scale = 1.0, distribution='normal'):
    
    receptive_field_size = np.prod(shape[:-2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
    
    if mode == 'fan_in':
        scale /= max(1., fan_in)
    elif mode == 'fan_out':
        scale /= max(1., fan_out)
    else:
        scale /= max(1., float(fan_in + fan_out) / 2)
        
    if distribution == 'normal':
        stddev = np.sqrt(scale)
        return truncated_normal(shape, 0., stddev).astype(np.float32)
    else:
        limit = np.sqrt(3. * scale)
        return np.random.uniform(-limit, limit, shape).astype(np.float32)

def elu(x, alpha=1.):
    """Exponential linear unit.
        Arguments
            x: A tenor or variable to compute the activation function for.
            alpha: A scalar, slope of positive section.
        Returns
            A tensor.
    """
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)

def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
        Arguments
            x: A tensor or variable to compute the activation function for.
        References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)

def get_devices(allow_soft_placement=True, allow_growth=True):
    """ Return gpu arguments and list of available compute resources 
        allow_soft_placement automatically chooses an existing and 
             supported device to run the operations in case the specified 
             one doesn't exist
        allow_growth attempts to allocate only as much GPU memory based on 
            runtime allocations
    """
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == 'GPU']

    if devices == []:
        devices=['/cpu:0']
    print 'Using device(s):', devices
    
    input_dev, seg_dev, odf_dev, output_dev = ['/'+str(devices[0][-5:-1])+str(i) for i in [0,1,1,0]]
    
    config = tf.ConfigProto(allow_soft_placement = allow_soft_placement)
    config.gpu_options.allow_growth = allow_growth
    return config, devices, input_dev, seg_dev, odf_dev, output_dev