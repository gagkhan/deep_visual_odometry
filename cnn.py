#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Project
# TensorFlow CNN

import tensorflow as tf
import numpy as np
import time

####################################
# TODO: Build your own LeNet model #
####################################
class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0, x_stride, y_stride,resp_norm=False):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, x_stride, y_stride, 1], padding="SAME")
            if resp_norm:
                resp_norm = tf.nn.local_response_normalization(conv_out + bias)
                cell_out = tf.nn.relu(resp_norm)
            else:
                cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

#### shouldnt need
class norm_layer(object):
    def __init__(self, input_x, is_training):
        """
        :param input_x: The input that needed for normalization.
        :param is_training: To control the training or inference phase
        """
        with tf.variable_scope('batch_norm'):
            batch_mean, batch_variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            ema = tf.train.ExponentialMovingAverage(decay=0.99)

            def True_fn():
                ema_op = ema.apply([batch_mean, batch_variance])
                with tf.control_dependencies([ema_op]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

            def False_fn():
                return ema.average(batch_mean), ema.average(batch_variance)

            mean, variance = tf.cond(is_training, True_fn, False_fn)

            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, keep_prob=1.0, activation_function=None, index=0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # added in dropout
            dropped = tf.nn.dropout(input_x,keep_prob)
            cell_out = tf.add(tf.matmul(dropped, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out