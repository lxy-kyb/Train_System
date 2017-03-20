import numpy as np
import tensorflow as tf

def add_layer(inputs, in_size, out_size, layer_name, keep_prob, activation_function = None):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram('weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([out_size]) + 0.1, name='b')
            tf.summary.histogram('biases', biases)
        with tf.name_scope('xW_plus_b'):
            xW_plus_b = tf.nn.xw_plus_b(inputs, Weights, biases, name='xW_plus_b')
            tf.summary.histogram('xW_plus_b', xW_plus_b)
            xW_plus_b = tf.nn.dropout(xW_plus_b, tf.Variable(keep_prob, 'keep_prob'))
        if activation_function is None:
            outputs = xW_plus_b
        else:
            outputs = activation_function(xW_plus_b)
        tf.summary.histogram('outputs', outputs)
        return outputs


        