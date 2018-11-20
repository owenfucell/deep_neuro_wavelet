#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:20:30 2018

@author: jannes
"""

import tensorflow as tf
import numpy as np

import sklearn.metrics # for confusion matrix in recall macro

DECAY = .999

def init_weights(shape, dist='random_normal', normalized=True):
    """Initializes network weights.
    
    Args:
        shape: A tensor. Shape of the weights.
        dist: A str. Distribution at initialization, one of 'random_normal' or 
            'truncated_normal'.
        normalized: A boolean. Whether weights should be normalized.
        
    Returns:
        A tf.variable.
    """
    # Normalized if normalized set to True
    if normalized == True:
        denom = np.prod(shape[:-1])
        std = 1 / denom
    else:
        std = .1
    
    # Draw from random or truncated normal
    if dist == 'random_normal':
        weights = tf.random_normal(shape, stddev=std)
    elif dist == 'truncated_normal':
        weights = tf.truncated_normal(shape, stddev=0.1)
    
    return tf.Variable(weights)


def init_biases(shape):
    """Initialize biases. """
    biases = tf.constant(0., shape=shape)
    return tf.Variable(biases)


def conv2d(x, W):
    """2D convolution. """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, pool_dim):
    """Max pooling. """
    patch_height = pool_dim[0]
    patch_width = pool_dim[1]
    return tf.nn.max_pool(x, 
                          ksize=[1, patch_height, patch_width, 1], 
                          strides=[1, patch_height, patch_width, 1], 
                          padding='SAME')





def fully_connected(x_in, bn, units, training, nonlin='leaky_relu', 
                    weights_dist='random_normal', normalized_weights=True):
    """Adds fully connected layer.
    
    Args:
        x_in: A tensor. Input layer.
        bn: A boolean. Indicating whether batch-norm. should be applied.
        units: An int. Number of output units.
        training: A boolean. Indicates training (True) or test (False).
        nonlin: A str. One of 'leaky_relu' or 'elu'; non-linearity to use.
        
    Returns:
        out: Fully-connected output layer.
        weights: Weights for the output layer.
    """
    # Fully-connected layer (BN)
    shape_in = x_in.get_shape().as_list()
    dim = shape_in[1] * shape_in[2] * shape_in[3]
    weights = init_weights([dim, units],
                          dist=weights_dist,
                          normalized=normalized_weights)
    flat = tf.reshape(x_in, [-1, dim])
    h_conv = tf.matmul(flat, weights)
    
    # Batch-normalize fully-connected layer
    if bn == True:
        
        # Manual batch-normalization
# =============================================================================
#         batch_mean, batch_var = tf.nn.moments(h_conv, [0])    
#         h_conv_hat = (h_conv - batch_mean) / tf.sqrt(batch_var + 1e-3)
#         scale = tf.Variable(tf.ones([units]))
#         beta = tf.Variable(tf.zeros([units]))
#         if nonlin == 'leaky_relu':
#             layer_bn = tf.nn.leaky_relu(h_conv_hat)
#         elif nonlin == 'elu':
#             layer_bn = tf.nn.elu(h_conv_hat)
#         else:
#             raise ValueError('Non-linearity "' + nonlin + '" not supported.')
#         out = scale * layer_bn + beta
# =============================================================================
        
        if nonlin == 'leaky_relu':
            layer_bn = tf.nn.leaky_relu(h_conv)
            out = tf.contrib.layers.batch_norm(
                    layer_bn,
                    data_format='NHWC',
                    center=True,
                    scale=True,
                    is_training=training,
                    decay=DECAY)
        elif nonlin == 'elu':
            layer_bn = tf.nn.elu(h_conv)
            out = tf.contrib.layers.batch_norm(
                    layer_bn,
                    data_format='NHWC',
                    center=True,
                    scale=True,
                    is_training=training,
                    decay=DECAY,
                    renorm=True)
        else:
            raise ValueError('Non-linearity "' + nonlin + '" not supported.')
    else:
        if nonlin == 'leaky_relu':
            out = tf.nn.leaky_relu(h_conv)
        elif nonlin == 'elu':
            out = tf.nn.elu(h_conv)
        else:
            raise ValueError('Non-linearity \'' + nonlin + '\' not supported.')
    
    # Return fully-connected, batch-normalized output layer + weights
    return out, weights


def l2_loss(weights_cnn, l2_regularization_penalty, y_, y_conv, name):
    """Implements L2 loss for an arbitrary number of weights.
    
    Args:
        weights: A dict. One key/value pair per layer in the network.
        l2_regularization_penalty: An int. Scales the l2 loss arbitrarily.
        y_:
        y_conv:
        name: 
            
    Returns:
        L2 loss.        
    """
    weights = {}
    for key, value in weights_cnn.items():
        weights[key] = tf.nn.l2_loss(value)
    
    l2_loss = l2_regularization_penalty * sum(weights.values())
    
    unregularized_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    return tf.add(unregularized_loss, l2_loss, name=name)

def recall_macro(y_true, y_pred):
    ''' 
    Returns the recall macro (average of accuracy of each class) and its error bar.
    
    Args :
        y_true : Ndarray. Ground truth (correct) labels.
        y_pred : Predicted labels, as returned by a classifier.
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    classes = confusion_matrix.shape[0]
    n_test = np.sum(confusion_matrix, axis = 1)
    
        
    #fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    #tn = confusion_matrix.sum() - (fp + fn + tp)
    
    #
    recall_macro_per_class = tp/(tp+fn)
    #print(recall_macro_per_class)
    #
    recall_macro = np.mean(recall_macro_per_class)
    
    # l'écart-type théorique du recall macro, soit un interval de confiance de 0.68 = erf( 1/np.sqrt(2)) --> 1 fois l'écart-type
    error_bar = np.sqrt( np.sum(recall_macro_per_class * (1 - recall_macro_per_class)/n_test) ) /classes 
    
    return(recall_macro, error_bar)


def mean_accuracy_per_class(y_true, y_pred):
    ''' 
    Returns the average of accuracy of each class and its error bar.
    It is call AUC for binary classfieur and recall macro when there is more than 2 class (multi-class).
    
    Args :
        y_true : Ndarray. Ground truth (correct) labels.
        y_pred : Predicted labels, as returned by a classifier.
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    classes = confusion_matrix.shape[0]
    n_test = np.sum(confusion_matrix, axis = 1)
    
        
    #fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    #tn = confusion_matrix.sum() - (fp + fn + tp)
    
    #
    recall_macro_per_class = tp/(tp+fn)
    #print(recall_macro_per_class)
    #
    recall_macro = np.mean(recall_macro_per_class)
    
    # l'écart-type théorique du recall macro, soit un interval de confiance de 0.68 = erf( 1/np.sqrt(2)) --> 1 fois l'écart-type
    error_bar = np.sqrt( np.sum(recall_macro_per_class * (1 - recall_macro_per_class)/n_test) ) /classes 
    
    return(recall_macro, error_bar)




