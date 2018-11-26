#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rudy 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pywt

import lib.matnpy.matnpyio as io
import lib.matnpy.matnpy as matnpy
import lib.cnn.helpers as hlp
import lib.cnn.cnn as cnn
#from lib.cnn_1.confusion_matrix import plot_confusion_matrix

#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import os.path
import sys
import datetime



def wavelet_decomposition(data, depth_wav , w_name = 'db4', mode ='smooth'):
    w = pywt.Wavelet(w_name)
    a = data.copy()
    ca = []
    cd = []
    for i in range(depth_wav):
        (a, d) = pywt.dwt(a, w, mode)
        a = a[:,:,1:-2]
        d = d[:,:,1:-2]
        ca.append(a)
        cd.append(d)
    return(ca, cd)

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
        std = 1e-3 / denom
    else:
        std = 1e-4
    
    # Draw from random or truncated normal
    if dist == 'random_normal':
        weights = tf.random_normal(shape, stddev=std)
    elif dist == 'truncated_normal':
        weights = tf.truncated_normal(shape, stddev=0.1)
    
    return tf.Variable(weights)

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

def l2_loss(weights, l2_regularization_penalty, y_, y_conv, name):
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
    w = {}
    for key, value in weights.items():
        w[key] = tf.nn.l2_loss(value)
    
    l2_loss = l2_regularization_penalty * sum(w.values())
    
    unregularized_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    return tf.add(unregularized_loss, l2_loss, name=name)



#########
# PATHS #
#########

param_index = int(sys.argv[1])
base_path = '/home/rudy.noyelle/wavelet/'
file = base_path + 'scripts/_params/training.txt'
# Get current params from file
with open(file, 'rb') as f:
    params = np.loadtxt(f, dtype='object', delimiter='\n')
params = params.tolist()
curr_params = eval(params[param_index-1])


##########
# PARAMS #
##########
sess = '01' # '01' or '02'
sess_no = curr_params[0]
decode_for = curr_params[1]
target_area = curr_params[2]
align_on ,from_time, to_time =curr_params[3], curr_params[4], curr_params[5]
cortex_name = curr_params[6]
curr_str_to_print = curr_params[7] #test numéro X

mode = 'smooth'
elec_type = 'grid'  # any one of single|grid|average

#only_correct_trials = False  
if decode_for == 'stim':
    only_correct_trials = True
else:
    only_correct_trials = False


#data path
raw_path = (base_path + 'data/raw/'+ sess_no +'/session01/')
#raw_path = ('/home/rudy.noyelle/wavelet/'+ 'data/raw/'+ sess_no +'/session01/')
rinfo_path = raw_path + 'recording_info.mat'
tinfo_path = raw_path + 'trial_info.mat'



#CNN PARAMS

n_layers = 6
depth_wav = 7
in_1, out_1 = 1, 6
in_2, out_2 = 7, 21
in_3, out_3 = 22, 41
in_4, out_4 = 42, 69
in_5, out_5 = 70,97 
in_6, out_6 = 98, 122
fc_units = 100


pool_dim = [1, 2] #size of max_pool
patch_dim = [1, 7] # taille de la fenêtre de convolution
patch_dim5 = [1, 3]
patch_dim6 = [1, 2]
nonlin = 'elu' # leaky_relu'

n_iterations = 2000

learning_rate = 1e-4
size_of_batches = 15

keep_prob_train = .2
l2_regularization_penalty = 0#0.001
amplify_input = True # q**..
q = 4

dist = 'random_normal'
normalized_weights = True

bn = False # Indicating whether batch-norm shoud be applied
batch_norm = 'renorm'  # 'after'
DECAY = .999



########
# DATA #
########

# train/test size, random split 
# train_size = .8
# test_size = .2
seed = np.random.randint(1,10000)
n_splits = 5


# Auto-define number of classes
classes = 2 if decode_for == 'resp' else 5

# Load data and targets
data = matnpy.get_subset_by_areas(sess_no, raw_path, 
                         align_on, from_time, to_time, 
                         target_area,
                         only_correct_trials = only_correct_trials, renorm = False, elec_type = elec_type )
n_chans = data.shape[1]

targets = io.get_targets(decode_for, raw_path,n_chans, elec_type=elec_type,
                        only_correct_trials=only_correct_trials,
                        onehot=True)

# indices = np.arange(len(data))
# train, test, train_labels, test_labels, idx_train, idx_test = (
#         train_test_split(
#             data, 
#             targets, 
#             indices,
#             test_size=test_size, 
#             random_state=seed)
#         )

# wavelet decomposition        
depth_wav = 7
ca, cd = wavelet_decomposition(data, depth_wav=depth_wav) # ca size = depth_wav * n_trials * n_chans * 2*8-i

## RESHAPE DATA

depth_wav = 7
ca_train = 7 *[0]
cd_train = 7 *[0]
for i in range(depth_wav):
    ca_train[i] = np.reshape(np.array(ca[i]),(len(ca[i]), len(ca[i][0]), len(ca[i][0][0]), 1))
    cd_train[i] = np.reshape(np.array(cd[i]),(len(cd[i]), len(cd[i][0]), len(cd[i][0][0]), 1))


ca = ca_train
cd = cd_train




## AMPLIFY INPUT    
if amplify_input == True :
    for i in range(len(ca)):
        cd[i] = 10000 * (q**(len(ca)-i-1)) *cd[i]**2

# slit train/ test
ca_train = [ca[i][idx_train, :, : , :] for i in range(len(ca))]
cd_train = [cd[i][idx_train, :, : , :] for i in range(len(cd))]

ca_test = [ca[i][idx_test, :, : , :] for i in range(len(ca))]
cd_test = [cd[i][idx_test, :, : , :] for i in range(len(cd))]


############## 
# CREATE CNN #
##############

if elec_type == 'single':
    n_chans = 1
x_1 = tf.placeholder(tf.float32, shape=[None, n_chans , 128, 1])
x_2 = tf.placeholder(tf.float32, shape=[None, n_chans , 64, 1])
x_3 = tf.placeholder(tf.float32, shape=[None, n_chans , 32, 1])
x_4 = tf.placeholder(tf.float32, shape=[None, n_chans , 16, 1])
x_5 = tf.placeholder(tf.float32, shape=[None, n_chans , 8, 1])
x_6 = tf.placeholder(tf.float32, shape=[None, n_chans , 4, 1])



keep_prob = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32, shape=[None, classes])

training = tf.placeholder_with_default(True, shape=())

weights = {}

# first_layer
weights_1 = init_weights([patch_dim[0], patch_dim[1], in_1, out_1], 
                    dist=dist,
                    normalized=normalized_weights)
conv_1 = conv2d(x_1, weights_1)
conv_1_elu = tf.nn.elu(conv_1)
if bn == True:
    conv_1_bn_elu = tf.contrib.layers.batch_norm(
        conv_1_elu,
        data_format='NHWC',
        center=True,
        scale=True,
        is_training=training,
        decay=DECAY,
        renorm=True)
    maxpool_1_bn_elu = max_pool(conv_1_bn_elu, pool_dim)
else:
    maxpool_1_bn_elu = max_pool(conv_1_elu, pool_dim)

weights[0] = weights_1    

# second layer
concat_2 = tf.concat((maxpool_1_bn_elu, x_2), axis = 3)
weights_2 = init_weights([patch_dim[0], patch_dim[1], in_2, out_2], 
                    dist=dist,
                    normalized=normalized_weights)
conv_2 = conv2d(concat_2, weights_2)
conv_2_elu = tf.nn.elu(conv_2)
if bn == True:
    conv_2_bn_elu = tf.contrib.layers.batch_norm(
        conv_2_elu,
        data_format='NHWC',
        center=True,
        scale=True,
        is_training=training,
        decay=DECAY,
        renorm=True)
    maxpool_2_bn_elu = max_pool(conv_2_bn_elu, pool_dim)
else:
    maxpool_2_bn_elu = max_pool(conv_2_elu, pool_dim)

weights[1] = weights_2  

# 3 layer
concat_3 = tf.concat((maxpool_2_bn_elu, x_3), axis = 3)
weights_3 = init_weights([patch_dim[0], patch_dim[1], in_3, out_3], 
                    dist=dist,
                    normalized=normalized_weights)
conv_3 = conv2d(concat_3, weights_3)
conv_3_elu = tf.nn.elu(conv_3)
if bn == True:
    conv_3_bn_elu = tf.contrib.layers.batch_norm(
        conv_3_elu,
        data_format='NHWC',
        center=True,
        scale=True,
        is_training=training,
        decay=DECAY,
        renorm=True)
    maxpool_3_bn_elu = max_pool(conv_3_bn_elu, pool_dim)
else:
    maxpool_3_bn_elu = max_pool(conv_3_elu, pool_dim)

weights[2] = weights_3  

# 4 layer
concat_4 = tf.concat((maxpool_3_bn_elu, x_4), axis = 3)
weights_4 = init_weights([patch_dim[0], patch_dim[1], in_4, out_4], 
                    dist=dist,
                    normalized=normalized_weights)
conv_4 = conv2d(concat_4, weights_4)
conv_4_elu = tf.nn.elu(conv_4)
if bn == True:
    conv_4_bn_elu = tf.contrib.layers.batch_norm(
        conv_4_elu,
        data_format='NHWC',
        center=True,
        scale=True,
        is_training=training,
        decay=DECAY,
        renorm=True)
    maxpool_4_bn_elu = max_pool(conv_4_bn_elu, pool_dim)
else:
    maxpool_4_bn_elu = max_pool(conv_4_elu, pool_dim)

weights[3] = weights_4  

# 5 layer
concat_5 = tf.concat((maxpool_4_bn_elu, x_5), axis = 3)
weights_5 = init_weights([patch_dim5[0], patch_dim5[1], in_5, out_5], 
                    dist=dist,
                    normalized=normalized_weights)
conv_5 = conv2d(concat_5, weights_5)
conv_5_elu = tf.nn.elu(conv_5)
if bn == True:
    conv_5_bn_elu = tf.contrib.layers.batch_norm(
        conv_5_elu,
        data_format='NHWC',
        center=True,
        scale=True,
        is_training=training,
        decay=DECAY,
        renorm=True)
    maxpool_5_bn_elu = max_pool(conv_5_bn_elu, pool_dim)
else:
    maxpool_5_bn_elu = max_pool(conv_5_elu, pool_dim)

weights[4] = weights_5 

# _6 layer
concat_6 = tf.concat((maxpool_5_bn_elu, x_6), axis = 3)
weights_6 = init_weights([patch_dim6[0], patch_dim6[1], in_6, out_6], 
                    dist=dist,
                    normalized=normalized_weights)
conv_6 = conv2d(concat_6, weights_6)
conv_6_elu = tf.nn.elu(conv_6)
if bn == True:
    conv_6_bn_elu = tf.contrib.layers.batch_norm(
        conv_6_elu,
        data_format='NHWC',
        center=True,
        scale=True,
        is_training=training,
        decay=DECAY,
        renorm=True)
    maxpool_6_bn_elu = max_pool(conv_6_bn_elu, pool_dim)
else:
    maxpool_6_bn_elu = max_pool(conv_6_elu, pool_dim)

weights[5] = weights_6 
# FC

fc1, weights[n_layers] = cnn.fully_connected(maxpool_6_bn_elu,
                                        bn=True, 
                                        units=fc_units,
                                        training=training,
                                        nonlin=nonlin,
                                        weights_dist=dist,
                                        normalized_weights=normalized_weights)

# Dropout (BN)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# Readout
weights[n_layers+1] = init_weights([fc_units, classes])
y_conv = tf.matmul(fc1_drop, weights[n_layers+1])
softmax_probs = tf.contrib.layers.softmax(y_conv)
weights_shape = [tf.shape(el) for el in weights.values()]

# LOSS
loss = l2_loss(weights, 
            l2_regularization_penalty, 
            y_, 
            y_conv, 
            'loss')


# Optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction = tf.argmax(y_conv, 1)


#ind_test = hlp.subset_test(test_labels, classes)

kf = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=seed)

acc_training_list = []
#acc_balanced_list = []

n_test_list = []
confusion_matrix_list = []
y_true_list = []
y_pred_list = []
#idx_test_list = []

cross_validation_i = 0
#for train_index, test_index in kf.split(data):  KFOLD
for train_index, test_index in kf.split(data, np.argmax(targets[:,:], axis=1)):
    cross_validation_i += 1
    #print('####################################')
    #print('         NUMERO   ', cross_validation_i, '/5')
    #print('####################################')

    ca_train = [ca[i][train_index, :, : , :] for i in range(len(ca))]
    cd_train = [cd[i][train_index, :, : , :] for i in range(len(cd))]
    train_labels = targets[train_index]

    ca_test = [ca[i][test_index, :, : , :] for i in range(len(ca))]
    cd_test = [cd[i][test_index, :, : , :] for i in range(len(cd))]
    test_labels = targets[test_index]




    #ind_test = hlp.subset_test(test_labels, classes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Number of batches to train on
        for i in range(n_iterations):
            ind_train = hlp.subset_train(train_labels, classes, size_of_batches)

            ## Every n iterations, print training accuracy
            if i % 50 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x_1: cd_train[1][ind_train,:,:],
                    x_2: cd_train[2][ind_train,:,:],
                    x_3: cd_train[3][ind_train,:,:],
                    x_4: cd_train[4][ind_train,:,:],
                    x_5: cd_train[5][ind_train,:,:],
                    x_6: cd_train[6][ind_train,:,:],
                    y_: train_labels[ind_train,:],
                    keep_prob: 1.0
                    })
                print('step %d, training accuracy: %g' % (
                        i, train_accuracy))

            # Training

            train_step.run(feed_dict={
                    x_1: cd_train[1][ind_train,:,:],
                    x_2: cd_train[2][ind_train,:,:],
                    x_3: cd_train[3][ind_train,:,:],
                    x_4: cd_train[4][ind_train,:,:],
                    x_5: cd_train[5][ind_train,:,:],
                    x_6: cd_train[6][ind_train,:,:],
                    y_: train_labels[ind_train,:],
                    keep_prob: keep_prob_train
                    })

        ### TRAINING ACCURACY last step

        y_pred = prediction.eval(feed_dict={
                        x_1: cd_train[1][:,:,:],
                        x_2: cd_train[2][:,:,:],
                        x_3: cd_train[3][:,:,:],
                        x_4: cd_train[4][:,:,:],
                        x_5: cd_train[5][:,:,:],
                        x_6: cd_train[6][:,:,:],
                        y_: train_labels[:,:],
                        keep_prob: 1.0
                        })
        # get mean of accuracy per class
        acc_train, error_bar = mean_accuracy_per_class(np.argmax(train_labels[:,:],1), y_pred)
        # append to list in order to save it
        mean_per_class_accuracy_train_per_fold.append(acc_train)

        ### TEST ACCURACY ###

        # Print test accuracy on balanced test base
        y_pred =  prediction.eval(feed_dict={
            x_1: cd_test[1][:,:,:],
            x_2: cd_test[2][:,:,:],
            x_3: cd_test[3][:,:,:],
            x_4: cd_test[4][:,:,:],
            x_5: cd_test[5][:,:,:],
            x_6: cd_test[6][:,:,:],
            y_: test_labels[:,:],
            keep_prob: 1.0
            })
        
        acc, error_bar = cnn.mean_accuracy_per_class(np.argmax(test_labels[:,:],1), y_pred)
        print('mean acc : %g +-' (acc, error_bar))         
        

        table_confusion = confusion_matrix(
            np.argmax(test_labels[:,:],1),
            y_pred)

        # append
        y_pred_list.append( list(y_pred) )
        y_true_list.append( list( np.argmax(test_labels[:,:],1)) )
        confusion_matrix_list.append(list(table_confusion)) 
        



cnf = np.array(confusion_matrix_list)
mean_accuracy_per_class_per_fold = []
error_bar_per_fold = []
for k in range(cnf.shape[0]):
    # Calculate average of accuracy of class on the k fold 
    n_test = np.sum(cnf[k], axis =1)
    tp = np.diag(cnf[k])
    fn = cnf[k].sum(axis=1) - np.diag(cnf[k])
    
    accuracy_per_class = tp/(tp+fn)
    
    mean_accuracy_per_class = np.mean( accuracy_per_class )
    
    error_bar = np.sqrt( np.sum( accuracy_per_class * (1 - accuracy_per_class)/n_test) ) /classes
    
    mean_accuracy_per_class_per_fold.append(mean_accuracy_per_class)
    error_bar_per_fold.append(error_bar)
    
# result = mean on folds 
mean_accuracy_per_class = np.mean(mean_accuracy_per_class_per_fold)
error_bar = np.sqrt(np.sum( np.array(error_bar_per_fold) **2 ))/len(error_bar_per_fold)

    

#acc_mean =  np.mean(acc_balanced_list)

channels_in = [in_1, in_2, in_3, in_4, in_5, in_6]
channels_out = [out_1, out_2, out_3, out_4, out_5, out_6]

time = str(datetime.datetime.now())
result = [str(sess_no), decode_for, only_correct_trials,
          str(target_area), cortex_name, elec_type,
          interval,
          mean_accuracy_per_class, error_bar, list(np.sum(targets, axis=0)), #### !!!!!
          mean_per_class_accuracy_train_per_fold, mean_accuracy_per_class_per_fold, # !!!!
          seed, n_splits,
          data.shape[0], n_chans, to_time - from_time,
          y_true_list, y_pred_list, ## !!!!!!!
          amplify_input, q,
          n_layers,
          str(patch_dim), str(patch_dim5), str(patch_dim6),  str(pool_dim), 
          str(channels_in), str(channels_out), ### !!!! 
          nonlin, fc_units,
          n_iterations, size_of_batches, learning_rate,
          dist, normalized_weights,
          batch_norm,
          keep_prob_train,
          l2_regularization_penalty,
          time]

df = pd.DataFrame([result],
                columns=['session', 'decode_for', 'only_correct_trials',
                         'areas', 'cortex', 'elec_type',
                         'interval',
                         'mean_per_class_accuracy', 'error_bar', 'n_test_per_class',
                         'mean_per_class_accuracy_train_per_fold', 'mean_per_class_accuracy_test_per_fold',
                         'seed', 'n_splits',
                         'data_size', 'n_chans', 'window_size',
                         'y_true_per_fold', 'y_pred_per_fold',
                         'amplify_input', 'q',
                         'n_layers', 'patch_dim', 'patchl_dim5', 'patch_dim6', 'pool_dim',
                         'channels_in', 'channels_out', 
                         'nonlin', 'fc_units',
                         'n_iterations', 'size_of_batches', 'learning_rate',
                         'weights_dist', 'normalized_weights',
                         'batch_norm',
                         'keep_prob_train', 
                         'l2_regularization_penalty',
                         'time'],
                index=[0])



# Save to file
file_name = (base_path + 'results/training/'
             + sess_no + '_training_'+decode_for+'.csv')
file_exists = os.path.isfile(file_name)
if file_exists :
    with open(file_name, 'a') as f:
        df.to_csv(f, mode ='a', index=False, header=False)
else:
    with open(file_name, 'w') as f:
        df.to_csv(f, mode ='w', index=False, header=True)
        
        


            
            

