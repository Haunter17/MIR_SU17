import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Functions for initializing neural nets parameters
def init_weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
  return tf.Variable(initial)

def init_bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

print('==> Experiment 2b')
filepath = '../taylorswift_out/s71d7.mat'
print('==> Loading data from {}'.format(filepath))
# benchmark
t_start = time.time()

# reading data
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))
print('--Time elapsed for loading data: {t:.2f} \
    seconds'.format(t = t_end - t_start))
del f
print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of test samples: {}'.format(X_val.shape[0]))

# Neural-network model set-up

'''
	CNN config parameters
'''
total_features = X_train.shape[1]
num_freq = 121
num_frames = int(total_features / num_freq)
num_classes = int(max(y_train.max(), y_test.max()) + 1)
k1 = 16
k2 = 32
l = 5

# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_test_OHEnc = tf.one_hot(y_test.copy(), num_classes)

# Set-up input and output label
x = tf.placeholder(tf.float64, [None, num_freq])
y_ = tf.placeholder(tf.float32, [None, num_classes])

# first convolutional layer
W_conv1 = init_weight_variable([num_freq, 1, 1, k1])
b_conv1 = init_bias_variable([k1])
x_image = tf.reshape(x, [-1, num_freq, num_frames, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# second layer
W_conv2 = init_weight_variable([1, l, k1, k2])
b_conv2 = init_bias_variable([k2])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

# softmax layer
W_sm = init_weight_variable([])

print('==> Done.')

