import numpy as np
import tensorflow as tf
import matplotlib
import h5py
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# reading data
filepath = 'exp2d_test.mat'
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))
del f
print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of validation samples: {}'.format(X_val.shape[0]))

# Neural-network model set-up
num_training_vec, total_features = X_train.shape
num_freq = 121
num_frames = 16
num_classes = 71
k1 = 12
k2 = 4

sess = tf.InteractiveSession()

# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_val_OHEnc = tf.one_hot(y_val.copy(), num_classes)
y_train = sess.run(y_train_OHEnc)[:, 0, :]
y_val = sess.run(y_val_OHEnc)[:, 0, :]

x = tf.placeholder(tf.float32, shape=[None, 1936])
y_ = tf.placeholder(tf.float32, shape=[None, 71])

W_conv1 = weight_variable([121, 1, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,121,16,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([1, 16, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)

#W_fc1 = weight_variable([7 * 7 * 64, 1024])
#b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_conv2, [-1, 64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([64, 71])
b_fc2 = bias_variable([71])

y_conv = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

train_acc_list = []
val_acc_list = []
train_err_list = []
val_err_list = []

batch_size = 1000

for i in range(0, num_training_vec, batch_size):

    batch_end_point = min(i + batch_size, num_training_vec)
    train_batch_data = X_train[i : batch_end_point]
    train_batch_label = y_train[i : batch_end_point]
    train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label})
    
    train_acc = accuracy.eval(feed_dict={x:train_batch_data, y_: train_batch_label})
    train_err = cross_entropy.eval(feed_dict={x:train_batch_data, y_: train_batch_label})
    val_acc = accuracy.eval(feed_dict={x:X_val, y_:y_val})
    val_err = cross_entropy.eval(feed_dict={x:X_val, y_:y_val})
    print("step %d, t acc %g, v acc %g, t err %g, v err %g"%(i, train_acc, val_acc, train_err, val_err))

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_err_list.append(train_err)
    val_err_list.append(val_err)

    #train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Reports
print('-- Training accuracy: {:.4f}'.format(train_acc_list[-1]))
print('-- Validation accuracy: {:.4f}'.format(val_acc_list[-1]))
print('-- Training error: {:.4E}'.format(train_err_list[-1]))
print('-- Validation error: {:.4E}'.format(val_err_list[-1]))

print('==> Generating error plot...')
x_list = range(0, len(train_acc_list))
train_err_plot, = plt.plot(x_list, train_err_list, 'b.')
val_err_plot, = plt.plot(x_list, val_err_list, '.', color='orange')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs')
plt.legend((train_err_plot, val_err_plot), ('training', 'validation'), loc='best')
plt.savefig('exp1b_error_{}.png'.format(l), format='png')
plt.close()

print('==> Done.')