import numpy as np
import tensorflow as tf
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

# reading data
filepath = 'exp2_d15_1s_2.mat'
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))
del f

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

W_conv1 = weight_variable([121, 1, 1, 12])
b_conv1 = bias_variable([12])

x_image = tf.reshape(x, [-1,121,16,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1_flat = tf.reshape(h_conv1, [-1, 16*12])

W_fc2 = weight_variable([16*12, 71])
b_fc2 = bias_variable([71])

y_conv = tf.matmul(h_conv1_flat, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

plotx = []
train_acc_list = []
val_acc_list = []
train_err_list = []
train_err_avg_list = []

val1_err_list = []
val2_err_list = []
val3_err_list = []
val4_err_list = []

batch_size = 1000

#duplicate validation dataset

X_val = X_val[:25000]
y_val = y_val[:25000]

X_val_2 = np.vstack((X_val,X_val))
y_val_2 = np.vstack((y_val,y_val))

X_val_3 = np.vstack((X_val_2,X_val))
y_val_3 = np.vstack((y_val_2,y_val))

X_val_4 = np.vstack((X_val_3,X_val))
y_val_4 = np.vstack((y_val_3,y_val))

print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of validation samples 1: {}'.format(X_val.shape[0]))
print('-- Number of validation samples 2: {}'.format(X_val_2.shape[0]))
print('-- Number of validation samples 3: {}'.format(X_val_3.shape[0]))
print('-- Number of validation samples 4: {}'.format(X_val_4.shape[0]))

for epoch in range(300):

    batch_trainErr = []

    for i in range(0, num_training_vec, batch_size):
      batch_end_point = min(i + batch_size, num_training_vec)
      train_batch_data = X_train[i : batch_end_point]
      train_batch_label = y_train[i : batch_end_point]
      train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label})

    for i in range(0, num_training_vec, batch_size):
      batch_end_point = min(i + batch_size, num_training_vec)
      train_batch_data = X_train[i : batch_end_point]
      train_batch_label = y_train[i : batch_end_point]
      batch_err = cross_entropy.eval(feed_dict={x: train_batch_data, y_: train_batch_label})
      batch_trainErr.append(batch_err*(batch_end_point-i))
    
    plotx.append(epoch)
    train_acc = accuracy.eval(feed_dict={x: X_train, y_: y_train})
    train_err_avg = np.sum(batch_trainErr)/num_training_vec
    train_err = cross_entropy.eval(feed_dict={x: X_train, y_: y_train})
    val_acc = accuracy.eval(feed_dict={x:X_val, y_:y_val})
    val1_err = cross_entropy.eval(feed_dict={x:X_val, y_:y_val})    
    val2_err = cross_entropy.eval(feed_dict={x:X_val_2, y_:y_val_2})
    val3_err = cross_entropy.eval(feed_dict={x:X_val_3, y_:y_val_3})
    val4_err = cross_entropy.eval(feed_dict={x:X_val_4, y_:y_val_4})
    
    print("Epoch: %d, v1 %g, v2 %g, v3 %g, v4 %g"%(epoch, val1_err, val2_err, val3_err, val4_err))
    print("====> t err avg %g"%train_err_avg)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_err_list.append(train_err)
    train_err_avg_list.append(train_err_avg)
    val1_err_list.append(val1_err)
    val2_err_list.append(val2_err)
    val3_err_list.append(val3_err)
    val4_err_list.append(val4_err)

print('==> Generating error plot...')
errfig = plt.figure()
valErrPlot1 = errfig.add_subplot(111)
valErrPlot1.set_xlabel('Number of Epochs')
valErrPlot1.set_ylabel('Cross-Entropy Error')
valErrPlot1.set_title('Error vs Number of Epochs')
valErrPlot1.scatter(plotx, val1_err_list, c='blue')
valErrPlot2 = errfig.add_subplot(111)
valErrPlot2.scatter(plotx, val2_err_list, c='red')
valErrPlot3 = errfig.add_subplot(111)
valErrPlot3.scatter(plotx, val3_err_list, c='green')
valErrPlot4 = errfig.add_subplot(111)
valErrPlot4.scatter(plotx, val4_err_list, c='yellow')
errfig.savefig('exp2j.png')