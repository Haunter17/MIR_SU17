import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Functions for initializing neural nets parameters
def init_weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial)

def init_bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')

def loadData(filepath):
  print('==> Experiment 2_0c')
  print('==> Loading data from {}'.format(filepath))
  # benchmark
  t_start = time.time()

  # reading data
  f = h5py.File(filepath)
  X_train = np.array(f.get('trainingFeatures'))
  y_train = np.array(f.get('trainingLabels'))
  X_val = np.array(f.get('validationFeatures'))
  y_val = np.array(f.get('validationLabels'))
  t_end = time.time()
  print('--Time elapsed for loading data: {t:.2f} \
      seconds'.format(t = t_end - t_start))
  del f
  print('-- Number of training samples: {}'.format(X_train.shape[0]))
  print('-- Number of validation samples: {}'.format(X_val.shape[0]))
  print('Shape of X_train: %s'%str(X_train.shape))
  print('Shape of y_train: %s'%str(y_train.shape))
  print('Shape of X_val: %s'%str(X_val.shape))
  print('Shape of y_val: %s'%str(y_val.shape))

  return [X_train, y_train, X_val, y_val]

def runNeuralNet(num_freq, X_train, y_train, X_val, y_val, batch_size, num_epochs, pooling_strategy):

  # Neural-network model set-up
  num_training_vec, total_features = X_train.shape
  num_frames = int(total_features / num_freq)
  print('-- Num frames: {}'.format(num_frames))
  num_classes = int(max(y_train.max(), y_val.max()) + 1)

  k1 = 5
  k2 = 0
  l = num_frames

  print("Num Classes: %g"%(num_classes))

  print_freq = 1

  # Transform labels into on-hot encoding form
  y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
  y_val_OHEnc = tf.one_hot(y_val.copy(), num_classes)

  # Set-up input and output label
  x = tf.placeholder(tf.float32, [None, total_features])
  y_ = tf.placeholder(tf.float32, [None, num_classes])

  
  # go straight from input to output, densely connected to SM layer
  '''
  W_sm = init_weight_variable([total_features, num_classes])
  b_sm = init_bias_variable([num_classes])
  y_conv = tf.matmul(x, W_sm) + b_sm
  '''

  print("Running single convolutional layer with %g 1x1 filters"%(k1))
  
  # single convolutional layer
  W_conv1 = init_weight_variable([1, 1, 1, k1]) # Old: [num_freq, 1, 1, k1]
  b_conv1 = init_bias_variable([k1])
  x_image = tf.reshape(x, [-1, num_freq, num_frames, 1])
  h_conv1 = conv2d(x_image, W_conv1) + b_conv1 # tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1), no non-linearity

  h_conv1_flat = tf.reshape(h_conv1, [-1, k1 * num_freq * num_frames]) #tf.reshape(h_conv1, [-1, num_frames * k1]) --- use this type of thing to make multiple scaled versions of data? enhance dataset?

  W_sm = init_weight_variable([k1 * num_freq * num_frames, num_classes])
  b_sm = init_bias_variable([num_classes])
  y_conv = tf.matmul(h_conv1_flat, W_sm) + b_sm
  

  '''
  # One hidden layer then softmax
  numHiddenUnits = 100
  W_1 = init_weight_variable([total_features, numHiddenUnits])
  b_1 = init_bias_variable([numHiddenUnits])

  W_sm = init_weight_variable([numHiddenUnits, num_classes])
  b_sm = init_bias_variable([num_classes])

  hiddenActivation = tf.nn.relu(tf.matmul(x, W_1) + b_1)
  y_conv = tf.matmul(hiddenActivation, W_sm) + b_sm
  '''
  
  # second layer
  #W_conv2 = init_weight_variable([1, l, k1, k2])
  #b_conv2 = init_bias_variable([k2])
  #h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
  #h_conv2_flat = tf.reshape(h_conv2, [-1, (num_frames - l + 1) * k2])

  #h_pool2 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

  # softmax layer
  #W_sm = init_weight_variable([(num_frames - l + 1) * k2, num_classes])
  #b_sm = init_bias_variable([num_classes])

  #y_conv = tf.matmul(h_conv2_flat, W_sm) + b_sm

  # evaluations
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  
  y_train = sess.run(y_train_OHEnc)[:, 0, :]
  y_val = sess.run(y_val_OHEnc)[:, 0, :]
  
  # print("h_conv1 %s"%str(h_conv1.eval(feed_dict={x:X_train, y_:y_train})))
  # print("W_sm is: %s"%str(W_sm.eval()))
  # print("h_conv1_flat is: %s"%str(h_conv1_flat.eval(feed_dict={x:X_train, y_:y_train})))
  # print("y_conv: %s"%str(y_conv.eval(feed_dict={x: X_train, y_: y_train})))
  # print("y_ is : %s"%str(y_.eval(feed_dict={x:X_train, y_:y_train})))

  train_acc_list = []
  val_acc_list = []
  train_err_list = []
  val_err_list = []
  epoch_numbers = []

  # benchmark
  t_start = time.time()
  for epoch in range(num_epochs):
    epochStart = time.time()
    for i in range(0, num_training_vec, batch_size):
      batch_end_point = min(i + batch_size, num_training_vec)
      train_batch_data = X_train[i : batch_end_point]
      train_batch_label = y_train[i : batch_end_point]
      train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label})
    epochEnd = time.time()
    # printing and recording data
    if (epoch + 1) % print_freq == 0:
      train_acc = accuracy.eval(feed_dict={x:X_train, y_: y_train})
      train_acc_list.append(train_acc)
      val_acc = accuracy.eval(feed_dict={x: X_val, y_: y_val})
      val_acc_list.append(val_acc)
      train_err = cross_entropy.eval(feed_dict={x: X_train, y_: y_train})
      train_err_list.append(train_err)
      val_err = cross_entropy.eval(feed_dict={x: X_val, y_: y_val})
      val_err_list.append(val_err)  
      epoch_numbers += [epoch]    
      #print("-- epoch: %d, training error %g"%(epoch + 1, train_err))
      print("epoch: %d, time: %g, t acc, v acc, t cost, v cost: %g, %g, %g, %g"%(epoch+1, epochEnd - epochStart, train_acc, val_acc, train_err, val_err))

  t_end = time.time()
  print('--Time elapsed for training: {t:.2f} \
      seconds'.format(t = t_end - t_start))

  return [train_acc_list, val_acc_list, train_err_list, val_err_list, epoch_numbers]
  
'''
Our Main
Command Line Arguments: (1) Length of horizontal window
'''

# load the data
[X_train, y_train, X_val, y_val] = loadData('/pylon2/ci560sp/cstrong/exp2/exp2_d15_1s_2.mat')


batchSize = 1000
numEpochs = 300
poolingStrategy = 'MAX'

[train_acc_list, val_acc_list, train_err_list, val_err_list, epoch_numbers] = runNeuralNet(121, X_train, y_train, X_val, y_val, batchSize, numEpochs, poolingStrategy)



# Reports
print('-- Training accuracy: {:.4f}'.format(train_acc_list[-1]))
print('-- Validation accuracy: {:.4f}'.format(val_acc_list[-1]))
print('-- Training error: {:.4E}'.format(train_err_list[-1]))
print('-- Validation error: {:.4E}'.format(val_err_list[-1]))

print('==> Generating error plot...')
x_list = epoch_numbers
train_err_plot, = plt.plot(x_list, train_err_list, 'b.')
val_err_plot, = plt.plot(x_list, val_err_list, '.', color='orange')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs')
plt.legend((train_err_plot, val_err_plot), ('training', 'validation'), loc='best')
plt.savefig('exp2_0c_k1=5.png', format='png')
plt.close()

print('==> Done.')
'''

y_ = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


y_ = np.array([[0], [1], [2], [3], [3]])
x = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29]])
x_val = np.array([[5, 6, 7, 8, 9, 10], [9, 10, 11, 12, 13, 14], [11, 12, 13, 14, 15, 16]])
y_val = np.array([[1], [3], [2]])
runNeuralNet(2, x, y_, x_val, y_val, 1, 300, 'MAX')



'''
'''

K1 = 5
--Time elapsed for training: 2883.65       seconds
-- Training accuracy: 0.9682
-- Validation accuracy: 0.8862
-- Training error: 1.3700E-01
-- Validation error: 4.2459E-01


K1 = 10
--Time elapsed for training: 3057.72       seconds
-- Training accuracy: 0.2608
-- Validation accuracy: 0.8906
-- Training error: 8.5545E+00
-- Validation error: 4.2435E-01
==> Generating error plot...

'''



