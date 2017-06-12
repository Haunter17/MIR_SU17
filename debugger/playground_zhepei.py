import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from tensorflow.python import debug as tf_debug

# Functions for initializing neural nets parameters
def init_weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)

def init_bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')

def get_preactv_stats(Z, X, y):
	Z_mat = sess.run(Z, feed_dict={x: X, y_: y})
	for sample_num in range(Z_mat.shape[0]):
		print('==> Sample #{}'.format(sample_num + 1))
		Z_max = Z_mat[sample_num].max(axis=2)
		Z_min = Z_mat[sample_num].min(axis=2)
		Z_mean = Z_mat[sample_num].mean(axis=2)
		for i in range(Z_max.shape[0]):
			for j in range(Z_max.shape[1]):
				print('-- row {} column {}'.format(i + 1, j + 1))
				print('-- Min: {:.2f}'.format(Z_min.item(i, j)))
				print('-- Max: {:.2f}'.format(Z_max.item(i, j)))
				print('-- Mean: {:.2f}'.format(Z_mean.item(i, j)))

print('==> Debugger playground...')
filepath = '/pylon2/ci560sp/haunter/exp2_d15_1s_2.mat'
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

# Neural-network model set-up
num_training_vec, total_features = X_train.shape
num_freq = 121
num_frames = int(total_features / num_freq)
num_classes = int(max(y_train.max(), y_val.max()) + 1)
k = 12
filter_row, filter_col = 121, 1

print('-- Filter size  is {} x {}'.format(filter_row, filter_col))

batch_size = 1000
num_epochs = 1
print_freq = 1


# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_val_OHEnc = tf.one_hot(y_val.copy(), num_classes)

# Set-up input and output label
x = tf.placeholder(tf.float32, [None, total_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])

# first convolutional layer
W_conv1 = init_weight_variable([filter_row, filter_col, 1, k])
b_conv1 = init_bias_variable([k])
x_image = tf.reshape(x, [-1, num_freq, num_frames, 1])
z_conv1 = conv2d(x_image, W_conv1) + b_conv1
z_conv1 = tf.identity(z_conv1, name='z_conv1')
h_conv1 = tf.nn.relu(z_conv1, name='h_conv1')
h_conv1_flat = tf.reshape(h_conv1, [-1, (num_freq - filter_row + 1) * (num_frames - filter_col + 1) * k])

# softmax layer
W_sm = init_weight_variable([(num_freq - filter_row + 1) * (num_frames - filter_col + 1) * k, num_classes])
b_sm = init_bias_variable([num_classes])

y_conv = tf.matmul(h_conv1_flat, W_sm) + b_sm

# evaluations
cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# debugging session
debug = 0
if debug:
	sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

y_train = sess.run(y_train_OHEnc)[:, 0, :]
y_val = sess.run(y_val_OHEnc)[:, 0, :]

train_acc_list = []
val_acc_list = []
train_err_list = []
val_err_list = []

epoch_tf = tf.Variable(-1, name='epoch_tensor')
batch_tf = tf.Variable(-1, name='batch_tensor')
sess.run([epoch_tf.initializer, batch_tf.initializer])

# benchmark
t_start = time.time()
for epoch in range(num_epochs):
	for i in range(0, num_training_vec, batch_size):
		batch_end_point = min(i + batch_size, num_training_vec)
		train_batch_data = X_train[i : batch_end_point]
		train_batch_label = y_train[i : batch_end_point]
		if batch_end_point == num_training_vec:
			sess.run([epoch_tf.assign(epoch), batch_tf.assign(i), train_step], \
				feed_dict={x: train_batch_data, y_: train_batch_label})
		else:
			train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label})
	get_preactv_stats(tf.get_default_graph().get_tensor_by_name('z_conv1:0'), \
		train_batch_data, train_batch_label)		

	if (epoch + 1) % print_freq == 0:
		train_acc = accuracy.eval(feed_dict={x:X_train, y_: y_train})
		train_acc_list.append(train_acc)
		val_acc = accuracy.eval(feed_dict={x: X_val, y_: y_val})
		val_acc_list.append(val_acc)
		train_err = cross_entropy.eval(feed_dict={x: X_train, y_: y_train})
		train_err_list.append(train_err)
		val_err = cross_entropy.eval(feed_dict={x: X_val, y_: y_val})
		val_err_list.append(val_err)      
		print("-- epoch: %d, training error %g"%(epoch + 1, train_err))

t_end = time.time()
print('--Time elapsed for training: {t:.2f} \
		seconds'.format(t = t_end - t_start))

# Reports
print('-- Training accuracy: {:.4f}'.format(train_acc_list[-1]))
print('-- Validation accuracy: {:.4f}'.format(val_acc_list[-1]))
print('-- Training error: {:.4E}'.format(train_err_list[-1]))
print('-- Validation error: {:.4E}'.format(val_err_list[-1]))

print('==> Generating error plot...')
x_list = range(0, print_freq * len(train_acc_list), print_freq)
train_err_plot, = plt.plot(x_list, train_err_list, 'b.')
val_err_plot, = plt.plot(x_list, val_err_list, '.', color='orange')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs with Filter Size of {} x {}'.format(filter_row, filter_col))
plt.legend((train_err_plot, val_err_plot), ('training', 'validation'), loc='best')
plt.savefig('exp2_0d_error_{}x{}.png'.format(filter_row, filter_col), format='png')
plt.close()

print('==> Done.')
