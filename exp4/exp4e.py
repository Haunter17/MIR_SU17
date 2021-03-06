import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

FAST_FLAG = 0
# Functions for initializing neural nets parameters
def init_weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)

def init_bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial)

# ==============================================
# ==============================================
# 					main driver
# ==============================================
# ==============================================
print('==> Experiment 4e: Dropout')
filepath = '/pylon2/ci560sp/haunter/exp3_taylorswift_d15_1s_C1C8.mat'
if FAST_FLAG:
	filepath = '/pylon2/ci560sp/haunter/exp3_small.mat'
print('==> Loading data from {}...'.format(filepath))
# benchmark
t_start = time.time()

# ==============================================
# 				reading data
# ==============================================
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

# ==============================================
# Neural-network model set-up
# ==============================================

num_training_vec, total_features = X_train.shape
num_freq = 169
num_frames = int(total_features / num_freq)
num_classes = int(max(y_train.max(), y_val.max()) + 1)

num_layers = 2
fac = 0.5

try:
	fac = float(sys.argv[1])
	num_layers = int(sys.argv[2])
	assert(num_layers >= 2 and num_layers <= 4)
except Exception, e:
	print('-- {}'.format(e))
print('-- Decreasing factor = {}'.format(fac))
print('-- Number of layers = {}'.format(num_layers))

size_list = []
for num in range(num_layers + 1):
	size_list.append(int(total_features * np.power(fac, num)))
print('-- Layer sizes = {}'.format(size_list))

batch_size = 1000

num_epochs = 500
print_freq = 10
if FAST_FLAG:
	num_epochs = 5
	print_freq = 1

# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_val_OHEnc = tf.one_hot(y_val.copy(), num_classes)

# reset placeholders
x = tf.placeholder(tf.float32, [None, total_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])

W_ae_list = [init_weight_variable([size_list[i], size_list[i + 1]]) \
	for i in range(num_layers)]
b_ae_list = [init_bias_variable([size_list[i + 1]])\
	for i in range(num_layers)]
a_list = [tf.nn.relu(tf.matmul(x, W_ae_list[0]) + b_ae_list[0])]
for i in range(num_layers - 1):
	a_list.append(tf.nn.relu(tf.matmul(a_list[-1], W_ae_list[i + 1]) + b_ae_list[i + 1]))

# dropout
keep_prob = tf.placeholder(tf.float32)
a_drop = tf.nn.dropout(a_list[-1], keep_prob)
W_sm = init_weight_variable([size_list[-1], num_classes])
b_sm = init_bias_variable([num_classes])
y_sm = tf.matmul(a_drop, W_sm) + b_sm

cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_sm))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_sm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
y_train = sess.run(y_train_OHEnc)[:, 0, :]
y_val = sess.run(y_val_OHEnc)[:, 0, :]

# evaluation metrics
train_acc_list = []
val_acc_list = []
train_err_list = []
val_err_list = []

print('==> Training the full network...')
t_start = time.time()
for epoch in range(num_epochs):
	for i in range(0, num_training_vec, batch_size):
		batch_end_point = min(i + batch_size, num_training_vec)
		train_batch_data = X_train[i : batch_end_point]
		train_batch_label = y_train[i : batch_end_point]
		train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label, keep_prob: 0.5})
	if (epoch + 1) % print_freq == 0:
		train_acc = accuracy.eval(feed_dict={x:X_train, y_: y_train, keep_prob: 1.0})
		train_acc_list.append(train_acc)
		val_acc = accuracy.eval(feed_dict={x: X_val, y_: y_val, keep_prob: 1.0})
		val_acc_list.append(val_acc)
		train_err = cross_entropy.eval(feed_dict={x: X_train, y_: y_train, keep_prob: 1.0})
		train_err_list.append(train_err)
		val_err = cross_entropy.eval(feed_dict={x: X_val, y_: y_val, keep_prob: 1.0})
		val_err_list.append(val_err)      
		print("-- epoch: %d, training error %g"%(epoch + 1, train_err))

t_end = time.time()
print('--Time elapsed for training: {t:.2f} \
		seconds'.format(t = t_end - t_start))

# ==============================================
# Reports
print('-- Training accuracy: {:.4f}'.format(train_acc_list[-1]))
print('-- Validation accuracy: {:.4f}'.format(val_acc_list[-1]))
print('-- Training error: {:.4E}'.format(train_err_list[-1]))
print('-- Validation error: {:.4E}'.format(val_err_list[-1]))

print('==> Generating error plot...')
x_list = range(0, print_freq * len(train_acc_list), print_freq)
train_err_plot = plt.plot(x_list, train_err_list, 'b-', label='training')
val_err_plot = plt.plot(x_list, val_err_list, '-', color='orange', label='validation')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs with {} Layers and Decreasing Factor {}'.format(num_layers, fac))
plt.legend(loc='best')
plt.savefig('exp4e_L{}F{}.png'.format(num_layers, fac), format='png')
plt.close()

print('==> Done.')
