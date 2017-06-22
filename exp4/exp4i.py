import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# usage: python exp4i.py 0.5 2 0 0 1 0
# system arg
num_layers = 2
fac = 0.5
SMALL_FLAG = 1
FAST_FLAG = 1
BN_FLAG = 1
SYS_FLAG = 0 # 0 for bridges, 1 for supermic
try:
	fac = float(sys.argv[1])
	num_layers = int(sys.argv[2])
	assert(num_layers >= 2 and num_layers <= 4)
	SMALL_FLAG = int(sys.argv[3])
	FAST_FLAG = int(sys.argv[4])
	BN_FLAG = int(sys.argv[5])
	SYS_FLAG = int(sys.argv[6])
except Exception, e:
	print('-- {}'.format(e))

print('-- Decreasing factor = {}'.format(fac))
print('-- Number of layers = {}'.format(num_layers))
print('-- SMALL FLAG: {}'.format(SMALL_FLAG))
print('-- FAST FLAG: {}'.format(FAST_FLAG))
print('-- NORMALIZATION FLAG: {}'.format(BN_FLAG))
print('-- SYS FLAG: {}'.format(SYS_FLAG))


# Functions for initializing neural nets parameters
def init_weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)

def init_bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial)

def batch_nm(x, eps=1e-5):
	# batch normalization to have zero mean and unit variance
	mu, var = tf.nn.moments(x, [0])
	return tf.nn.batch_normalization(x, mu, var, None, None, eps)


# ==============================================
# ==============================================
# 					main driver
# ==============================================
# ==============================================
print('==> Experiment 4i: Early Stopping...')
sys_path = '/pylon2/ci560sp/haunter/'
if SYS_FLAG:
	sys_path = '/scratch/zwang3/'
filename = 'exp3_taylorswift_d15_1s_C1C8.mat'
if SMALL_FLAG:
	filename = 'exp3_small.mat'
filepath = sys_path + filename
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

size_list = []
for num in range(num_layers + 1):
	size_list.append(int(total_features * np.power(fac, num)))
print('-- Layer sizes = {}'.format(size_list))

batch_size = 1000

num_epochs = 500
print_freq = 5
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
a_list = []

if BN_FLAG:
	a_list.append(tf.nn.relu(batch_nm(tf.matmul(x, W_ae_list[0]) + b_ae_list[0])))
else:
	a_list.append(tf.nn.relu(tf.matmul(x, W_ae_list[0]) + b_ae_list[0]))
for i in range(num_layers - 1):
	a_i = 0
	if BN_FLAG:
		# batch normalization
		a_i = tf.nn.relu(batch_nm(tf.matmul(a_list[-1], W_ae_list[i + 1]) + b_ae_list[i + 1]))
	else:
		a_i = tf.nn.relu(tf.matmul(a_list[-1], W_ae_list[i + 1]) + b_ae_list[i + 1])
	a_list.append(a_i)

# dropout
keep_prob = tf.placeholder(tf.float32)
a_drop = tf.nn.dropout(a_list[-1], keep_prob)
W_sm = init_weight_variable([size_list[-1], num_classes])
b_sm = init_bias_variable([num_classes])
y_sm = tf.matmul(a_drop, W_sm) + b_sm

cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_sm))
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_sm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
if SYS_FLAG:
	sess.run(tf.initialize_all_variables())
else:
	sess.run(tf.global_variables_initializer())
y_train = sess.run(y_train_OHEnc)[:, 0, :]
y_val = sess.run(y_val_OHEnc)[:, 0, :]

# evaluation metrics
train_acc_list = []
val_acc_list = []
train_err_list = []
val_err_list = []

# saver setup
varsave_list = W_ae_list + b_ae_list + [W_sm, b_sm]
saver = tf.train.Saver(varsave_list)
save_path = './4imodel_{}+{}'.format(num_layers, fac)
if not SYS_FLAG:
	save_path += '.ckpt'
opt_val_err = np.inf
opt_epoch = -1
step_counter = 0
max_counter = 50

print('==> Training the full network...')
t_start = time.time()
for epoch in range(num_epochs):
	for i in range(0, num_training_vec, batch_size):
		batch_end_point = min(i + batch_size, num_training_vec)
		train_batch_data = X_train[i : batch_end_point]
		train_batch_label = y_train[i : batch_end_point]
		train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label, keep_prob: 0.5})
	if (epoch + 1) % print_freq == 0:
		# evaluate metrics
		train_acc = accuracy.eval(feed_dict={x:train_batch_data, y_: train_batch_label, keep_prob: 1.0})
		train_acc_list.append(train_acc)
		val_acc = accuracy.eval(feed_dict={x: X_val, y_: y_val, keep_prob: 1.0})
		val_acc_list.append(val_acc)
		train_err = cross_entropy.eval(feed_dict={x: train_batch_data, y_: train_batch_label, keep_prob: 1.0})
		train_err_list.append(train_err)
		val_err = cross_entropy.eval(feed_dict={x: X_val, y_: y_val, keep_prob: 1.0})
		val_err_list.append(val_err)
		print("-- epoch: %d, training error %g, validation error %g"%(epoch + 1, train_err, val_err))
		# save screenshot of the model
		if val_err < opt_val_err:
			step_counter = 0	
			saver.save(sess, save_path)
			print('==> New optimal validation error found. Model saved.')
			opt_val_err, opt_epoch = val_err, epoch + 1
	if step_counter > max_counter:
		print('==> Step counter exceeds maximum value. Stop training at epoch {}.'.format(epoch + 1))
		break
	step_counter += 1      
t_end = time.time()
print('--Time elapsed for training: {t:.2f} \
		seconds'.format(t = t_end - t_start))

# ==============================================
# Restore model & Evaluations
# ==============================================
saver.restore(sess, save_path)
print('==> Model restored to epoch {}'.format(opt_epoch))

train_acc = accuracy.eval(feed_dict={x:X_train, y_: y_train, keep_prob: 1.0})
val_acc = accuracy.eval(feed_dict={x: X_val, y_: y_val, keep_prob: 1.0})
train_err = cross_entropy.eval(feed_dict={x: X_train, y_: y_train, keep_prob: 1.0})
val_err = cross_entropy.eval(feed_dict={x: X_val, y_: y_val, keep_prob: 1.0})
print('-- Training accuracy: {:.4f}'.format(train_acc))
print('-- Validation accuracy: {:.4f}'.format(val_acc))
print('-- Training error: {:.4E}'.format(train_err))
print('-- Validation error: {:.4E}'.format(val_err))

print('-- Training accuracy --')
print([float('{:.4f}'.format(x)) for x in train_acc_list])
print('-- Validation accuracy --')
print([float('{:.4f}'.format(x)) for x in val_acc_list])
print('-- Training error --')
print([float('{:.4E}'.format(x)) for x in train_err_list])
print('-- Validation error --')
print([float('{:.4E}'.format(x)) for x in val_err_list])

print('==> Generating error plot...')
x_list = range(0, print_freq * len(train_acc_list), print_freq)
train_err_plot = plt.plot(x_list, train_err_list, 'b-', label='training')
val_err_plot = plt.plot(x_list, val_err_list, '-', color='orange', label='validation')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs with {} Layers and Decreasing Factor {}'.format(num_layers, fac))
plt.legend(loc='best')
plt.savefig('exp4i_L{}F{}BN{}.png'.format(num_layers, fac, BN_FLAG), format='png')
plt.close()

print('==> Finished!')
