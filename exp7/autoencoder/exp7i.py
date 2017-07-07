import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# usage: python exp7i.py 64 4 0 0 0
# system arg
nhidden = 64
nlayer = 1
SMALL_FLAG = 1
FAST_FLAG = 1
SYS_FLAG = 0 # 0 for bridges, 1 for supermic

try:
	nhidden = int(sys.argv[1])
	nlayer = int(sys.argv[2])
	SMALL_FLAG = int(sys.argv[3])
	FAST_FLAG = int(sys.argv[4])
	SYS_FLAG = int(sys.argv[5])
	
except Exception, e:
	print('-- {}'.format(e))

print('-- Number of features = {}'.format(nhidden))
print('-- Number of layers = {}'.format(nlayer))
print('-- SMALL FLAG: {}'.format(SMALL_FLAG))
print('-- FAST FLAG: {}'.format(FAST_FLAG))
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
print('==> Experiment 7i: Autoencoder number of layers...')
sys_path = '/pylon2/ci560sp/haunter/'
if SYS_FLAG:
	sys_path = '/scratch/zwang3/'
filename = 'exp7g.mat'
if SMALL_FLAG:
	filename = 'exp7g_small.mat'
filepath = sys_path + filename
print('==> Loading data from {}...'.format(filepath))
# benchmark
t_start = time.time()

# ==============================================
# 				reading data
# ==============================================
train_frac = 0.8
f = h5py.File(filepath)
X = np.array(f.get('data'))
t_end = time.time()
print('--Time elapsed for loading data: {t:.2f} \
		seconds'.format(t = t_end - t_start))
del f
# shuffle and split
np.random.shuffle(X)
total_samples, total_features = X.shape
X_train = X[:int(train_frac * total_samples), :]
X_val = X[int(train_frac):, :]

num_training_vec = X_train.shape[0]
num_val_vec = X_val.shape[0]
print('-- Number of training samples: {}'.format(num_training_vec))
print('-- Number of validation samples: {}'.format(num_val_vec))

size_list = [total_features]
for i in range(nlayer):
	size_list.append(nhidden)
print('-- Layer sizes: {}'.format(size_list))
# ==============================================
# Neural-network model set-up
# ==============================================
batch_size = 1000

num_epochs = 500
print_freq = 5
if FAST_FLAG:
	num_epochs = 5
	print_freq = 1

# reset placeholders
x = tf.placeholder(tf.float32, [None, total_features])

# autoencoder: x -> a1 -> a2 ... ->x
W_list = [init_weight_variable([size_list[i], size_list[i + 1]]) \
	for i in range(nlayer)]
b_list = [init_bias_variable([size_list[i + 1]])\
	for i in range(nlayer)]
a_list = [tf.nn.relu(tf.matmul(x, W_list[0]) + b_list[0])]
for i in range(nlayer - 1):
	a_i = tf.nn.relu(tf.matmul(a_list[-1], W_list[i + 1])+ b_list[i + 1])
	a_list.append(a_i)

W_ad = init_bias_variable([nhidden, total_features])
b_ad = init_bias_variable([total_features])
h = tf.nn.relu(tf.matmul(a_list[-1], W_ad + b_ad))
error = None
if SYS_FLAG:
	error = tf.reduce_mean(tf.square(x - h))
else:
	error = tf.losses.mean_squared_error(x, h)
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)

sess = tf.InteractiveSession()
if SYS_FLAG:
	sess.run(tf.initialize_all_variables())
else:
	sess.run(tf.global_variables_initializer())

# evaluation metrics
train_err_list = []
val_err_list = []

# saver setup
varsave_list = W_list + b_list + [W_ad, b_ad]
saver = tf.train.Saver(varsave_list)
save_path = './out/7imodel_{}_{}.ckpt'.format(nhidden, nlayer)
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
		train_step.run(feed_dict={x: train_batch_data})
	if (epoch + 1) % print_freq == 0:
		# evaluate metrics
		train_err = error.eval(feed_dict={x: train_batch_data})
		train_err_list.append(train_err)
		val_err = error.eval(feed_dict={x: X_val})
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

train_err = error.eval(feed_dict={x: X_train})
val_err = error.eval(feed_dict={x: X_val})
print('-- Training error: {:.4E}'.format(train_err))
print('-- Validation error: {:.4E}'.format(val_err))

print('-- Training error --')
print([float('{:.6E}'.format(x)) for x in train_err_list])
print('-- Validation error --')
print([float('{:.6E}'.format(x)) for x in val_err_list])

# ==============================================
# Saving weights
# ==============================================
model_path = './out/7i_{}_{}'.format(nhidden, nlayer)
W_data = []
b_data = []
for i in range(nlayer):
	W, b = sess.run([W_list[i], b_list[i]], feed_dict={x: X_train})
	W_data.append(W)
	b_data.append(b)
from scipy.io import savemat
savemat(model_path, {'W_list': W_data, 'b_list': b_data, 'num_layer': nlayer})
print('==> Autoencoder weights saved to {}.mat'.format(model_path))

print('==> Finished!')
