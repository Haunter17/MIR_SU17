import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# usage: python exp11a.py bigk.r.i.t 0 0
# system arg

artist = ''
SMALL_FLAG = 1
FAST_FLAG = 1
try:
	artist = sys.argv[1]
	SMALL_FLAG = int(sys.argv[2])
	FAST_FLAG = int(sys.argv[3])
except Exception, e:
	print('-- {}'.format(e))

print('-- Artist: {}'.format(artist))
print('-- SMALL FLAG: {}'.format(SMALL_FLAG))
print('-- FAST FLAG: {}'.format(FAST_FLAG))


# Functions for initializing neural nets parameters
def init_weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)

def init_bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')

def batch_nm(x, eps=1e-5):
	# batch normalization to have zero mean and unit variance
	mu, var = tf.nn.moments(x, [0])
	return tf.nn.batch_normalization(x, mu, var, None, None, eps)

def max_pool(x, p):
  return tf.nn.max_pool(x, ksize=[1, p, p, 1],
                        strides=[1, p, p, 1], padding='VALID')

def batch_eval(data, label, metric, batch_size=256):
	value = 0.
	for i in range(0, data.shape[0], batch_size):
		batch_end_point = min(i + batch_size, data.shape[0])
		batch_data = data[i : batch_end_point]
		batch_label = label[i : batch_end_point]
		value += batch_data.shape[0] * metric.eval(feed_dict={x: batch_data, y_: batch_label, keep_prob: 1.0})
	value = value / data.shape[0]
	return value

def MRR_batch(data, label, batch_size=256):
	value = 0.
	for i in range(0, data.shape[0], batch_size):
		batch_end_point = min(i + batch_size, data.shape[0])
		batch_data = data[i : batch_end_point]
		batch_label = label[i : batch_end_point]
		batch_pred = sess.run(y_sm, feed_dict={x: batch_data, y_: batch_label, keep_prob: 1.0})
		value += batch_data.shape[0] * MRR(batch_pred, batch_label)
	value = value / data.shape[0]
	return value

def MRR(pred, label):
	'''
		pred, label are np arrays with dimension m x k
	'''
	pred_rank = np.argsort(-pred, axis=1)
	groundtruth = np.argmax(label, axis=1)
	rank = np.array([np.where(pred_rank[index] == groundtruth[index])[0].item(0) + 1 \
		for index in range(label.shape[0])])
	return np.mean(1. / rank)
		

# ==============================================
# ==============================================
# 					main driver
# ==============================================
# ==============================================
print('==> Experiment 11a: MNIST Mirror on Full Window...')
sys_path = '/pylon2/ci560sp/haunter/'
filename = artist + '_data.mat'
if SMALL_FLAG:
	filename = artist + '_data_small.mat'
filepath = sys_path + filename
print('==> Loading data from {}...'.format(filepath))
# benchmark
t_start = time.time()

# ==============================================
# 				reading data
# ==============================================
f = h5py.File(filepath)
D_train = np.array(f.get('DTrain'))
D_val = np.array(f.get('DVal'))
np.random.shuffle(D_train)
np.random.shuffle(D_val)
X_train = D_train[:, :-1]
y_train = D_train[:, -1]
X_val = D_val[:, :-1]
y_val = D_val[:, -1]

t_end = time.time()
print('--Time elapsed for loading data: {t:.2f} \
		seconds'.format(t = t_end - t_start))
del D_train, D_val, f
print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of validation samples: {}'.format(X_val.shape[0]))

# ==============================================
# Neural-network model set-up
# ==============================================

num_train, total_features = X_train.shape
num_freq = 121
num_frames = int(total_features / num_freq)
num_classes = int(max(y_train.max(), y_val.max()) + 1)


# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_val_OHEnc = tf.one_hot(y_val.copy(), num_classes)

# reset placeholders
x = tf.placeholder(tf.float32, [None, total_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])

# ==============================================
# First layer
# ==============================================
r1, c1, k1 = 5, 5, 32
W_conv1 = init_weight_variable([r1, c1, 1, k1])
b_conv1 = init_bias_variable([k1])
x_image = tf.reshape(x, [-1, num_freq, num_frames, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 2)
h1r, h1c = (num_freq - r1 + 1) / 2, (num_frames - c1 + 1) / 2
# ==============================================
# Second layer
# ==============================================
r2, c2, k2 = 5, 5, 64
W_conv2 = init_weight_variable([r2, c2, k1, k2])
b_conv2 = init_bias_variable([k2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 2)
h2r, h2c = (h1r - r2 + 1) / 2, (h1c - c2 + 1) / 2
h_pool2_flat = tf.reshape(h_pool2, [-1, h2r * h2c * k2])

# ==============================================
# Dense layer
# ==============================================
nhidden = 1024
W_fc1 = init_weight_variable([h2r * h2c * k2, nhidden])
b_fc1 = init_bias_variable([nhidden])
h_fc1 = tf.nn.relu(batch_nm(tf.matmul(h_pool2_flat, W_fc1) + b_fc1))
# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_sm = init_weight_variable([nhidden, num_classes])
b_sm = init_bias_variable([num_classes])
y_sm = tf.matmul(h_fc1_drop, W_sm) + b_sm

cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_sm))
train_step = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_sm, 1), tf.argmax(y_, 1))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
y_train = sess.run(y_train_OHEnc)
y_val = sess.run(y_val_OHEnc)

# evaluation metrics
train_err_list = []
val_err_list = []
val_mrr_list = []

# saver setup
varsave_list = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_sm, b_sm]
saver = tf.train.Saver(varsave_list)
save_path = './out/11amodel_{}.ckpt'.format(artist)
opt_val_err = np.inf
opt_epoch = -1
step_counter = 0
max_counter = 5000

batch_size = 256
max_epochs = 500
print_freq = 200
num_iter = 0

if FAST_FLAG:
	max_epochs = 1
	print_freq = 10
print('==> Training the full network...')
t_start = time.time()
for epoch in range(max_epochs):
	if step_counter <= max_counter:
		for i in range(0, num_train, batch_size):
			batch_end_point = min(i + batch_size, num_train)
			train_batch_data = X_train[i : batch_end_point]
			train_batch_label = y_train[i : batch_end_point]
			train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label, keep_prob: 0.5})
			if (num_iter + 1) % print_freq == 0:
				# evaluate metrics
				train_err = cross_entropy.eval(feed_dict={x: train_batch_data, y_: train_batch_label, keep_prob: 1.0})
				train_err_list.append(train_err)
				val_err = batch_eval(X_val, y_val, cross_entropy)
				val_err_list.append(val_err)
				val_mrr = MRR_batch(X_val, y_val)
				val_mrr_list.append(val_mrr)
				print("-- epoch %d, iter %d, training error %g, validation error %g"%(epoch + 1, num_iter + 1, train_err, val_err))
				# save screenshot of the model
				if val_err < opt_val_err:
					step_counter = 0	
					saver.save(sess, save_path)
					print('==> New optimal validation error found. Model saved.')
					opt_val_err, opt_epoch, opt_iter = val_err, epoch + 1, num_iter + 1
			if step_counter > max_counter:
				print('==> Step counter exceeds maximum value. Stop training at epoch {}, iter {}.'.format(epoch + 1, num_iter + 1))
				break
			step_counter += 1
			num_iter += 1
	else:
		break    

t_end = time.time()
print('--Time elapsed for training: {t:.2f} \
		seconds'.format(t = t_end - t_start))

# ==============================================
# Restore model & Evaluations
# ==============================================
saver.restore(sess, save_path)
print('==> Model restored to epoch {}, iter {}'.format(opt_epoch, opt_iter))

from scipy.io import savemat
model_path = './out/11a_{}'.format(artist)
W1, W2 = sess.run([W_conv1, W_conv2], feed_dict={x: X_train, y_: y_train, keep_prob: 1.0})
savemat(model_path, {'W1': W1, 'W2': W2})
print('==> CNN filters saved to {}.mat'.format(model_path))

print('-- Final Validation error: {:.4E}'.format(batch_eval(X_val, y_val, cross_entropy)))
print('-- Final Validation MRR: {:.3f}'.format(MRR_batch(X_val, y_val)))

print('-- Training error --')
print([float('{:.4E}'.format(e)) for e in train_err_list])
print('-- Validation error --')
print([float('{:.4E}'.format(e)) for e in val_err_list])
print('-- Validaiton MRR --')
print([float('{:.3f}'.format(e)) for e in val_mrr_list])

print('==> Generating error plot...')
x_list = range(0, print_freq * len(train_err_list), print_freq)
train_err_plot = plt.plot(x_list, train_err_list, 'b', label='training')
val_err_plot = plt.plot(x_list, val_err_list , color='orange', label='validation')
plt.xlabel('Number of Iterations')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs for {}'.format(artist))
plt.legend(loc='best')
plt.savefig('./out/exp11a_{}.png'.format(artist), format='png')
plt.close()

print('==> Finished!')
