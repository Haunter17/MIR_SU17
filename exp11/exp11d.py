import numpy as np
import tensorflow as tf
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
from scipy.io import savemat

# usage: python exp11b.py deathcabforcutie 0 2 [5, 5] [5, 5] [32, 64] 5by5

# ================ Sys arg ==========================
artist = ''
FAST_FLAG = 1
NUM_CONV_LAYER = 2
HEIGHT_LIST = [5, 5]
WIDTH_LIST = [5, 5]
FILNUM_LIST = [32, 64]
stamp = time.strftime('%H-%M-%S') # optional stamp marker

try:
	artist = sys.argv[1]
	FAST_FLAG = int(sys.argv[2])
	NUM_CONV_LAYER = int(sys.argv[3])
	HEIGHT_LIST_RAW = sys.argv[4].strip('[]').split(',')
	HEIGHT_LIST = [int(element) for element in HEIGHT_LIST_RAW]
	WIDTH_LIST_RAW = sys.argv[5].strip('[]').split(',')
	WIDTH_LIST = [int(element) for element in WIDTH_LIST_RAW]
	FILNUM_LIST_RAW = sys.argv[6].strip('[]').split(',')
	FILNUM_LIST = [int(element) for element in FILNUM_LIST_RAW]
	assert(len(HEIGHT_LIST) == NUM_CONV_LAYER)
	assert(len(WIDTH_LIST) == NUM_CONV_LAYER)
	assert(len(FILNUM_LIST) == NUM_CONV_LAYER)
	stamp = sys.argv[7]
except Exception, e:
	print('-- {}'.format(e))

print('-- Artist: {}'.format(artist))
print('-- FAST FLAG: {}'.format(FAST_FLAG))
print('-- Number of convolutional layers: {}'.format(NUM_CONV_LAYER))
print('-- Stamp (marker) for this run: {}'.format(stamp))
FILNUM_LIST.insert(0, 1) # setting first layer to have one input channel

# ================ Functions for preprocessing data ==========================
def preprocessQ(Q, downsamplingRate):
	'''
	Take in CQT matrix Q
	'''
	# take the abs to make it real, then log it
	Q = np.log10(1 + 1000000 *abs(Q))
	# downsample
	numCols = np.shape(Q)[1]
	# keep only 1 out of every downRate cols
	Q = Q[:,range(0, numCols, downsamplingRate)]
	return Q

def loadData(filepath, datapath, downsamplingRate):
	'''
	filepath: gives the path to a file which holds a list of filenames and labels, 
			each of which has one CQT matrix. It also holds the labels for each of these files
	datapath: gives a path to the folder containing the raw
			  CQT data (which the filepaths retrieved from the 'filepath' file)
				will descend from
	Load and return four variables from the file with path filepath
	X_train: input data for training, a stack of CQT matrices, already pre-processed
	y_train: labels for X_train
	X_val: input data for validation, a stack of CQT matrices, already pre-processed
	y_val: labels for X_val
	'''
	print('==> Experiment 11')
	print('==> Loading data from {}'.format(filepath))
	# benchmark
	t_start = time.time()

	# reading data
	fileAndLabelData = loadmat(filepath)

	trainFileList = [str(i[0]) for i in fileAndLabelData['trainFileList'][0]] # un-nest it
	valFileList = [str(i[0]) for i in fileAndLabelData['valFileList'][0]]

	y_train = np.array(fileAndLabelData['trainLabels']).squeeze().reshape((-1,1)) # reshape into cols
	y_val = np.array(fileAndLabelData['valLabels']).squeeze().reshape((-1,1))

	# loop through each of the training and validation files and pull the CQT out
	X_train = []
	for i in range(len(trainFileList)):
		print('Loading training data: file %d from %s'%(i + 1, datapath + trainFileList[i]))
		data = loadmat(datapath + trainFileList[i])
		rawQ = data['Q']['c'][0][0]
		X_train.append(preprocessQ(rawQ, downsamplingRate)) # not sure why so nested

	X_val = []

	for i in range(len(valFileList)):
		print('Loading validation data: file %d from %s'%(i + 1, datapath + valFileList[i]))
		data = loadmat(datapath + valFileList[i])
		rawQ = data['Q']['c'][0][0]
		X_val.append(preprocessQ(rawQ, downsamplingRate))


	t_end = time.time()
	print('--Time elapsed for loading data: {t:.2f} \
	seconds'.format(t = t_end - t_start))
	print('-- Number of training songs: {}'.format(len(X_train)))
	print('-- Number of validation songs: {}'.format(len(X_val)))
	return [X_train, y_train, X_val, y_val]

def loadReverbSamples(filepath):
	data = loadmat(filepath)
	return data['reverbSamples']

def getMiniBatch(cqtList, labelMatrix, batchNum, batchWidth, makeNoisy=False, reverbMatrix=[]):
	'''
	Inputs
		cqtList: a list, where each element is a cqt matrix
		batchNum: the number to get in this batch
		labelList: nxk matrix of labels
	'''
	# pick batchNum random songs to sample one sample from (allow repeats)
	songNums = np.random.randint(len(cqtList), size=batchNum)
	labels = (np.array(labelMatrix))[songNums, :] # grab the labels for each random song
	batch = np.zeros((batchNum, batchWidth * cqtList[0].shape[0]))
	# for each song, pull out a single sample
	for i in range(batchNum):
		songNum = songNums[i]
		curCQT = cqtList[songNum]
		startCol = np.random.randint(np.shape(curCQT)[1] - batchWidth)
		curSample = curCQT[:, startCol: startCol + batchWidth]
		if makeNoisy:
			curSample = addReverbNoise(curSample, reverbMatrix)
		# make it have a global mean of 0 before appending
		curSample = curSample - np.mean(curSample)
		# string out the sample into a row and add it to the batch
		curSample = np.reshape(curSample, (1,-1))
		batch[i, :] = curSample
	return [batch, labels]

def addReverbNoise(Q, reverbMatrix):
	# pick a random column and add it to each column of Q
	randCol = np.random.randint(np.shape(reverbMatrix)[1])
	col = np.reshape(reverbMatrix[:, randCol], (-1, 1)) # reshape because it converts to a row
	return Q + col

# ================== Functions for initializing neural nets parameters ===============================
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

print('==> Experiment 11b: CNN on Full Window with Delta Reverb...')
datapath = '/pylon2/ci560sp/cstrong/exp11/new_data/' + artist + '_out/'
filepath = datapath + 'FilesAndLabels.mat'
reverbpath = datapath + 'reverbSamples.mat'

# ==============================================
# 				reading data
# ==============================================
# benchmark
downsamplingRate = 12
sixSecFull = 1449
t_start = time.time()
print("==> Loading Training and Validation Data")
[X_train, y_train, X_val_raw, y_val_raw] = loadData(filepath, datapath, downsamplingRate)
print("==> Loading Reverb Data")
reverbSamples = loadReverbSamples(reverbpath)
t_end = time.time()
print('--Time elapsed for loading data: {t:.2f} \
		seconds'.format(t = t_end - t_start))

# ==============================================
# Neural-network model set-up
# ==============================================
num_frames = sixSecFull / downsamplingRate
num_freq = X_train[0].shape[0]
total_features = num_frames * num_freq
num_classes = int(max(y_train.max(), y_val_raw.max()) + 1)

# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_val_OHEnc = tf.one_hot(y_val_raw.copy(), num_classes)

# reset placeholders
x = tf.placeholder(tf.float32, [None, total_features])
y_ = tf.placeholder(tf.float32, [None, num_classes])

# ==============================================
# Convolutional layers
# ==============================================
W_conv_list = []
b_conv_list = []
h_list = [tf.reshape(x, [-1, num_freq, num_frames, 1])]
h_row_list = [num_freq]
h_col_list = [num_frames]

for layer_num in range(NUM_CONV_LAYER):
	r, c= HEIGHT_LIST[layer_num], WIDTH_LIST[layer_num]
	k_in, k_out = FILNUM_LIST[layer_num], FILNUM_LIST[layer_num + 1]
	W_conv = init_weight_variable([r, c, k_in, k_out]])
	b_conv = init_bias_variable([k_out])
	h_conv = tf.nn.relu(conv2d(h_list[-1], W_conv) + b_conv)
	h_pool = max_pool(h_conv, 2)
	hr = (h_row_list[layer_num] - r + 1) / 2
	hc = (h_col_list[layer_num] - c + 1) / 2
	W_conv_list.append(W_conv)
	b_conv_list.append(b_conv)
	h_list.append(h_pool)
	h_row_list.append(hr)
	h_col_list.append(hc)
h_pool_flat = tf.reshape(h_list[-1], [-1, hr * hc * k_out])

# ==============================================
# Dense layer
# ==============================================
nhidden = 1024
W_fc1 = init_weight_variable([hr * hc * k_out, nhidden])
b_fc1 = init_bias_variable([nhidden])
h_fc1 = tf.nn.relu(batch_nm(tf.matmul(h_pool_flat, W_fc1) + b_fc1))
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

# ==============================================
# Training
# ==============================================

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
y_train = sess.run(y_train_OHEnc)[:, 0, :]
y_val_raw = sess.run(y_val_OHEnc)[:, 0, :]

[X_val, y_val] = getMiniBatch(X_val_raw, y_val_raw, 1000, num_frames)

# evaluation metrics
train_err_list = []
val_err_list = []
val_mrr_list = []

# saver setup
varsave_list = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_sm, b_sm]
saver = tf.train.Saver(varsave_list)
save_path = './out/11bmodel_{}_{}.ckpt'.format(artist, stamp)
opt_val_err = np.inf
opt_iter = -1
step_counter = 0
max_counter = 200
batch_size = 256
max_iter = 10000
print_freq = 2

if FAST_FLAG:
	max_iter = 10
	print_freq = 1
print('==> Training the full network...')
t_start = time.time()
for num_iter in range(max_iter):
	if step_counter > max_counter:
		print('==> Step counter exceeds maximum value. Stop training at iter {}.'.format(num_iter + 1))
		break
	[train_batch_data, train_batch_label] = getMiniBatch(X_train, y_train, batch_size, num_frames, \
		makeNoisy=True, reverbMatrix=reverbSamples)
	train_step.run(feed_dict={x: train_batch_data, y_: train_batch_label, keep_prob: 0.5})
	if (num_iter + 1) % print_freq == 0:
		# evaluate metrics
		train_err = cross_entropy.eval(feed_dict={x: train_batch_data, y_: train_batch_label, keep_prob: 1.0})
		train_err_list.append(train_err)
		val_err = batch_eval(X_val, y_val, cross_entropy)
		val_err_list.append(val_err)
		val_mrr = MRR_batch(X_val, y_val)
		val_mrr_list.append(val_mrr)
		print("-- iter %d, training error %g, validation error %g"%(num_iter + 1, train_err, val_err))
		# save screenshot of the model
		if val_err < opt_val_err:
			step_counter = 0	
			saver.save(sess, save_path)
			print('==> New optimal validation error found. Model saved.')
			opt_val_err, opt_iter = val_err, num_iter + 1
	step_counter += 1

t_end = time.time()
print('--Time elapsed for training: {t:.2f} \
		seconds'.format(t = t_end - t_start))

# ==============================================
# Restore model & Evaluations
# ==============================================
saver.restore(sess, save_path)
print('==> Model restored to iter {}'.format(opt_iter))

model_path = './out/11b_{}_{}'.format(artist, stamp)
W1, W2 = sess.run([W_conv1, W_conv2])
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
plt.title('Error vs Number of Iterations for {}'.format(artist))
plt.legend(loc='best')
plt.savefig('./out/exp11b_{}_{}.png'.format(artist, stamp), format='png')
plt.close()
print('==> Finished!')
