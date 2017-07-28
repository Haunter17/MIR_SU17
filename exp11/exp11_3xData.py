import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from scipy.io import savemat

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

def getMiniBatch(cqtList, labelMatrix, batchNum, batchWidth, makeNoisy=False, reverbMatrix=[], addAWGN=False, SNR=0):
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
        if addAWGN:
			curSample = addWhiteNoise(curSample, SNR)
            
		# make it have a global mean of 0 before appending
		curSample = curSample - np.mean(curSample)
		# string out the sample into a row and add it to the batch
        # IMPORTANT: Transpose is added for MLP code only.
		curSample = np.reshape(np.transpose(curSample), (1,-1))
		batch[i, :] = curSample
	return [batch, labels]

def addReverbNoise(Q, reverbMatrix):
	# pick a random column and add it to each column of Q
	randCol = np.random.randint(np.shape(reverbMatrix)[1])
	col = np.reshape(reverbMatrix[:, randCol], (-1, 1)) # reshape because it converts to a row
	return Q + col
    
def addWhiteNoise(Q, SNR):
	# find the power of Q, use it to set the power of the noise (which decides the variance)
	Pq = np.mean(np.square(Q)) # power equal to the average of the coefficients squared
	Pn = Pq / (10.**(SNR / 10.))
	variance = np.sqrt(Pn)
	# generate and add on noise
	noise = np.random.normal(0, variance, Q.shape)
	return Q + noise

# Functions for initializing neural nets parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

def MRR_batch(data, label, batch_size=500):
	value = 0.
	for i in range(0, data.shape[0], batch_size):
		batch_end_point = min(i + batch_size, data.shape[0])
		batch_data = data[i : batch_end_point]
		batch_label = label[i : batch_end_point]
		batch_pred = sess.run(y, feed_dict={x: batch_data, y_: batch_label})
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
# Download data from .mat file into numpy array
print('==> Experiment 11')
artist = 'deathcabforcutie'
datapath = '/pylon2/ci560sp/mint96/' + artist + '_out/'
filepath = datapath + 'FilesAndLabels.mat'
reverbpath = datapath + 'reverbSamples.mat'

downsamplingRate = 7
#sixSecFull = 1449
t_start = time.time()
print("==> Loading Training and Validation Data")
[X_train, y_train, X_val, y_val] = loadData(filepath, datapath, downsamplingRate)
print("==> Loading Reverb Data")
reverbSamples = loadReverbSamples(reverbpath)
t_end = time.time()
print('--Time elapsed for loading data: {t:.2f} \
		seconds'.format(t = t_end - t_start))

'''
    NN config parameters
'''

numFreqBins = 121
sub_window_size = 32  
num_features = numFreqBins*sub_window_size
num_frames = 32
hidden_layer_size = 64
num_classes = 87
print("Number of features:", num_features)
print("Number of songs:",num_classes)

# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), num_classes)
y_val_OHEnc = tf.one_hot(y_val.copy(), num_classes)

plotx = []
ploty_train = []
ploty_val = []
plot_val_mrr = []
        
 # Set-up NN layers
x = tf.placeholder(tf.float64, [None, num_features])
W1 = weight_variable([num_features, hidden_layer_size])
b1 = bias_variable([hidden_layer_size])

OpW1 = tf.placeholder(tf.float64, [num_features, hidden_layer_size])
Opb1 = tf.placeholder(tf.float64, [hidden_layer_size])

# Hidden layer activation function: ReLU
h = tf.matmul(x, W1) + b1
h1 = tf.nn.relu(h)

W2 = weight_variable([hidden_layer_size, num_classes])
b2 = bias_variable([num_classes])

OpW2 = tf.placeholder(tf.float64, [hidden_layer_size, num_classes])
Opb2 = tf.placeholder(tf.float64, [num_classes])

# Softmax layer (Output)
y = tf.matmul(h1, W2) + b2

# NN desired value (labels)
y_ = tf.placeholder(tf.float64, [None, num_classes])

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
sess.run(tf.global_variables_initializer())

y_train = sess.run(y_train_OHEnc)[:, 0, :]
y_val = sess.run(y_val_OHEnc)[:, 0, :]

[X_val, y_val] = getMiniBatch(X_val, y_val, 1000, num_frames)

# Training
numTrainingVec = len(X_train)
batch_size = 300
max_iter = 50
print_freq = 2

startTime = time.time()
for num_iter in range(max_iter):
	# Get batch data for training. Comment out if not needed.
    # Get clean audio batch data
    [dataClean, labelClean] = getMiniBatch(X_train, y_train, batch_size, num_frames, \
        makeNoisy=False, reverbMatrix=reverbSamples,  addAWGN=False, SNR=10)

	# Get reverb-added batch data
    [dataRev, labelRev] = getMiniBatch(X_train, y_train, batch_size, num_frames, \
        makeNoisy=True, reverbMatrix=reverbSamples,  addAWGN=False, SNR=10)

	# Get AWGN-added batch data
    [dataAwgn, labelAwgn] = getMiniBatch(X_train, y_train, batch_size, num_frames, \
        makeNoisy=False, reverbMatrix=reverbSamples,  addAWGN=True, SNR=10)

	# Concatenate all training data and labels together
    trainBatchData = np.concatenate((dataClean, dataRev, dataAwgn), axis=0)
    trainBatchLabel = np.concatenate((labelClean, labelRev, labelAwgn), axis=0)
    train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

    # Print accuracy
    if (num_iter + 1) % print_freq == 0:
        plotx.append(num_iter)
        train_error = cross_entropy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
        val_error = cross_entropy.eval(feed_dict={x:X_val, y_: y_val})
        train_acc = accuracy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
        val_acc = accuracy.eval(feed_dict={x:X_val, y_: y_val})
        val_mrr = MRR_batch(X_val, y_val)
		plot_val_mrr.append(val_mrr)
        ploty_train.append(train_error)
        ploty_val.append(val_error)
        print("Step: %d, train acc %g, val acc %g, train error %g, val error %g, val mrr %g"%(num_iter, train_acc, val_acc, train_error, val_error, val_mrr))

endTime = time.time()
print("Elapse Time:", endTime - startTime)
print("Best validation error: %g at epoch %d"%(bestValErr, bestValStep))

saveweight = {}
saveweight['W1'] = np.array(W1.eval())
saveweight['b1'] = np.array(b1.eval())

savemat('exp11_s7_3x_snr10_weight.mat',saveweight)

print('==> Generating error plot...')
errfig = plt.figure()
trainErrPlot = errfig.add_subplot(111)
trainErrPlot.set_xlabel('Number of Epochs')
trainErrPlot.set_ylabel('Cross-Entropy Error')
trainErrPlot.set_title('Error vs Number of Epochs')
trainErrPlot.scatter(plotx, ploty_train)
valErrPlot = errfig.add_subplot(111)
valErrPlot.scatter(plotx, ploty_val)
errfig.savefig('exp11_s7_3x_snr10.png')