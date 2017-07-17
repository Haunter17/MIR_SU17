import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import scipy.io as sp
from random import randint


# Functions for initializing neural nets parameters
def init_weight_variable(shape, nameIn):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial, name=nameIn)

def init_bias_variable(shape, nameIn):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial, name=nameIn)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')

def preprocessQ(Q, downsamplingRate):
	'''
	Take in CQT matrix Q
	'''
	# take the abs to make it real, then log it
	Q = np.log10(abs(Q))
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
	print('==> Experiment 11 =D')
	print('==> Loading data from {}'.format(filepath))
	# benchmark
	t_start = time.time()

	# reading data
	fileAndLabelData = sp.loadmat(filepath)

	trainFileList = [str(i[0]) for i in fileAndLabelData['trainFileList'][0]] # un-nest it
	valFileList = [str(i[0]) for i in fileAndLabelData['valFileList'][0]]

	y_train = np.array(fileAndLabelData['trainLabels']).squeeze().reshape((-1,1)) # reshape into cols
	y_val = np.array(fileAndLabelData['valLabels']).squeeze().reshape((-1,1))

	# loop through each of the training and validation files and pull the CQT out
	X_train = []
	for i in range(len(trainFileList)):
		print('Loading training data: file %d from %s'%(i, datapath + trainFileList[i]))
		data = sp.loadmat(datapath + trainFileList[i])
		rawQ = data['Q']['c'][0][0]
		print(np.shape(rawQ))
		print(np.shape(preprocessQ(rawQ, downsamplingRate)))
		X_train.append(preprocessQ(rawQ, downsamplingRate)) # not sure why so nested

	X_val = []

	for i in range(len(valFileList)):
		print('Loading validation data: file %d from %s'%(i, datapath + valFileList[i]))
		data = sp.loadmat(datapath + valFileList[i])
		rawQ = data['Q']['c'][0][0]
		X_val.append(preprocessQ(rawQ, downsamplingRate))


	t_end = time.time()
	print('--Time elapsed for loading data: {t:.2f} \
	seconds'.format(t = t_end - t_start))
	print('-- Number of training songs: {}'.format(len(X_train)))
	print('-- Number of validation songs: {}'.format(len(X_val)))
	return [X_train, y_train, X_val, y_val]

def loadReverbSamples(filepath):
	data = sp.loadmat(filepath)
	return data['reverbSamples']

def getMiniBatch(cqtList, labelMatrix, batchNum, batchWidth, makeNoisy, reverbMatrix=[]):
	'''
	Inputs
		cqtList: a list, where each element is a cqt matrix
		batchNum: the number to get in this batch
		labelList: nxk matrix of labels
	'''
	# pick batchNum random songs to sample one sample from (allow repeats)
	songNums = [randint(0,len(cqtList) - 1) for i in range(batchNum)]
	labels = (np.array(labelMatrix))[songNums, :] # grab the labels for each random song
	batch = np.array([])
	# for each song, pull out a single sample
	for i in range(batchNum):
		songNum = songNums[i]
		curCQT = cqtList[songNum]
		startCol = randint(0, np.shape(curCQT)[1] - batchWidth)
		curSample = curCQT[:, startCol: startCol + batchWidth]
		if makeNoisy:
			curSample = addReverbNoise(curSample, reverbMatrix)
		# make it have a global mean of 0 before appending
		curSample = curSample - np.mean(curSample)
		# string out the sample into a row and add it to the batch
		curSample = np.reshape(curSample, (1,-1))
		batch = np.vstack((batch, curSample)) if batch.size else curSample # if and else to deal the first loop through when batch is empty
	return [batch, labels]

# x = np.array([[1,2,3,4],[5,6,7,8]])
# y = np.array([[9,10,11,12], [15,17,18,23]])
# [z,l] = getMiniBatch([x,y],[[0,1],[1,0]], 10, 3, False)

def addReverbNoise(Q, reverbMatrix):
	# pick a random column and add it to each column of Q
	randCol = randint(0, np.shape(reverbMatrix)[1] - 1)
	col = np.reshape(reverbMatrix[:, randCol], (-1, 1)) # reshape because it converts to a row
	return Q + col


#self, X_train, y_train, X_val, y_val, num_freq, filter_row, filter_col, k1, k2, learningRate, pooling_strategy):

# set up property that makes it only be set once
# we'll use this to avoid adding tensors to the graph multiple times
import functools
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model:
	def __init__(self, num_frames, X_train, y_train, X_val, y_val, reverbSamples, filter_row, filter_col, k1, learningRate, debug):
		'''
		Initializer for the model
		'''
		# store the data
		self.X_train, self.y_train, self.X_val, self.y_val, self.reverbSamples = X_train, y_train, X_val, y_val, reverbSamples
		
		# store the parameters sent to init that define our model
		self.num_frames, self.filter_row, self.filter_col, self.k1, self.learningRate, self.debug = num_frames, filter_row, filter_col, k1, learningRate, debug

		# find num_training_vec, total_features, num_frames, num_classes, and l from the shape of the data
		# and store them
		self.storeParamsFromData()

		# Set-up and store the input and output placeholders
		x = tf.placeholder(tf.float32, [None, self.total_features])
		y_ = tf.placeholder(tf.float32, [None, self.num_classes])
		self.x = x
		self.y_ = y_

		# Setup and store tensor that performs the one-hot encoding
		y_train_OHEnc = tf.one_hot(self.y_train.copy(), self.num_classes)
		y_val_OHEnc = tf.one_hot(self.y_val.copy(), self.num_classes)
		self.y_train_OHEnc = y_train_OHEnc
		self.y_val_OHEnc = y_val_OHEnc

		# create each lazy_property
		# each lazy_property will add tensors to the graph
		self.y_conv
		self.cross_entropy
		self.train_step
		self.accuracy
		
		# properties for use in debugging
		if self.debug:
			self.grads_and_vars

		# print to the user that the network has been set up, along with its properties
		print("Setting up Single Conv Layer Neural net with %g x %g filters, k1 = %g, learningRate = %g"%(filter_row, filter_col, k1, learningRate))

	def storeParamsFromData(self):
		'''
		Calculate and store parameters from the raw data

		total_features: The number of CQT coefficients total (incldues all context frames)
		num_training_vec: The number of training examples in your dataset
		num_frames: The number of context frames in each training example (total_features / num_freq)
		num_classes: The number of songs we're distinguishing between in our output
		l: The length of our second convolutional kernel - for now, its equal to num_frames
		'''
		# Neural-network model set-up
		# calculating some values which will be nice as we set up the model
		num_freq = self.X_train[0].shape[0]
		total_features = num_freq * self.num_frames
		print('-- Num freq: {}'.format(num_freq))
		num_classes = int(max(self.y_train.max(), self.y_val.max()) + 1)

		# store what will be helpful later
		self.num_freq = num_freq
		self.total_features = total_features
		self.num_classes = num_classes

	@lazy_property
	def y_conv(self):
		# reshape the input into the form of a spectrograph 
		x_image = tf.reshape(self.x, [-1, self.num_freq, self.num_frames, 1])
		x_image = tf.identity(x_image, name="x_image")

		# first convolutional layer parameters
		self.W_conv1 = init_weight_variable([self.filter_row, self.filter_col, 1, self.k1], "W_conv1")
		self.b_conv1 = init_bias_variable([self.k1], "b_conv1")

		# tensor that computes the output of the first convolutional layer
		h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
		h_conv1 = tf.identity(h_conv1, name="h_conv_1")
		
		# flatten out the output of the first convolutional layer to pass to the softmax layer
		h_conv1_flat = tf.reshape(h_conv1, [-1, (self.num_freq - self.filter_row + 1) * (self.num_frames - self.filter_col + 1) * self.k1])
		h_conv1_flat = tf.identity(h_conv1_flat, name="h_conv1_flat")

		# softmax layer parameters
		self.W_sm = init_weight_variable([(self.num_freq - self.filter_row + 1) * (self.num_frames - self.filter_col + 1) * self.k1, self.num_classes], "W_sm")
		self.b_sm = init_bias_variable([self.num_classes], "b_sm")

		# the output of the layer - un-normalized and without a non-linearity
		# since cross_entropy_with_logits takes care of that
		y_conv = tf.matmul(h_conv1_flat, self.W_sm) + self.b_sm
		y_conv = tf.identity(y_conv, name="y_conv")
		
		return  y_conv # would want to softmax it to get an actual prediction

	@lazy_property
	def cross_entropy(self):
		'''
		Create a tensor that computes the cross entropy cost
		Use the placeholder y_ as the labels, with input y_conv
		
		Note that softmax_cross_entropy_with_logits takes care of normalizing
		y_conv to make it a probability distribution

		This tensor can be accessed using: self.cross_entropy
		'''
		cross_entropy = tf.reduce_mean(
		    tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
		cross_entropy = tf.identity(cross_entropy, name="cross_entropy")
		return cross_entropy

	@lazy_property
	def optimizer(self):
		'''
		Create a tensor that represents the optimizer. This tensor can
		be accessed using: self.optimizer
		'''
		optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
		return optimizer

	@lazy_property
	def train_step(self):
		'''
		Creates a tensor that represents a single training step. This tensor
		can be passed a feed_dict that has x and y_, and it will compute the gradients
		and perform a single step.

		This tensor can be accessed using: self.train_step
		'''
		return self.optimizer.minimize(self.cross_entropy)

	@lazy_property
	def accuracy(self):
		'''
		Create a tensor that computes the accuracy, using the placeholder y_ as the labeled data
		and y_conv for the predictions of the network.

		This tensor can be accessed using: self.accuracy
		'''
		correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy

	'''
	Properties that we'll use for debugging
	'''
	@lazy_property
	def grads_and_vars(self):
  		grads_and_vars = self.optimizer.compute_gradients(self.cross_entropy, tf.trainable_variables())
		return grads_and_vars

	def train(self, batch_size, num_batches, print_freq, debug_out='debug.txt'):
		'''
		Train the Network on the data that will have been loaded when the NN is initialized
		Trained on: self.X_train, and a OH encoding of self.y_train
		Trains with batch_size batches for num_epochs epochs

		Debugging info is written to debug.txt (can add params to have more places to write out
			to)
		'''
		print("==> Entered Training")
		# Starting an interactive session and initializing the parameters
		#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
		sess = tf.InteractiveSession()

		sess.run(tf.global_variables_initializer())
		
		# replace it with the one-hot encoded one --- should I replace?
		y_trainOH = sess.run(self.y_train_OHEnc)[:, 0, :]
		y_valOH = sess.run(self.y_val_OHEnc)[:, 0, :]
		
		# lists to record accuracy at several points during training
		train_acc_list = []
		val_acc_list = []
		train_acc_on_batch_list = []
		# lists to record the error at several points during training
		train_err_list = []
		val_err_list = []
		train_err_on_batch_list = []
		# track which batches you record data during
		batch_numbers = []

		# record the start time
		t_start = time.time()

		# keep track of the best weights and biases
		best_weights = [self.W_conv1.eval(), self.W_sm.eval()]
		best_biases = [self.b_conv1.eval(), self.b_sm.eval()]
		# validation batch size just to get statistics with
		valStatisticBatchSize = 500
		[val_batch_data, val_batch_label] = getMiniBatch(X_val, y_valOH, valStatisticBatchSize, self.num_frames, False)
		bestValidationError = self.evalByBatch(self.cross_entropy, val_batch_data, val_batch_label, 5000);

		for batchNum in range(num_batches):
			print('Batch %g'%(batchNum))
			batchStart = time.time()
			fetchMiniStart = time.time()
			[train_batch_data, train_batch_label] = getMiniBatch(X_train, y_trainOH, batch_size, self.num_frames, True, self.reverbSamples) 
			fetchMiniEnd = time.time()
			self.train_step.run(feed_dict={self.x: train_batch_data, self.y_: train_batch_label})
			batchEnd = time.time()

			print(batchEnd - batchStart)
			# print and record data now that we've trained on our full training set
			if (batchNum + 1) % print_freq == 0:
				# timing for the measurements of cost and accuracy
				evaluationStart = time.time()

				# val batch for evaluation
				[val_batch_data, val_batch_label] = getMiniBatch(X_val, y_valOH, valStatisticBatchSize, self.num_frames, False)

				# evaluate training error and accuracy only on the most recent batch

				# start with accuracy
				train_acc = self.evalByBatch(self.accuracy, train_batch_data, train_batch_label, 5000)
				train_acc_list.append(train_acc)
				val_acc = self.evalByBatch(self.accuracy, val_batch_data, val_batch_label, 5000)
				val_acc_list.append(val_acc)
				# Now we compute the error on each set:
				train_err = self.evalByBatch(self.cross_entropy, train_batch_data, train_batch_label, 5000)
				train_err_list.append(train_err)
				val_err = self.evalByBatch(self.cross_entropy, val_batch_data, val_batch_label, 5000)
				val_err_list.append(val_err)

				# if the validation error is better then store the current weights
				if (val_err < bestValidationError):
					# update the best error, weights, and biases
					print('new best =D ');
					bestValidationError = val_err
					best_weights = [self.W_conv1.eval(), self.W_sm.eval()]
					best_biases = [self.b_conv1.eval(), self.b_sm.eval()]

				# keep track of which epochs we have data for
				batch_numbers += [batchNum]    
				# this marks the end of our evaluation
				evaluationEnd = time.time()
				# print a summary of our NN at this epoch
				print("batch: %d, time (train, evaluation, fetchMini): (%g, %g, %g), t acc, v acc, t cost, v cost: %.5f, %.5f, %.5f, %.5f"%(batchNum+1, batchEnd - batchStart, evaluationEnd - evaluationStart, fetchMiniEnd - fetchMiniStart, train_acc, val_acc, train_err, val_err))
			# debugging print outs
			if self.debug:
				# print out step / current value ratio for each parameter in our network
				# based on training data from the most recent batch
				# to the file with name debug_out
				self.debug_WriteGradAndVar(train_batch_data, train_batch_label, epoch, debug_out)

		# record the total time spent training the neural network
		t_end = time.time()
		print('--Time elapsed for training for %g epochs: %g'%(num_epochs, t_end - t_start))

		# return the lists of logged data
		return [train_acc_list, val_acc_list, train_err_list, val_err_list, epoch_numbers, best_weights, best_biases]

	def evalByBatch(self, toEval, x, y_, batchSize):
		weightedAvg = 0.0

		for i in range(0, len(x), batchSize):
			batch_end_point = min(i + batchSize, len(x))
			batch_data = x[i : batch_end_point]
			batch_label = y_[i : batch_end_point]
			curAmount = toEval.eval(feed_dict={self.x: batch_data, self.y_: batch_label})
			# weight by the length of the batch and keep adding on
			weightedAvg = weightedAvg + curAmount * float(batch_end_point - i) / len(x)
		return weightedAvg

	def debug_WriteGradAndVar(self, xDebug, yDebug, epoch, debug_out):
		'''
		Helper function that prints the ratio of the training step that would be taken
		on input data and labels xDebug and yDebug to the magnitude of each parameter
		in the network. This gives us a sense of how much each parameter is changing.

		Inputs:
			xDebug: input data to calculate the gradient from
			yDebug: labels for the input data
			epoch: the number of the epoch (to print out to the file)
			debug_out: the file to write to - if it doesn't exist it will be created
		'''
		file_object = open(debug_out, 'a+')

		# record which epoch this is
		file_object.write("Epoch: %d\n"%(epoch))
		# find the current learning rate - this will be used with the gradient to find the step size
		curLearningRate = self.optimizer._lr
		# print each gradient and the variables they are associated with
		# the gradients are stored in tuples, where the first element is a tensor
		# that computes the gradient, and the second is the parameter that gradient
		# is associated with
		for gv in self.grads_and_vars:

			curGrads = gv[0].eval(feed_dict={self.x: xDebug, self.y_: yDebug})
			curSteps = curGrads * curLearningRate # scale down the graident by the learning rate
			curVars = gv[1].eval()

			# How much, compared to the magnitude of the weight, are we stepping
			stepToVarRatio = np.absolute(np.divide(curSteps, curVars))
			  
			# print the name of the variable, then all the step ratios (step amount / current value)
			# these values will have been averaged across the training examples
			curName = gv[1].name
			file_object.write("Variable: " + curName + "\n")
			for index, step in np.ndenumerate(stepToVarRatio):
			  file_object.write(str(index) + ": " + str(step) + "\n")
			
			# print summary statistics for this layer
			maxVal = np.amax(stepToVarRatio)
			thirdQuartile = np.percentile(stepToVarRatio, 75)
			mean = np.mean(stepToVarRatio)
			median = np.median(stepToVarRatio)
			firstQuartile = np.percentile(stepToVarRatio, 25)
			minVal = np.amin(stepToVarRatio)

			file_object.write("Statistics: (%g, %g, %g, %g, %g, %g)\n"%(minVal, firstQuartile, median, mean, thirdQuartile, maxVal))
			file_object.write("---------------------------------------\n")

		# close the file
		file_object.close()




def makeTrainingPlots(epochs, paramValues, trainingMetricLists, validationMetricLists, paramName, metricName, titles, filenames):
    '''
    Plots of the given training and validation metrics versus epoch number. One plot per list 
    in trainingMetricLists and validationMetricLists. Assume there will be the same number of sublists
    in both those parameters. Titles will hold a list of strings that will be used for the titles
    of the graphs. The last title will be for the plot with all the validation curves. Filenames is a list of filenames to save your plots to
    Input:
            epochs: a list of the epochs on which data was taken - assume all of them took
            data at the same epoch numbers
            paramValues: the values of the param that we were varying (to label the curves in our validation plot)
            trainingMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
            validationMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
            paramName: name of the parameter you're varying (e.g. learningRate or kernel height)
            metricName: the name of the metric (e.g. accuracy, or cross-entropy error), to be used on the y-axis
            titles: titles for the graph (will include info on the params used). 
            		*The last title will be for the validation plot
            filename: the filenames to write the graphs to (will include info on the params used)
            		  * the last filename will be for the validation plot
    Output:
            Write a png file for each list in trainingMetricLists/validationMetricLists with the desired plot
    '''

    # figure with all the validation curves
    validationFig = plt.figure(figsize=(7, 4))
    validationPlot = validationFig.add_subplot(111)

    # go through each setup and make a plot for each
    for i in range(len(trainingMetricLists)):
    	# pull out the list we're concerned with
    	trainingMetric = trainingMetricLists[i]
    	validationMetric = validationMetricLists[i]
    	# make the figure, add plots, axis lables, a title, and legend
    	fig = plt.figure(figsize=(7, 4))
    	myPlot = fig.add_subplot(111)
    	myPlot.plot(epochs, trainingMetric, '-', label="Training")
    	myPlot.plot(epochs, validationMetric, '-', label="Validation")
    	myPlot.set_xlabel("Epoch Number")
    	myPlot.set_ylabel(metricName)
    	myPlot.set_title(titles[i])
    	myPlot.legend(loc="best", frameon=False)
    	# Write the figure
    	fig.savefig(filenames[i])

    	# update the figure with all the validation curves
    	validationPlot.plot(epochs, validationMetric, '-', label=(paramName + " = " + str(paramValues[i])))

    # finish labeling + write the validation plot
    validationPlot.set_xlabel("Epoch Number")
    validationPlot.set_ylabel(metricName)
    validationPlot.set_title(titles[-1])
    validationPlot.legend(loc="best", frameon=False)

    validationFig.savefig(filenames[-1])

def makeBestResultPlot(paramValues, trainingMetricLists, validationMetricLists, paramName, metricName, title, filename):
	'''
	Plot the "best" value of the training and validation metric against the param that led to it
	Best is assumed to be the largest value of the metric
	Input:
		trainingMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
		validationMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
		paramName: 
		metricName: 
		title: the title of the graph (will include info on the params used)
		filename: the filename to write the graph to (will include info on the params used)
	Output:
		Write a png file with the desired plot
	Is there a way to call the other one to do this? if didn't assume epoch number then yes - oh well
	'''

	bestTrainingMetrics = [max(curList) for curList in trainingMetricLists]
	bestValidationMetrics = [max(curList) for curList in validationMetricLists]
	# make the figure, add plots, axis lables, a title, and legend
	fig = plt.figure(figsize=(7, 4))
	myPlot = fig.add_subplot(111)
	myPlot.plot(paramValues, bestTrainingMetrics, '-', label="Training")
	myPlot.plot(paramValues, bestValidationMetrics, '-', label="Validation")
	myPlot.set_xlabel(paramName)
	myPlot.set_ylabel(metricName)
	myPlot.set_title(title)
	myPlot.legend(loc="best", frameon=False)
	# Write the figure
	fig.savefig(filename)



def makeEndResultPlot(paramValues, trainingMetricLists, validationMetricLists, paramName, metricName, title, filename):
	'''
	Plot the final value of the training and validation metric against the param that led to it
	Input:
		trainingMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
		validationMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
		paramName: 
		metricName: 
		title: the title of the graph (will include info on the params used)
		filename: the filename to write the graph to (will include info on the params used)
	Output:
		Write a png file with the desired plot
	Is there a way to call the other one to do this? if didn't assume epoch number then yes - oh well
	'''

	finalTrainingMetrics = [curList[-1] for curList in trainingMetricLists]
	finalValidationMetrics = [curList[-1] for curList in validationMetricLists]
	# make the figure, add plots, axis lables, a title, and legend
	fig = plt.figure(figsize=(7, 4))
	myPlot = fig.add_subplot(111)
	myPlot.plot(paramValues, finalTrainingMetrics, '-', label="Training")
	myPlot.plot(paramValues, finalValidationMetrics, '-', label="Validation")
	myPlot.set_xlabel(paramName)
	myPlot.set_ylabel(metricName)
	myPlot.set_title(title)
	myPlot.legend(loc="best", frameon=False)
	# Write the figure
	fig.savefig(filename)

'''
Our main, with 121x1 convolutional layer.
'''

# read in command line parameters
try:
	# read in a list of the row numbers for the kernels
	filterRowsString = sys.argv[1]
	filterRowsIn = map(int, filterRowsString.strip('[]').split(','))
	# read in a list of the col numbers for the kernels
	filterColsString = sys.argv[2]
	# map it from a string into a list of ints
	filterColsIn = map(int, filterColsString.strip('[]').split(','))
	# read in the number of epochs

	k1sString = sys.argv[3]
	k1sIn = map(int, k1sString.strip('[]').split(','))

	numBatches = int(sys.argv[4])

	finalPlotName = sys.argv[5]
except Exception, e:
  print('-- {}'.format(e))



# filepath to the data you want to laod
filepath = '/pylon2/ci560sp/cstrong/exp11/new_data/deathcabforcutie_out/FilesAndLabels.mat'
datapath = '/pylon2/ci560sp/cstrong/exp11/'
reverbPath = '/pylon2/ci560sp/cstrong/exp11/new_data/deathcabforcutie_out/reverbSamples_dcfc.mat'

# define the configurations we're going to be looking at
# in this exp: just change the number of rows in a vertical kernel
filterCols = filterColsIn
filterRows = filterRowsIn
k1s = k1sIn

learningRates = [0.0005] * len(filterRows)

# set training parameters
batchSize = 256
print_freq = 5

# make lists to store data
train_acc_lists = []
val_acc_lists = []
train_err_lists = []
val_err_lists = []
epoch_number_lists = []
best_weights_lists = []
best_biases_lists = []

# load data
downsamplingRate = 12
sixSecondsNoDownsampling = 1449

print("==> Loading Training and Validation Data")
[X_train, y_train, X_val, y_val] = loadData(filepath, datapath, downsamplingRate)
print("==> Loading Reverb Data")
reverbSamples = loadReverbSamples(reverbPath)


# loop through the setups and make a model each time
for i in range(len(filterRows)):
	# create the model - this will create the TF graph as well as load the data
	m = Model(int(np.floor(sixSecondsNoDownsampling) / downsamplingRate), X_train, y_train, X_val, y_val, reverbSamples, filterRows[i], filterCols[i], k1s[i], learningRates[i], False)
	# actually train the model (on the data it already loaded)
	[train_acc_list, val_acc_list, train_err_list, val_err_list, epoch_numbers, best_weights, best_biases] = m.train(batchSize, numBatches, print_freq)
	# store the new data
	train_acc_lists.append(train_acc_list)
	val_acc_lists.append(val_acc_list)
	train_err_lists.append(train_err_list)
	val_err_lists.append(val_err_list)
	epoch_number_lists.append(epoch_numbers)
	best_weights_lists.append(best_weights)
	best_biases_lists.append(best_biases)
	# grab these to store when we save the model
	num_freq = m.num_freq
	num_frames = m.num_frames

	del m # clear out the model to avoid huge buildup of memory

# printing
print("filterRows = %s"%(filterRows))
print("filterCols = %s"%(filterCols))
print("k1s = %s"%(k1s))
print("learningRates = %s"%(learningRates))
print("train_acc_lists = %s"%(str(train_acc_lists)))
print("val_acc_lists = %s"%(str(val_acc_lists)))
print("train_err_lists = %s"%(str(train_err_lists)))
print("val_err_lists = %s"%(str(val_err_lists)))
print("epoch_number_lists = %s"%(str(epoch_number_lists)))
print("best_weights_lists = %s"%(str(best_weights_lists)))
print("best_biases_lists = %s"%(str(best_biases_lists)))
bestValues = [min(curList) for curList in val_err_lists]
print("bestValues = %s"%str(bestValues))

modelFiles = ['exp9_Model_SingleLayerCNN_%gx%g_k1=%g_LR=%f_Epochs=%g'%(filterRows[i], filterCols[i], k1s[i], learningRates[i], numEpochs) for i in range(len(filterRows))]
for i in range(len(filterRows)):
	learningRate = learningRates[i]
	# pull out the weights - separate into convolutional and softmax weights
	curWeights = best_weights_lists[i]
	curW_conv1 = np.squeeze(curWeights[0], axis=2)
	curW_sm = np.squeeze(curWeights[1])

	curBiases = best_biases_lists[i]
	# just take the number of frames and frequencies from the most recent model, since they should all be the same
	sp.savemat(modelFiles[i], {'W_conv1': curW_conv1, 'b_conv1': curBiases[0], 'W_sm': curW_sm, 'b_sm': curBiases[1], 'rows_in_region': num_freq, 'cols_in_region' : num_frames, 'learning_rate': learningRate, 'epochs': numEpochs, 'lowest_val_err': bestValues[i]})

# plotting
trainingPlotTitles = ['Single Layer CNN with %gx%g kernels and k1=%g, LR=%f'%(filterRows[i], filterCols[i], k1s[i], learningRates[i]) for i in range(len(filterRows))]
trainingPlotTitles.append('Exp 9 Single Layer CNN, Validation Cross-Entropy Cost vs. Epoch')
trainingPlotFiles = ['exp9_training_%gx%g_k1=%g_LR=%f_%gEpochs.png'%(filterRows[i], filterCols[i], k1s[i], learningRates[i], numEpochs) for i in range(len(filterRows))]
trainingPlotFiles.append('exp9_validationCurves_%gEpochs'%(numEpochs))
makeTrainingPlots(epoch_number_lists[0], zip(filterRows, filterCols), train_err_lists, val_err_lists, "Shape of Kernel", "Cross Entropy Cost", trainingPlotTitles, trainingPlotFiles)



