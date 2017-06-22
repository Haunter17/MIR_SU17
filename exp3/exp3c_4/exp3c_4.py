import numpy as np
import tensorflow as tf
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


# Functions for initializing neural nets parameters
def init_weight_variable(shape, nameIn):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial, name=nameIn)

def init_bias_variable(shape, nameIn):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial, name=nameIn)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')

def loadData(filepath):
	'''
	Load and return four variables from the file with path filepath
	X_train: input data for training
	y_train: labels for X_train
	X_val: input data for validation
	y_val: labels for X_val
	'''
	print('==> Experiment 2l')
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
	def __init__(self, num_freq, X_train, y_train, X_val, y_val, filter_row, filter_col, k1, learningRate, debug):
		'''
		Initializer for the model
		'''
		# store the data
		self.X_train, self.y_train, self.X_val, self.y_val = X_train, y_train, X_val, y_val
		
		# store the parameters sent to init that define our model
		self.num_freq, self.filter_row, self.filter_col, self.k1, self.learningRate, self.debug = num_freq, filter_row, filter_col, k1, learningRate, debug

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
		num_training_vec, total_features = self.X_train.shape
		num_frames = int(total_features / self.num_freq)
		print('-- Num frames: {}'.format(num_frames))
		num_classes = int(max(self.y_train.max(), self.y_val.max()) + 1)
		l = num_frames

		# store what will be helpful later
		self.total_features = total_features
		self.num_training_vec = num_training_vec
		self.num_frames = num_frames
		self.num_classes = num_classes
		self.l = l

	@lazy_property
	def y_conv(self):
		# reshape the input into the form of a spectrograph 
		x_image = tf.reshape(self.x, [-1, self.num_freq, self.num_frames, 1])
		x_image = tf.identity(x_image, name="x_image")

		# first convolutional layer parameters
		W_conv1 = init_weight_variable([self.filter_row, self.filter_col, 1, self.k1], "W_conv1")
		b_conv1 = init_bias_variable([self.k1], "b_conv1")

		# tensor that computes the output of the first convolutional layer
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_conv1 = tf.identity(h_conv1, name="h_conv_1")
		
		# flatten out the output of the first convolutional layer to pass to the softmax layer
		h_conv1_flat = tf.reshape(h_conv1, [-1, (self.num_freq - self.filter_row + 1) * (self.num_frames - self.filter_col + 1) * self.k1])
		h_conv1_flat = tf.identity(h_conv1_flat, name="h_conv1_flat")

		# softmax layer parameters
		W_sm = init_weight_variable([(self.num_freq - self.filter_row + 1) * (self.num_frames - self.filter_col + 1) * self.k1, self.num_classes], "W_sm")
		b_sm = init_bias_variable([self.num_classes], "b_sm")

		# the output of the layer - un-normalized and without a non-linearity
		# since cross_entropy_with_logits takes care of that
		y_conv = tf.matmul(h_conv1_flat, W_sm) + b_sm
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

	def train(self, batch_size, num_epochs, print_freq, debug_out='debug.txt'):
		'''
		Train the Network on the data that will have been loaded when the NN is initialized
		Trained on: self.X_train, and a OH encoding of self.y_train
		Trains with batch_size batches for num_epochs epochs

		Debugging info is written to debug.txt (can add params to have more places to write out
			to)
		'''

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
		# track which epochs you record data during
		epoch_numbers = []

		# record the start time
		t_start = time.time()
		for epoch in range(num_epochs):
			epochStart = time.time()
			# train by systematically pulling batches of batch_size from
			# the training set and taking a training step on each batch
			for i in range(0, self.num_training_vec, batch_size):
			  batch_end_point = min(i + batch_size, self.num_training_vec)
			  train_batch_data = self.X_train[i : batch_end_point]
			  train_batch_label = y_trainOH[i : batch_end_point]
			  self.train_step.run(feed_dict={self.x: train_batch_data, self.y_: train_batch_label})
			epochEnd = time.time()

			# print and record data now that we've trained on our full training set
			if (epoch + 1) % print_freq == 0:
				# timing for the measurements of cost and accuracy
				evaluationStart = time.time()

				# compute training (on the most recent batch and the full data set)
				# and validation cost and accuracy, then print them and add them to the list
				# we start with accuracy:
				train_acc = self.evalByBatch(self.accuracy, X_train, y_trainOH, 5000)
				train_acc_list.append(train_acc)
				val_acc = self.evalByBatch(self.accuracy, X_val, y_valOH, 5000)
				val_acc_list.append(val_acc)
				# Now we compute the error on each set:
				train_err = self.evalByBatch(self.cross_entropy, X_train, y_trainOH, 5000)
				train_err_list.append(train_err)
				val_err = self.evalByBatch(self.cross_entropy, X_val, y_valOH, 5000)
				val_err_list.append(val_err)

				# keep track of which epochs we have data for
				epoch_numbers += [epoch]    
				# this marks the end of our evaluation
				evaluationEnd = time.time()
				# print a summary of our NN at this epoch
				print("epoch: %d, time (train, evaluation): (%g, %g), t acc, v acc, t cost, v cost: %.5f, %.5f, %.5f, %.5f"%(epoch+1, epochEnd - epochStart, evaluationEnd - evaluationStart, train_acc, val_acc, train_err, val_err))
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
		return [train_acc_list, val_acc_list, train_err_list, val_err_list, epoch_numbers]

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
    	myPlot.plot(epochs, trainingMetric, '.', label="Training")
    	myPlot.plot(epochs, validationMetric, '.', label="Validation")
    	myPlot.set_xlabel("Epoch Number")
    	myPlot.set_ylabel(metricName)
    	myPlot.set_title(titles[i])
    	myPlot.legend(loc="best", frameon=False)
    	# Write the figure
    	fig.savefig(filenames[i])

    	# update the figure with all the validation curves
    	validationPlot.plot(epochs, validationMetric, '.', label=(paramName + " = " + str(paramValues[i])))

    # finish labeling + write the validation plot
    validationPlot.set_xlabel("Epoch Number")
    validationPlot.set_ylabel(metricName)
    validationPlot.set_title(titles[-1])
    validationPlot.legend(loc="best", frameon=False)

    validationFig.savefig(filenames[-1])

def makeBestResultPlot(paramValues, trainingMetricLists, validationMetricLists, bestFunction, paramName, metricName, title, filename):
	'''
	Plot the "best" value of the training and validation metric against the param that led to it
	Best is assumed to be the largest value of the metric
	Input:
		trainingMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
		validationMetricLists: a list of lists, where each list represents some metric on the progress of training throughout training
		bestFunction: function that takes in a list (of values of the metric) and returns the "best" one. Often min or max will suffice
		paramName: name of the parameter you varied in this experiment (e.g. height of kernel)
		metricName: name of the metric you're using (e.g. cross-entropy error)
		title: the title of the graph (will include info on the params used)
		filename: the filename to write the graph to (will include info on the params used)
	Output:
		Write a png file with the desired plot
	Is there a way to call the other one to do this? if didn't assume epoch number then yes - oh well
	'''

	bestTrainingMetrics = [bestFunction(curList) for curList in trainingMetricLists]
	bestValidationMetrics = [bestFunction(curList) for curList in validationMetricLists]
	# make the figure, add plots, axis lables, a title, and legend
	fig = plt.figure(figsize=(7, 4))
	myPlot = fig.add_subplot(111)
	myPlot.plot(paramValues, bestTrainingMetrics, '.', label="Training")
	myPlot.plot(paramValues, bestValidationMetrics, '.', label="Validation")
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
	myPlot.plot(paramValues, finalTrainingMetrics, label="Training")
	myPlot.plot(paramValues, finalValidationMetrics, label="Validation")
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
	filterColsString = sys.argv[1]
	# map it from a string into a list of ints
	filterColsIn = map(int, filterColsString.strip('[]').split(','))

	# read in k1 as well
	k1sString = sys.argv[2]
	k1sIn = map(int, k1sString.strip('[]').split(','))

	# read in the learning rates
	learningRatesString = sys.argv[3]
	learningRatesIn = map(float, learningRatesString.strip('[]').split(','))

	# read in the number of epochs
	numEpochs = int(sys.argv[4])

	finalPlotName = sys.argv[5]
except Exception, e:
  print('-- {}'.format(e))



# filepath to the data you want to laod
filepath = '/pylon2/ci560sp/cstrong/exp3/exp3_taylorswift_d15_1s_C1C8.mat'

# define the configurations we're going to be looking at
# in this exp: just change the number of rows in a vertical kernel
filterCols = filterColsIn
filterRows = [1] * len(filterColsIn)

k1s = k1sIn
learningRates = learningRatesIn

# set training parameters
batchSize = 1000
print_freq = 1

# make lists to store data
train_acc_lists = []
val_acc_lists = []
train_err_lists = []
val_err_lists = []
epoch_number_lists = []

# load data
[X_train, y_train, X_val, y_val] = loadData(filepath)

# loop through the setups and make a model each time
for i in range(len(filterRows)):
	# create the model - this will create the TF graph as well as load the data
	m = Model(169, X_train, y_train, X_val, y_val, filterRows[i], filterCols[i], k1s[i], learningRates[i], False)
	# actually train the model (on the data it already loaded)
	[train_acc_list, val_acc_list, train_err_list, val_err_list, epoch_numbers] = m.train(1000, numEpochs, print_freq)
	# store the new data
	train_acc_lists.append(train_acc_list)
	val_acc_lists.append(val_acc_list)
	train_err_lists.append(train_err_list)
	val_err_lists.append(val_err_list)
	epoch_number_lists.append(epoch_numbers)

	del m # clear out the model to avoid huge buildup of memory

	# print what you have so far in case it crashes
	print("So far after %g models we have:"%(i+1))
	print("Filter Rows: %s"%(filterRows))
	print("Filter Cols: %s"%(filterCols))
	print("K1s: %s"%(k1s))
	print("Learning Rates: %s"%(learningRates))
	print("Train acc list: %s"%(str(train_acc_lists)))
	print("Val acc list: %s"%(str(val_acc_lists)))
	print("Train err list: %s"%(str(train_err_lists)))
	print("Val err list: %s"%(str(val_err_lists)))
	print("Epoch number lists: %s"%(str(epoch_number_lists)))

# printing
print("Filter Rows: %s"%(filterRows))
print("Filter Cols: %s"%(filterCols))
print("K1s: %s"%(k1s))
print("Learning Rates: %s"%(learningRates))
print("Train acc list: %s"%(str(train_acc_lists)))
print("Val acc list: %s"%(str(val_acc_lists)))
print("Train err list: %s"%(str(train_err_lists)))
print("Val err list: %s"%(str(val_err_lists)))
print("Epoch number lists: %s"%(str(epoch_number_lists)))
# plotting
trainingPlotTitles = ['Single Layer CNN with %gx%g kernels, k1=%g and LR=%g'%(filterRows[i], filterCols[i], k1s[i], learningRates[i]) for i in range(len(filterRows))]
trainingPlotTitles.append('Exp 3c_4, Validation Cross-Entropy Cost vs. Epoch')
trainingPlotFiles = ['exp3c_4_training_%gx%g_k1=%g_LR=%f_%gEpochs.png'%(filterRows[i], filterCols[i], k1s[i], learningRates[i], numEpochs) for i in range(len(filterRows))]
trainingPlotFiles.append('exp3c_4_validationCurves_%gEpochs'%(numEpochs))
makeTrainingPlots(epoch_number_lists[0], k1s, train_err_lists, val_err_lists, "k1", "Cross Entropy Cost", trainingPlotTitles, trainingPlotFiles)
makeBestResultPlot(k1s, train_err_lists, val_err_lists, min, "k1", "Cross Entropy Cost", 'Cost vs. k1', finalPlotName)

