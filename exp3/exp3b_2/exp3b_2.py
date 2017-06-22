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
    	myPlot.plot(epochs, trainingMetric, label="Training")
    	myPlot.plot(epochs, validationMetric, label="Validation")
    	myPlot.set_xlabel("Epoch Number")
    	myPlot.set_ylabel(metricName)
    	myPlot.set_title(titles[i])
    	myPlot.legend(loc="best", frameon=False)
    	# Write the figure
    	fig.savefig(filenames[i])

    	# update the figure with all the validation curves
    	validationPlot.plot(epochs, validationMetric, label=(paramName + " = " + str(paramValues[i])))

    # finish labeling + write the validation plot
    validationPlot.set_xlabel("Epoch Number")
    validationPlot.set_ylabel(metricName)
    validationPlot.set_title(titles[-1])
    validationPlot.legend(loc="best", frameon=False)

    validationFig.savefig(filenames[-1])

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
filterRows = filterRowsIn
filterCols = [1] * len(filterRows)
k1s = k1sIn
learningRates = learningRatesIn

# set training parameters
batchSize = 1000
print_freq = 10

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
trainingPlotTitles = ['Single Layer CNN with %gx%g kernels and k1=%g'%(filterRows[i], filterCols[i], k1s[i]) for i in range(len(filterRows))]
trainingPlotTitles.append('Exp 3b, Validation Cross-Entropy Cost vs. Epoch')
trainingPlotFiles = ['exp3b_training_%gx%g_k1=%g_%gEpochs.png'%(filterRows[i], filterCols[i], k1s[i], numEpochs) for i in range(len(filterRows))]
trainingPlotFiles.append('exp3b_validationCurves_%gEpochs'%(numEpochs))
makeTrainingPlots(epoch_number_lists[0], filterRows, train_err_lists, val_err_lists, "Height of Kernel", "Cross Entropy Cost", trainingPlotTitles, trainingPlotFiles)
makeEndResultPlot(filterRows, train_err_lists, val_err_lists, "Height of Vertical Kernel", "Cross Entropy Cost", 'Cost vs. Height of Filter, k1=%g'%k1s[0], finalPlotName)

'''
Filter Rows: [169, 164, 145, 24]
Filter Cols: [1, 1, 1, 1]
K1s: [39, 39, 39, 39]
Learning Rates: [0.0001, 0.0001, 0.0001, 0.0001]
Train acc list: [[0.022914268619934102, 0.05531110050729101, 0.080474586354611882, 0.12920571365047526, 0.17742223632121365, 0.22728664477050464, 0.26276530956520855, 0.28791145152893016, 0.30956523375305511, 0.33045000175102779, 0.34977940109554428, 0.36773266932716314, 0.38598082480287565, 0.40393408732540825, 0.42178327520330422, 0.4397365448508645, 0.45595519275920832, 0.47190786546942304, 0.48536847521568405, 0.49859780525605507, 0.51055507898566432, 0.52196885238192192, 0.53211055744371782, 0.54160469186237137, 0.55039341820367205, 0.55875424133852425, 0.56697054224651622, 0.57444674269216411, 0.58147192025536298, 0.5886994807305751, 0.59552808309307315, 0.60231042809214519, 0.60876319384519595, 0.6146319684258893, 0.62054120957831083, 0.62660659804659957, 0.63233081492038623, 0.6379741079276583, 0.64332248633785005, 0.64869979740899242, 0.65388629686341038, 0.65877212776455463, 0.66331104082763703, 0.66799449111584208, 0.67269529257327487, 0.67689305769912556, 0.68113131002116811, 0.68537533145999152, 0.68938228608677599, 0.69349332760935734, 0.69727479171042706, 0.70164022725648123, 0.70580910132183838, 0.70937084432793807, 0.71319855301604684, 0.716922179436146, 0.72067474653035346, 0.72447932804694859, 0.72809309990229065, 0.7315796825872527, 0.73496796520409924, 0.73827529830309069, 0.74136869764520164, 0.74438113521071492, 0.74756126873473161, 0.75073560005335815, 0.75380586927627447, 0.75665641669288974, 0.7596110594561486, 0.76253098533037289, 0.76565328638157515, 0.76824364497247755, 0.77105372817468254, 0.77408352699729677, 0.77656402190158724, 0.77935675120676162, 0.7819124031822613, 0.78460684435065753, 0.78722612221439303, 0.78967192338099113, 0.79189801334680643, 0.79440742715344437, 0.7966277216711698, 0.79913713902825934, 0.80126493000777543, 0.80369917889512676, 0.80610450161690383, 0.80816869688323267, 0.81053355562564178, 0.81296201498439014, 0.81521701024111015, 0.81731010852615138, 0.81921818587484818, 0.82110891949606701, 0.82328874665801444, 0.82512743993909943, 0.82703550674775972, 0.82881060137969376, 0.83068977752522699, 0.83259784446932961], [0.026840281149740453, 0.09354780420207516, 0.20922351971365416, 0.31141549231757942, 0.39438792007884921, 0.45372331656141623, 0.50016473663022876, 0.53465467175258485, 0.56420093847410568, 0.58828317777720451, 0.60832375351664136, 0.62577976229192933, 0.64092873039764986, 0.65458591545351674, 0.6666530645687897, 0.67844843893528284, 0.68897176179008612, 0.69943727301986325, 0.70901813689131199, 0.7179745204001402, 0.726537722560188, 0.73468463751505886, 0.74281420748659321, 0.75057950028707365, 0.75776080204615892, 0.76461252591192386, 0.77145847903008535, 0.77734459531006772, 0.78312664621753036, 0.78878726379631614, 0.79422818798654327, 0.79972112646371663, 0.80494231600853205, 0.80979344813786969, 0.8147602310523866, 0.8192817981543975, 0.82365301384079492, 0.82777561851454584, 0.83169585068535812, 0.83547731428050021, 0.83915468699170481, 0.84259500406779708, 0.84600062652530528, 0.84926749909578003, 0.85254592383246086, 0.85547163532158943, 0.85865174448148485, 0.86172780793523285, 0.8646246138284529, 0.86714559585897355, 0.8697128100778686, 0.87234365542003656, 0.87482991586377412, 0.87731620641259767, 0.87929367428120286, 0.88144458997083419, 0.88357816798343636, 0.88545155906307238, 0.88744635987959652, 0.88938912323094976, 0.89130298754725146, 0.89325731303343925, 0.895148052531065, 0.89703300077690273, 0.89878495099375, 0.90063520029373911, 0.90232356905021149, 0.90402349477196153, 0.9056193321752527, 0.90715157342491015, 0.90892665933235539, 0.91039530878252439, 0.91177144282862876, 0.91322272607994537, 0.91469137164329695, 0.91625252003800661, 0.91763442476623291, 0.91893540308934818, 0.92030573459369625, 0.92167030414489148, 0.92321989212795208, 0.9245035103681446, 0.92564257753224877, 0.92691461177348378, 0.92819821872818942, 0.92935463300066201, 0.93057466454544235, 0.93170215510925836, 0.93280652802974329, 0.93379525873789804, 0.93484758533328127, 0.93585366335072828, 0.93690599210974368, 0.9380566333355389, 0.93903958103286722, 0.94001095409261626, 0.94106906707971594, 0.942005777179591, 0.94297715605026589, 0.94393118480664762], [0.085325728524219852, 0.25050736306637961, 0.41901368022620533, 0.51913564757960728, 0.58429356009371725, 0.63263149301183697, 0.67138855381754547, 0.70492443345484412, 0.73214053405123247, 0.7557602080420196, 0.77526883567979155, 0.79204833719868795, 0.80689085594253984, 0.82055384523922381, 0.83235500765065706, 0.84310382227807257, 0.85251122529293999, 0.86095880001111502, 0.86853328254757223, 0.87565097848216111, 0.88235816434599135, 0.88861432929008077, 0.89370255306738644, 0.89940364125187822, 0.90443402275006513, 0.9096089500669009, 0.91439069850359223, 0.91875036266067367, 0.92299439518854054, 0.92707651594290907, 0.9308637617720652, 0.9343271920718621, 0.93763453882537606, 0.94053712786486388, 0.94348018730465411, 0.9463422864515092, 0.94886905655385234, 0.95166756522255735, 0.9541249395699547, 0.95635680943659074, 0.9585539885714599, 0.96065865568315423, 0.96256095384236195, 0.96458466876664162, 0.96632506712228694, 0.96823891784954708, 0.96973068663935136, 0.97128605955734681, 0.97278361014071835, 0.97433320386922484, 0.97572666427998944, 0.97703919527971805, 0.97824186549019809, 0.97945031334102395, 0.98062405681004094, 0.98176312468099602, 0.98278655402999449, 0.98380419225816651, 0.98484495499553038, 0.98581055531808537, 0.98673568432808445, 0.98743531663095707, 0.9882968327882754, 0.98910633189224539, 0.98984064637643832, 0.99049980399034476, 0.99110113365962316, 0.99166199738682581, 0.99219972045479687, 0.99267384591235297, 0.99316530317053109, 0.99353536264555964, 0.99399215904201466, 0.99444893246116817, 0.99488836272425163, 0.99517746380357286, 0.99560533622048575, 0.99588287313230295, 0.99610837710713696, 0.9964148343081215, 0.99665767379635262, 0.99690052636321791, 0.99710868099608996, 0.99729948690596826, 0.99750186261575513, 0.99769844879693892, 0.99781987989650289, 0.99791817557187312, 0.99809741961332155, 0.99819571873506285, 0.99831713094506658, 0.99842699953483105, 0.9985252883174589, 0.9986235950387935, 0.99871033260158915, 0.99884330981691005, 0.99890690672047378, 0.99895894310382838, 0.99903410430585882, 0.99912085017509811], [0.076525447140415176, 0.14917113352665504, 0.42252338082762297, 0.63056151662467208, 0.75169542830697134, 0.82711647680839961, 0.87145321196501069, 0.90138687401856088, 0.9238906075180674, 0.94026537599206228, 0.95280662562936191, 0.96281535232406046, 0.97036093429872317, 0.97666334673893374, 0.98124274456369143, 0.98560818963792807, 0.98882877724264551, 0.99132662539811622, 0.99338503774598208, 0.99494040198256917, 0.99613151695095703, 0.99701039214798093, 0.99779674893501513, 0.99843277220654347, 0.99889535276112118, 0.9992885281294136, 0.9995949725612483, 0.99975107141527819, 0.99986669716808441, 0.9999360663683956, 0.99996496763704046, 0.99998808934123062, 0.99999386890568531, 0.99999386890568531, 0.99999386890568531, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.9999996501933256, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014, 0.99999964847014]]
Val acc list: [[0.023588992593733885, 0.054138227515531996, 0.080950057195172051, 0.13026757344527667, 0.1809798434284704, 0.22838802834384145, 0.25872060170334815, 0.28179503645247078, 0.30233721724828289, 0.3217148583528861, 0.3416070723124256, 0.362108634244245, 0.38093107785735375, 0.39676089782896157, 0.41394484309150098, 0.43066837782204287, 0.44601070073118138, 0.45952494265056615, 0.47234857702562605, 0.48402118945015865, 0.49498966741104744, 0.50440090011703198, 0.51329753987387017, 0.52195044849304006, 0.53053563937194526, 0.53810523835050883, 0.54520089430535756, 0.55198510539530421, 0.55840370559108676, 0.56426707519352115, 0.56991380806308101, 0.57530327375835766, 0.58074686991769386, 0.58610922014864375, 0.59126846935079846, 0.59612979459912385, 0.60112655273871829, 0.60600143960955433, 0.61047008244578826, 0.61480329837639958, 0.61909589200788995, 0.62346974469486438, 0.62784359771760334, 0.63236639509608117, 0.63621212241321889, 0.64036930337679021, 0.644580660126163, 0.64857536315427233, 0.65271900768412283, 0.65641577146174956, 0.66034275568418788, 0.66406663619171502, 0.66776341886254864, 0.67089146490590879, 0.67400596227305987, 0.67789233262277282, 0.68180576025270945, 0.68538067383079659, 0.68857642748337522, 0.69183987996364094, 0.69495439418358418, 0.69823137192048912, 0.70172504868102925, 0.7050562058002896, 0.70859047987132828, 0.71144771756078273, 0.71450804211235619, 0.71721630821041038, 0.7202631247132717, 0.72317449381596577, 0.72557130857919527, 0.72811706860860193, 0.73035137845253117, 0.73315446159974185, 0.73588979939651988, 0.73850328701312895, 0.74064281036892277, 0.74358126214725495, 0.74641139612413798, 0.74910613251021696, 0.75211230202881396, 0.75468514461763214, 0.75706840702415779, 0.75957357116360757, 0.76161829975983808, 0.76401511862326799, 0.76601921580308152, 0.76814522185727008, 0.76998682749024727, 0.7722211560917861, 0.77419819379599286, 0.77610752021048601, 0.77800331112106724, 0.77989909580063532, 0.78167299524493628, 0.78339277030122267, 0.7854104046056567, 0.78687287743256107, 0.78874156065951095, 0.79028527165347706], [0.027204527112750884, 0.091552913401208469, 0.20989057505141101, 0.3064673332090484, 0.38554868052031854, 0.4438982539233246, 0.48670237656275533, 0.52029840768113222, 0.54579671143245723, 0.56707012302089566, 0.58617694829760625, 0.60299527845278345, 0.61824279405639115, 0.63200077932687559, 0.64365984952897914, 0.6545200008649168, 0.66429682989805428, 0.67373512740022068, 0.68198178998957271, 0.6907972045934494, 0.69829908030454879, 0.70571972698525343, 0.7126799669468239, 0.72005999680395039, 0.72669524781534267, 0.73266696340406767, 0.73828661683940799, 0.74401457689911565, 0.74899780663105042, 0.75358829980242115, 0.75817880734063592, 0.7622005866777728, 0.76631715066279782, 0.7703795533644624, 0.77436071224272573, 0.77817933320637567, 0.78140218137028872, 0.78477397367040191, 0.78790201574270191, 0.79116546199195437, 0.79456436120085838, 0.79789550381767793, 0.80067149010320326, 0.80371827881510027, 0.80645361767728463, 0.80906708531590965, 0.81168057603833987, 0.81419925472870913, 0.81632524208985846, 0.81823457218484652, 0.82046888889903269, 0.82268966288391476, 0.82469379177409641, 0.82683331570456398, 0.82920305839456332, 0.83131550198037307, 0.83311648877518574, 0.83474144323741806, 0.83640702126434507, 0.8381132167799219, 0.83979235743019609, 0.84141731351313753, 0.84296104595082877, 0.84430161045866037, 0.84557448236981814, 0.84672552263887813, 0.84764630933221696, 0.84902753647610152, 0.85016498771264981, 0.85149204866584727, 0.85269722549438254, 0.85386179282158781, 0.85510757398584691, 0.85620443704856353, 0.85715232956591536, 0.85831687382868604, 0.85944080421873204, 0.86057827620810523, 0.86175636659656685, 0.86267719178659008, 0.86350321576649813, 0.86449173738865803, 0.86535835889860291, 0.86633333763649034, 0.86707812149286301, 0.86794474448791992, 0.86856765484140941, 0.86927181200789494, 0.87000304548340823, 0.87070719364882421, 0.87149258251244932, 0.8722508969072944, 0.87300921846296164, 0.87352377455873265, 0.87414666785926276, 0.87476957623690743, 0.87540604136202194, 0.87602892843153901, 0.8764622392187964, 0.87696327446806455], [0.085973889726869798, 0.25112392339710549, 0.40698459839694306, 0.50037911769667331, 0.56566184557681343, 0.61213565721192609, 0.65075550060232956, 0.68267239293327908, 0.70827902202783888, 0.7290514364375309, 0.74557183321810361, 0.76023709764331815, 0.77196388075584432, 0.78287817681291594, 0.79234359147114497, 0.80067148231605101, 0.80779419690709564, 0.81484925364662819, 0.8205772200729462, 0.82704996030853173, 0.83218212863655372, 0.83709764494796235, 0.84136314366778375, 0.84534430567123908, 0.84929838505071231, 0.8526566186985387, 0.85619090600644465, 0.85916999438617148, 0.8622032293918499, 0.86501981526719762, 0.8679041386218852, 0.87011137178228048, 0.87225089458277139, 0.87466127777419878, 0.87680077345525431, 0.87845282495996824, 0.8804975712032026, 0.88239335149200526, 0.88473600632138905, 0.8861713791807867, 0.8881619729221919, 0.89012544964313145, 0.89170978695247816, 0.89299620795955037, 0.89428260497237932, 0.89550132515654901, 0.89677424606994371, 0.89811482523518427, 0.89921166663463792, 0.90045747121846653, 0.90136473740106704, 0.90251576669321276, 0.90340950692827393, 0.90424905559014757, 0.90511569692310856, 0.90581982710575593, 0.90661875863411534, 0.90736354927680385, 0.90824373567004912, 0.90916455094856397, 0.9098009871139956, 0.91046450710441262, 0.91108741838771123, 0.91165614545881246, 0.91218425539639902, 0.91295611336641636, 0.9136060919770439, 0.91414773961392537, 0.91471648743785139, 0.91510919552946535, 0.91551544005480046, 0.91605707835484762, 0.91647686148991558, 0.9168289357501912, 0.9174111898988081, 0.91783099194645379, 0.91814244942835055, 0.91845389076772621, 0.91875178301118732, 0.91904970223848648, 0.91955074187852004, 0.91987573428965519, 0.92014655956092262, 0.92043091339481808, 0.92062049335757223, 0.92102674998979805, 0.92118924428731952, 0.92169027459051933, 0.92204235011636848, 0.92236731888839552, 0.9226652452571461, 0.92294961813921639, 0.92326106699454968, 0.92342356843352258, 0.92369438527839398, 0.92391104176971439, 0.92400583526370406, 0.92427666421546639, 0.92461519490056987, 0.92499436657783363], [0.075668936139092702, 0.14963167059616539, 0.41444586142017842, 0.60957634228175439, 0.72669523874324571, 0.79606742745114223, 0.83387477698040458, 0.85823564488223236, 0.874078989919998, 0.88462767938327302, 0.89375451514837401, 0.9005793398269375, 0.9061583731585392, 0.91028847907865673, 0.91361964462431311, 0.91621956536240656, 0.91858929814089818, 0.92060694674760601, 0.92246212087980839, 0.92370793686025698, 0.9249672814033334, 0.92622661731984668, 0.9270932349943285, 0.92824425969554136, 0.92894841961271257, 0.92970673378801927, 0.93070878998428197, 0.93134523528638047, 0.93206292024065285, 0.93278060278000408, 0.93328164794078017, 0.93379621026756432, 0.93429723780070728, 0.93474410242978045, 0.93498783467482571, 0.93550240139237539, 0.93597636966945019, 0.9363961515389444, 0.93666697829532386, 0.93695135705327171, 0.93735758288556725, 0.93770964871944695, 0.93781797091166175, 0.93799401097973856, 0.93817004297655449, 0.93834609295613924, 0.93857628898700152, 0.93864400318236874, 0.93884710767212043, 0.93892836105189426, 0.93913148055419071, 0.93928044373856623, 0.93948356508110997, 0.93957836203605616, 0.93976792953678412, 0.93982210787310405, 0.9399710486839451, 0.94007937491178994, 0.94006583940787825, 0.94009290014905866, 0.94016061986516797, 0.94028249002332087, 0.94035021413019582, 0.94041791182790646, 0.94044498906674334, 0.94043144952720104, 0.94052623586036854, 0.94058038410702927, 0.94059392548681875, 0.94074290665396243, 0.94091892250825759, 0.94102727110963669, 0.94105435825998163, 0.94113558926622098, 0.94118978374506201, 0.94121686098389901, 0.94128455868160954, 0.94140642664437957, 0.94154182572029566, 0.94154185800533774, 0.94152830635890483, 0.94162309927822108, 0.94167727980992355, 0.94174496136511321, 0.94178560016189083, 0.941812679240975, 0.94185330996649208, 0.94186685354166466, 0.94193454536349763, 0.94194808893866988, 0.94198871343317381, 0.94200227718649765, 0.9419751617867409, 0.94192101205496848, 0.94190746040853557, 0.94197518232002775, 0.94201581488579245, 0.9419751801246451, 0.94197518819590564, 0.94201582295705311]]
Train err list: [[4.2383888321347891, 4.1716434582161934, 3.9789989102097598, 3.7137076111723331, 3.4339613192199274, 3.1915876708839002, 2.9992258679891672, 2.8477191628192196, 2.7250861845596637, 2.6220793327437875, 2.5325305744568691, 2.4524361196169808, 2.3791908399665957, 2.3111629292880416, 2.2473602284251903, 2.1871755561709838, 2.1302293768623812, 2.0763054299005375, 2.0252778762475194, 1.9770389802706145, 1.9314799369839664, 1.8884489184790003, 1.8477729378168748, 1.80926571346483, 1.77274993183659, 1.7380610897287818, 1.7050566163388192, 1.6735963393316684, 1.6435569045105971, 1.6148179321191349, 1.5872766565205831, 1.5608296455762478, 1.5353856438109179, 1.5108575281317622, 1.4871741837635348, 1.4642748196309521, 1.4421007387379574, 1.4205990173412644, 1.3997273724561172, 1.3794476398931681, 1.3597241610286721, 1.3405261074080421, 1.3218220023073766, 1.303585875845539, 1.2857946134330294, 1.2684304068601062, 1.2514696742065801, 1.2348960313784094, 1.2186886753761428, 1.2028295138367358, 1.187306582984569, 1.1721040575877708, 1.1572099661529183, 1.1426123495735343, 1.1283002912821398, 1.1142625411693863, 1.1004853357508415, 1.0869615666302437, 1.0736830783562086, 1.060644434213422, 1.0478359150302303, 1.0352599550144357, 1.0229162835493908, 1.0107972351798047, 0.99889239250784101, 0.98719487048242294, 0.97570224620915313, 0.96441259416392611, 0.95331674255567456, 0.94241109215425856, 0.93169358194562601, 0.92115952039090432, 0.91079797999106948, 0.90060533269156717, 0.89057580862913288, 0.88070432790607833, 0.87098979885572969, 0.86142171076837359, 0.85199359089679105, 0.84270628976851147, 0.83355922618474276, 0.82455327638328824, 0.81568866225772108, 0.80696262020814669, 0.79836910269028249, 0.78991438711619855, 0.78159118373848002, 0.77339316823684345, 0.76531715190805782, 0.75736210429851569, 0.74952758638188621, 0.74180777568984047, 0.73420432647674938, 0.72671334000282928, 0.71933743799362526, 0.71206889588961508, 0.70490635859454076, 0.69784722362050078, 0.69088775069220243, 0.68403043964359123], [4.2096628194898864, 3.9805948215218598, 3.4993294011396205, 2.9634642217308009, 2.5961728280733793, 2.3373398667588785, 2.141770518266044, 1.9863140296610422, 1.8581359657349543, 1.7499566689634314, 1.6570487609351376, 1.5760094774039639, 1.5043620274413203, 1.4403013969038569, 1.3824635875001809, 1.3298076371065717, 1.2815248653451963, 1.2370040836595821, 1.1957524637451424, 1.1573702362867977, 1.1215035614975255, 1.0878768455802379, 1.0562384448697084, 1.0263868917084396, 0.9981444426997661, 0.97135608863296796, 0.94588785601436709, 0.92161411425806916, 0.89842535837577431, 0.876232979178035, 0.85496713272932423, 0.83454596807446224, 0.81492390726108765, 0.79604528859494694, 0.7778650647311206, 0.76033181838754516, 0.74340958244821254, 0.72706664372301355, 0.71126905702676291, 0.69597445005958347, 0.68115570299070494, 0.66677047002112111, 0.65282373667412597, 0.63930050936719007, 0.62619272531116577, 0.61347357453541385, 0.60113967607744112, 0.58916375756836525, 0.57753237244102018, 0.56623421255571393, 0.55525695673814168, 0.54459009785109802, 0.53422539651510037, 0.52415524716417217, 0.51435335600188448, 0.504818673234112, 0.49554044891077448, 0.48650499010448067, 0.47770636927686627, 0.46913525990358651, 0.4607884654946795, 0.45265482961368131, 0.44472822014046554, 0.43700271496022203, 0.42946951470984857, 0.42211990195992616, 0.41495279197839713, 0.40796247308139516, 0.401141297861068, 0.39448401074492084, 0.38798274571367924, 0.38163151644922322, 0.37542818075891687, 0.3693664655231455, 0.36343993741204361, 0.35764570847926597, 0.35198365380729846, 0.34644183851541355, 0.34102347404083, 0.33571684999840479, 0.3305231219808607, 0.32543963980923984, 0.32045659040378949, 0.31557380715624334, 0.31079815041805292, 0.30611987065727558, 0.30153302773044488, 0.29703825152875174, 0.29263071945765029, 0.28831447969071672, 0.28407880653256939, 0.27992460520774964, 0.27585414333681502, 0.27185749415008631, 0.26793527627308483, 0.26408854010408422, 0.26031042848771202, 0.25660129254508784, 0.25295801132915513, 0.24937686069019174], [4.0891572745002982, 3.2927559413499323, 2.5812187082569009, 2.1333564970150589, 1.8211690038102288, 1.591507796053619, 1.4136705508574632, 1.2719858917586389, 1.1566407465071922, 1.0606811185902665, 0.9791187708131448, 0.90848664192204431, 0.84638409371828194, 0.79113920973242891, 0.7415499971940902, 0.69669528894196586, 0.6558689180274152, 0.61851157457723238, 0.58417377334020737, 0.55248414971743565, 0.52314691978969519, 0.49591163076377953, 0.4705678203866418, 0.44694442527221878, 0.42489026697903548, 0.40428613943093744, 0.38502172762367104, 0.36698450152990858, 0.3500716432692359, 0.3341819418984856, 0.31923001756002201, 0.30513190771807763, 0.29182259300695829, 0.27923557615603284, 0.26731620924663146, 0.25601869235424124, 0.24529912931729111, 0.23511447281887615, 0.22542777484035761, 0.21621331011928691, 0.20743912966914635, 0.19907393877613466, 0.19109757578798789, 0.18348962219569107, 0.17622279447256362, 0.16927827270894472, 0.1626406499046657, 0.15629641574899564, 0.15022390271277888, 0.14441013186264495, 0.1388468042212731, 0.13351801405240488, 0.12841264221749774, 0.12351733466601275, 0.11882498725364833, 0.11432551992509639, 0.11000638753272235, 0.10586324213349282, 0.10188403620931946, 0.098066412285726337, 0.09439997860613758, 0.090876060416083898, 0.087491512409040184, 0.084237114532507984, 0.081109451740761826, 0.078101173077499378, 0.075208934536311681, 0.072427732633012401, 0.06975101343426017, 0.06717539267094115, 0.064698572773403129, 0.062316029547227869, 0.060024449627347987, 0.057817622076054349, 0.055691960226522307, 0.053647696657778111, 0.051678655839268478, 0.049783793588400363, 0.047958887799878898, 0.04620263822344977, 0.044509722500338501, 0.04287863692404871, 0.041308823261049406, 0.039796601678871717, 0.038339132774859029, 0.036933456675736394, 0.035579062963249537, 0.034274438051491427, 0.033016126366402429, 0.031804161303793078, 0.030634645341277956, 0.029506900098225063, 0.02841986957268991, 0.027371524292078637, 0.026361027115698755, 0.025386189835257419, 0.024447148083209056, 0.023541128565404235, 0.022668537285583691, 0.021826662563864903], [4.1920110689760319, 3.7867021956174511, 2.9237250904291709, 1.9181240526578529, 1.2724546156676517, 0.89146769161547579, 0.65608837771444251, 0.50030587209573141, 0.39167327901854665, 0.31310114544402484, 0.25451713267759174, 0.20955465526990075, 0.17428968502962108, 0.14614736076950421, 0.12334136918688327, 0.10462700917530464, 0.08912237676965723, 0.07618093443885024, 0.065310482863696492, 0.056128935559722738, 0.048331110973610369, 0.041675603154818094, 0.035974001261024832, 0.031075430980022446, 0.026857099281016903, 0.023220881783818155, 0.020082670239665298, 0.017370893689576668, 0.015025220289575673, 0.012994274877053824, 0.011234613269353946, 0.0097096219440664031, 0.0083876473330061815, 0.0072417985120783535, 0.0062486991612880688, 0.0053883973740012454, 0.0046435129238619251, 0.0039989341114645095, 0.0034414701418916012, 0.0029597825465264549, 0.0025438113596088913, 0.0021849029423784785, 0.0018753670786189852, 0.0016086343840813187, 0.0013787411613050296, 0.0011809505287352663, 0.0010108346577922052, 0.0008646968762852402, 0.00073929696511180663, 0.0006317883009069621, 0.000539700833824498, 0.00046088905951789915, 0.00039350456911672042, 0.00033591920379332445, 0.00028674964693318626, 0.00024478837681098141, 0.00020899910804691321, 0.00017849954238790928, 0.00015251776071403369, 0.00013039150195132772, 0.00011155146149291187, 9.5516286170945297e-05, 8.1870071916954881e-05, 7.0256329686866037e-05, 6.0367583154957537e-05, 5.1943738944917687e-05, 4.4767286480996871e-05, 3.8646006426461545e-05, 3.3422299919737076e-05, 2.8957991255773425e-05, 2.5139907583332067e-05, 2.1871337498134001e-05, 1.9068794024822229e-05, 1.666381533562852e-05, 1.4598099527831975e-05, 1.2823013356385062e-05, 1.1297797949253031e-05, 9.9839813312913427e-06, 8.8516644198234745e-06, 7.8739153868926751e-06, 7.0265019758988193e-06, 6.2907695952288707e-06, 5.6496445408319086e-06, 5.0880611574265252e-06, 4.5939975844128499e-06, 4.1557684660809296e-06, 3.7658383049133527e-06, 3.4170523037872199e-06, 3.1035903700329316e-06, 2.8219455511164372e-06, 2.5688813856409852e-06, 2.3408543065827184e-06, 2.1360937504986641e-06, 1.9516674048943569e-06, 1.7859783733840887e-06, 1.6367771171237924e-06, 1.5027101222169413e-06, 1.380975681889753e-06, 1.271544201106198e-06, 1.1723652795303707e-06]]
Val err list: [[4.2381244630393118, 4.1729646366500814, 3.9830164760593321, 3.7206510185433443, 3.4448797955129193, 3.2065177769640805, 3.0177291797902006, 2.8695132376481056, 2.7498976995234443, 2.6498042383401628, 2.5631309671115807, 2.4858998543856377, 2.4155121796981653, 2.3503461657643849, 2.2894201075936458, 2.2321301949862318, 2.1780951378495548, 2.1270886553598114, 2.078980215837348, 2.0336610904404746, 1.9910054997185975, 1.9508546956659152, 1.9130336744996326, 1.8773578287087855, 1.84365290722988, 1.8117311857023293, 1.7814402130853446, 1.7526483085713114, 1.7252472881221679, 1.699089362398928, 1.6740671277059114, 1.6500809791996995, 1.6270374158469352, 1.6048515267599937, 1.5834599940880913, 1.5627948783508376, 1.5427874694497135, 1.5233952704417124, 1.5045750286943049, 1.486284835123445, 1.4684916813729216, 1.4511706724467508, 1.4342865962246594, 1.417820906938704, 1.4017555206649261, 1.3860714485812915, 1.370754750600655, 1.3557859792725881, 1.3411518930678583, 1.3268302878066891, 1.3128118084977889, 1.2990916933815524, 1.2856529434018347, 1.2724880568062467, 1.2595789816330216, 1.2469044293449185, 1.2344612932685599, 1.2222487690027506, 1.2102689953927888, 1.1985175019159773, 1.1869931862060097, 1.1757077869234074, 1.1646500160680948, 1.1538033008885378, 1.1431628158815426, 1.1327162421727255, 1.1224688776729297, 1.1124127359537994, 1.1025419429198022, 1.0928515333639117, 1.0833337653330248, 1.0739917286864704, 1.064808971560824, 1.0557886680922881, 1.0469212417754408, 1.0382089327402149, 1.0296489678952021, 1.0212262630682185, 1.0129456464550293, 1.0047999905449976, 0.99679038802537312, 0.98890738030862302, 0.98115174989376963, 0.97352126658175886, 0.96601603434979744, 0.95864702195700735, 0.95140103210466354, 0.94428112290944088, 0.93727477441279838, 0.93038398056476579, 0.9236079012246059, 0.91694581019840937, 0.91039418438095376, 0.90394675620598397, 0.89760821966034476, 0.89137080173272387, 0.88523944406852406, 0.87920617620111507, 0.87326547052449333, 0.86742548019988552], [4.2101950439339069, 3.984896769387765, 3.5072373090954159, 2.9782670718362385, 2.6205339148687443, 2.3724335189656514, 2.1878271634704491, 2.0429280868086441, 1.9245275302279901, 1.8251797559266232, 1.7402103635423067, 1.6663320981181593, 1.6011987727249628, 1.5430952354986351, 1.490739396793356, 1.4431544758751544, 1.3995891321571943, 1.35951497275578, 1.3224854372573334, 1.2881343579217397, 1.2561532951437013, 1.2262878574099216, 1.1983062547770087, 1.1720312217270847, 1.1473024492338177, 1.123956505829484, 1.1018580067081551, 1.0809007324024111, 1.0609538369312561, 1.0419526384396205, 1.0238161127263297, 1.0064678869607997, 0.9898682895415446, 0.97395827846813288, 0.95869330851825862, 0.94400819410625358, 0.92988437336685215, 0.9162801964132411, 0.90316691757372014, 0.89048830354966602, 0.87821032138677135, 0.86625634825199938, 0.85469878564455859, 0.84363726069746159, 0.83298438507226158, 0.82270081021806141, 0.81277112556735831, 0.80318145441500155, 0.79392862223091165, 0.7849799054323201, 0.77631828337730224, 0.76790646711339572, 0.75974372471286733, 0.75184186238300343, 0.74415278751792546, 0.73669592496853142, 0.72946104451540983, 0.72243810540068998, 0.71562685800934001, 0.70900586324605408, 0.70257961487875975, 0.69632864093976798, 0.69024956173401408, 0.68433484470272177, 0.67859002168123539, 0.6729956024801883, 0.66755078189771544, 0.66225300441130752, 0.65709803773604991, 0.65208321244134737, 0.64719619808185669, 0.64243352736027914, 0.63780624764981431, 0.63329561733499595, 0.6289077213326818, 0.62463072270668285, 0.62047150263922168, 0.61641650189688069, 0.61247167726364227, 0.60862020934962091, 0.6048722500119742, 0.60121917962227289, 0.59766323685116574, 0.59419513288524173, 0.59082138026319819, 0.58753494589523236, 0.58432387034924804, 0.58119380863448178, 0.5781359393797334, 0.57515719531998055, 0.57224567451633057, 0.56940652374042977, 0.56663670096001217, 0.56392309218467518, 0.56127107552130728, 0.55867943998380598, 0.55614602948020275, 0.55367728154781437, 0.55126191906016775, 0.54889753191626833], [4.0937292066765636, 3.3097940605751628, 2.6126023105701504, 2.1820486074356098, 1.8866591335647274, 1.6710793744212125, 1.5052560582448673, 1.3748159707528826, 1.2703377425689995, 1.1849030932707565, 1.1135209467310054, 1.0527078908955003, 0.99998711872018198, 0.95360918675118833, 0.91237460063733478, 0.8753719111199677, 0.84196033763570399, 0.81162039925096208, 0.78392984441320812, 0.75855870109223777, 0.73523282579243909, 0.71370288640883806, 0.69374315618061067, 0.67516369003991827, 0.65782416039253933, 0.64162035673362783, 0.62647841445752162, 0.61231675929695706, 0.59906649686393709, 0.58665574185187808, 0.57501323211362265, 0.5640854007980054, 0.55382368336931587, 0.54416150646289274, 0.53506812767403422, 0.5264894957016043, 0.51839525969066447, 0.51075809047573661, 0.50353536565414092, 0.49669727191366131, 0.49022030883330769, 0.48407713663550911, 0.47824866426441764, 0.47272617630188368, 0.46748182644708203, 0.46249519471434647, 0.4577560736807334, 0.45325255767519879, 0.44897446725301599, 0.44489389234394489, 0.44102224049634298, 0.43734821982301597, 0.43386059108398789, 0.43054072298178137, 0.42738905572294605, 0.42440858978244206, 0.42157459724583513, 0.41889617521149952, 0.41635604861846476, 0.41395234977058698, 0.41168811899548213, 0.40953771689449275, 0.40751649993830474, 0.40560766615158372, 0.40380743256034685, 0.40212104153547884, 0.40054179161267245, 0.39906281560849888, 0.39767738718302592, 0.396390705971428, 0.39519321653750211, 0.39407381716586576, 0.39303514300362796, 0.3920679944151208, 0.39117215448037446, 0.39035851499419888, 0.3896059230972429, 0.38892523776011401, 0.38831158226262891, 0.38775824350172228, 0.38726253810611788, 0.3868135918666683, 0.38643545402748047, 0.38609896802835064, 0.38582174217645421, 0.38559271302386955, 0.38540938568262451, 0.38526940872333199, 0.38517854739250512, 0.38512666138780294, 0.3851223029878284, 0.38514957276819944, 0.38522054305211934, 0.38533531244456376, 0.38548822474497751, 0.38567971438088766, 0.38589843268862473, 0.38616119324031001, 0.38646082437228263, 0.38678836526778781], [4.19311214782166, 3.7988293119014602, 2.9597369424055229, 1.9910528361248487, 1.3778671850402413, 1.0243702259795686, 0.81450553086167909, 0.68208340711446924, 0.59340939333503184, 0.53066222733039026, 0.48446758582020949, 0.44990524448484248, 0.42368070257406121, 0.40350877079322117, 0.38779988738028909, 0.37552053115308748, 0.36594513273680807, 0.35854632820110383, 0.35291463372999488, 0.34872961373618955, 0.34573644292456596, 0.3437347271185166, 0.34256664131576459, 0.34210905157955401, 0.34225411389788618, 0.342930235056639, 0.34410357338715031, 0.34571630979149931, 0.34774272249574295, 0.35014633204965134, 0.35287915469959336, 0.35591873815524311, 0.35921657494793036, 0.36275500856245502, 0.36652121033070345, 0.37051788514193251, 0.37470138704462957, 0.37905738012088941, 0.38357823303262123, 0.38824522616015833, 0.39303511938066271, 0.39796014997453161, 0.4030017236074328, 0.40812786473779528, 0.413299342112737, 0.41859965654484133, 0.42403619714398488, 0.42956500737977787, 0.43514016598653543, 0.44076384709065014, 0.44643096970447727, 0.4521418679459176, 0.45788126195346196, 0.463639770949756, 0.46940735203965933, 0.47516705324184644, 0.48091117700806235, 0.48660646942206548, 0.49225770515102091, 0.49786574721607552, 0.50340629313570895, 0.5088860395217869, 0.51430120535357116, 0.51965138411395362, 0.52494902004659427, 0.53018223777350426, 0.53534529831205158, 0.54043176613304955, 0.54545193700745975, 0.55039901195723351, 0.55525228609412169, 0.56001040773703725, 0.56467234552093848, 0.56923436523839799, 0.57369048557525837, 0.5780418900172225, 0.58230047794620887, 0.58648115505945919, 0.59059734302059874, 0.59464601238207038, 0.59865189841606936, 0.60258705378895971, 0.60645313499136821, 0.61024622569917386, 0.6139718927317922, 0.61762822127628914, 0.62121498781686069, 0.62474078285437129, 0.62821063982705594, 0.6316525385989592, 0.63504310104805894, 0.63840463757192112, 0.64173912951649492, 0.64506772919551436, 0.6483706735722643, 0.65164302878849378, 0.65492530006484728, 0.65817298513103972, 0.66139711800072021, 0.66461470909436848]]
Epoch number lists: [[9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199, 209, 219, 229, 239, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 369, 379, 389, 399, 409, 419, 429, 439, 449, 459, 469, 479, 489, 499, 509, 519, 529, 539, 549, 559, 569, 579, 589, 599, 609, 619, 629, 639, 649, 659, 669, 679, 689, 699, 709, 719, 729, 739, 749, 759, 769, 779, 789, 799, 809, 819, 829, 839, 849, 859, 869, 879, 889, 899, 909, 919, 929, 939, 949, 959, 969, 979, 989, 999], [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199, 209, 219, 229, 239, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 369, 379, 389, 399, 409, 419, 429, 439, 449, 459, 469, 479, 489, 499, 509, 519, 529, 539, 549, 559, 569, 579, 589, 599, 609, 619, 629, 639, 649, 659, 669, 679, 689, 699, 709, 719, 729, 739, 749, 759, 769, 779, 789, 799, 809, 819, 829, 839, 849, 859, 869, 879, 889, 899, 909, 919, 929, 939, 949, 959, 969, 979, 989, 999], [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199, 209, 219, 229, 239, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 369, 379, 389, 399, 409, 419, 429, 439, 449, 459, 469, 479, 489, 499, 509, 519, 529, 539, 549, 559, 569, 579, 589, 599, 609, 619, 629, 639, 649, 659, 669, 679, 689, 699, 709, 719, 729, 739, 749, 759, 769, 779, 789, 799, 809, 819, 829, 839, 849, 859, 869, 879, 889, 899, 909, 919, 929, 939, 949, 959, 969, 979, 989, 999], [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199, 209, 219, 229, 239, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 369, 379, 389, 399, 409, 419, 429, 439, 449, 459, 469, 479, 489, 499, 509, 519, 529, 539, 549, 559, 569, 579, 589, 599, 609, 619, 629, 639, 649, 659, 669, 679, 689, 699, 709, 719, 729, 739, 749, 759, 769, 779, 789, 799, 809, 819, 829, 839, 849, 859, 869, 879, 889, 899, 909, 919, 929, 939, 949, 959, 969, 979, 989, 999]]
'''


