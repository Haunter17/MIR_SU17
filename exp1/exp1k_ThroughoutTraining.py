import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

startTime = time.time()

print('==> Experiment 1i')

def loadData(filepath):

	print('==> Loading data from {}'.format(filepath))
	f = h5py.File(filepath)
	X_train = np.array(f.get('trainingFeatures'))
	y_train = np.array(f.get('trainingLabels'))
	X_test = np.array(f.get('validationFeatures'))
	y_test = np.array(f.get('validationLabels'))
	del f
	print('==> Data sizes:',X_train.shape, y_train.shape, X_test.shape, y_test.shape)

	# Transform labels into on-hot encoding form
	enc = OneHotEncoder()
	y_train = enc.fit_transform(y_train.copy()).astype(int).toarray()
	y_test = enc.fit_transform(y_test.copy()).astype(int).toarray()

	return [X_train, y_train, X_test, y_test]

# Neural-network model set-up
# Functions for initializing neural nets parameters
def init_weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
  return tf.Variable(initial)

def init_bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
  return tf.Variable(initial)


def runNeuralNet(num_features, hidden_layer_size, X_train, y_train, X_test, y_test, batchSize, numEpochs):

	'''
		NN config parameters
	'''

	num_classes = y_test.shape[1]
	
	print('==> Creating Neural net with %d features, %d hidden units, and %d classes'%(num_features, hidden_layer_size, num_classes))

	# Set-up NN layers
	x = tf.placeholder(tf.float64, [None, num_features])
	W1 = init_weight_variable([num_features, hidden_layer_size])
	b1 = init_bias_variable([hidden_layer_size])

	# Hidden layer activation function: ReLU
	h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

	W2 = init_weight_variable([hidden_layer_size, num_classes])
	b2 = init_bias_variable([num_classes])

	# Softmax layer (Output), dtype = float64
	y = tf.matmul(h1, W2) + b2

	# NN desired value (labels)
	y_ = tf.placeholder(tf.float64, [None, num_classes])

	# Loss function
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	sess = tf.InteractiveSession()

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
	sess.run(tf.global_variables_initializer())


	'''
		Training config
	'''
	numTrainingVec = len(X_train)
	print_freq = 5

	train_accuracies = []
	test_accuracies = []
	train_costs = []
	test_costs = []

	print('Training with %d samples, a batch size of %d, for %d epochs'%(numTrainingVec, batchSize, numEpochs))


	for epoch in range(numEpochs):

		epochStart = time.time()
		for i in range(0,numTrainingVec,batchSize):

			# Batch Data
			batchEndPoint = min(i+batchSize, numTrainingVec)
			trainBatchData = X_train[i:batchEndPoint]
			trainBatchLabel = y_train[i:batchEndPoint]

			train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

		epochEnd = time.time()

		# calculate the accuracies and costs at this epoch
		train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
		test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
		train_cost = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
		test_cost = cross_entropy.eval(feed_dict={x: X_test, y_: y_test})
		# update the lists
		train_accuracies += [train_accuracy]
		test_accuracies += [test_accuracy]
		train_costs += [train_cost]
		test_costs += [test_cost]
		# Print accuracy

		if (epoch + 1) % print_freq == 0:

			print("epoch: %d, time: %g, t acc, v acc, t cost, v cost: %g, %g, %g, %g"%(epoch+1, epochEnd - epochStart, train_accuracy, test_accuracy, train_cost, test_cost))

	# Validation
	train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
	test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
	train_cost = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
	test_cost = cross_entropy.eval(feed_dict={x:X_test, y_: y_test})
	print("test accuracy %g"%(test_accuracy))
	return [train_accuracies, test_accuracies, train_costs, test_costs]

''' 
our main
'''
'''
Plot the cost at each epoch for each of these downsampling amounts
'''


[X_train, y_train, X_test, y_test] = loadData('pylon2/ci560sp/cstrong/taylorswift_smallDataset_71_7.mat')
numEpochs = 5

numTrainingSamples = X_train.shape[0]

# leave the testing data the same, downsample the training data
print("==> Starting Downsampling Tests for exp1c")
# set the rates we want to test at
batchSizes = [100, 500, 1000, 5000, 10000]

matplotlib.rcParams.update({'font.size': 8})


trainingAccuracyLists = []
testAccuracyLists = []
trainingCostLists = []
testCostLists = []

times = []
for curSize in batchSizes:
	startOfLoop = time.time()
	print("==> Test with Batch Size of %d"%(curSize))
	[trainingAccuracies, testAccuracies, trainingCosts, testCosts] = runNeuralNet(121, 100, X_train, y_train, X_test, y_test, curSize, numEpochs)
	# store the data at each epoch
	trainingAccuracyLists += [trainingAccuracies]
	testAccuracyLists += [testAccuracies]
	trainingCostLists += [trainingCosts]
	testCostLists += [testCosts]

	endOfTraining = time.time()

	times += [endOfTraining - startOfLoop]

	endOfLoop = time.time()
	print("Test with Batch Size of %d took: %g"%(curSize, endOfLoop - startOfLoop))

#track the time of the whole experiment	
endTime = time.time()
print("Whole experiment Took: %g"%(endTime - startTime))


'''
Printing results
'''
print("--------------------------")
print("Summary Of Results")
print("--------------------------")
print("Batch Sizes: %s"%str(batchSizes))
print("Training Accuracy Lists: %s"%str(trainingAccuracyLists))
print("Test Accuracy Lists: %s"%str(testAccuracyLists))
print("Training Cost Lists: %s"%str(trainingCostLists))
print("Test Cost Lists: %s"%str(testCostLists))


'''
Plotting results
'''
#setup the figure, will add plots to it in the loop
fig = plt.figure(figsize=(8,11))
trainingPlot = fig.add_subplot(311)
trainingPlot.set_xlabel("Epoch Numbers")
trainingPlot.set_ylabel("Cross-Entropy Error")

trainingPlot.set_title("Error vs. Epoch Number")

for i in range(len(trainingAccuracyLists)):
	curSize = batchSizes[i]
	trainingCosts = trainingCostLists[i]
	testCosts = testCostLists[i]
	numEpochs = len(trainingAccuracies)
	epochNumbers = range(numEpochs)	
	# only put on validation cost
	trainingPlot.plot(epochNumbers, testCosts, label="Validation, Batchsize = %d"%(curSize), marker="o", markersize="3", ls="None")


# plots have already been added to the figure during the loop
trainingPlot.legend(loc="upper right", frameon=False)

# add in the final values
finalTrainingAccuracies = [curList[-1] for curList in trainingAccuracyLists]
finalTestAccuracies = [curList[-1] for curList in testAccuracyLists]
finalTrainingCosts = [curList[-1] for curList in trainingCostLists]
finalTestCosts = [curList[-1] for curList in testCostLists]

errPlot = fig.add_subplot(312)
errPlot.plot(batchSizes, finalTrainingCosts, label="Training", marker="o", markersize="3",  ls="None")
errPlot.plot(batchSizes, finalTestCosts, label="Validation", marker="o", markersize="3", ls="None")
errPlot.set_xlabel("Batch Size")
errPlot.set_ylabel("Cross-Entropy Error")
errPlot.legend(loc="lower right", frameon=False)
errPlot.set_title("Final Error vs. Batch Size")
# plot the time versus batch size
timePlot = fig.add_subplot(313)
timePlot.plot(batchSizes, times, label="Time", marker="o", markersize="3",  ls="None")
timePlot.set_xlabel("Batch Size")
timePlot.set_ylabel("Time to Make and Train NN (S)")
timePlot.set_title("Time vs. Batch Size")

fig.tight_layout()
fig.savefig('exp1k_BatchSize.png')







