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


def runNeuralNet(num_features, hidden_layer_size, X_train, y_train, X_test, y_test, batchSize, numEpochs, regularizationType, regularizationScale):

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
	
	# add on regularization, None, L1 or L2 as options
	if regularizationType == "None":
		total_cost = cross_entropy
	if regularizationType == "L1":
		total_cost = cross_entropy + 0.5 * regularizationScale * (tf.reduce_mean(tf.abs(W1)) + tf.reduce_mean(tf.abs(W2))) # the 1/2 factor to average between the two weight matrices
	elif regularizationType == "L2":
		total_cost = cross_entropy + 0.5 * regularizationScale * (tf.reduce_mean(tf.square(W1)) + tf.reduce_mean(tf.square(W2))) 
	else:
		total_cost = cross_entropy

	train_step = tf.train.AdamOptimizer(1e-4).minimize(total_cost)

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
	# track which epochs you grab data on
	epoch_numbers = []

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


		# Print accuracy

		if (epoch + 1) % print_freq == 0:
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
			epoch_numbers += [epoch]
			print("epoch: %d, time: %g, t acc, v acc, t cost, v cost: %g, %g, %g, %g"%(epoch+1, epochEnd - epochStart, train_accuracy, test_accuracy, train_cost, test_cost))

	# Validation
	train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
	test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
	train_cost = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
	test_cost = cross_entropy.eval(feed_dict={x:X_test, y_: y_test})
	print("test accuracy %g"%(test_accuracy))
	return [train_accuracies, test_accuracies, train_costs, test_costs, epoch_numbers]

''' 
our main  - compare regularization methods
'''


'''
 set params
'''
numEpochs = 800
regularizationScale = 1.0
batchSize = 1000

[X_train, y_train, X_test, y_test] = loadData('/pylon2/ci560sp/cstrong/taylorswift_smallDataset_71_7.mat')


numTrainingSamples = X_train.shape[0]

print("==> Starting Regularization Tests for exp1n")

matplotlib.rcParams.update({'font.size': 8})

regularizationTypes = ["None", "L1", "L2"]
trainingAccuracyLists = []
testAccuracyLists = []
trainingCostLists = []
testCostLists = []
epochNumberLists = []
times = []

for curRegularizationType in regularizationTypes:
	startOfLoop = time.time()
	print("==> Test with Regularization type %s"%(curRegularizationType))
	[trainingAccuracies, testAccuracies, trainingCosts, testCosts, epochNumbers] = runNeuralNet(121, 100, X_train, y_train, X_test, y_test, batchSize, numEpochs, regularizationScale, curRegularizationType)
	# store the data at each epoch
	trainingAccuracyLists += [trainingAccuracies]
	testAccuracyLists += [testAccuracies]
	trainingCostLists += [trainingCosts]
	testCostLists += [testCosts]
	epochNumberLists += [epochNumbers]

	endOfTraining = time.time()

	times += [endOfTraining - startOfLoop]

	endOfLoop = time.time()
	print("Test with Regularization type Size %s took: %g"%(curRegularizationType, endOfLoop - startOfLoop))

#track the time of the whole experiment	
endTime = time.time()
print("Whole experiment Took: %g"%(endTime - startTime))


'''
Printing results
'''
print("--------------------------")
print("Summary Of Results - exp1n, Regularization")
print("--------------------------")
print("Epoch Numbers: %s"%str(epochNumbers))
print("Regularization Types: %s"%str(regularizationTypes))
print("Regularizatoin Scale: %g"%regularizationScale)
print("Training Accuracy Lists: %s"%str(trainingAccuracyLists))
print("Test Accuracy Lists: %s"%str(testAccuracyLists))
print("Training Cost Lists: %s"%str(trainingCostLists))
print("Test Cost Lists: %s"%str(testCostLists))


'''
Plotting results
'''
#setup the figure, will add plots to it in the loop
numPlots = 1 + len(regularizationTypes) # 1 fig for all the validations together, then a fig for the training of each regularization type
fig = plt.figure(figsize=(8,4 * numPlots))

trainingPlot = fig.add_subplot(int("%d11"%(numPlots)))
trainingPlot.set_xlabel("Epoch Numbers")
trainingPlot.set_ylabel("Cross-Entropy Error")
trainingPlot.set_title("Validation Error vs. Epoch Number for all Regularization Types, scale = %g"%regularizationScale)
# create one plot with all the validation curves
# and a plot for each regularization type with both validation and training costs
for i in range(len(trainingAccuracyLists)):
	curRegularizationType = regularizationTypes[i]
	trainingCosts = trainingCostLists[i]
	testCosts = testCostLists[i]
	epochNumbers = epochNumberLists[i] # it may not save every epoch
	# put validation cost onto the training plot
	trainingPlot.plot(epochNumbers, testCosts, label="Validation, Type = %s"%(curRegularizationType), marker="o", markersize="3", ls="None")

	# create a subplot with both validation and training costs
	newPlot = fig.add_subplot(int("%d1%d"%(numPlots, 2+i))) # 2 + i so that starts at 2, so won't interfere with the training plot
	newPlot.set_xlabel("Epoch Numbers")
	newPlot.set_ylabel("Cross-Entropy Error")
	newPlot.set_title("Error vs. Epoch Number, Regularization: %s"%(curRegularizationType))
	newPlot.plot(epochNumbers, trainingCosts, label="Training, Type = %s"%(curRegularizationType), marker="o", markersize="3", ls="None")
	newPlot.plot(epochNumbers, testCosts, label="Validation, Type = %s"%(curRegularizationType), marker="o", markersize="3", ls="None")
	newPlot.legend(loc="upper right", frameon=False)

trainingPlot.legend(loc="upper right", frameon=False)


fig.tight_layout()
fig.savefig('exp1n_RegularizationTypes.png')







