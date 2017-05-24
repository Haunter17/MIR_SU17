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


def runNeuralNet(num_features, hidden_layer_size, X_train, y_train, X_test, y_test):


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
	batchSize = 1000
	numEpochs = 2
	print_freq = 5

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
	        train_accuracy = accuracy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
	        test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
	        train_cost = cross_entropy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
	        test_cost = cross_entropy.eval(feed_dict={x: X_test, y_: y_test})
	        print("epoch: %d, time: %d, t acc, v acc, t cost, v cost: %g, %g, %g, %g"%(epoch+1, epochEnd - epochStart, train_accuracy, test_accuracy, train_cost, test_cost))

	# Validation
	train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
	test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
	train_cost = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
	test_cost = cross_entropy.eval(feed_dict={x:X_test, y_: y_test})
	print("test accuracy %g"%(test_accuracy))
	return [train_accuracy, test_accuracy, train_cost, test_cost]


''' 
our main
'''
[X_train, y_train, X_test, y_test] = loadData('taylorswift_smallDataset_71_7.mat')

numTrainingSamples = X_train.shape[0]

# leave the testing data the same, downsample the training data
print("==> Starting Downsampling Tests for exp1c")
# set the rates we want to test at
downsamplingRates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
trainingAccuracies = []
testAccuracies = []
trainingCosts = []
testCosts = []

for curRate in downsamplingRates:
	startOfLoop = time.time()
	print("==> Test with Downsampling Rate of %d"%(curRate))
	# downsample then ron on the downsampled training data
	X_train_downsampled = X_train[:numTrainingSamples / curRate,:]
	y_train_downsampled = y_train[:numTrainingSamples / curRate, :]
	[trainingAccuracy, testAccuracy, trainingCost, testCost] = runNeuralNet(121, 20, X_train_downsampled, y_train_downsampled, X_test, y_test)
	# track the final accuracies
	trainingAccuracies += [trainingAccuracy]
	testAccuracies += [testAccuracy]
	trainingCosts += [trainingCost]
	testCosts += [testCost]

	endOfLoop = time.time()
	print("Test with downsampling of %d took: %d"%(curRate, endOfLoop - startOfLoop))

endTime = time.time()
print("Whole experiment Took: %d"%(endTime - startTime))


'''
Printing results
'''
print("--------------------------")
print("Summary Of Results")
print("--------------------------")
print("Downsampling Rates: %s"%str(downsamplingRates))
print("Training Accuracies: %s"%str(trainingAccuracies))
print("Test Accuracies: %s"%str(testAccuracies))
print("Training Costs: %s"%str(trainingCosts))
print("Test Costs: %s"%str(testCosts))

'''
Plotting results
'''

recipOfDownsampling = map(lambda x:1.0/x, downsamplingRates)

matplotlib.rcParams.update({'font.size': 8})

plt.plot(downsamplingRates, trainingAccuracies, label="Training Accuracy", marker="o", ls="None")
plt.plot(downsamplingRates, testAccuracies, label="Test Accuracy", marker="o", ls="None")
plt.xlabel("Downsampling of Training Data")
plt.ylabel("Accuracy")
plt.legend(loc="upper left", frameon=False)
plt.show()

fig = plt.figure()
accPlot = fig.add_subplot(221)

accPlot.plot(downsamplingRates, trainingAccuracies, label="Training", marker="o", ls="None")
accPlot.plot(downsamplingRates, testAccuracies, label="Validation", marker="o", ls="None")
accPlot.set_xlabel("Downsampling Rate")
accPlot.set_ylabel("Accuracy (%)")
accPlot.legend(loc="upper right", frameon=False)
accPlot.set_title("Accuracy vs. Downsampling Rate")

errPlot = fig.add_subplot(222)
errPlot.plot(downsamplingRates, trainingCosts, label="Training", marker="o", ls="None")
errPlot.plot(downsamplingRates, testCosts, label="Validation", marker="o", ls="None")
errPlot.set_xlabel("Downsampling Rate")
errPlot.set_ylabel("Error")
errPlot.legend(loc="lower right", frameon=False)
errPlot.set_title("Error vs. Downsampling Rate")

reciprocalErrPlot = fig.add_subplot(223)
reciprocalErrPlot.plot(recipOfDownsampling, trainingCosts, label="Training", marker="o", ls="None")
reciprocalErrPlot.plot(recipOfDownsampling, testCosts, label="Validation", marker="o", ls="None")
reciprocalErrPlot.set_xlabel("Percent Of Data Remaining after Downsampling")
reciprocalErrPlot.set_ylabel("Cross-Entropy Error")
reciprocalErrPlot.legend(loc="lower left", frameon=False)
reciprocalErrPlot.set_title("Error vs. Percent of Data Remaining")

fig.tight_layout()
fig.savefig('exp1i_AccuracyErrorAndRecip.png')





