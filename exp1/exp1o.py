import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import visualizationHelper

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
	return [train_accuracies, test_accuracies, train_costs, test_costs, epoch_numbers, W1.eval(), W2.eval()]

''' 
our main  - compare regularization methods
'''


'''
 set params
'''
startExperiment = time.time()

numEpochs = 5
batchSize = 1000
numHiddenNeurons = 100

[X_train, y_train, X_test, y_test] = loadData('pylon2/ci560sp/cstrong/taylorswift_smallDataset_71_7.mat')

# cubic root
X_train_cr = np.cbrt(X_train)
X_test_cr = np.cbrt(X_test)

[trainingAccuracies, testAccuracies, trainingCosts, testCosts, epochNumbers, W1, W2] = runNeuralNet(121, numHiddenNeurons, X_train_cr, y_train, X_test_cr, y_test, batchSize, numEpochs)

#W1cols = [W1[:,i] for i in range(W1.shape[1])]
# make them intro matrices by nesting them again
#W1colsAsMatrices = [[curCol.tolist()] for curCol in W1cols]
#visualizeWeights(W1colsAsMatrices, 10, 'testvisualize.png')
visualizationHelper.visualizeColVecs(W1, 5, 'testvisualizecols.png')

visualizationHelper.visualizeColVecsGroupedByOctave(W1, 5, 'testvisualizecols_grouped.png')

visualizationHelper.visualizeWeights([W1, W2], 1, 'testvisualizeweights.png')


endExperiment = time.time()

'''
Printing results
'''
print("--------------------------")
print("Summary Of Results - exp1o, Visualization")
print("--------------------------")
print("Epoch Numbers: %s"%str(epochNumbers))
print("Training Accuracy: %s"%str(trainingAccuracies))
print("Test Accuracy: %s"%str(testAccuracies))
print("Training Cost: %s"%str(trainingCosts))
print("Test Cost: %s"%str(testCosts))
print("Time: %g"%(endExperiment - startExperiment))


'''
Plotting results
'''







