import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time

startTime = time.time()

print('==> Experiment 1g')

def loadData(filepath):

	print('==> Loading data from {}'.format(filepath))
	f = h5py.File(filepath)
	data_train = np.array(f.get('trainingSet'))
	X_train = data_train[:, :-1]
	y_train = data_train[:, -1].reshape(-1, 1)
	data_test = np.array(f.get('testSet'))
	X_test = data_test[:, :-1]
	y_test = data_test[:, -1].reshape(-1, 1)
	del data_train, data_test, f
	print('-- Number of training samples: {}'.format(X_train.shape[0]))
	print('-- Number of test samples: {}'.format(X_test.shape[0]))

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
	numEpochs = 800
	print_freq = 5
	train_accuracies = []
	test_accuracies = []

	print('Training with %d samples, a batch size of %d, for %d epochs'%(numTrainingVec, batchSize, numEpochs))

	for epoch in range(numEpochs):
	    for i in range(0,numTrainingVec,batchSize):

	        # Batch Data
	        batchEndPoint = min(i+batchSize, numTrainingVec)
	        trainBatchData = X_train[i:batchEndPoint]
	        trainBatchLabel = y_train[i:batchEndPoint]

	        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

		train_accuracy = accuracy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
	    test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
	    # keep a list of the training and testing accuracies for each epoch
	    train_accuracies += [train_accuracy]
	    test_accuracies += [test_accuracy]
	    # Print accuracy
	    if (epoch + 1) % print_freq == 0:
	        print("epoch: %d, training accuracy, test accuracy: %g, %g"%(epoch+1, train_accuracy, test_accuracy))

	# Validation
	train_accuracy = accuracy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
	test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
	print("test accuracy %g"%(test_accuracy))
	return [train_accuracies, test_accuracies]


''' 
our main
'''
[X_train, y_train, X_test, y_test] = loadData('taylorswift_smallDataset_71_7.mat')

[train_accuracies, test_accuracies] = runNeuralNet(121, 20, X_train, y_train, X_test, y_test)


endTime = time.time()
print("Experiment took: %d"%(endTime - startTime))

'''
Printing results
'''
print("--------------------------")
print("Summary Of Results")
print("Training Accuracies: %s"%str(train_accuracies))
print("Testing Accuracies: %s"%str(test_accuracies))
print("--------------------------")


'''
Plotting results
'''
numEpochs = len(train_accuracies)
epochNumbers = range(numEpochs)
plt.plot(epochNumbers, train_accuracies, label="Training Accuracy", marker="o", ls="None")
plt.plot(epochNumbers, test_accuracies, label="Test Accuracy", marker="o", ls="None")
plt.xlabel("Epoch Number (starting at 0)")
plt.ylabel("Accuracy")
plt.legend(loc="upper left", frameon=False)
plt.show()








