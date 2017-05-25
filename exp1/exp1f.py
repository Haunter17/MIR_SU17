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


def runNeuralNet(num_features, hidden_layer_size, X_train, y_train, X_test, y_test, cpuOnly):

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

	# restrict to CPU only if desired
	if cpuOnly:
		config = tf.ConfigProto(device_count = {'GPU': 0}, log_device_placement=True)
	else:
		config = tf.ConfigProto(log_device_placement=True)
	sess = tf.InteractiveSession(config=config)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
	sess.run(tf.global_variables_initializer())


	'''
		Training config
	'''
	numTrainingVec = len(X_train)
	batchSize = 1000
	numEpochs = 300
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
	        print("epoch: %d, time: %g, t acc, v acc, t cost, v cost: %g, %g, %g, %g"%(epoch+1, epochEnd - epochStart, train_accuracy, test_accuracy, train_cost, test_cost))

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

### CODE FOR RUNNING ON THE GPU NODE
#withGPUStart = time.time()
#[trainingAccuracy, testAccuracy, trainingCost, testCost] = runNeuralNet(121, 100, X_train, y_train, X_test, y_test, False)
#withGPUEnd = time.time()
#
#cpuOnlyStart = time.time()
#[trainingAccuracy, testAccuracy, trainingCost, testCost] = runNeuralNet(121, 100, X_train, y_train, X_test, y_test, True)
#cpuOnlyEnd = time.time()
#print("Timing for Creating and Training Neural Net, doesn't include loading data")
#print("GPU: %g"%(withGPUEnd - withGPUStart))
#print("CPU Only on GPU Node: %g"%(cpuOnlyEnd - cpuOnlyStart))
###
### CODE FOR RUNNING ON THE NON-GPU NODE
cpuNodeStart = time.time()
# last parameter of False just runs it normally, with no restrictions
[trainingAccuracy, testAccuracy, trainingCost, testCost] = runNeuralNet(121, 100, X_train, y_train, X_test, y_test, False)
cpuNodeEnd = time.time()
print("Timing for Creating and Training Neural Net, doesn't include loading data")
print("Running on CPU Node: %g"%(cpuNodeEnd - cpuNodeStart))

'''
Printing results
'''


print("C")
'''
Plotting results
'''


'''
--------------------------
Summary Of Results
--------------------------
Timing for Creating and Training Neural Net, doesn't include loading data
GPU: 516.153
CPU Only on GPU Node: 844.12
Running on CPU Node: 1027.41

test accuracy 0.509728 GPU+CPU
test accuracy 0.508567 CPU on GPU Node
test accuracy 0.510065 CPU on CPU Node


'''

