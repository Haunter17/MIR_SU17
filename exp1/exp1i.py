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
	numEpochs = 500
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
	        train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
	        test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
	        train_cost = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
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
[X_train, y_train, X_test, y_test] = loadData('taylorswift_fullDataset_71_1.mat')

numTrainingSamples = X_train.shape[0]

# leave the testing data the same, downsample the training data
print("==> Starting Downsampling Tests for exp1c")
# set the rates we want to test at
downsamplingRates = [1, 2, 3, 4, 5, 6, 7]
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
	[trainingAccuracy, testAccuracy, trainingCost, testCost] = runNeuralNet(121, 100, X_train_downsampled, y_train_downsampled, X_test, y_test)
	# track the final accuracies
	trainingAccuracies += [trainingAccuracy]
	testAccuracies += [testAccuracy]
	trainingCosts += [trainingCost]
	testCosts += [testCost]

	endOfLoop = time.time()
	print("Test with downsampling of %d took: %g"%(curRate, endOfLoop - startOfLoop))

endTime = time.time()
print("Whole experiment Took: %g"%(endTime - startTime))


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

fig = plt.figure()

errPlot = fig.add_subplot(211)
errPlot.plot(downsamplingRates, trainingCosts, label="Training", marker="o", markersize="3",  ls="None")
errPlot.plot(downsamplingRates, testCosts, label="Validation", marker="o", markersize="3", ls="None")
errPlot.set_xlabel("Downsampling Rate")
errPlot.set_ylabel("Cross-Entropy Error")
errPlot.legend(loc="lower right", frameon=False)
errPlot.set_title("Error vs. Downsampling Rate")

reciprocalErrPlot = fig.add_subplot(212)
reciprocalErrPlot.plot(recipOfDownsampling, trainingCosts, label="Training", marker="o", markersize="3", ls="None")
reciprocalErrPlot.plot(recipOfDownsampling, testCosts, label="Validation", marker="o", markersize="3", ls="None")
reciprocalErrPlot.set_xlabel("Percent Of Data Remaining after Downsampling")
reciprocalErrPlot.set_ylabel("Cross-Entropy Error")
reciprocalErrPlot.legend(loc="lower left", frameon=False)
reciprocalErrPlot.set_title("Error vs. Percent of Data Remaining")

fig.tight_layout()
fig.savefig('exp1i_DownsamplingErrorAndRecip.png')


'''

--------------------------
Summary Of Results - on 'taylorswift_smallDataset_71_7.mat'
--------------------------
Downsampling Rates: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
Training Accuracies: [0.53892874164415927, 0.47806151379465373, 0.44131513920560839, 0.40775863754570507, 0.38405699690818662, 0.36050740577797336, 0.34082704580061229, 0.32178401736308004, 0.30807303123625152, 0.29447133010705384, 0.24769027716673986, 0.20829056068827301, 0.18637335777574091, 0.174585716380701, 0.14137661321861558, 0.13651918846247862]
Test Accuracies: [0.50163891570141617, 0.449460062741313, 0.41443955115830139, 0.38324987934362942, 0.36179818211068215, 0.33886341698841699, 0.32238376769626764, 0.30372727638352648, 0.29301399613899615, 0.28020933880308885, 0.23180099742599738, 0.19884169884169883, 0.17611305501930505, 0.16318774131274127, 0.13811132561132555, 0.12854428088803091]
Training Costs: [1.6813306359695026, 1.9332665070077102, 2.0811193325782944, 2.2077154246943871, 2.3001199834867907, 2.3931665543962617, 2.47497979334159, 2.5465880652459698, 2.5939233912394779, 2.6515009136545671, 2.8707026074927713, 3.0627334249113582, 3.1785790370267559, 3.2983200331840048, 3.5141303311235879, 3.6721708876656765]
Test Costs: [1.8309723669637192, 2.0354896755905214, 2.1655903244656738, 2.286901244334369, 2.3709173844094029, 2.4572944759312487, 2.5266386179264084, 2.5972954569980988, 2.6425106567188346, 2.6976812797701628, 2.9020364739279425, 3.0784706741561965, 3.1932050531873348, 3.3134997833991515, 3.5219883284574927, 3.6808464251794963]

--------------------------
Summary Of Results - on 'taylorswift_fullDataset_71_1.mat'
--------------------------
Downsampling Rates: [1, 2, 3, 4, 5, 6, 7]
Training Accuracies: [0.70347195138004071, 0.68398063133901887, 0.66931156243682466, 0.65691157287270208, 0.6423487389163075, 0.63290903224995576, 0.62029016678724236]
Test Accuracies: [0.62700824917300024, 0.61137173563002156, 0.60194687952013237, 0.59278200953474036, 0.58412633889263188, 0.57335989692483869, 0.56595172876409028]
Training Costs: [1.043051784263896, 1.109848570166041, 1.1607781967484514, 1.210971953046859, 1.2654602865163078, 1.3074660106709701, 1.3508221276479093]
Test Costs: [1.3791842702070281, 1.4245322893977455, 1.4611670058514004, 1.4864197609766745, 1.5192002942371901, 1.5590648155526432, 1.5838801702834713]
'''


