import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

startTime = time.time()

print('==> Experiment 1j')

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

print("==> Starting Downsampling Tests for exp1c")
# set the rates we want to test at
files = ['/pylon2/ci560sp/cstrong/bigk.r.i.t._smallDataset_71_7.mat', '/pylon2/ci560sp/cstrong/chromeo_smallDataset_44_7.mat', '/pylon2/ci560sp/cstrong/deathcabforcutie_smallDataset_87_7.mat', '/pylon2/ci560sp/cstrong/foofighters_smallDataset_87_7.mat', '/pylon2/ci560sp/cstrong/kanyewest_smallDataset_92_7.mat', '/pylon2/ci560sp/cstrong/maroon5_smallDataset_66_7.mat', '/pylon2/ci560sp/cstrong/onedirection_smallDataset_60_7.mat', '/pylon2/ci560sp/cstrong/taylorswift_smallDataset_71_7.mat', '/pylon2/ci560sp/cstrong/t.i_smallDataset_154_7.mat', '/pylon2/ci560sp/cstrong/tompetty_smallDataset_193_7.mat']
numSongs = [71, 44, 87, 87, 92, 66, 60, 71, 154, 193] # track the number of songs for each file

trainingAccuracies = []
testAccuracies = []
trainingCosts = []
testCosts = []

for curFileName in files:
	print("==> Test with Filename %s"%(curFileName))
	startOfLoop = time.time()

	[X_train, y_train, X_test, y_test] = loadData(curFileName)
	# run with this set of data
	[trainingAccuracy, testAccuracy, trainingCost, testCost] = runNeuralNet(121, 100, X_train, y_train, X_test, y_test)
	# track the final accuracies
	trainingAccuracies += [trainingAccuracy]
	testAccuracies += [testAccuracy]
	trainingCosts += [trainingCost]
	testCosts += [testCost]

	# time how long this run took
	endOfLoop = time.time()
	print("Test with file %s took: %g"%(curFileName, endOfLoop - startOfLoop))

endTime = time.time()
print("Whole experiment Took: %g"%(endTime - startTime))

'''
Printing results
'''
print("--------------------------")
print("Summary Of Results")
print("--------------------------")
print("Filenames: %s"%str(files))
print("Training Accuracies: %s"%str(trainingAccuracies))
print("Test Accuracies: %s"%str(testAccuracies))
print("Training Costs: %s"%str(trainingCosts))
print("Test Costs: %s"%str(testCosts))

'''
Plotting results
'''

trainingError = map(lambda x: 1.0 - x, trainingAccuracies)
validationError = map(lambda x: 1.0 - x, testAccuracies)

matplotlib.rcParams.update({'font.size': 8})

fig = plt.figure()

errPlot = fig.add_subplot(111)
errPlot.plot(numSongs, trainingCosts, label="Training", marker="o", ls="None")
errPlot.plot(numSongs, testCosts, label="Validation", marker="o", ls="None")
errPlot.set_xlabel("Number of songs")
errPlot.set_ylabel("Cross Entropy error")
errPlot.legend(loc="upper left", frameon=False)
errPlot.set_title("Error vs. Number of Songs")

fig.tight_layout()
fig.savefig('exp1j_PerformanceAcrossArtists.png')

'''
--------------------------
Summary Of Results - for 300 epochs
--------------------------

Filenames: ['bigk.r.i.t._smallDataset_71_7.mat', 'chromeo_smallDataset_44_7.mat', 'deathcabforcutie_smallDataset_87_7.mat', 'foofighters_smallDataset_87_7.mat', 'kanyewest_smallDataset_92_7.mat', 'maroon5_smallDataset_66_7.mat', 'onedirection_smallDataset_60_7.mat', 'taylorswift_smallDataset_71_7.mat', 't.i_smallDataset_154_7.mat', 'tompetty_smallDataset_193_7.mat']
Training Accuracies: [0.58595070684843586, 0.60337084524426965, 0.71044389478549086, 0.60007004404165865, 0.56118181425548153, 0.56451058821420785, 0.5986581029533623, 0.54863434723630988, 0.46090013956588533, 0.5567197749613505]
Test Accuracies: [0.56793070603643292, 0.58072204520936299, 0.67555961289688715, 0.56432122788635419, 0.53601238082160263, 0.53326174263350323, 0.55787824497522975, 0.51083393661518661, 0.44379958312350248, 0.52012423083954251]
Training Costs: [1.6256964327979653, 1.4740867976275571, 1.1182194621819614, 1.519509695555127, 1.7502016665566253, 1.6169096778482237, 1.4587339405195838, 1.6475863633655878, 2.3200866634944766, 1.860562361959889]
Test Costs: [1.7118313871313515, 1.5639550092026051, 1.2812777912222009, 1.6638050544617826, 1.8812740629121854, 1.7815370122594094, 1.6176581532431178, 1.7977739956364229, 2.4125679386842025, 2.0531056328918629]

--------------------------
Summary Of Results - for 500 epochs
--------------------------
Filenames: ['/pylon2/ci560sp/cstrong/bigk.r.i.t._smallDataset_71_7.mat', '/pylon2/ci560sp/cstrong/chromeo_smallDataset_44_7.mat', '/pylon2/ci560sp/cstrong/deathcabforcutie_smallDataset_87_7.mat', '/pylon2/ci560sp/cstrong/foofighters_smallDataset_87_7.mat', '/pylon2/ci560sp/cstrong/kanyewest_smallDataset_92_7.mat', '/pylon2/ci560sp/cstrong/maroon5_smallDataset_66_7.mat', '/pylon2/ci560sp/cstrong/onedirection_smallDataset_60_7.mat', '/pylon2/ci560sp/cstrong/taylorswift_smallDataset_71_7.mat', '/pylon2/ci560sp/cstrong/t.i_smallDataset_154_7.mat', '/pylon2/ci560sp/cstrong/tompetty_smallDataset_193_7.mat']
Training Accuracies: [0.62329018320182061, 0.6443203428741453, 0.74180553048590481, 0.63185595188483124, 0.5950173478642693, 0.60083224050735073, 0.63335233013883863, 0.57782204352979893, 0.48824670768358985, 0.58359035061548525]
Test Accuracies: [0.59894035004167256, 0.61373842795855427, 0.70336013361934746, 0.59118731011493231, 0.56193565387141975, 0.55837021495831851, 0.58240188007282423, 0.53616171975547045, 0.4669287764471548, 0.54187467229696573]
Training Costs: [1.4551154411996254, 1.3062200897227911, 0.98120497863219691, 1.3762278199774451, 1.6054248795566881, 1.474320393325727, 1.316650381966914, 1.5149553378963669, 2.1766357215014094, 1.7267926325108442]
Test Costs: [1.5747762993712215, 1.4308506139477666, 1.1742349778311969, 1.5556437116366797, 1.7744765023242113, 1.6793058814773629, 1.5170491874112881, 1.6960415289355657, 2.2870295671366909, 1.9465227103381111]

'''



