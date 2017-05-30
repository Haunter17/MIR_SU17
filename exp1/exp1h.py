import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math

# Download data from .mat file into numpy array
print('==> Experiment 1h')
filepath = '(separate features & labels) exp1a_smallDataset_71_7.mat'
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

# Functions for initializing neural nets parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

'''
	NN config parameters
'''
num_features = 121
hidden_layer_size = 100
num_classes = y_test.shape[1]

# log transformation
X_train_log = np.log10(X_train)
X_test_log = np.log10(X_test)

# cubic root
X_train_cr = np.cbrt(X_train)
X_test_cr = np.cbrt(X_test)

''' No preprocessing '''
print("No Preprocessing")
# Set-up NN layers
x = tf.placeholder(tf.float64, [None, num_features])
W1 = weight_variable([num_features, hidden_layer_size])
b1 = bias_variable([hidden_layer_size])

# Hidden layer activation function: ReLU
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_variable([hidden_layer_size, num_classes])
b2 = bias_variable([num_classes])

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

# Training
numTrainingVec = len(X_train)
batchSize = 1000
numEpochs = 300

plotx = []
ploty_train_noprep = []
ploty_test_noprep = []

noisy = False

startTime = time.time()
for epoch in range(numEpochs):
    for i in range(0,numTrainingVec,batchSize):

        # Batch Data
        batchEndPoint = min(i+batchSize, numTrainingVec)
        trainBatchData = X_train[i:batchEndPoint]
        trainBatchLabel = y_train[i:batchEndPoint]

        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

    if epoch%10 == 0 or epoch == numEpochs-1:

        # Evaluation
        #train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
        #validation_accuracy = accuracy.eval(feed_dict={x:X_test, y_: y_test})
        train_error = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
        validation_error = cross_entropy.eval(feed_dict={x:X_test, y_: y_test})

        # Save accuracy and error values for plotting
        plotx.append(epoch)
        ploty_train_noprep.append(train_error)
        ploty_test_noprep.append(validation_error)

        if noisy:
            print("epoch: %d, training accuracy %g"%(epoch, train_accuracy))

endTime = time.time()
print("Elapse Time:", endTime - startTime)

# Save plot
# Error plot
errfig = plt.figure()
trainacc = errfig.add_subplot(111)
trainacc.set_xlabel('Number of epochs')
trainacc.set_ylabel('Cross-Entropy Error')
trainacc.set_title('Error vs Number of Epochs (No Preprocessing)')
trainacc.scatter(plotx, ploty_train_noprep)
validacc = errfig.add_subplot(111)
validacc.scatter(plotx, ploty_test_noprep, c='r')
errfig.savefig('exp1h_noprep.png')

# Validation
train_error = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
validation_error = cross_entropy.eval(feed_dict={x:X_test, y_: y_test})

print("Train error:", train_error)
print("Validation error:", validation_error)

''' Logarithm '''
print("Logarithm Base 10")

# Set-up NN layers
x = tf.placeholder(tf.float64, [None, num_features])
W1 = weight_variable([num_features, hidden_layer_size])
b1 = bias_variable([hidden_layer_size])

# Hidden layer activation function: ReLU
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_variable([hidden_layer_size, num_classes])
b2 = bias_variable([num_classes])

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

# Training
numTrainingVec = len(X_train_log)
batchSize = 1000
numEpochs = 300

ploty_train_log = []
ploty_test_log = []

noisy = False

startTime = time.time()
for epoch in range(numEpochs):
    for i in range(0,numTrainingVec,batchSize):

        # Batch Data
        batchEndPoint = min(i+batchSize, numTrainingVec)
        trainBatchData = X_train_log[i:batchEndPoint]
        trainBatchLabel = y_train[i:batchEndPoint]

        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

    if epoch%10 == 0 or epoch == numEpochs-1:

        # Evaluation
        #train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
        #validation_accuracy = accuracy.eval(feed_dict={x:X_test, y_: y_test})
        train_error = cross_entropy.eval(feed_dict={x:X_train_log, y_: y_train})
        validation_error = cross_entropy.eval(feed_dict={x:X_test_log, y_: y_test})

        # Save accuracy and error values for plotting
        ploty_train_log.append(train_error)
        ploty_test_log.append(validation_error)

        if noisy:
            print("epoch: %d, training accuracy %g"%(epoch, train_accuracy))

endTime = time.time()
print("Elapse Time:", endTime - startTime)

# Save plot
# Error plot
errfig = plt.figure()
trainacc = errfig.add_subplot(111)
trainacc.set_xlabel('Number of epochs')
trainacc.set_ylabel('Cross-Entropy Error')
trainacc.set_title('Error vs Number of Epochs (Logarithm Base 10)')
trainacc.scatter(plotx, ploty_train_log)
validacc = errfig.add_subplot(111)
validacc.scatter(plotx, ploty_test_log, c='r')
errfig.savefig('exp1h_log.png')

# Validation
train_error = cross_entropy.eval(feed_dict={x:X_train_log, y_: y_train})
validation_error = cross_entropy.eval(feed_dict={x:X_test_log, y_: y_test})

print("Train error:", train_error)
print("Validation error:", validation_error)

''' Cubic Root '''
print("Cubic Root")

# Set-up NN layers
x = tf.placeholder(tf.float64, [None, num_features])
W1 = weight_variable([num_features, hidden_layer_size])
b1 = bias_variable([hidden_layer_size])

# Hidden layer activation function: ReLU
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_variable([hidden_layer_size, num_classes])
b2 = bias_variable([num_classes])

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

# Training
numTrainingVec = len(X_train_cr)
batchSize = 1000
numEpochs = 300

ploty_train_cr = []
ploty_test_cr = []

noisy = False

startTime = time.time()
for epoch in range(numEpochs):
    for i in range(0,numTrainingVec,batchSize):

        # Batch Data
        batchEndPoint = min(i+batchSize, numTrainingVec)
        trainBatchData = X_train_cr[i:batchEndPoint]
        trainBatchLabel = y_train[i:batchEndPoint]

        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

    if epoch%10 == 0 or epoch == numEpochs-1:

        # Evaluation
        #train_accuracy = accuracy.eval(feed_dict={x:X_train, y_: y_train})
        #validation_accuracy = accuracy.eval(feed_dict={x:X_test, y_: y_test})
        train_error = cross_entropy.eval(feed_dict={x:X_train_cr, y_: y_train})
        validation_error = cross_entropy.eval(feed_dict={x:X_test_cr, y_: y_test})

        # Save accuracy and error values for plotting
        ploty_train_cr.append(train_error)
        ploty_test_cr.append(validation_error)

        if noisy:
            print("epoch: %d, training accuracy %g"%(epoch, train_accuracy))

endTime = time.time()
print("Elapse Time:", endTime - startTime)

# Save plot
# Error plot
errfig = plt.figure()
trainacc = errfig.add_subplot(111)
trainacc.set_xlabel('Number of epochs')
trainacc.set_ylabel('Cross-Entropy Error')
trainacc.set_title('Error vs Number of Epochs (Cubic Root)')
trainacc.scatter(plotx, ploty_train_cr)
validacc = errfig.add_subplot(111)
validacc.scatter(plotx, ploty_test_cr, c='r')
errfig.savefig('exp1h_cr.png')

# Validation
train_error = cross_entropy.eval(feed_dict={x:X_train_cr, y_: y_train})
validation_error = cross_entropy.eval(feed_dict={x:X_test_cr, y_: y_test})

print("Train error:", train_error)
print("Validation error:", validation_error)

# Comparison Plot
errfig = plt.figure()
noprep = errfig.add_subplot(111)
noprep.set_xlabel('Number of epochs')
noprep.set_ylabel('Cross-Entropy Error')
noprep.set_title('Error vs Number of Epochs')
noprep.scatter(plotx, ploty_test_noprep)
logT = errfig.add_subplot(111)
logT.scatter(plotx, ploty_test_log, c='r')
cubicRoot = errfig.add_subplot(111)
cubicRoot.scatter(plotx, ploty_test_cr, c='g')
errfig.savefig('exp1h_comparison.png')
