import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder

# Download data from .mat file into numpy array
# small dataset can be downloaded at https://drive.google.com/file/d/0BxtJa9dtWREBTUxlYVgtOVdpQ1k/view?usp=sharing

print('==> Experiment 1a')
filepath = 'exp1a_smallDataset.mat'
print('==> Loading data from {}'.format(filepath))
f = h5py.File(filepath)
data_train = np.array(f.get('trainingSet'))
X_train = data_train[:, :-1]
y_train = data_train[:, -1].reshape(-1, 1)
data_test = np.array(f.get('testSet'))
X_test = data_test[:, :-1]
y_test = data_test[:, -1].reshape(-1, 1)
del data_train, data_test, f
print('==> Data sizes:',X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Transform labels into on-hot encoding form
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.copy()).astype(int).toarray()
y_test = enc.fit_transform(y_test.copy()).astype(int).toarray()

# Neural-network model set-up

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
num_featuers = 121
hidden_layer_size = 20
num_classes = y_test.shape[1]

# Set-up NN layers
x = tf.placeholder(tf.float64, [None, num_featuers])
W1 = init_weight_variable([num_featuers, hidden_layer_size])
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

# Training
numTrainingVec = len(X_train)
batchSize = 1000
numEpochs = 20
noisy = True
for epoch in range(numEpochs):
    for i in range(0,numTrainingVec,batchSize):

        # Batch Data
        batchEndPoint = min(i+batchSize, numTrainingVec)
        trainBatchData = X_train[i:batchEndPoint]
        trainBatchLabel = y_train[i:batchEndPoint]

        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

    # Print accuracy
    if noisy:
        train_accuracy = accuracy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
        print("epoch: %d, training accuracy %g"%(epoch, train_accuracy))

# Validation
print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: y_test}))