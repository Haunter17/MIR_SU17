import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import scipy.io

# Functions for initializing neural nets parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')

# Download data from .mat file into numpy array
print('==> Experiment 8a')
filepath = '/scratch/ttanpras/exp8a_d7_1s.mat'
print('==> Loading data from {}'.format(filepath))
f = h5py.File(filepath)
data_train = np.array(f.get('trainingFeatures'))
data_val = np.array(f.get('validationFeatures'))
X_test = np.array(f.get('window_testFeatures'))
y_test = np.array(f.get('window_testLabels'))
del f

print('==> Data sizes:',data_train.shape, data_val.shape,  X_test.shape, y_test.shape)

# Transform labels into on-hot encoding form
enc = OneHotEncoder()
y_test = enc.fit_transform(y_test.copy()).astype(int).toarray()

'''
    NN config parameters
'''
sub_window_size = 32  
num_features = 169*sub_window_size
num_frames = 32
hidden_layer_size = 4000
num_classes = 71
print("Number of features:", num_features)
print("Number of songs:",num_classes)

# Reshape input features
X_train = np.reshape(data_train,(-1, num_features))
X_val = np.reshape(data_val,(-1, num_features))
print("Input sizes:", X_train.shape, X_val.shape)

y_train = []
y_val = []
# Add Labels
for label in range(num_classes):
    for sampleCount in range(X_train.shape[0]//num_classes):
        y_train.append([label])
    for sampleCount in range(X_val.shape[0]//num_classes):
        y_val.append([label])

X_train = np.concatenate((X_train, y_train), axis=1)
X_val = np.concatenate((X_val, y_val), axis=1)

# Shuffle
np.random.shuffle(X_train)
np.random.shuffle(X_val)

# Separate coefficients and labels
y_train = X_train[:, -1].reshape(-1, 1)
X_train = X_train[:, :-1]
y_val = X_val[:, -1].reshape(-1, 1)
X_val = X_val[:, :-1]
print('==> Data sizes:',X_train.shape, y_train.shape,X_val.shape, y_val.shape)

y_train = enc.fit_transform(y_train.copy()).astype(int).toarray()
y_val = enc.fit_transform(y_val.copy()).astype(int).toarray()

plotx = []
ploty_train = []
ploty_val = []
        
 # Set-up NN layers
x = tf.placeholder(tf.float64, [None, num_features])
W1 = weight_variable([num_features, hidden_layer_size])
b1 = bias_variable([hidden_layer_size])

OpW1 = tf.placeholder(tf.float64, [num_features, hidden_layer_size])
Opb1 = tf.placeholder(tf.float64, [hidden_layer_size])

# Hidden layer activation function: ReLU
h = tf.matmul(x, W1) + b1
h1 = tf.nn.sigmoid(h)

W2 = weight_variable([hidden_layer_size, num_classes])
b2 = bias_variable([num_classes])

OpW2 = tf.placeholder(tf.float64, [hidden_layer_size, num_classes])
Opb2 = tf.placeholder(tf.float64, [num_classes])

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
sess.run(tf.initialize_all_variables())

# Training
numTrainingVec = len(X_train)
batchSize = 500
numEpochs = 500
bestValErr = 10000
bestValEpoch = 0

startTime = time.time()
for epoch in range(numEpochs):
    for i in range(0,numTrainingVec,batchSize):

        # Batch Data
        batchEndPoint = min(i+batchSize, numTrainingVec)
        trainBatchData = X_train[i:batchEndPoint]
        trainBatchLabel = y_train[i:batchEndPoint]

        train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

    # Print accuracy
    if epoch%5 == 0 or epoch == numEpochs-1:
        plotx.append(epoch)
        train_error = cross_entropy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
        val_error = cross_entropy.eval(feed_dict={x:X_val, y_: y_val})
        train_acc = accuracy.eval(feed_dict={x:trainBatchData, y_: trainBatchLabel})
        val_acc = accuracy.eval(feed_dict={x:X_val, y_: y_val})
        ploty_train.append(train_error)
        ploty_val.append(val_error)
        print("epoch: %d, train acc %g, val acc %g, train error %g, val error %g"%(epoch, train_acc, val_acc, train_error, val_error))

        if val_error < bestValErr:
            bestValErr = val_error
            bestValEpoch = epoch
            OpW1 = W1
            Opb1 = b1
            OpW2 = W2
            Opb2 = b2

endTime = time.time()
print("Elapse Time:", endTime - startTime)
print("Best validation error: %g at epoch %d"%(bestValErr, bestValEpoch))

# Restore best model for early stopping
W1 = OpW1
b1 = Opb1
W2 = OpW2
b2 = Opb2

print('==> Generating error plot...')
errfig = plt.figure()
trainErrPlot = errfig.add_subplot(111)
trainErrPlot.set_xlabel('Number of Epochs')
trainErrPlot.set_ylabel('Cross-Entropy Error')
trainErrPlot.set_title('Error vs Number of Epochs')
trainErrPlot.scatter(plotx, ploty_train)
valErrPlot = errfig.add_subplot(111)
valErrPlot.scatter(plotx, ploty_val)
errfig.savefig('exp8a_sgmd_4000.png')

'''
GENERATING REPRESENTATION OF NOISY FILES
'''
namelist = ['orig','comp5','comp10','str5','str10','ampSat_(-15)','ampSat_(-10)','ampSat_(-5)', \
            'ampSat_(5)','ampSat_(10)','ampSat_(15)','pitchShift_(-1)','pitchShift_(-0.5)', \
            'pitchShift_(0.5)','pitchShift_(1)','rev_dkw','rev_gal','rev_shan0','rev_shan1', \
            'rev_gen','crowd-15','crowd-10','crowd-5','crowd0','crowd5','crowd10','crowd15', \
            'crowd100','rest-15','rest-10','rest-5','rest0','rest5','rest10','rest15', \
            'rest100','AWGN-15','AWGN-10','AWGN-5','AWGN0','AWGN5','AWGN10','AWGN15', 'AWGN100']
outdir = '/scratch/ttanpras/taylorswift_noisy_processed/'

repDict = {}
repRowDict = {}
repZeroDict = {}

# Loop over each CQT files, not shuffled
for count in range(len(namelist)):

    name = namelist[count]
    filename = outdir + name + '.mat'
    cqt = scipy.io.loadmat(filename)['Q']
    cqt = np.transpose(np.array(cqt))

    # Group into windows of 32 without overlapping
    # Discard any leftover frames
    num_windows = cqt.shape[0] // 32
    cqt = cqt[:32*num_windows]
    X = np.reshape(cqt,(num_windows, num_features))

    # Feed window through model (Only 1 layer of weight w/o non-linearity)
    rep = h.eval(feed_dict={x:X})

    # Threshold output with median of neuron
    medRow = np.median(rep,axis=0)
    repRow = np.array([ [ int(rep[w][i]>medRow[i]) for i in range(hidden_layer_size)] for w in range(num_windows)])

    # Threshold output with 0
    repZero = np.array([ [ int(rep[w][i]>0) for i in range(hidden_layer_size)] for w in range(num_windows)])

    # Put the output representation into a dictionary
    repDict['n'+str(count)] = rep
    repRowDict['r'+str(count)] = repRow
    repZeroDict['z'+str(count)] = repZero

scipy.io.savemat('exp8a_sgmd_repNon.mat',repDict)
scipy.io.savemat('exp8a_sgmd_repRow.mat',repRowDict)
scipy.io.savemat('exp8a_sgmd_repZero.mat',repZeroDict)