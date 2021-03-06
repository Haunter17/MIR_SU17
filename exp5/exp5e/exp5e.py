import numpy as np
import tensorflow as tf
import h5py
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Download data from .mat file into numpy array
print('==> Experiment 5e')
filepath = '/pylon2/ci560sp/mint96/exp5e_d7_1s_temporal.mat'
print('==> Loading data from {}'.format(filepath))
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))
X_test = np.array(f.get('window_testFeatures'))
y_test = np.array(f.get('window_testLabels'))
del f
print('==> Data sizes:',X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

# Transform labels into on-hot encoding form
enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.copy()).astype(int).toarray()
y_val = enc.fit_transform(y_val.copy()).astype(int).toarray()
y_test = enc.fit_transform(y_test.copy()).astype(int).toarray()

# Functions for initializing neural nets parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')

hidden_size_list = [20, 40, 80, 160, 320]
plot_train_all = []
plot_val_all = []
plot_test_all = []

for hidden_layer_size in hidden_size_list:
    '''
        NN config parameters
    '''
    num_features = 34
    num_frames = 169
    num_classes = y_test.shape[1]
    print("Number of features:", num_features)
    print("Number of songs:",num_classes)
    plotx = []
    ploty_train = []
    ploty_val = []
    ploty_test = []
            
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
    y_group = tf.reshape(tf.reduce_mean(y, 0),[-1, num_classes])

    # NN desired value (labels)
    y_ = tf.placeholder(tf.float64, [None, num_classes])

    # Loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy_group = -tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y_group)))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction_group = tf.equal(tf.argmax(y_group, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    accuracy_group = tf.reduce_mean(tf.cast(correct_prediction_group, tf.float64))
    sess.run(tf.global_variables_initializer())

    # Training
    numTrainingVec = len(X_train)
    batchSize = 500
    numEpochs = 400

    startTime = time.time()
    for epoch in range(numEpochs):
        for i in range(0,numTrainingVec,batchSize):

            # Batch Data
            batchEndPoint = min(i+batchSize, numTrainingVec)
            trainBatchData = X_train[i:batchEndPoint]
            trainBatchLabel = y_train[i:batchEndPoint]

            train_step.run(feed_dict={x: trainBatchData, y_: trainBatchLabel})

        # Print accuracy
        if epoch%50 == 0 or epoch == numEpochs-1:
            plotx.append(epoch)
            train_error = cross_entropy.eval(feed_dict={x:X_train, y_: y_train})
            validation_error = cross_entropy.eval(feed_dict={x:X_val, y_: y_val})
            train_acc = accuracy.eval(feed_dict={x:X_train, y_: y_train})
            val_acc = accuracy.eval(feed_dict={x:X_val, y_: y_val})
            ploty_train.append(train_error)
            ploty_val.append(validation_error)
            print("epoch: %d, train acc %g, val acc %g, train error %g, val error %g"%(epoch, train_acc, val_acc, train_error, validation_error))

            # Evaluating multi-frame validation set
            total_error = 0
            total_acc = 0
            for t in range(len(X_test)):
                this_x = X_test[t]
                this_y = [y_test[t]]
                x_image = np.reshape(this_x, (-1, num_features))

                total_error = total_error + cross_entropy_group.eval(feed_dict={x:x_image, y_: this_y})
                total_acc = total_acc + accuracy_group.eval(feed_dict={x:x_image, y_: this_y})
            ploty_test.append(total_error/len(X_test))
            print("====> Window acc %g, err: %g"%(total_acc/len(X_test),ploty_test[-1]))

    endTime = time.time()
    print("Elapse Time:", endTime - startTime)
    
    plot_train_all.append(ploty_train)
    plot_val_all.append(ploty_val)
    plot_test_all.append(ploty_test)

    print('==> Generating error plot...')
    errfig = plt.figure()
    trainErrPlot = errfig.add_subplot(111)
    trainErrPlot.set_xlabel('Number of Epochs')
    trainErrPlot.set_ylabel('Cross-Entropy Error')
    trainErrPlot.set_title('Error vs Number of Epochs')
    trainErrPlot.scatter(plotx, ploty_train)
    valErrPlot = errfig.add_subplot(111)
    valErrPlot.scatter(plotx, ploty_val, c='r')
    testErrPlot = errfig.add_subplot(111)
    testErrPlot.scatter(plotx, ploty_test, c='g')
    errfig.savefig('exp5e_hidden'+str(hidden_layer_size)+'_error.png')
    print("============================================")

#END OF BIG FOR-LOOP

print('==> Generating FINAL TRAINING error plot...')
errfig = plt.figure()
hidden20Plot = errfig.add_subplot(111)
hidden20Plot.set_xlabel('Number of Epochs')
hidden20Plot.set_ylabel('Cross-Entropy Error')
hidden20Plot.set_title('Error vs Number of Epochs')
hidden20Plot.scatter(plotx, plot_train_all[0])
hidden40Plot = errfig.add_subplot(111)
hidden40Plot.scatter(plotx, plot_train_all[1], c='r')
hidden80Plot = errfig.add_subplot(111)
hidden80Plot.scatter(plotx, plot_train_all[2], c='g')
hidden160Plot = errfig.add_subplot(111)
hidden160Plot.scatter(plotx, plot_train_all[3], c='yellow')
hidden320Plot = errfig.add_subplot(111)
hidden320Plot.scatter(plotx, plot_train_all[4], c='magenta')
errfig.savefig('exp5e_final_train_error.png')

print('==> Generating FINAL VALIDATION error plot...')
errfig = plt.figure()
hidden20Plot = errfig.add_subplot(111)
hidden20Plot.set_xlabel('Number of Epochs')
hidden20Plot.set_ylabel('Cross-Entropy Error')
hidden20Plot.set_title('Error vs Number of Epochs')
hidden20Plot.scatter(plotx, plot_val_all[0])
hidden40Plot = errfig.add_subplot(111)
hidden40Plot.scatter(plotx, plot_val_all[1], c='r')
hidden80Plot = errfig.add_subplot(111)
hidden80Plot.scatter(plotx, plot_val_all[2], c='g')
hidden160Plot = errfig.add_subplot(111)
hidden160Plot.scatter(plotx, plot_val_all[3], c='yellow')
hidden320Plot = errfig.add_subplot(111)
hidden320Plot.scatter(plotx, plot_val_all[4], c='magenta')
errfig.savefig('exp5e_final_val_error.png')

print('==> Generating FINAL TEST error plot...')
errfig = plt.figure()
hidden20Plot = errfig.add_subplot(111)
hidden20Plot.set_xlabel('Number of Epochs')
hidden20Plot.set_ylabel('Cross-Entropy Error')
hidden20Plot.set_title('Error vs Number of Epochs')
hidden20Plot.scatter(plotx, plot_test_all[0])
hidden40Plot = errfig.add_subplot(111)
hidden40Plot.scatter(plotx, plot_test_all[1], c='r')
hidden80Plot = errfig.add_subplot(111)
hidden80Plot.scatter(plotx, plot_test_all[2], c='g')
hidden160Plot = errfig.add_subplot(111)
hidden160Plot.scatter(plotx, plot_test_all[3], c='yellow')
hidden320Plot = errfig.add_subplot(111)
hidden320Plot.scatter(plotx, plot_test_all[4], c='magenta')
errfig.savefig('exp5e_final_test_error.png')